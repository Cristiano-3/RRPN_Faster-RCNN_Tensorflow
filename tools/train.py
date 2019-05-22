# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
sys.path.append("../")
import time

from libs.networks import build_whole_network
from data.io.read_tfrecord import next_batch
from help_utils import tools
from libs.box_utils.coordinate_convert import back_forward_convert
from libs.box_utils.show_box_in_tensor import *

os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP

# 省略了 with tf.Graph().as_default()
def train():

    # get Network Class
    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=True)
    # get batch
    with tf.name_scope('get_batch'):
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                       is_training=True)

        # convert gtboxes_and_labels
        gtboxes_and_label = tf.py_func(back_forward_convert, inp=[tf.squeeze(gtboxes_and_label_batch, 0)], Tout=tf.float32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 6])

    # draw ground-truth
    with tf.name_scope('draw_gtboxes'):
        gtboxes_in_img = draw_box_with_color_rotate(img_batch, 
                                                    tf.reshape(gtboxes_and_label, [-1, 6])[:, :-1],
                                                    text=tf.shape(gtboxes_and_label)[0])

    # default regularizers
    biases_regularizer = tf.no_regularizer
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY)

    # list as many types of layers as possible, even if they are not used now
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                         slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                         weights_regularizer=weights_regularizer,
                         biases_regularizer=biases_regularizer,
                         biases_initializer=tf.constant_initializer(0.0)):

        # build network                 
        final_boxes, final_scores, final_category, loss_dict = \
            faster_rcnn.build_whole_detection_network(input_img_batch=img_batch,
                                                      gtboxes_batch=gtboxes_and_label)

    # draw detections
    dets_in_img = draw_boxes_with_categories_and_scores_rotate(img_batch=img_batch,
                                                               boxes=final_boxes,
                                                               labels=final_category,
                                                               scores=final_scores)

    # ----------------------------------------------------------------------------------------------------build loss
    # weight decay loss
    weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())

    # rpn losses
    rpn_location_loss = loss_dict['rpn_loc_loss']
    rpn_cls_loss = loss_dict['rpn_cls_loss']
    rpn_total_loss = rpn_location_loss + rpn_cls_loss

    # fastrcnn losses
    fastrcnn_cls_loss = loss_dict['fastrcnn_cls_loss']
    fastrcnn_loc_loss = loss_dict['fastrcnn_loc_loss']
    fastrcnn_total_loss = fastrcnn_cls_loss + fastrcnn_loc_loss

    # total loss
    total_loss = rpn_total_loss + fastrcnn_total_loss + weight_decay_loss
    # ____________________________________________________________________________________________________build loss

    # ---------------------------------------------------------------------------------------------------add summary
    tf.summary.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
    tf.summary.scalar('RPN_LOSS/location_loss', rpn_location_loss)
    tf.summary.scalar('RPN_LOSS/rpn_total_loss', rpn_total_loss)

    tf.summary.scalar('FAST_LOSS/fastrcnn_cls_loss', fastrcnn_cls_loss)
    tf.summary.scalar('FAST_LOSS/fastrcnn_location_loss', fastrcnn_loc_loss)
    tf.summary.scalar('FAST_LOSS/fastrcnn_total_loss', fastrcnn_total_loss)

    tf.summary.scalar('LOSS/total_loss', total_loss)
    tf.summary.scalar('LOSS/regular_weights', weight_decay_loss)

    tf.summary.image('img/gtboxes', gtboxes_in_img)
    tf.summary.image('img/dets', dets_in_img)

    # ___________________________________________________________________________________________________add summary
    # create global step
    global_step = slim.get_or_create_global_step()

    # learning rate setting
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    tf.summary.scalar('lr', lr)

    # optimizer with lr
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)

    # ---------------------------------------------------------------------------------------------compute gradients
    # compute gradients
    gradients = faster_rcnn.get_gradients(optimizer, total_loss)

    # enlarge gradients for bias
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = faster_rcnn.enlarge_gradients_for_bias(gradients)

    # clipping gradients
    if cfgs.GRADIENT_CLIPPING_BY_NORM:
        with tf.name_scope('clip_gradients'):
            gradients = slim.learning.clip_gradient_norms(gradients,
                                                          cfgs.GRADIENT_CLIPPING_BY_NORM)
    # _____________________________________________________________________________________________compute gradients

    # train_op, ie. apply gradients
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)

    # summary_op                                         
    summary_op = tf.summary.merge_all()

    # init_op
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    # restorer and Saver
    restorer, restore_ckpt = faster_rcnn.get_restorer()
    saver = tf.train.Saver(max_to_keep=10)

    # GPU Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # start a session
    with tf.Session(config=config) as sess:

        # init
        sess.run(init_op)

        # restore
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        # Session对象是支持多线程的, 可以在同一个会话（Session）中创建多个线程，并行执行;
        # 创建线程协调器(管理器), 并启动(Tensor/训练数据)入队线程,
        # 入队具体使用多少个线程在read_tfrecord.py->next_batch->tf.train.batch中定义;
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # construct summary writer
        summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
        tools.mkdir(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

        # train MAX_ITERATION steps
        for step in range(cfgs.MAX_ITERATION):

            # time of this step
            training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            # no show & no summary
            if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:

                # global_step is added to run, global_stepnp will be used in Save
                _, global_stepnp = sess.run([train_op, global_step])

            else:
                # show 
                if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                    
                    start = time.time()

                    # middle infos are added to run, losses and img_name_batch
                    _global_step, _img_name_batch, _rpn_location_loss, _rpn_classification_loss, \
                    _rpn_total_loss, _fast_rcnn_location_loss, _fast_rcnn_classification_loss, \
                    _fast_rcnn_total_loss, _total_loss, _ = \
                        sess.run([global_step, img_name_batch, rpn_location_loss, rpn_cls_loss,
                                  rpn_total_loss, fastrcnn_loc_loss, fastrcnn_cls_loss,
                                  fastrcnn_total_loss, total_loss, train_op])

                    # # show boxes, scores, category infos
                    # final_boxes_r, _final_scores_r, _final_category_r = sess.run([final_boxes_r, final_scores_r, final_category_r])
                    # print('*'*100)
                    # print(_final_boxes_r)
                    # print(_final_scores_r)
                    # print(_final_category_r)

                    end = time.time()

                    # print
                    print(""" {}: step {}: image_name:{} |\t
                                                    rpn_loc_loss:{} |\t rpn_cla_loss:{} |\t
                                                    rpn_total_loss:{} |
                                                    fast_rcnn_loc_loss:{} |\t fast_rcnn_cla_loss:{} |\t
                                                    fast_rcnn_total_loss:{} |\t
                                                    total_loss:{} |\t pre_cost_time:{}s""" \
                          .format(training_time, _global_step, str(_img_name_batch[0]), 
                                _rpn_location_loss, _rpn_classification_loss, _rpn_total_loss, 
                                _fast_rcnn_location_loss, _fast_rcnn_classification_loss, _fast_rcnn_total_loss, 
                                _total_loss, (end - start)))

                # summary
                else:
                    if step % cfgs.SMRY_ITER == 0:
                        # summary_op is added to run
                        _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])

                        # add summary
                        summary_writer.add_summary(summary_str, global_stepnp)
                        summary_writer.flush()

            # save
            if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or (step == cfgs.MAX_ITERATION - 1):

                # mkdir
                save_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # save checkpoint
                save_ckpt = os.path.join(save_dir, 'voc_' + str(global_stepnp) + 'model.ckpt')
                saver.save(sess, save_ckpt)
                print(' weights had been saved')


        # 协调器发出终止所有线程的命令
        coord.request_stop()
        # 把开启的线程加入主线程，等待threads结束
        coord.join(threads)


if __name__ == '__main__':

    train()
