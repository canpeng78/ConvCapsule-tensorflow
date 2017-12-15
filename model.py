"""
Copyright (C) 2017 Can Peng
License: MIT Software License
Author: Can Peng (can.peng78@gmail.com)
"""

from __future__ import division
import os
import time
from capsLayers import CapsuleConv2d
from scipy.misc import imsave
from ops import *
from utils import *


class Caps2Layers(object):
    """
    The simple 2-capsule-layer network in the paper 'Dynamic Routing Between Capsules', Sara Sabour, Nicholas Frosst and
    Geoffrey Hinton, 2017
    arXiv:1710.09829v2 [cs.CV] 7 Nov 2017
    """

    def __init__(self, sess, input_height=28, input_width=28, input_channel=1,
                 primary_capsize=8, primary_cap_channels=32,
                 cat_capsize=16, cat_cap_num=10,
                 batch_size=256, weight_decay=0.0001,
                 model_name=None):
        self.tf_session = sess
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.prim_capsize = primary_capsize
        self.prim_capchan = primary_cap_channels
        self.cat_capsize = cat_capsize
        self.cat_cap_num = cat_cap_num
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.primcaps = CapsuleConv2d(self.prim_capsize, name="PrimCaps")
        self.catcaps = CapsuleConv2d(self.cat_capsize, routing="Dynamic", name="CatCaps")

        self._build_model()

    def _predictor(self, input):
        eps = 1e-20
        conv = leaky_relu(conv2d_layer(input, [9, 9, 1, 256], scope='convlayer', padding='VALID'), leak=0.0)
        primcaps = self.primcaps(conv, self.prim_capchan, [9, 9, 256], [2, 2])
        self.prcap_sum = hist_summary("primary_caps", primcaps)
        pcshape = primcaps.get_shape().as_list()
        catcaps = self.catcaps(primcaps, self.cat_cap_num,
                               [pcshape[1], pcshape[2], self.prim_capchan], [1, 1])
        catcaps = tf.reshape(catcaps, shape=[self.batch_size, self.cat_capsize, self.cat_cap_num])
        self.catcap_sum = hist_summary("out_caps", catcaps)
        out = tf.sqrt(tf.reduce_sum(tf.square(catcaps), axis=1) + eps)  # in shape [batch_size, self.cat_cap_num]
        outsum = tf.reduce_sum(out, axis=1, keep_dims=True)
        out /= outsum
        return out, catcaps

    def _decoder(self, catcaps):
        catcapshape = catcaps.get_shape().as_list()
        masked = tf.expand_dims(self.label, axis=1) * catcaps
        masked = tf.reshape(masked, shape=[-1, catcapshape[1]*catcapshape[2]])
        fc1 = leaky_relu(linear(masked, 512, scope="FC1"), leak=0.0)
        fc2 = leaky_relu(linear(fc1, 1024, scope="FC2"), leak=0.0)
        fc3 = linear(fc2, self.input_height*self.input_width, scope="FC3")
        fc3 = tf.sigmoid(fc3)
        return tf.reshape(fc3, shape=[-1, self.input_height, self.input_width, 1])

    @staticmethod
    def _xentropy(predict, label):
        xentrpy = -label * tf.log(predict)
        return tf.reduce_mean(tf.reduce_sum(xentrpy, [1]))

    @staticmethod
    def _marginloss(catcaps, label):
        eps = 1e-10
        vlen = tf.sqrt(tf.reduce_sum(tf.square(catcaps), axis=1) + eps)
        loss = label * tf.square(tf.maximum(0.0, 0.9 - vlen)) + 0.5 * (1 - label) * tf.square(tf.maximum(0.0, vlen - 0.1))
        return tf.reduce_mean(tf.reduce_sum(loss, [1]))

    def _build_model(self):
        self.input = tf.placeholder(tf.float32,
                                    [self.batch_size, self.input_height, self.input_width, self.input_channel],
                                    name='input')
        self.label = tf.placeholder(tf.float32, [self.batch_size, self.cat_cap_num], name='label')
        self.pred, self.ctcaps = self._predictor(self.input)
        self.xentrpy_loss = self._xentropy(self.pred, self.label)
        self.xentrpy_sum = scl_summary("xentropy_loss", self.xentrpy_loss)
        self.marginloss = self._marginloss(self.ctcaps, self.label)
        self.mrgnloss_sum = scl_summary("margin_loss", self.marginloss)
        self.recon = self._decoder(self.ctcaps)
        
        self.recon_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.recon - self.input), axis=[-1, -2, -3])))
        self.reconloss_sum = scl_summary("recon_loss", self.recon_loss)

        self.p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        p_w = filter(lambda x: x.name.endswith('w:0'), self.p_vars)
        self.loss = self.marginloss + self.weight_decay * tf.reduce_mean(
            tf.pack(map(lambda x: tf.nn.l2_loss(x), p_w))) + 0.005 * self.recon_loss
        self.totloss_sum = scl_summary("total_loss", self.loss)
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.label, 1))
        self.numcorrect = tf.reduce_sum(tf.cast(correct_pred, tf.int32), axis=0)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.saver = tf.train.Saver(max_to_keep=5)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(self.model_name, self.prim_capsize, self.prim_capchan, self.cat_capsize,
                                       self.cat_cap_num)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            init_op = tf.global_variables_initializer()
            self.tf_session.run(init_op)
            self.saver.restore(self.tf_session, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, checkpoint_dir, step):
        model_name = "caps2layer.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.tf_session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config, data_dir):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_steps, config.decay_rate,
                                                   staircase=True)
        optim = tf.train.AdamOptimizer(learning_rate, beta1=config.beta1).minimize(self.loss, var_list=self.p_vars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N, global_step=global_step)
                                       
        self.summaries = merge_all(key=tf.GraphKeys.SUMMARIES)
        self.writer = SummaryWriter(os.path.join(config.log_dir, self.model_dir), self.tf_session.graph)

        start_time = time.time()
        could_load, checkpoint_counter = self.load(config.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            global_step.assign(counter).op.run()
            print("[*] Load SUCCESS")
        else:
            counter = 0
            print("Restart ...")
            tf.global_variables_initializer().run()

        trX, trY = load_mnist(data_dir)
        teX, teY = load_mnist(data_dir, is_train=False)

        for epoch in xrange(config.epoch):
            num_batches = trX.shape[0] // self.batch_size
            seed = np.random.randint(1000)
            np.random.seed(seed)
            np.random.shuffle(trX)
            np.random.seed(seed)
            np.random.shuffle(trY)

            for idx in xrange(0, num_batches):
                data = trX[idx * self.batch_size:(idx + 1) * self.batch_size, :]
                label = trY[idx * self.batch_size:(idx + 1) * self.batch_size, :]
                _, loss, xentropy_loss, recon_loss, summaries = self.tf_session.run([optim, self.loss, self.xentrpy_loss, self.recon_loss, self.summaries],
                                                             feed_dict={self.input: data,
                                                                        self.label: label})
                self.writer.add_summary(summaries, counter)
                counter += 1
                if counter % 100 == 0:
                    print("Epoch: [%d], Batch: [%4d/%4d], Elapsed time: %4.4f, Loss: %.8f, Xentropy_loss: %.8f, recon_loss: %.8f" %
                          (epoch, idx, num_batches, time.time() - start_time, loss, xentropy_loss, recon_loss))
                if counter % 500 == 0:
                    self.validate(teX, teY)
                    self.save(config.checkpoint_dir, counter)

    def validate(self, teX, teY):
        seed = np.random.randint(1000)
        np.random.seed(seed)
        np.random.shuffle(teX)
        np.random.seed(seed)
        np.random.shuffle(teY)

        output, xentropy_loss, correct, accu, recon_loss = self.tf_session.run([self.pred, self.xentrpy_loss, self.numcorrect, self.accuracy, self.recon_loss],
                                                          feed_dict={self.input: teX[0:self.batch_size, :],
                                                                     self.label: teY[0:self.batch_size, :]})
        print("Xentropy_loss: %.8f, correct prediction: %4d, accuracy: %.8f, recon_loss: %.8f" % (xentropy_loss, correct, accu, recon_loss))

    def test(self, config, data_dir, path):
        could_load, checkpoint_counter = self.load(config.checkpoint_dir)
        if could_load:
            print("[*] Load SUCCESS")
        else:
            print("Missing model!")
            pass
        teX, teY = load_mnist(data_dir, is_train=False)
        #seed = np.random.randint(1000)
        seed = 13
        np.random.seed(seed)
        np.random.shuffle(teX)
        np.random.seed(seed)
        np.random.shuffle(teY)

        pred, recon, correct, accu, recon_loss = self.tf_session.run([self.pred, self.recon, self.numcorrect, self.accuracy, self.recon_loss],
                                                               feed_dict={self.input: teX[0:self.batch_size, :],
                                                                          self.label: teY[0:self.batch_size, :]})
        print("correct prediction: %4d, accuracy: %.8f, recon_loss: %.8f" % (correct, accu, recon_loss))

        frame_rows = int(np.floor(np.sqrt(self.batch_size)))
        frame_cols = int(np.ceil(self.batch_size / frame_rows))
        
        savepath = os.path.join(config.log_dir, path)
        imgarray = merge(recon*255, [frame_rows, frame_cols])
        imsave(savepath, imgarray)

class Caps3Layers(Caps2Layers):
    """
    A test model for testing convolutional capsule layers
    Insert a hidden convolutional capsule layer
    """

    def __init__(self, sess, input_height=28, input_width=28, input_channel=1,
                 primary_capsize=8, primary_cap_channels=32,
                 hidden_capsize=8, hidden_cap_channel=64,
                 cat_capsize=16, cat_cap_num=10,
                 batch_size=64, weight_decay=0.0001,
                 model_name=None):
        self.hidden_capsize = hidden_capsize
        self.hidden_capchan = hidden_cap_channel
        self.hiddencaps = CapsuleConv2d(self.hidden_capsize, routing="Dynamic", name="HiddenCaps")
        super(Caps3Layers, self).__init__(sess, input_height, input_width, input_channel,
                                          primary_capsize, primary_cap_channels,
                                          cat_capsize, cat_cap_num,
                                          batch_size, weight_decay, model_name)

    def _predictor(self, input):
        eps = 1e-20
        conv = leaky_relu(conv2d_layer(input, [9, 9, 1, 256], scope='convlayer', padding='VALID'), leak=0.0)
        prcaps = self.primcaps(conv, self.prim_capchan, [9, 9, 256], [2, 2], stddev=0.1)
        self.prcap_sum = hist_summary("primary_caps", prcaps)
        hidcaps = self.hiddencaps(prcaps, self.hidden_capchan, [2, 2, self.prim_capchan], [1, 1], stddev=0.1)
        self.hidcap_sum = hist_summary("hidden_caps", hidcaps)
        hcshape = hidcaps.get_shape().as_list()
        catcaps = self.catcaps(hidcaps, self.cat_cap_num,
                               [hcshape[1], hcshape[2], self.hidden_capchan], [1, 1], stddev=0.1)
        catcaps = tf.reshape(catcaps, shape=[self.batch_size, self.cat_capsize, self.cat_cap_num])
        self.catcap_sum = hist_summary("out_caps", catcaps)
        out = tf.sqrt(tf.reduce_sum(tf.square(catcaps), axis=1) + eps)  # in shape [batch_size, self.cat_cap_num]
        outsum = tf.reduce_sum(out, axis=1, keep_dims=True)
        out /= outsum
        return out, catcaps



