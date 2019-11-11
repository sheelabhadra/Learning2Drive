# MIT License
# https://github.com/hardmaru/WorldModelsExperiments/tree/master/carracing
# Original author: hardmaru
# Edited by Roma Sokolkov and Antonin Raffin
# VAE model

import os

import cloudpickle
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def conv_to_fc(input_tensor):
    """
    Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network

    :param input_tensor: (TensorFlow Tensor) The convolutional input tensor
    :return: (TensorFlow Tensor) The fully connected output tensor
    """
    n_hidden = np.prod([v.value for v in input_tensor.get_shape()[1:]])
    input_tensor = tf.reshape(input_tensor, [-1, n_hidden])
    return input_tensor


class ConvVAE(object):
    """
    VAE model.

    :param z_size: (int) latent space dimension
    :param batch_size: (int)
    :param learning_rate: (float)
    :param kl_tolerance: (float)
    :param is_training: (bool)
    :param beta: (float) weight for KL loss
    :param reuse: (bool)
    """

    def __init__(self, z_size=512, batch_size=100, learning_rate=0.0001,
                 kl_tolerance=0.5, is_training=True, beta=1.0, reuse=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.kl_tolerance = kl_tolerance
        self.beta = beta
        self.reuse = reuse
        self.graph = None
        self.input_tensor = None
        self.output_tensor = None

        with tf.variable_scope('conv_vae', reuse=self.reuse):
            self._build_graph()

        with self.graph.as_default():
            self.params = tf.trainable_variables()

        self._init_session()

    def _build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, shape=[None, 80, 160, 3])

            # Encoder
            h = tf.layers.conv2d(self.input_tensor, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
            h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
            # h = tf.reshape(h, [-1, 3 * 8 * 256])
            h = conv_to_fc(h)

            # VAE
            self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
            self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random_normal([self.batch_size, self.z_size])
            # self.epsilon = tf.random_normal([None, self.z_size])
            # self.z = self.mu + self.sigma * self.epsilon
            if self.is_training:
                self.z = self.mu + self.sigma * self.epsilon
            else:
                self.z = self.mu

            # Decoder
            h = tf.layers.dense(self.z, 3 * 8 * 256, name="dec_fc")
            h = tf.reshape(h, [-1, 3, 8, 256])
            h = tf.layers.conv2d_transpose(h, 128, 4, strides=2, activation=tf.nn.relu, name="dec_deconv1")
            h = tf.layers.conv2d_transpose(h, 64, 4, strides=2, activation=tf.nn.relu, name="dec_deconv2")
            h = tf.layers.conv2d_transpose(h, 32, 5, strides=2, activation=tf.nn.relu, name="dec_deconv3")
            self.output_tensor = tf.layers.conv2d_transpose(h, 3, 4, strides=2, activation=tf.nn.sigmoid,
                                                            name="dec_deconv4")

            # train ops
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                # reconstruction loss
                self.r_loss = tf.reduce_sum(
                    tf.square(self.input_tensor - self.output_tensor),
                    reduction_indices=[1, 2, 3]
                )
                self.r_loss = tf.reduce_mean(self.r_loss)

                # augmented kl loss per dim
                self.kl_loss = - 0.5 * tf.reduce_sum(
                    (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
                    reduction_indices=1
                )
                if self.kl_tolerance > 0:
                    self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                self.kl_loss = tf.reduce_mean(self.kl_loss)

                self.loss = self.r_loss + self.beta * self.kl_loss

                # training
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.loss)  # can potentially clip gradients here.

                self.train_op = self.optimizer.apply_gradients(
                    grads, global_step=self.global_step, name='train_step')

            # initialize vars
            self.init = tf.global_variables_initializer()

    def _init_session(self):
        """Launch tensorflow session and initialize variables"""
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.init)

    def close_sess(self):
        """ Close tensorflow session """
        self.sess.close()

    def encode(self, input_tensor):
        """
        :param input_tensor: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.sess.run(self.z, feed_dict={self.input_tensor: input_tensor})

    def decode(self, z):
        """
        :param z: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.sess.run(self.output_tensor, feed_dict={self.z: z})

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def set_params(self, params):
        assign_ops = []
        for param, loaded_p in zip(self.params, params):
            assign_ops.append(param.assign(loaded_p))
        self.sess.run(assign_ops)

    def get_params(self):
        return self.sess.run(self.params)

    def set_model_params(self, params):
        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                pshape = self.sess.run(var).shape
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op = var.assign(p.astype(np.float) / 10000.)
                self.sess.run(assign_op)
                idx += 1

    def save_checkpoint(self, model_save_path):
        sess = self.sess
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'vae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0)  # just keep one

    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    @staticmethod
    def _save_to_file(save_path, data=None, params=None):
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"

            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params), save_path)

    def save(self, save_path):
        """
        Save to a pickle file.

        :param save_path: (str)
        """
        data = {
            "z_size": self.z_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "is_training": self.is_training,
            "kl_tolerance": self.kl_tolerance
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @staticmethod
    def _load_from_file(load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file:
                data, params = cloudpickle.load(file)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    @classmethod
    def load(cls, load_path, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(data['z_size'], data['batch_size'],
                    data['learning_rate'], data['kl_tolerance'],
                    data['is_training'])
        model.__dict__.update(data)
        model.__dict__.update(kwargs)

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
