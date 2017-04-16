# -*- coding: utf-8 -*-

"""
Varational Auto Encoder Example.
����Զ�������
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# ����ģ��

class Layer:
  def __init__(self, input, n_output):
    self.input = input
    W = tf.Variable(tf.truncated_normal([ int(self.input.get_shape()[1]), n_output ], stddev = 0.001))#tf.shape(input)[0]
    b = tf.Variable(tf.constant(0., shape = [ n_output ]))

    self.raw_output = tf.matmul(input, W) + b
    self.output = tf.nn.relu(self.raw_output)


# ������X
n_X = 784 # 28 * 28
n_z = 20 # latent variable count
X = tf.placeholder(tf.float32, shape = [ None, n_X ])

# Encoder���������

## \mu(X) ���ö�������
ENCODER_HIDDEN_COUNT = 400
mu = Layer(Layer(X, ENCODER_HIDDEN_COUNT).output, n_z).raw_output

## \Sigma(X) ���ö�������
log_sigma = Layer(Layer(X, ENCODER_HIDDEN_COUNT).output, n_z).raw_output # Ϊ��ѵ������nan? ����ʵ���ʱ��ֱ��������������sigma���㲻�����ģ������ָ��!!!
sigma = tf.exp(log_sigma)

## KLD = D[N(mu(X), sigma(X))||N(0, I)] = 1/2 * sum(sigma_i + mu_i^2 - log(sigma_i) - 1)
KLD = 0.5 * tf.reduce_sum(sigma + tf.pow(mu, 2) - log_sigma - 1, reduction_indices = 1) # reduction_indices = 1������ÿ����������һ��KLD


# epsilon = N(0, I) ����ģ��
epsilon = tf.random_normal(tf.shape(sigma), name = 'epsilon')

# z = mu + sigma^ 0.5 * epsilon
z = mu + tf.exp(0.5 * log_sigma) * epsilon

# Decoder��������̣� ||f(z) - X|| ^ 2 �ؽ���X��X��ŷʽ���룬���ӳ����������ʹ��crossentropy
def buildDecoderNetwork(z):
  # ����һ�����������磬��Ϊ������������Աƽ��κκ���
  DECODER_HIDDEN_COUNT = 400
  layer1 = Layer(z, DECODER_HIDDEN_COUNT)
  layer2 = Layer(layer1.output, n_X)
  return layer2.raw_output

reconstructed_X = buildDecoderNetwork(z)

reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(reconstructed_X, X), reduction_indices = 1)

loss = tf.reduce_mean(reconstruction_loss + KLD)

# minimize loss
n_steps = 100000
learning_rate = 0.01
batch_size = 100

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for step in xrange(1, n_steps):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([ optimizer, loss ], feed_dict = { X: batch_x })

    if step % 100 == 0:
      print('Step', step, ', Loss:', l)