import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=5, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=100, type=int)
parser.add_argument('-n', '--n_events', default=10**4, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr')
parser.add_argument('-i', '--input_path', default='../../data/junipr/final_reclustered_practice.out')
parser.add_argument('-m', '--model_path', default='../../data/junipr/JUNIPR_p10_q5_LR0.0001_E0_B800')
parser.add_argument('-x', '--times', action='store_false')
args = vars(parser.parse_args())

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']
batch_size = args['batch_size']

n_events = args['n_events']
n_batches = n_events//batch_size

# path to where to save data
data_path = args['data_path']

# path to input jets file
input_path = args['input_path']

# path to saved model
model_path = args['model_path']
model_basename = os.path.splitext(os.path.basename(model_path))[0]

# boolean indicating if we are in the p times q or p plus q framework
times = args['times']

print('JUNIPR started', flush=True)

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from utilities import load_data
import numpy as np
from matplotlib import pyplot as plt
import encoding
import sys

print('done importing', flush=True)

# Load in jets from file

[daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(input_path, 
    n_events=n_events, batch_size=batch_size, split_p_q=(not times),
    p_granularity=p_granularity, q_granularity=q_granularity)

if not times:
  discrete_p_splittings, discrete_q_splittings = discrete_splittings

print('data loaded', flush=True)

# temporary hack having to do with mask values; this will change later.
for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] =0

class JUNIPR_functions():
  def activation_average(x):
      total = tf.keras.backend.clip(tf.keras.backend.sum(x, axis=-1, keepdims=True), tf.keras.backend.epsilon(), 1)
      return x/total

  def categorical_crossentropy2(target, output):
      sums = tf.keras.backend.sum(output, axis=-1, keepdims=True)
      sums = tf.keras.backend.clip(sums, tf.keras.backend.epsilon(), 1)
      output = output/sums
      output = tf.keras.backend.clip(output, tf.keras.backend.epsilon(), 1)
      length = tf.keras.backend.sum(tf.keras.backend.ones_like(target[0,:,0]))
      return -tf.keras.backend.sum(target*tf.keras.backend.log(output), axis=-1)*length/tf.keras.backend.sum(target)

with tf.keras.utils.CustomObjectScope({'activation_average': JUNIPR_functions.activation_average,
    'categorical_crossentropy2': JUNIPR_functions.categorical_crossentropy2}):
  model = tf.keras.models.load_model(model_path)
  print(model.summary(), flush=True)

# Validate Model

  charge_out     = np.zeros((2, 100, q_granularity))

  if times:
    for n in range(len(daughters)):
      e, m, b = model.predict_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]])
      I, T = e.shape[:2]
      for i in range(I):
        for t in range(T):
          if discrete_splittings[n][1][i,t]==False:
            pxq = b[i,t].reshape((p_granularity**4, q_granularity))
            charge_out[0,t]+=np.mean(pxq, axis=0)
            charge_out[1,t]+=np.ones(q_granularity)
