import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=100, type=int)
parser.add_argument('-n', '--n_events', default=10**5, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr/longer_training')
parser.add_argument('-i', '--input_path', default='../../data/junipr/final_reclustered_practice.out')
parser.add_argument('-m', '--model_path', required=True)
parser.add_argument('-e', '--n_epochs', default=1, type=int)
parser.add_argument('-x', '--times', action='store_true')
args = vars(parser.parse_args())

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']
batch_size = args['batch_size']
n_epochs = args['n_epochs']
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

import tensorflow as tf
from utilities import load_data
import numpy as np
import time

# Load in jets from file

[daughters, endings, mothers, discrete_splittings,
    mother_momenta] = load_data(input_path, 
    n_events=n_events, batch_size=batch_size, split_p_q=(not times),
    p_granularity=p_granularity, q_granularity=q_granularity)

if not times:
  discrete_p_splittings, discrete_q_splittings = discrete_splittings

print('data loaded', flush=True)

# temporary hack having to do with mask values; this will change later.
for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] = 0
    if times:
      discrete_splittings[i][0][discrete_splittings[i][0]==q_granularity*p_granularity**4] = 0
    else:
      discrete_p_splittings[i][0][discrete_p_splittings[i][0]==p_granularity**4] = 0
      discrete_q_splittings[i][0][discrete_q_splittings[i][0]==q_granularity] = 0

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

# Train Model

with tf.keras.utils.CustomObjectScope({'activation_average': JUNIPR_functions.activation_average,
    'categorical_crossentropy2': JUNIPR_functions.categorical_crossentropy2}):
  model = tf.keras.models.load_model(model_path)
  print(model.summary(), flush=True)
  for lr in [1e-2, 1e-3, 1e-4]:
    print('Using learning rate ', lr, flush=True)
    if times:
      model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
          loss=['binary_crossentropy', JUNIPR_functions.categorical_crossentropy2, 'sparse_categorical_crossentropy'])
    else:
      model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
        loss=['binary_crossentropy', JUNIPR_functions.categorical_crossentropy2,
        'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'])
    for epoch in range(n_epochs):
      print("Epoch: ", epoch, flush=True)
      l = 0 
      for n in range(len(daughters)):
        start_time = time.time()
        if times:
          batch_loss = model.train_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]], y=[endings[n][0], mothers[n][0], discrete_splittings[n][0]])[0]
        else:
          batch_loss = model.train_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]], 
            y=[endings[n][0], mothers[n][0], np.ma.masked_array(discrete_p_splittings[n][0], mask =
               discrete_p_splittings[n][1]), discrete_q_splittings[n][0]])[0]
        print("Batch {}: {}, took {} seconds".format(n, batch_loss, time.time() - start_time), flush=True)
        l += batch_loss
        if n%100==0 and n>0:
          model.save(os.path.join(data_path, 'JUNIPR{}_LR{}_E{}_B{}'.format('_plus_q' if not times else '', lr, epoch, n)))
          if n%1000==0:
            print("Batch: ", n, l/1000, flush=True)
            l=0
      model.save(os.path.join(data_path, 'JUNIPR{}_LR{}_E{}'.format('_plus_q' if not times else '', lr, epoch)))
