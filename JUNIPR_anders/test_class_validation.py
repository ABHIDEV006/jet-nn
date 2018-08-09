import argparse
import os
import tensorflow as tf
K = tf.keras.backend

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=100, type=int)
parser.add_argument('-n', '--n_events', default=10**4, type=int)
parser.add_argument('-d', '--data_path',
    default='../../data/junipr/u_training/p10_q21')
parser.add_argument('-i', '--input_path',
    default='../../data/junipr/final_reclustered_u_jets.out')
parser.add_argument('-m', '--model_path',
    default='../../data/junipr/u_training/p10_q21/JUNIPR_p10_q21_LR0.0001_E0')
parser.add_argument('-x', '--times', action='store_true')
args = vars(parser.parse_args())

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']
batch_size = args['batch_size']

n_events = args['n_events']

# path to where to save data
data_path = args['data_path']

# path to input jets file
input_path = args['input_path']

# path to the model
model_path = args['model_path']

# boolean indicating if we are in the p times q or p plus q framework
times = args['times']
import JUNIPR_class
from utilities import load_data

def compile_jets(data_path, n_events, p_granularity, q_granularity, batch_size,
    split_p_q):
  # Load in jets from file
  [daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(data_path, 
      n_events=n_events, batch_size=batch_size, split_p_q=split_p_q,
      p_granularity=p_granularity, q_granularity=q_granularity)
  if split_p_q:
    discrete_p_splittings, discrete_q_splittings = discrete_splittings

  # temporary hack having to do with mask values; this will change later.
  if split_p_q:
    for i in range(len(mothers)):
      mothers[i][0][mothers[i][0]==-1] = 0
      discrete_p_splittings[i][0][discrete_p_splittings[i][0]==p_granularity**4] = 0
      discrete_q_splittings[i][0][discrete_q_splittings[i][0]==q_granularity] = 0
  else:
    for i in range(len(mothers)):
      mothers[i][0][mothers[i][0]==-1] = 0
      discrete_splittings[i][0][discrete_splittings[i][0]==q_granularity*p_granularity**4] = 0

  # this unpacking is necessary to remove it from the tuple and put it into a
  # list
  x = [[*a] for a in zip(daughters, mother_momenta, [m[1] for m in mothers])]
  if split_p_q:
    y = [[*a] for a in zip([e[0] for e in endings], [m[0] for m in mothers], [d[0] for d in
        discrete_p_splittings], [q[0] for q in discrete_q_splittings])]
  else:
    y = [[*a] for a in zip([e[0] for e in endings], [m[0] for m in mothers], [d[0] for d in
        discrete_splittings])]
  return x, y


[daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(input_path, 
        n_events=n_events, batch_size=batch_size, split_p_q=(not times),
        p_granularity=p_granularity, q_granularity=q_granularity)

x = [[*a] for a in zip(daughters, mother_momenta, [m[1] for m in mothers])]
#x = list(zip(daughters, mother_momenta, [m[1] for m in mothers]))

if not times:
  model = JUNIPR_class.JUNIPR_plus(model_path=model_path,
      p_granularity=p_granularity, q_granularity=q_granularity)
  (discrete_p_splittings, discrete_q_splittings) = discrete_splittings
  model.validate(data_path, x, endings, mothers, discrete_p_splittings,
      discrete_q_splittings)
else:
  model = JUNIPR_class.JUNIPR_times(model_path=model_path,
      p_granularity=p_granularity, q_granularity=q_granularity)
  model.validate(data_path, x, endings, mothers, discrete_splittings)
