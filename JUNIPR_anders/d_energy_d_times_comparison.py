import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-n', '--n_events', default=10**5, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr/u_d_0.03_energy_discrimination')
parser.add_argument('-i', '--input_path', default='../../data/junipr/final_reclustered_d_jets_0.03.out')
parser.add_argument('--times_model', default='../../data/junipr/d_training_0.03_energy/p10_q21/JUNIPR_p10_q21_LR1e-05_E0')
parser.add_argument('--plus_model', default='../../data/junipr/u_training_0.03_energy/p10_q21/JUNIPR_p10_q21_LR1e-05_E0')
args = vars(parser.parse_args())

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']
batch_size = args['batch_size']

n_events = args['n_events']

# path to where to save data
data_path = args['data_path']
if not os.path.exists(data_path):
  os.makedirs(data_path)

# path to input jets file
input_path = args['input_path']

# path to models
times_model_path = args['times_model']
plus_model_path = args['plus_model']

import JUNIPR_class
from utilities import load_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

times_log_probs = [[0] * 100, [0] * 100]
plus_log_probs = [[0] * 100, [0] * 100]

# loading each of the trained models
times_model = JUNIPR_class.JUNIPR_times(p_granularity, q_granularity, model_path=times_model_path).model
plus_model = JUNIPR_class.JUNIPR_plus(p_granularity, q_granularity, model_path=plus_model_path).model

for model, split, log_probs in [(plus_model, True, plus_log_probs), (times_model, False, times_log_probs)]:
  [daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(input_path, 
      n_events=n_events, batch_size=batch_size, split_p_q=split,
      p_granularity=p_granularity, q_granularity=q_granularity)
  if split:
    discrete_p_splittings, discrete_q_splittings = discrete_splittings

  for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] = 0


  if split:
    for n in range(len(daughters)):
      e, m, b, q = model.predict_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]])
      I, T = e.shape[:2]
      for i in range(I):
        for t in range(T):
          if endings[n][1][i,t][0] == False:
            log_probs[1][t] += 1
            if endings[n][0][i,t][0] == 0:
              log_probs[0][t] += np.log10(1 - e[i,t][0])
            else:
              log_probs[0][t] += np.log10(e[i,t][0])
          for j in range(100):
            if mothers[n][1][i,t][j] == True:
              mother_index = np.nonzero(mothers[n][0][i,t])[0][0]
              log_probs[0][t] += np.log10(m[i,t,mother_index])
              log_probs[1][t] += 1
          if discrete_p_splittings[n][1][i,t]==False:
            log_probs[0][t] += (np.log10(b[i,t][discrete_p_splittings[n][0][i,t]]) +
                np.log10(q[i,t][discrete_q_splittings[n][0][i,t]]))
            log_probs[1][t] += 1
  else:
    for n in range(len(daughters)):
      e, m, b = model.predict_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]])
      I, T = e.shape[:2]
      for i in range(I):
        for t in range(T):
          if endings[n][1][i,t][0] == False:
            log_probs[1][t] += 1
            if endings[n][0][i,t][0] == 0:
              log_probs[0][t] += np.log10(1 - e[i,t][0])
            else:
              log_probs[0][t] += np.log10(e[i,t][0])
          for j in range(100):
            if mothers[n][1][i,t][j] == True:
              mother_index = np.nonzero(mothers[n][0][i,t])[0][0]
              log_probs[0][t] += np.log10(m[i,t,mother_index])
              log_probs[1][t] += 1
          if discrete_splittings[n][1][i,t]==False:
            log_probs[0][t] += np.log10(b[i,t][discrete_splittings[n][0][i,t]])
            log_probs[1][t] += 1

# plot log_probs
def compute_masked_avg(a):
 # taking the average of both subarrays of a (a[i][0] contains values, a[i][1]
 # contains counts), differencing, and then masking where there were no counts
  return np.ma.array((np.array(a[0]) / np.clip(np.array(a[1]), 0.1,
    np.inf)), mask=(a[1] == 0))

plt.plot(compute_masked_avg(times_log_probs), label='times')
plt.plot(compute_masked_avg(plus_log_probs), label='plus')
plt.legend()
plt.title('log probability vs. timestep')
plt.ylabel('log probability')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'times_plus_comparison.png'))
plt.close()
