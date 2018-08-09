import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-n', '--n_events', default=10**5, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr')
parser.add_argument('--down_path', default='../../data/junipr/final_reclustered_d_jets.out')
parser.add_argument('--up_path', default='../../data/junipr/final_reclustered_u_jets.out')
parser.add_argument('--down_model',
                default='../../data/junipr/d_training/p10_q21/JUNIPR_p10_q21_LR0.0001_E0')
parser.add_argument('--up_model', default='../../data/junipr/u_training/p10_q21/JUNIPR_p10_q21_LR0.0001_E0')
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
down_path = args['down_path']
up_path = args['up_path']

# path to models
down_model_path = args['down_model']
up_model_path = args['up_model']

# boolean indicating if we are in the p times q or p plus q framework
times = args['times']
import JUNIPR_class
from utilities import load_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys

# the first array is for the up model, the second for the down. In the first
# entry of each subarray is the sum of the log probabilities over timesteps. In
# the second entry of each subarray is the number of jets which reach at least
# that timestep (we take a[0] / np.clip(a[1], 0.1, np.inf) to get the average
# likelihood at that timestep).
# 100 is the max length of a jet
zeros = [0] * 100
u_e_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]
u_m_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]
u_b_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]
u_q_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]
d_e_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]
d_m_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]
d_b_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]
d_q_log_probs = [[zeros[:], zeros[:]], [zeros[:], zeros[:]]]

# loading each of the trained models
u_model = JUNIPR_class.JUNIPR_energy(p_granularity, q_granularity, model_path=up_model_path).model
d_model = JUNIPR_class.JUNIPR_energy(p_granularity, q_granularity, model_path=down_model_path).model

for path, log_probs in [(up_path, [u_e_log_probs, u_m_log_probs, u_b_log_probs, u_q_log_probs]), (down_path, [d_e_log_probs, d_m_log_probs, d_b_log_probs, d_q_log_probs])]:
  [daughters, endings, mothers, (discrete_p_splittings, discrete_q_splittings), mother_momenta] = load_data(path, 
      n_events=n_events, batch_size=batch_size, split_p_q=True,
      p_granularity=p_granularity, q_granularity=q_granularity)

  for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] = 0

  n_timesteps = [None] * len(daughters)

  for n in range(len(daughters)):
    for i_m, model in enumerate([u_model, d_model]):
      e, m, b, q = model.predict_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]])
      I, T = e.shape[:2]
      if i_m == 0:
        n_timesteps[n] = T
      for i in range(I):
        for t in range(T):
          if endings[n][1][i,t][0] == False:
            log_probs[0][i_m][1][t] += 1
            if endings[n][0][i,t][0] == 0:
              log_probs[0][i_m][0][t] += np.log10(1-e[i,t][0])
            else:
              log_probs[0][i_m][0][t] += np.log10(e[i,t][0])
          mother_index = np.nonzero(mothers[n][0][i,t])[0]
          if len(mother_index) != 0:
            log_probs[1][i_m][0][t] += np.log10(m[i,t,mother_index[0]])
            log_probs[1][i_m][1][t] += 1
          # this also means that discrete_q_splittings[n][1][i,t] == False
          if discrete_p_splittings[n][1][i,t]==False:
            log_probs[2][i_m][0][t] += np.log10(b[i,t][discrete_p_splittings[n][0][i,t]])
            log_probs[2][i_m][1][t] += 1
            log_probs[3][i_m][0][t] += np.log10(q[i,t][discrete_q_splittings[n][0][i,t]])
            log_probs[3][i_m][1][t] += 1
  edges = np.linspace(0, 99, num=100)
  plt.hist(n_timesteps, bins=edges, histtype='step')

plt.savefig(os.path.join(data_path, 'timestep_histogram.png'))
plt.close()

def compute_masked_diff_avg(a):
 # taking the average of both subarrays of a (a[i][0] contains values, a[i][1]
 # contains counts), differencing, and then masking where there were no counts
  return (np.array(a[0][0]) / np.clip(np.array(a[0][1]), 0.1,
    np.inf)) - (np.array(a[1][0]) / np.clip(np.array(a[1][1]), 0.1, np.inf))

u_e_log_likelihood = compute_masked_diff_avg(u_e_log_probs)
u_m_log_likelihood = compute_masked_diff_avg(u_m_log_probs)
u_b_log_likelihood = compute_masked_diff_avg(u_b_log_probs)
u_q_log_likelihood = compute_masked_diff_avg(u_q_log_probs)
d_e_log_likelihood = compute_masked_diff_avg(d_e_log_probs)
d_m_log_likelihood = compute_masked_diff_avg(d_m_log_probs)
d_b_log_likelihood = compute_masked_diff_avg(d_b_log_probs)
d_q_log_likelihood = compute_masked_diff_avg(d_q_log_probs)

# endings
plt.plot(u_e_log_likelihood, label='up')
plt.plot(d_e_log_likelihood, label='down')
plt.title('ending log likelihood vs. timestep')
plt.legend()
plt.ylabel('end log likelihood')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'end_log_likelihood.png'))
plt.close()

#branchings
plt.plot(u_b_log_likelihood, label='up')
plt.plot(d_b_log_likelihood, label='down')
plt.title('branching log likelihood vs. timestep')
plt.legend()
plt.ylabel('branch log likelihood')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'branch_log_likelihood.png'))
plt.close()

# mothers
plt.plot(u_m_log_likelihood, label='up')
plt.plot(d_m_log_likelihood, label='down')
plt.title('mothers log likelihood vs. timestep')
plt.legend()
plt.ylabel('mothers log likelihood')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'mothers_log_likelihood.png'))
plt.close()

# charges
plt.plot(u_q_log_likelihood, label='up')
plt.plot(d_q_log_likelihood, label='down')
plt.title('charges log likelihood vs. timestep')
plt.legend()
plt.ylabel('charges log likelihood')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'charges_log_likelihood.png'))
plt.close()

