import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-n', '--n_events', default=10**5, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr/u_d_0.03_energy_discrimination')
parser.add_argument('--down_path', default='../../data/junipr/final_reclustered_d_jets_0.03.out')
parser.add_argument('--up_path', default='../../data/junipr/final_reclustered_u_jets_0.03.out')
parser.add_argument('--down_model', default='../../data/junipr/d_training_0.03_energy/p10_q21/JUNIPR_p10_q21_LR1e-05_E0')
parser.add_argument('--up_model', default='../../data/junipr/u_training_0.03_energy/p10_q21/JUNIPR_p10_q21_LR1e-05_E0')
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

def roc_auc(x, y, rnge, bins):
  uhist, edges = np.histogram(x, bins=bins,
      range=rnge)
  dhist, edges = np.histogram(y, bins=bins,
      range=rnge)
  dtotal = sum(dhist)
  utotal = sum(uhist)
  dbehindcut = [dhist[0]]
  uaftercut = [utotal - uhist[0]]
  for j in range(1, len(dhist)):
    dbehindcut.append(dbehindcut[j-1] + dhist[j])
    uaftercut.append(uaftercut[j-1] - uhist[j])
  dbehindcut = np.array(dbehindcut)
  uaftercut = np.array(uaftercut)
  sens = dbehindcut / dtotal
  spec = uaftercut / utotal
  # calculate AUC using trapezoids
  auc = np.sum((spec[:-1] + spec[1:]) * np.diff(sens) / 2)
  return sens, spec, auc

# the first array is for the up model, the second for the down. In the first
# entry of each subarray is the sum of the log probabilities over timesteps. In
# the second entry of each subarray is the number of jets which reach at least
# that timestep (we take a[0] / np.clip(a[1], 0.1, np.inf) to get the average
# likelihood at that timestep).
# 100 is the max length of a jet
u_e_log_probs = None
u_m_log_probs = None
u_b_log_probs = None
u_q_log_probs = None
u_log_probs = [u_e_log_probs,u_m_log_probs,u_b_log_probs,u_q_log_probs]
d_e_log_probs = None
d_m_log_probs = None
d_b_log_probs = None
d_q_log_probs = None
d_log_probs = [d_e_log_probs,d_m_log_probs,d_b_log_probs,d_q_log_probs]

# loading each of the trained models
u_model = JUNIPR_class.JUNIPR_energy(p_granularity, q_granularity, model_path=up_model_path).model
d_model = JUNIPR_class.JUNIPR_energy(p_granularity, q_granularity, model_path=down_model_path).model
ud_path_probs = [[up_path, u_log_probs], [down_path, d_log_probs]]

for ud in range(len(ud_path_probs)):
  [daughters, endings, mothers, (discrete_p_splittings, discrete_q_splittings), 
      mother_momenta] = load_data(ud_path_probs[ud][0], 
      n_events=n_events, batch_size=batch_size, split_p_q=True,
      p_granularity=p_granularity, q_granularity=q_granularity)

  #zeros = [[0] * 100 for d in daughters]
  for i in range(len(ud_path_probs[ud][1])):
    ud_path_probs[ud][1][i] = np.zeros((2, len(daughters), 100))

  for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] = 0

  for n in range(len(daughters)):
    for i_m, model in enumerate([u_model, d_model]):
# for charge
      #e, m, b, q = model.predict_on_batch(x=[daughters[n], mother_momenta[n],
      #  mothers[n][1], discrete_q_splittings[n][0]])
# for branching
      e, m, b, q = model.predict_on_batch(x=[daughters[n], mother_momenta[n],
        mothers[n][1], discrete_p_splittings[n][0]])
# for standard
      #e, m, b, q = model.predict_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]])
      I, T = e.shape[:2]
      for i in range(I):
        for t in range(T):
          if endings[n][1][i,t][0] == False:
            if endings[n][0][i,t][0] == 0:
              ud_path_probs[ud][1][0][i_m][n][t] = np.log10(1-e[i,t][0])
            else:
              ud_path_probs[ud][1][0][i_m][n][t] = np.log10(e[i,t][0])
          mother_index = np.nonzero(mothers[n][0][i,t])[0]
          if len(mother_index) != 0:
            ud_path_probs[ud][1][1][i_m][n][t] = np.log10(m[i,t,mother_index[0]])
          # this also means that discrete_q_splittings[n][1][i,t] == False
          if discrete_p_splittings[n][1][i,t]==False:
            ud_path_probs[ud][1][2][i_m][n][t] = np.log10(b[i,t][discrete_p_splittings[n][0][i,t]])
            ud_path_probs[ud][1][3][i_m][n][t] = np.log10(q[i,t][discrete_q_splittings[n][0][i,t]])


log_probs = [np.array(ud_path_probs[0][1]), np.array(ud_path_probs[1][1])]

u_log_probs_diffs = log_probs[0][:, 0] - log_probs[0][:, 1]
d_log_probs_diffs = log_probs[1][:, 0] - log_probs[1][:, 1]

# combining all of the observables to get the total log likelihood ratio at each
# timestep
combinedaucs = [0] * 100
combined_u_diffs = np.sum(u_log_probs_diffs, axis=0)
combined_d_diffs = np.sum(d_log_probs_diffs, axis=0)

aucs = [[0] * 100 for ob in range(4)]
for t in range(100):
  # getting discrimination power from all observables combined at each timestep
  combinedaucs[t] = roc_auc(combined_u_diffs[:, t], combined_d_diffs[:, t], (-5,
    5), 1000)[2]
  for i in range(len(aucs)):
    # getting discrimination power from each observable separately at
    # just one timestep
    aucs[i][t] = roc_auc(u_log_probs_diffs[i, :, t], d_log_probs_diffs[i, :, t], (-5, 5),
        1000)[2]
    #uhist, edges = np.histogram(u_log_probs_diffs[i, :, t], bins=1000,
    #    range=(-5,5))
    #dhist, edges = np.histogram(d_log_probs_diffs[i, :, t], bins=1000,
    #    range=(-5,5))
    #dtotal = sum(dhist)
    #utotal = sum(uhist)
    #dbehindcut = [dhist[0]]
    #uaftercut = [utotal - uhist[0]]
    #for j in range(1, len(dhist)):
    #  dbehindcut.append(dbehindcut[j-1] + dhist[j])
    #  uaftercut.append(uaftercut[j-1] - uhist[j])
    #dbehindcut = np.array(dbehindcut)
    #uaftercut = np.array(uaftercut)
    #sens = dbehindcut / dtotal
    #spec = uaftercut / utotal
    ## calculate AUC using trapezoids
    #aucs[i][t] = np.sum((spec[:-1] + spec[1:]) * np.diff(sens) / 2)

# this turns these arrays into a running sum of log likelihood ratios so we can
# see how predictions really changes over time.
for t in range(1, 100):
  combined_u_diffs[:, t] += combined_u_diffs[:, t-1]
  combined_d_diffs[:, t] += combined_d_diffs[:, t-1]
  for i, ob in enumerate(u_log_probs_diffs):
    u_log_probs_diffs[i, :, t] += u_log_probs_diffs[i, :, t-1]
    d_log_probs_diffs[i, :, t] += d_log_probs_diffs[i, :, t-1]

cum_combinedaucs = [0] * 100
cum_aucs = [[0] * 100 for ob in range(4)]
# doing the same for loop again but now the arrays are cumulative.
for t in range(100):
  cum_combinedaucs[t] = roc_auc(combined_u_diffs[:, t], combined_d_diffs[:, t], (-5,
    5), 1000)[2]
  for i in range(len(aucs)):
    cum_aucs[i][t] = roc_auc(u_log_probs_diffs[i, :, t], d_log_probs_diffs[i, :, t], (-5, 5),
        1000)[2]

# endings
plt.plot(aucs[0])
plt.title('ending AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'end_auc.png'))
plt.close()

#mothers
plt.plot(aucs[1])
plt.title('mothers AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'mothers_auc.png'))
plt.close()

# branching
plt.plot(aucs[2])
plt.title('branch AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'branch_auc.png'))
plt.close()

# charges
plt.plot(aucs[3])
plt.title('charges AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'charges_auc.png'))
plt.close()

# combined
plt.plot(combinedaucs)
plt.title('AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'combined_auc.png'))
plt.close()

# cumulative aucs
# endings
plt.plot(cum_aucs[0])
plt.title('cumulative ending AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'cum_end_auc.png'))
plt.close()

#mothers
plt.plot(cum_aucs[1])
plt.title('cumulative mothers AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'cum_mothers_auc.png'))
plt.close()

# branching
plt.plot(cum_aucs[2])
plt.title('cumulative branch AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'cum_branch_auc.png'))
plt.close()

# charges
plt.plot(cum_aucs[3])
plt.title('cumulative charges AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'cum_charges_auc.png'))
plt.close()

# combined
plt.plot(cum_combinedaucs)
plt.title('cumulative AUC vs. timestep')
plt.ylabel('AUC')
plt.xlabel('timestep')
plt.savefig(os.path.join(data_path, 'cum_combined_auc.png'))
plt.close()
