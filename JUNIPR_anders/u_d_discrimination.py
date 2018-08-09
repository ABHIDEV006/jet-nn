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
if not os.path.exists(data_path):
  os.makedirs(data_path)

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

# the first array contains the total probabilities from each jet being evaluated
# on the up model. the second array contains those from being evaluated on the
# down model
u_log_probs = [None, None]
d_log_probs = [None, None]
#u_log_probs = [[0]*n_events, [0]*n_events]
#d_log_probs = [[0]*n_events, [0]*n_events]

# loading each of the trained models
if times:
  u_model = JUNIPR_class.JUNIPR_times(p_granularity, q_granularity, model_path=up_model_path).model
  d_model = JUNIPR_class.JUNIPR_times(p_granularity, q_granularity, model_path=down_model_path).model
else:
  u_model = JUNIPR_class.JUNIPR_plus(p_granularity, q_granularity, model_path=up_model_path).model
  d_model = JUNIPR_class.JUNIPR_plus(p_granularity, q_granularity, model_path=down_model_path).model

for path, log_probs in [(up_path, u_log_probs), (down_path, d_log_probs)]:
  [daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(path, 
      n_events=n_events, batch_size=batch_size, split_p_q=(not times),
      p_granularity=p_granularity, q_granularity=q_granularity)
  if not times:
    discrete_p_splittings, discrete_q_splittings = discrete_splittings

  for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] = 0

  zeros = [0] * len(daughters)
  log_probs[0] = zeros[:]
  log_probs[1] = zeros[:]

  if times:
    for n in range(len(daughters)):
      for i_m, model in enumerate([u_model, d_model]):
        e, m, b = model.predict_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]])
        I, T = e.shape[:2]
        for i in range(I):
          for t in range(T):
            if endings[n][1][i,t][0] == False:
              if endings[n][0][i,t][0] == 0:
                log_probs[i_m][n] += np.log10(1 - e[i,t][0])
              else:
                log_probs[i_m][n] += np.log10(e[i,t][0])
            mother_index = np.nonzero(mothers[n][0][i,t])[0]
            if len(mother_index) != 0:
              log_probs[i_m][n] += np.log10(m[i,t,mother_index[0]])
            if discrete_splittings[n][1][i,t]==False:
              log_probs[i_m][n] += np.log10(b[i,t][discrete_splittings[n][0][i,t]])
  else:
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
                log_probs[i_m][n] += np.log10(1 - e[i,t][0])
              else:
                log_probs[i_m][n] += np.log10(e[i,t][0])
            mother_index = np.nonzero(mothers[n][0][i,t])[0]
            if len(mother_index) != 0:
              log_probs[i_m][n] += np.log10(m[i,t,mother_index[0]])
            # this also means that discrete_q_splittings[n][1][i,t] == False
            if discrete_p_splittings[n][1][i,t]==False:
              log_probs[i_m][n] += (np.log10(b[i,t][discrete_p_splittings[n][0][i,t]]) +
                  np.log10(q[i,t][discrete_q_splittings[n][0][i,t]]))

u_log_likelihood = np.array(u_log_probs[0]) - np.array(u_log_probs[1])
d_log_likelihood = np.array(d_log_probs[0]) - np.array(d_log_probs[1])

edges = np.linspace(-10, 10, num=100)

plt.hist(u_log_likelihood, label='u', bins=edges, histtype='step')
plt.hist(d_log_likelihood, label='d', bins=edges, histtype='step')
plt.legend()
plt.savefig(os.path.join(data_path, 'separation.png'))
plt.close()

dhist, edges = np.histogram(d_log_likelihood, bins=1000, range=(-10, 10))
uhist, edges = np.histogram(u_log_likelihood, bins=1000, range=(-10, 10))
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
#auc = np.sum(np.diff(spec) * sens[1:])
# this is how katie plots it
plt.plot(sens, spec, label='roc')
plt.title('roc (AUC: {})'.format(auc))
plt.xlabel('Down Quark Jet Identification')
plt.ylabel('Up Quark Jet Rejection')
plt.savefig(os.path.join(data_path, 'roc.png'))
plt.close()
# this is the form used in the example i saw
#plt.plot(1 - spec, sens)
plt.plot(sens, sens/np.sqrt(1 - spec))
plt.xlabel('Down Quark Jet Identification')
plt.ylabel('Up Quark Jet Rejection')
plt.savefig(os.path.join(data_path, 'sic.png'))
