import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=100, type=int)
parser.add_argument('-n', '--n_events', default=10**4, type=int)
parser.add_argument('-d', '--data_path', default='final_reclustered_practice_cut.out')
parser.add_argument('-m', '--model_path', required=True)
args = vars(parser.parse_args())


# # Load Data

# In[2]:

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']
batch_size = args['batch_size']

n_events = args['n_events']
n_batches = n_events//batch_size

# path to saved jets
data_path = args['data_path']

# path to saved model
model_path = args['model_path']

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

[daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(data_path, 
    n_events=n_events, batch_size=batch_size,
    p_granularity=p_granularity, q_granularity=q_granularity, reload=True)

print('data loaded', flush=True)

# temporary hack having to do with mask values; this will change later.
for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] =0

batch_number = 0
print(daughters[batch_number].shape, flush=True)
print(endings[batch_number][0].shape, flush=True)
print(mothers[batch_number][0].shape, flush=True)
print(discrete_splittings[batch_number][0].shape, flush=True)
print(mother_momenta[batch_number].shape, flush=True)

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

with tf.keras.utils.CustomObjectScope({'activation_average':
    JUNIPR_functions.activation_average,
    'categorical_crossentropy2': JUNIPR_functions.categorical_crossentropy2}):
  model = tf.keras.models.load_model(model_path)
  print(model.summary(), flush=True)

# Validate Model

  endings_out    = np.zeros((2, 100, 1))
  mothers_out    = np.zeros((2, 100, 100))
  branchings_out = np.zeros((2,100, p_granularity**4))
  charge_out     = np.zeros((2, 100, q_granularity))

  for n in range(len(daughters)):
      e, m, b = model.predict_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]])
      I, T = e.shape[:2]
      for i in range(I):
          for t in range(T):
              if endings[n][1][i,t][0] == False:
                  endings_out[0,t]+=e[i,t]
                  endings_out[1,t]+=1
              for j in range(100):
                  if mothers[n][1][i,t][j] == True:
                      mothers_out[0,t,j]+=m[i,t,j]
                      mothers_out[1,t,j]+=1
              if discrete_splittings[n][1][i,t]==False:
                  pxq = b[i,t].reshape((p_granularity**4, q_granularity))
                  branchings_out[0,t]+=np.mean(pxq, axis=1)
                  branchings_out[1,t]+=np.ones(p_granularity**4)
                  charge_out[0,t]+=np.mean(pxq, axis=0)
                  charge_out[1,t]+=np.ones(q_granularity)

  avg_e = endings_out[0]/np.clip(endings_out[1], 0.1 , np.inf)
  avg_m = mothers_out[0]/np.clip(mothers_out[1], 0.1 , np.inf)
  avg_b = branchings_out[0]/np.clip(branchings_out[1], 0.1 , np.inf)
  avg_q = charge_out[0]/np.clip(charge_out[1], 0.1 , np.inf)

  np.save('{}_avg_e.npy'.format(model_path), avg_e)
  np.save('{}_avg_m.npy'.format(model_path), avg_m)
  np.save('{}_avg_b.npy'.format(model_path), avg_b)
  np.save('{}_avg_q.npy'.format(model_path), avg_q)

  print(avg_e.shape)
  print(avg_m.shape)
  print(avg_b.shape)

# End Shower

  def average_endings_data(endings, maxlen=100):
      n_batches = len(endings)
      padded_endings = []
      padded_mask = []
      for i in range(n_batches):
          padded_endings.append(tf.keras.preprocessing.sequence.pad_sequences(endings[i][0], maxlen=maxlen, dtype='float32', padding='post', value=0))
          padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(endings[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))
      
      endings = np.asarray(padded_endings)
      mask = np.asarray(padded_mask)
      return np.ma.average(np.ma.masked_array(endings, mask = mask), axis=(0,1))

  def average_endings_output(out, data, maxlen=100):
      n_batches = len(data)
      n_batches = len(out)
      padded_endings = []
      padded_mask = []
      for i in range(n_batches):
          padded_endings.append(tf.keras.preprocessing.sequence.pad_sequences(out[i], maxlen=maxlen, dtype='float32', padding='post', value=0))
          padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(data[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))
      
      endings = np.asarray(padded_endings)
      mask = np.asarray(padded_mask)
      return np.ma.average(np.ma.masked_array(endings, mask = mask), axis=(0,1))

  avg_e_data = average_endings_data(endings)
  #np.save('{}_avg_e_data.npy'.format(model_path), avg_e_data)
  plt.plot(avg_e_data, label='Pythia')
  plt.plot(avg_e, label='JUNIPR')
  plt.ylim(0,1)
  plt.xlim(0,50)
  #plt.plot(average_endings_output(endings_out, endings), label='JUNIPR')
  plt.legend()
  plt.savefig('{}_avg_endings.png'.format(model_path))
  plt.close()


# Choose Parent

  def average_mothers_data(mothers, maxlen=100):
      n_batches = len(mothers)
      padded_mothers = []
      padded_mask = []
      for i in range(n_batches):
          padded_mothers.append(tf.keras.preprocessing.sequence.pad_sequences(mothers[i][0], maxlen=maxlen, dtype='float32', padding='post', value=0))
          padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(1-mothers[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))    
      average = np.ma.average(np.ma.masked_array(padded_mothers, mask = padded_mask), axis=(0,1,2))
      return average

  def average_mothers_output(out, data, maxlen=100):
      n_batches = len(out)
      padded_endings = []
      padded_mask = []
      for i in range(n_batches):
          padded_endings.append(tf.keras.preprocessing.sequence.pad_sequences(out[i], maxlen=maxlen, dtype='float32', padding='post', value=0))
          padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(1-data[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))
      
      endings = np.asarray(padded_endings)
      mask = np.asarray(padded_mask)
      return np.ma.average(np.ma.masked_array(endings, mask = mask), axis=(0,1,2))

  avg_m_data = average_mothers_data(mothers)
  #np.save('{}_avg_m_data.npy'.format(model_path), avg_m_data)
  plt.plot(avg_m_data, '.', label="Pythia")
  #plt.plot(avg_m, '.', label='JUNIPR')
  #plt.plot(average_mothers_output(mothers_out, mothers), '.', label='JUNIPR')
  plt.plot(np.mean(avg_m, axis=0), '.', label='JUNIPR')
  plt.ylim(0,1)
  plt.xlim(0,10)
  plt.legend()
  plt.savefig('{}_avg_mothers.png'.format(model_path))
  plt.close()

# Branching Function

  def average_branching_data(branchings, p_granularity=10, q_granularity=21, max_t=100, weighted_time_average=True):
      avg_branching = np.zeros((2, max_t, p_granularity**4))
      avg_charge = np.zeros((2, max_t, q_granularity))
      for i in range(len(branchings)):
          B, T = branchings[i][0].shape[:2]
          for b in range(B):
              for t in range(T):
                  if branchings[i][1][b,t]==False:
                      i_p, i_q = encoding.get_all_obs(branchings[i][0][b,t], [p_granularity**4, q_granularity])
                      #avg_branching[0, t, branchings[i][0][b,t]]+=1
                      #avg_branching[1, t]+= np.ones((granularity**4))
                      avg_branching[0, t, i_p] += 1
                      avg_branching[1, t] += np.ones(p_granularity**4)
                      avg_charge[0, t, i_q] += 1
                      avg_charge[1, t] += np.ones(q_granularity)
      if weighted_time_average:
          return (avg_branching[0]/np.sum(avg_branching[0])).reshape((max_t, 
            p_granularity, p_granularity, p_granularity, p_granularity)), avg_charge[0]/np.sum(avg_charge[0])
      else:
          avg_b = avg_branching[0]/np.clip(avg_branching[1], 0.1, np.inf)
          avg_b = avg_b/np.sum(avg_b)
          avg_q = avg_charge[0]/np.clip(avg_charge[1], 0.1, np.inf)
          avg_q = avg_q/np.sum(avg_q)
          return avg_b.reshape((max_t, p_granularity, p_granularity, p_granularity,
            p_granularity)), avg_q

  avg_b_data, avg_q_data = average_branching_data(discrete_splittings, p_granularity=p_granularity, q_granularity=q_granularity)

  z_JUNIPR = np.mean(avg_b.reshape((100, p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,2,3,4))
  theta_JUNIPR = np.mean(avg_b.reshape((100,p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,1,3,4))
  phi_JUNIPR = np.mean(avg_b.reshape((100,p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,1,2,4))
  delta_JUNIPR = np.mean(avg_b.reshape((100,p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,1,2,3))

  z_JUNIPR = z_JUNIPR/np.sum(z_JUNIPR)
  theta_JUNIPR = theta_JUNIPR/np.sum(theta_JUNIPR)
  phi_JUNIPR = phi_JUNIPR/np.sum(phi_JUNIPR)
  delta_JUNIPR = delta_JUNIPR/np.sum(delta_JUNIPR)

  z_data = np.mean(avg_b_data.reshape((100,p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,2,3,4))
  theta_data = np.mean(avg_b_data.reshape((100,p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,1,3,4))
  phi_data = np.mean(avg_b_data.reshape((100,p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,1,2,4))
  delta_data = np.mean(avg_b_data.reshape((100,p_granularity,p_granularity,p_granularity,p_granularity)), axis=(0,1,2,3))

  z_data = z_data/np.sum(z_data)
  theta_data = theta_data/np.sum(theta_data)
  phi_data = phi_data/np.sum(phi_data)
  delta_data = delta_data/np.sum(delta_data)

  plt.plot(z_JUNIPR, label='junipr')
  plt.plot(z_data, label='pythia')
  plt.ylim(0,max(z_data)*1.1)
  plt.legend()
  plt.savefig('{}_avg_z.png'.format(model_path))
  plt.close()

  plt.plot(theta_JUNIPR, label='junipr')
  plt.plot(theta_data, label='pythia')
  plt.ylim(0,max(theta_data)*1.1)
  plt.legend()
  plt.savefig('{}_avg_theta.png'.format(model_path))
  plt.close()

  plt.plot(phi_JUNIPR, label='junipr')
  plt.plot(phi_data, label='pythia')
  plt.ylim(0, max(phi_data)*1.1)
  plt.legend()
  plt.savefig('{}_avg_phi.png'.format(model_path))
  plt.close()

  plt.plot(delta_JUNIPR, label='junipr')
  plt.plot(delta_data, label="pythia")
  plt.ylim(0, max(delta_data)*1.1)
  plt.legend()
  plt.savefig('{}_avg_delta.png'.format(model_path))
  plt.close()

  plt.plot(np.mean(avg_q_data, axis=0), '.', label="Pythia")
  plt.plot(np.mean(avg_q, axis=0), '.', label="JUNIPR")
  plt.legend()
  plt.savefig('{}_avg_charge.png'.format(model_path))
  plt.close()
  #np.save('{}_avg_b_data.npy'.format(model_path), avg_b_data)
  #np.save('{}_avg_q_data.npy'.format(model_path), avg_q_data)
  sys.exit(0)

  def average_charge_data(charges, granularity=21, max_t=100, weighted_time_average=True):
      avg_charge = np.zeros((2, max_t, granularity))
      for i in range(len(charges)):
          B, T = charges[i][0].shape[:2]
          for b in range(B):
              for t in range(T):
                  if charges[i][1][b,t]==False:
                      avg_charge[0, t, charges[i][0][b,t]]+=1
                      avg_charge[1, t]+= np.ones((granularity))
      if weighted_time_average:
          return (avg_branching[0]/np.sum(avg_branching[0])).reshape((max_t,
            granularity, granularity, granularity, granularity))
      else:
          avg_b = avg_branching[0]/np.clip(avg_branching[1], 0.1, np.inf)
          avg_b = avg_b/np.sum(avg_b)
          return avg_b.reshape((max_t, granularity, granularity, granularity, granularity))

  plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(1,2,3)))
  plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(0,2,3)))
  plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(0,1,3)))
  plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(0,1,2)))
  plt.show()

  z_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(1,2,3))
  theta_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(0,2,3))
  phi_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(0,1,3))
  delta_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(0,1,2))

  z_JUNIPR = z_JUNIPR/np.sum(z_JUNIPR)
  theta_JUNIPR = theta_JUNIPR/np.sum(theta_JUNIPR)
  phi_JUNIPR = phi_JUNIPR/np.sum(phi_JUNIPR)
  delta_JUNIPR = delta_JUNIPR/np.sum(delta_JUNIPR)

  z_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(1,2,3))
  theta_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(0,2,3))
  phi_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(0,1,3))
  delta_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(0,1,2))

  z_data = z_data/np.sum(z_data)
  theta_data = theta_data/np.sum(theta_data)
  phi_data = phi_data/np.sum(phi_data)
  delta_data = delta_data/np.sum(delta_data)

  plt.plot(z_JUNIPR)
  plt.plot(z_data)
  plt.ylim(0,max(z_data)*1.1)
  plt.show()

  plt.plot(theta_JUNIPR)
  plt.plot(theta_data)
  plt.ylim(0,max(theta_data)*1.1)
  plt.show()

  plt.plot(phi_JUNIPR)
  plt.plot(phi_data)
  plt.ylim(0, max(phi_data)*1.1)
  plt.show()

  plt.plot(delta_JUNIPR)
  plt.plot(delta_data)
  plt.ylim(0, max(delta_data)*1.1)
  plt.show()

