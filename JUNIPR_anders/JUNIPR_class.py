import tensorflow as tf
K = tf.keras.backend
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utilities import load_data
import os
import time
import encoding

class JUNIPR_model():
  
  def __init__(self, p_granularity, q_granularity, model_path=None):
    self.model_basename = None
    self.p_granularity = p_granularity
    self.q_granularity = q_granularity
    if model_path is not None:
      self.model_basename = os.path.splitext(os.path.basename(model_path))[0]
      with tf.keras.utils.CustomObjectScope({'activation_average': self.activation_average,
        'categorical_crossentropy2': self.categorical_crossentropy2, 'tf' : tf}):
        self.model = tf.keras.models.load_model(model_path)
        print(self.model.summary(), flush=True)


  def train(self, lrs, n_epochs, x, y, data_path, loss_functions):
    path = os.path.join(data_path,
      'p{}_q{}'.format(self.p_granularity, self.q_granularity))
    if not os.path.exists(path):
      os.makedirs(path)
    for lr, n_epoch, x_batch, y_batch in zip(lrs, n_epochs, x, y):
      print('Using learning rate ', lr, flush=True)
      self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr), 
          loss=loss_functions)
      for epoch in range(n_epoch):
        print("Epoch: ", epoch, flush=True)
        for n in range(len(x_batch)):
          start_time = time.time()
          batch_loss = self.model.train_on_batch(x=x_batch[n], y=y_batch[n])
          print("Batch {}: {}, took {} seconds".format(n, batch_loss, time.time() - start_time), flush=True)
        self.model_basename = 'JUNIPR_p{}_q{}_LR{}_E{}'.format(self.p_granularity, 
          self.q_granularity, lr, epoch)
        self.model.save(os.path.join(path, self.model_basename))


  def validate(self):
    self.endings_out    = np.zeros((2, 100, 1))
    self.mothers_out    = np.zeros((2, 100, 100))
    self.branchings_out = np.zeros((2,100, self.p_granularity**4))
    self.charge_out     = np.zeros((2, 100, self.q_granularity))


  @staticmethod
  def activation_average(x):
      total = tf.keras.backend.clip(tf.keras.backend.sum(x, axis=-1, keepdims=True), tf.keras.backend.epsilon(), 1)
      return x/total


  @staticmethod
  def categorical_crossentropy2(target, output):
      sums = tf.keras.backend.sum(output, axis=-1, keepdims=True)
      sums = tf.keras.backend.clip(sums, tf.keras.backend.epsilon(), 1)
      output = output/sums
      output = tf.keras.backend.clip(output, tf.keras.backend.epsilon(), 1)
      length = tf.keras.backend.sum(tf.keras.backend.ones_like(target[0,:,0]))
      return -tf.keras.backend.sum(target*tf.keras.backend.log(output), axis=-1)*length/tf.keras.backend.sum(target)


  @staticmethod
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


  @staticmethod
  def average_endings_output(out, data, maxlen=100):
      n_batches = len(data)
      n_batches = len(out)
      padded_endings = []
      padded_mask = []
      for i in range(n_batches):
          padded_endings.append(tf.keras.preprocessing.sequence.pad_sequences(out[i], maxlen=maxlen, dtype='float32', padding='post', value=0))
          padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(data[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))


  @staticmethod
  def average_mothers_data(mothers, maxlen=100):
      n_batches = len(mothers)
      padded_mothers = []
      padded_mask = []
      for i in range(n_batches):
          padded_mothers.append(tf.keras.preprocessing.sequence.pad_sequences(mothers[i][0], maxlen=maxlen, dtype='float32', padding='post', value=0))
          padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(1-mothers[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))    
      average = np.ma.average(np.ma.masked_array(padded_mothers, mask = padded_mask), axis=(0,1,2))
      return average


  @staticmethod
  def average_mothers_output(out, data, maxlen=100):
      n_batches = len(out)
      padded_endings = []
      padded_mask = []
      for i in range(n_batches):
          padded_endings.append(tf.keras.preprocessing.sequence.pad_sequences(out[i], maxlen=maxlen, dtype='float32', padding='post', value=0))
          padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(1-data[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))

  # this expects that:
  # - the model was loaded from file OR
  # - at least an epoch of training has been performed OR
  # - self.model_basename has been set
  # so that the plots can be saved sensibly
  def plot(self, data_path, avg_m, avg_m_data, avg_e, avg_e_data, avg_b, avg_b_data, avg_q,
      avg_q_data):
    if self.model_basename is None:
      print('self.model_basename must be set')
      return

    p_granularity = self.p_granularity
    q_granularity = self.q_granularity

    # plot endings data
    plt.plot(avg_e_data, label='Pythia')
    plt.plot(avg_e, label='JUNIPR')
    plt.ylim(0,1)
    plt.xlim(0,50)
    plt.legend()
    plt.savefig(os.path.join(data_path, '{}_avg_endings.png'.format(self.model_basename)))
    plt.close()

    # plot mothers data
    plt.plot(avg_m_data, '.', label="Pythia")
    plt.plot(avg_m, '.', label='JUNIPR')
    plt.ylim(0,1)
    plt.xlim(0,10)
    plt.legend()
    plt.savefig(os.path.join(data_path, '{}_avg_mothers.png'.format(self.model_basename)))
    plt.close()

    # plot branching data
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
    plt.savefig(os.path.join(data_path, '{}_avg_z.png'.format(self.model_basename)))
    plt.close()

    plt.plot(theta_JUNIPR, label='junipr')
    plt.plot(theta_data, label='pythia')
    plt.ylim(0,max(theta_data)*1.1)
    plt.legend()
    plt.savefig(os.path.join(data_path, '{}_avg_theta.png'.format(self.model_basename)))
    plt.close()

    plt.plot(phi_JUNIPR, label='junipr')
    plt.plot(phi_data, label='pythia')
    plt.ylim(0, max(phi_data)*1.1)
    plt.legend()
    plt.savefig(os.path.join(data_path, '{}_avg_phi.png'.format(self.model_basename)))
    plt.close()

    plt.plot(delta_JUNIPR, label='junipr')
    plt.plot(delta_data, label="pythia")
    plt.ylim(0, max(delta_data)*1.1)
    plt.legend()
    plt.savefig(os.path.join(data_path, '{}_avg_delta.png'.format(self.model_basename)))
    plt.close()

    # plot charge data
    plt.plot(np.mean(avg_q_data, axis=0), '.', label="Pythia")
    plt.plot(np.mean(avg_q, axis=0), '.', label="JUNIPR")
    plt.legend()
    plt.savefig(os.path.join(data_path, '{}_avg_charge.png'.format(self.model_basename)))
    plt.close()


class JUNIPR_times(JUNIPR_model):
  
  def __init__(self, p_granularity, q_granularity, model_path=None):
    #super(JUNIPR_times, self).__init__(p_granularity, q_granularity, model_path)
    super().__init__(p_granularity, q_granularity, model_path)
    if model_path is None:
      # Define input to RNN cell
      input_daughters = tf.keras.Input((None,8), name='Input_Daughters')

      # Masking input daughters
      masked_input = tf.keras.layers.Masking(mask_value=-1, name='Masked_Input_Daughters')(input_daughters)

      # Define RNN cell
      rnn_cell = tf.keras.layers.SimpleRNN(100, name='RNN', activation='tanh', return_sequences=True, bias_initializer='glorot_normal')(masked_input)

      # End shower
      end_hidden = tf.keras.layers.Dense(100, name='End_Hidden_Layer', activation='relu')(rnn_cell)
      end_output = tf.keras.layers.Dense(1, name='End_Output_Layer', activation='sigmoid')(end_hidden)

      # Choose Mother
      mother_hidden = tf.keras.layers.Dense(100, name='Mother_Hidden_Layer', activation='relu')(rnn_cell)
      mother_output = tf.keras.layers.Dense(100, name='Mother_Output_Layer', activation='softmax')(mother_hidden)

      mother_weights = tf.keras.Input((None, 100), name='mother_weights')
      mother_weighted_output = tf.keras.layers.multiply([mother_weights, mother_output])

      normalization = tf.keras.layers.Activation(super().activation_average)(mother_weighted_output)

      # Branching Function
      input_mother_momenta = tf.keras.Input((None, 4), name='Input_Mother_Momenta')

      # Masking Mother Momenta for branching function
      masked_mother_momenta = tf.keras.layers.Masking(mask_value=-1, name='Masked_Mother_Momenta')(input_mother_momenta)

      # Merge rnn & mother momenta inputs to branching function
      branch_input = tf.keras.layers.concatenate([rnn_cell, masked_mother_momenta], axis=-1)
      branch_hidden = tf.keras.layers.Dense(100,   name='Branch_Hidden_Layer', activation='relu')(branch_input)
      branch_output = tf.keras.layers.Dense(p_granularity**4 * q_granularity, name='Branch_Output_Layer', activation='softmax')(branch_hidden)

      self.model = tf.keras.models.Model(
          inputs=[input_daughters, input_mother_momenta, mother_weights], 
          outputs=[end_output, normalization, branch_output])
      print(self.model.summary(), flush=True)

  def train(self, lrs, n_epochs, x, y, data_path):
    super().train(lrs, n_epochs, x, y, data_path, loss_functions=['binary_crossentropy',
      super().categorical_crossentropy2, 'sparse_categorical_crossentropy'])

  def validate(self, data_path, batches, endings, mothers, discrete_splittings):
    super().validate()
    p_granularity = self.p_granularity
    q_granularity = self.q_granularity
    for n in range(len(batches)):
      e, m, b = self.model.predict_on_batch(x=batches[n])
      I, T = e.shape[:2]
      for i in range(I):
        for t in range(T):
          if endings[n][1][i,t][0] == False:
            self.endings_out[0,t]+=e[i,t]
            self.endings_out[1,t]+=1
          for j in range(100):
            if mothers[n][1][i,t][j] == True:
              self.mothers_out[0,t,j]+=m[i,t,j]
              self.mothers_out[1,t,j]+=1
          if discrete_splittings[n][1][i,t]==False:
            pxq = b[i,t].reshape((p_granularity**4, q_granularity))
            self.branchings_out[0,t]+=np.mean(pxq, axis=1)
            self.branchings_out[1,t]+=np.ones(p_granularity**4)
            self.charge_out[0,t]+=np.mean(pxq, axis=0)
            self.charge_out[1,t]+=np.ones(q_granularity)
    avg_e = self.endings_out[0] / self.endings_out[1]
    avg_e_data = super().average_endings_data(endings)
    avg_m = self.mothers_out[0]/np.sum(self.mothers_out[0])
    avg_m = np.mean(avg_m, axis = 0)
    avg_m = avg_m / np.sum(avg_m)
    avg_m_data = super().average_mothers_data(mothers)
    avg_b_data, avg_q_data = self.average_branching_data(discrete_splittings,
        p_granularity=p_granularity, q_granularity=q_granularity)
    avg_b = self.branchings_out[0]/np.sum(self.branchings_out[0])
    avg_q = self.charge_out[0]/np.sum(self.charge_out[0])
    super().plot(data_path, avg_m, avg_m_data, avg_e, avg_e_data, avg_b, avg_b_data, avg_q,
        avg_q_data)


  @staticmethod
  def average_branching_data(branchings, p_granularity, q_granularity,
      max_t=100, weighted_time_average=True):
      avg_branching = np.zeros((2, max_t, p_granularity**4))
      avg_charge = np.zeros((2, max_t, q_granularity))
      for i in range(len(branchings)):
        B, T = branchings[i][0].shape[:2]
        for b in range(B):
          for t in range(T):
            if branchings[i][1][b,t]==False:
              i_p, i_q = encoding.get_all_obs(branchings[i][0][b,t], [p_granularity**4, q_granularity])
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


class JUNIPR_plus(JUNIPR_model):

  def __init__(self, p_granularity, q_granularity, model_path=None):
    super().__init__(p_granularity, q_granularity, model_path)
    if model_path is None:
      # Define input to RNN cell
      input_daughters = tf.keras.Input((None,8), name='Input_Daughters')

      # Masking input daughters
      masked_input = tf.keras.layers.Masking(mask_value=-1, name='Masked_Input_Daughters')(input_daughters)

      # Define RNN cell
      rnn_cell = tf.keras.layers.SimpleRNN(100, name='RNN', activation='tanh', return_sequences=True)(masked_input)

      # End shower
      end_hidden = tf.keras.layers.Dense(100, name='End_Hidden_Layer', activation='relu')(rnn_cell)
      end_output = tf.keras.layers.Dense(1, name='End_Output_Layer', activation='sigmoid')(end_hidden)

      # Choose Mother
      mother_hidden = tf.keras.layers.Dense(100, name='Mother_Hidden_Layer', activation='relu')(rnn_cell)
      mother_output = tf.keras.layers.Dense(100, name='Mother_Output_Layer', activation='softmax')(mother_hidden)

      mother_weights = tf.keras.Input((None, 100), name='mother_weights')
      mother_weighted_output = tf.keras.layers.multiply([mother_weights, mother_output])

      normalization = tf.keras.layers.Activation(super().activation_average)(mother_weighted_output)

      # Branching Function
      input_mother_momenta = tf.keras.Input((None, 4), name='Input_Mother_Momenta')

      # Masking Mother Momenta for branching function
      masked_mother_momenta = tf.keras.layers.Masking(mask_value=-1, name='Masked_Mother_Momenta')(input_mother_momenta)

      # Merge rnn & mother momenta inputs to branching function
      branch_input = tf.keras.layers.concatenate([rnn_cell, masked_mother_momenta], axis=-1)
      branch_hidden = tf.keras.layers.Dense(100,   name='Branch_Hidden_Layer', activation='relu')(branch_input)
      branch_output = tf.keras.layers.Dense(p_granularity**4, name='Branch_Output_Layer', activation='softmax')(branch_hidden)

      # Use merged rnn & mother momenta concatenation for charge function
      charge_hidden = tf.keras.layers.Dense(100, name='Charge_Hidden_Layer', activation='relu')(branch_input)
      charge_output = tf.keras.layers.Dense(q_granularity, name='Charge_Output_Layer', activation='softmax')(charge_hidden)

      self.model = tf.keras.models.Model(
          inputs=[input_daughters, input_mother_momenta, mother_weights], 
          outputs=[end_output, normalization, branch_output, charge_output])
      print(self.model.summary(), flush=True)

  def train(self, lrs, n_epochs, x, y, data_path):
    super().train(lrs, n_epochs, x, y, data_path, loss_functions=['binary_crossentropy',
      super().categorical_crossentropy2, 'sparse_categorical_crossentropy', 
      'sparse_categorical_crossentropy'])


  def validate(self, data_path, batches, endings, mothers, discrete_p_splittings,
      discrete_q_splittings):
    super().validate()
    p_granularity = self.p_granularity
    q_granularity = self.q_granularity
    for n in range(len(batches)):
      e, m, b, q = self.model.predict_on_batch(x=batches[n])
      I, T = e.shape[:2]
      for i in range(I):
        for t in range(T):
          if endings[n][1][i,t][0] == False:
            self.endings_out[0,t]+=e[i,t]
            self.endings_out[1,t]+=1
          for j in range(100):
            if mothers[n][1][i,t][j] == True:
              self.mothers_out[0,t,j]+=m[i,t,j]
              self.mothers_out[1,t,j]+=1
          if discrete_p_splittings[n][1][i,t]==False:
            self.branchings_out[0,t]+=b[i,t]
            self.branchings_out[1,t]+=np.ones(p_granularity**4)
            self.charge_out[0,t]+=q[i,t]
            self.charge_out[1,t]+=np.ones(q_granularity)
    avg_e = self.endings_out[0] / self.endings_out[1]
    avg_e_data = super().average_endings_data(endings)
    avg_m = self.mothers_out[0]/np.sum(self.mothers_out[0])
    avg_m = np.mean(avg_m, axis = 0)
    avg_m = avg_m / np.sum(avg_m)
    avg_m_data = super().average_mothers_data(mothers)
    avg_b_data, avg_q_data = self.average_branching_data(discrete_p_splittings,
        discrete_q_splittings, p_granularity=p_granularity, 
        q_granularity=q_granularity)
    avg_b = self.branchings_out[0]/np.sum(self.branchings_out[0])
    avg_q = self.charge_out[0]/np.sum(self.charge_out[0])
    super().plot(data_path, avg_m, avg_m_data, avg_e, avg_e_data, avg_b, avg_b_data, avg_q,
        avg_q_data)


  @staticmethod
  def average_branching_data(branchings, charges, p_granularity=10, q_granularity=21,
      max_t=100, weighted_time_average=True):
      avg_branching = np.zeros((2, max_t, p_granularity**4))
      avg_charge = np.zeros((2, max_t, q_granularity))
      for i in range(len(branchings)):
        B, T = branchings[i][0].shape[:2]
        for b in range(B):
          for t in range(T):
            if branchings[i][1][b,t]==False:
              avg_branching[0, t, branchings[i][0][b,t]]+=1
              avg_branching[1, t]+= np.ones(p_granularity**4)
              avg_charge[0, t, charges[i][0][b,t]]+=1
              avg_charge[1, t]+= np.ones(q_granularity)

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
 
class JUNIPR_energy(JUNIPR_plus):

  def __init__(self, p_granularity, q_granularity, model_path=None):
    JUNIPR_model.__init__(self, p_granularity, q_granularity, model_path)
    if model_path is None:
      # Define input to RNN cell
      input_daughters = tf.keras.Input((None,8), name='Input_Daughters')

      # Masking input daughters
      masked_input = tf.keras.layers.Masking(mask_value=-1, name='Masked_Input_Daughters')(input_daughters)

      # Define RNN cell
      rnn_cell = tf.keras.layers.SimpleRNN(100, name='RNN', activation='tanh', return_sequences=True)(masked_input)

      # End shower
      end_hidden = tf.keras.layers.Dense(100, name='End_Hidden_Layer', activation='relu')(rnn_cell)
      end_output = tf.keras.layers.Dense(1, name='End_Output_Layer', activation='sigmoid')(end_hidden)

      # Choose Mother
      mother_hidden = tf.keras.layers.Dense(100, name='Mother_Hidden_Layer', activation='relu')(rnn_cell)
      mother_output = tf.keras.layers.Dense(100, name='Mother_Output_Layer', activation='softmax')(mother_hidden)

      mother_weights = tf.keras.Input((None, 100), name='mother_weights')
      mother_weighted_output = tf.keras.layers.multiply([mother_weights, mother_output])

      normalization = tf.keras.layers.Activation(super().activation_average)(mother_weighted_output)

      # Branching Function
      input_mother_momenta = tf.keras.Input((None, 4), name='Input_Mother_Momenta')

      # Masking Mother Momenta for branching function
      masked_mother_momenta = tf.keras.layers.Masking(mask_value=-1, name='Masked_Mother_Momenta')(input_mother_momenta)

      # Merge rnn & mother momenta inputs to branching function
      branch_input = tf.keras.layers.concatenate([rnn_cell, masked_mother_momenta], axis=-1)
      branch_hidden = tf.keras.layers.Dense(100,   name='Branch_Hidden_Layer', activation='relu')(branch_input)
      branch_output = tf.keras.layers.Dense(p_granularity**4, name='Branch_Output_Layer', activation='softmax')(branch_hidden)

      # compute z in a lambda layer
      def z(x):
        K = tf.keras.backend
        shape = K.shape(x)
        # reshaping the tensor so that it has shape (batches, timesteps,
        # p_granularity, p_granularity, p_granularity, p_granularity), then
        # averaging over the last 3 axes to get non-normalized probabilities for
        # z, then returning the argmax over the last axis, reshaping so that
        # the shape is (batches, timesteps, 1), and casting the result to
        # tf.float32 so that it has the same type as the tensor it is
        # concatenated with.
          #(-1, -1, *[p_granularity]*4)), axis=[3, 4, 5]), axis=2), 
        return tf.cast(K.reshape(K.argmax(K.mean(K.reshape(x, 
          tf.stack([shape[0], shape[1], *[p_granularity]*4])), axis=[3, 4, 5]), axis=2), 
          tf.stack([shape[0], shape[1], 1])), tf.float32)

      z_layer = tf.keras.layers.Lambda(z, name='Z_lambda')(branch_output)

      # Use merged rnn & mother momenta concatenation for charge function
      charge_input = tf.keras.layers.concatenate([branch_input, z_layer], axis=-1)
      charge_hidden = tf.keras.layers.Dense(100, name='Charge_Hidden_Layer', activation='relu')(charge_input)
      charge_output = tf.keras.layers.Dense(q_granularity, name='Charge_Output_Layer', activation='softmax')(charge_hidden)

      self.model = tf.keras.models.Model(
          inputs=[input_daughters, input_mother_momenta, mother_weights], 
          outputs=[end_output, normalization, branch_output, charge_output])
      print(self.model.summary(), flush=True)


class JUNIPR_branching(JUNIPR_plus):

  def __init__(self, p_granularity, q_granularity, model_path=None):
    JUNIPR_model.__init__(self, p_granularity, q_granularity, model_path)
    if model_path is None:
      # Define input to RNN cell
      input_daughters = tf.keras.Input((None,8), name='Input_Daughters')

      # Masking input daughters
      masked_input = tf.keras.layers.Masking(mask_value=-1, name='Masked_Input_Daughters')(input_daughters)

      # Define RNN cell
      rnn_cell = tf.keras.layers.SimpleRNN(100, name='RNN', activation='tanh', return_sequences=True)(masked_input)

      # End shower
      end_hidden = tf.keras.layers.Dense(100, name='End_Hidden_Layer', activation='relu')(rnn_cell)
      end_output = tf.keras.layers.Dense(1, name='End_Output_Layer', activation='sigmoid')(end_hidden)

      # Choose Mother
      mother_hidden = tf.keras.layers.Dense(100, name='Mother_Hidden_Layer', activation='relu')(rnn_cell)
      mother_output = tf.keras.layers.Dense(100, name='Mother_Output_Layer', activation='softmax')(mother_hidden)

      mother_weights = tf.keras.Input((None, 100), name='mother_weights')
      mother_weighted_output = tf.keras.layers.multiply([mother_weights, mother_output])

      normalization = tf.keras.layers.Activation(super().activation_average)(mother_weighted_output)

      # Branching Function
      input_mother_momenta = tf.keras.Input((None, 4), name='Input_Mother_Momenta')

      # Masking Mother Momenta for branching function
      masked_mother_momenta = tf.keras.layers.Masking(mask_value=-1, name='Masked_Mother_Momenta')(input_mother_momenta)

      # Merge rnn & mother momenta inputs to branching function
      branch_input = tf.keras.layers.concatenate([rnn_cell, masked_mother_momenta], axis=-1)
      branch_hidden = tf.keras.layers.Dense(100,   name='Branch_Hidden_Layer', activation='relu')(branch_input)
      branch_output = tf.keras.layers.Dense(p_granularity**4, name='Branch_Output_Layer', activation='softmax')(branch_hidden)

      # compute q in a lambda layer
      #def q(x):
      #  K = tf.keras.backend
      #  shape = K.shape(x)
      #  # taking the argmax over the probabilities assigned to each of the
      #  # potential charges, reshaping so the tensor has the shape (batches,
      #  # timesteps, 1), and then casting the restult to tf.float32 so that it
      #  # has the same type ast the tensor it is concatenated with.
      #  return tf.cast(K.reshape(K.argmax(x, axis=2), 
      #    tf.stack([shape[0], shape[1], 1])), tf.float32)
      #q_layer = tf.keras.layers.Lambda(q, name='Q_lambda')(charge_output)

      # scaling the feature.
      #def divide(x):
      #  return tf.divide(x, p_granularity**4)

      def split_scale(x):
        i3 = tf.floordiv(x, p_granularity**3)
        i2 = tf.floordiv(tf.floormod(x, p_granularity**3), p_granularity**2)
        i1 = tf.floordiv(tf.floormod(x, p_granularity**2), p_granularity)
        i0 = tf.floormod(x, p_granularity)
        return tf.truediv(tf.concat([i3, i2, i1, i0], axis=-1),
                        tf.cast(p_granularity, tf.float32))


      # Use merged rnn & mother momenta concatenation for charge function
      input_branch = tf.keras.Input((None, 1), name='Input_Branching')
      masked_branch = tf.keras.layers.Masking(mask_value=p_granularity**4, name='Masked_Branching')(input_branch)
      scaled_masked_branch = tf.keras.layers.Lambda(split_scale, name='Scaled_Masked_Branch')(masked_branch)
      charge_input = tf.keras.layers.concatenate([branch_input, scaled_masked_branch], axis=-1)
      charge_hidden = tf.keras.layers.Dense(100, name='Charge_Hidden_Layer', activation='relu')(charge_input)
      charge_output = tf.keras.layers.Dense(q_granularity, name='Charge_Output_Layer', activation='softmax')(charge_hidden)

      self.model = tf.keras.models.Model(
          inputs=[input_daughters, input_mother_momenta, mother_weights, input_branch], 
          outputs=[end_output, normalization, branch_output, charge_output])
      print(self.model.summary(), flush=True)


class JUNIPR_charge(JUNIPR_plus):

  def __init__(self, p_granularity, q_granularity, model_path=None):
    JUNIPR_model.__init__(self, p_granularity, q_granularity, model_path)
    if model_path is None:
      # Define input to RNN cell
      input_daughters = tf.keras.Input((None,8), name='Input_Daughters')

      # Masking input daughters
      masked_input = tf.keras.layers.Masking(mask_value=-1, name='Masked_Input_Daughters')(input_daughters)

      # Define RNN cell
      rnn_cell = tf.keras.layers.SimpleRNN(100, name='RNN', activation='tanh', return_sequences=True)(masked_input)

      # End shower
      end_hidden = tf.keras.layers.Dense(100, name='End_Hidden_Layer', activation='relu')(rnn_cell)
      end_output = tf.keras.layers.Dense(1, name='End_Output_Layer', activation='sigmoid')(end_hidden)

      # Choose Mother
      mother_hidden = tf.keras.layers.Dense(100, name='Mother_Hidden_Layer', activation='relu')(rnn_cell)
      mother_output = tf.keras.layers.Dense(100, name='Mother_Output_Layer', activation='softmax')(mother_hidden)

      mother_weights = tf.keras.Input((None, 100), name='mother_weights')
      mother_weighted_output = tf.keras.layers.multiply([mother_weights, mother_output])

      normalization = tf.keras.layers.Activation(super().activation_average)(mother_weighted_output)

      # Branching Function
      input_mother_momenta = tf.keras.Input((None, 4), name='Input_Mother_Momenta')

      # Masking Mother Momenta for branching function
      masked_mother_momenta = tf.keras.layers.Masking(mask_value=-1, name='Masked_Mother_Momenta')(input_mother_momenta)

      # Merge rnn & mother momenta inputs to branching function
      charge_input = tf.keras.layers.concatenate([rnn_cell, masked_mother_momenta], axis=-1, name='Charge_Input')
      charge_hidden = tf.keras.layers.Dense(100, name='Charge_Hidden_Layer', activation='relu')(charge_input)
      charge_output = tf.keras.layers.Dense(q_granularity, name='Charge_Output_Layer', activation='softmax')(charge_hidden)

      # compute q in a lambda layer
      #def q(x):
      #  K = tf.keras.backend
      #  shape = K.shape(x)
      #  # taking the argmax over the probabilities assigned to each of the
      #  # potential charges, reshaping so the tensor has the shape (batches,
      #  # timesteps, 1), and then casting the restult to tf.float32 so that it
      #  # has the same type ast the tensor it is concatenated with.
      #  return tf.cast(K.reshape(K.argmax(x, axis=2), 
      #    tf.stack([shape[0], shape[1], 1])), tf.float32)
      #q_layer = tf.keras.layers.Lambda(q, name='Q_lambda')(charge_output)

      # scaling the feature.
      def divide(x):
        return tf.truediv(x, q_granularity)

      input_charge = tf.keras.Input((None, 1), name='Input_Charge')
      masked_charge = tf.keras.layers.Masking(mask_value=q_granularity, name='Masked_Charge')(input_charge)
      scaled_masked_charge = tf.keras.layers.Lambda(divide, name='Scaled_Masked_Charge')(masked_charge)
      branch_input = tf.keras.layers.concatenate([charge_input, scaled_masked_charge], axis=-1, name='Branch_Input')
      branch_hidden = tf.keras.layers.Dense(100,   name='Branch_Hidden_Layer', activation='relu')(branch_input)
      branch_output = tf.keras.layers.Dense(p_granularity**4, name='Branch_Output_Layer', activation='softmax')(branch_hidden)


      self.model = tf.keras.models.Model(
          inputs=[input_daughters, input_mother_momenta, mother_weights,
            input_charge], 
          outputs=[end_output, normalization, branch_output, charge_output])
      print(self.model.summary(), flush=True)
