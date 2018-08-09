import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=100, type=int)
parser.add_argument('-n', '--n_events', default=10**5, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr/final_reclustered_practice.out')
parser.add_argument('-e', '--n_epochs', default=1, type=int)
args = vars(parser.parse_args())

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']
batch_size = args['batch_size']
n_epochs = args['n_epochs']
n_events = args['n_events']
n_batches = n_events//batch_size

# path to saved jets
data_path = args['data_path']

print('JUNIPR started', flush=True)
import tensorflow as tf
from utilities import load_data
import numpy as np
import time

print('done importing', flush=True)

# Load in jets from file

[daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(data_path, 
    n_events=n_events, batch_size=batch_size,
    p_granularity=p_granularity, q_granularity=q_granularity)

print('data loaded', flush=True)

# temporary hack having to do with mask values; this will change later.
for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] =0
    discrete_splittings[i][0][discrete_splittings[i][0]==q_granularity*p_granularity**4] = 0

# Build Model

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

# Define input to RNN cell
input_daughters = tf.keras.Input((None,8), name='Input_Daughters')

# Masking input daughters
masked_input = tf.keras.layers.Masking(mask_value=-1, name='Masked_Input_Daughters')(input_daughters)

# Define RNN cell
rnn_cell = tf.keras.layers.SimpleRNN(100, name='RNN', activation='tanh', return_sequences=True, bias_initializer='glorot_normal')(masked_input)

# End shower
end_hidden = tf.keras.layers.Dense(100, name='End_Hidden_Layer', activation='relu')(rnn_cell)
end_output = tf.keras.layers.Dense(1, name='End_Output_Layer', activation='sigmoid')(end_hidden)

## Choose Mother
mother_hidden = tf.keras.layers.Dense(100, name='Mother_Hidden_Layer', activation='relu')(rnn_cell)
mother_output = tf.keras.layers.Dense(100, name='Mother_Output_Layer', activation='softmax')(mother_hidden)

mother_weights = tf.keras.Input((None, 100), name='mother_weights')
mother_weighted_output = tf.keras.layers.multiply([mother_weights, mother_output])

normalization = tf.keras.layers.Activation(activation_average)(mother_weighted_output)

## Branching Function
input_mother_momenta = tf.keras.Input((None, 4), name='Input_Mother_Momenta')

# Masking Mother Momenta for branching function
masked_mother_momenta = tf.keras.layers.Masking(mask_value=-1, name='Masked_Mother_Momenta')(input_mother_momenta)

# Merge rnn & mother momenta inputs to branching function
branch_input = tf.keras.layers.concatenate([rnn_cell, masked_mother_momenta], axis=-1)

branch_hidden = tf.keras.layers.Dense(100,   name='Branch_Hidden_Layer', activation='relu')(branch_input)
branch_output = tf.keras.layers.Dense(p_granularity**4 * q_granularity, name='Branch_Output_Layer', activation='softmax')(branch_hidden)


# In[13]:

model = tf.keras.models.Model(
    inputs=[input_daughters, input_mother_momenta, mother_weights], 
    outputs=[end_output, normalization, branch_output])
print(model.summary(), flush=True)


# ## Train Model

for lr in [1e-2, 1e-3, 1e-4]:
  print('Using learning rate ', lr, flush=True)
  model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr), loss=['binary_crossentropy', categorical_crossentropy2, 'sparse_categorical_crossentropy'])
  for epoch in range(n_epochs):
    print("Epoch: ", epoch, flush=True)
    l = 0 
    for n in ([n % len(daughters) for n in range(2*len(daughters))] if lr < 5e-2 else range(len(daughters))):
      start_time = time.time()
      batch_loss = model.train_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]], y=[endings[n][0], mothers[n][0], discrete_splittings[n][0]])[0]
      print("Batch {}: {}, took {} seconds".format(n, batch_loss, time.time() - start_time), flush=True)
      l += batch_loss
      if n%100==0 and n>0:
        model.save(os.path.join(data_path,
          'JUNIPR_p{}_q{}_LR{}_E{}_B{}'.format(p_granularity, q_granularity, lr, epoch, n)))
        if n%1000==0 and n>0:
          print("Batch: ", n, l/1000, flush=True)
          l=0
    model.save(os.path.join(data_path,
      'JUNIPR_p{}_q{}_LR{}_E{}'.format(p_granularity, q_granularity, lr, epoch)))
