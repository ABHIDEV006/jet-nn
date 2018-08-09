
# coding: utf-8

# # Setup

# In[1]:

import os
import tensorflow as tf
from utilities import *
from JUNIPR_utilities import *
from utilities_coordinates import *
from utilities_generator import * # dot producs and coordinate transformations
#from discrete_utilities import *
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# # Load Data

# In[2]:

# Setup parameters
granularity = 10
batch_size = 10

n_events = 5*10**5
n_batches = n_events//batch_size


# In[3]:

# path to saved jets
data_path = '/project/Chris/FASTJET3/shower_model/final_data/LEP1tev_dijets_j1_r0.1_p0_energycutFIXED_tree.dat'


# ## Load in jets from file

# In[4]:

[daughters, endings, mothers, discrete_splittings, mother_momenta] = load_data(data_path, n_events=n_events, batch_size=batch_size, granularity=granularity)


# In[5]:

for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] =0


# In[6]:

batch_number = 3
print(daughters[batch_number].shape)
print(endings[batch_number][0].shape)
print(mothers[batch_number][0].shape)
print(discrete_splittings[batch_number][0].shape)
print(mother_momenta[batch_number].shape)


# # Simple RNN

# ## Build Model

# In[7]:

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


# In[8]:

# Define input to RNN cell
input_daughters = tf.keras.Input((None,6), name='Input_Daughters')

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
input_mother_momenta = tf.keras.Input((None, 3), name='Input_Mother_Momenta')

# Masking Mother Momenta for branching function
masked_mother_momenta = tf.keras.layers.Masking(mask_value=-1, name='Masked_Mother_Momenta')(input_mother_momenta)

# Merge rnn & mother momenta inputs to branching function
branch_input = tf.keras.layers.concatenate([rnn_cell, masked_mother_momenta], axis=-1)

branch_hidden = tf.keras.layers.Dense(100,   name='Branch_Hidden_Layer', activation='relu')(branch_input)
branch_output = tf.keras.layers.Dense(granularity**4, name='Branch_Output_Layer', activation='softmax')(branch_hidden)


# In[ ]:

#model = tf.keras.models.Model(inputs=[input_daughters, input_mother_momenta], outputs=[end_output, mother_output, branch_output])
model = tf.keras.models.Model(
    inputs  = [input_daughters, input_mother_momenta, mother_weights], 
    outputs = [end_output, normalization, branch_output])
print(model.summary())


# ## Train Model

# In[ ]:

for lr in [1e-2, 1e-3, 1e-4]:
    print('Using learning rate ', lr)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr), loss=['binary_crossentropy', categorical_crossentropy2, 'sparse_categorical_crossentropy'])
    for epoch in range(5):
        print("Epoch: ", epoch)
        l = 0 
        for n in range(len(daughters)):
            l = l + model.train_on_batch(x=[daughters[n], mother_momenta[n], mothers[n][1]], y=[endings[n][0], mothers[n][0], discrete_splittings[n][0]])[0]
            if n%1000==0 and n>0:
                print("Batch: ", n, l/1000)
                l=0
        model.save_weights('JUNIPR_July12_LR'+str(lr)+"_E"+str(epoch))


# In[67]:

model.test_on_batch(x=[daughters[0], mother_momenta[0], mothers[0][1]], y=[endings[0][0], mothers[0][0]])


# In[18]:

model.compile(optimizer='sgd', loss=['binary_crossentropy', weighted_categorical_crossentropy(mothers[0][1]), 'sparse_categorical_crossentropy'])
model.test_on_batch(x=[daughters[0], mother_momenta[0]], y=[endings[0][0], mothers[0][0], discrete_splittings[0][0]])


# In[152]:

model.load_weights('JUNIPR_saved_weights_JULY11')


# In[150]:

model2 = tf.keras.models.load_model('JUNIPR_saved_JULY11')


# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# mothers[0].shape

# mother_outputs[0].shape

# mother_mask_padded[0].shape

# ## Validate Model

# In[121]:

endings_out    = np.zeros((2, 100, 1))
mothers_out    = np.zeros((2, 100, 100))
branchings_out = np.zeros((2,100, granularity**4))


# In[122]:

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
                branchings_out[0,t]+=b[i,t]
                branchings_out[1,t]+=np.ones_like(b[i,t])


# In[129]:

avg_e = endings_out[0]/np.clip(endings_out[1], 0.1 , np.inf)
avg_m = mothers_out[0]/np.clip(mothers_out[1], 0.1 , np.inf)
avg_b = branchings_out[0]/np.clip(branchings_out[1], 0.1 , np.inf)


# In[130]:

print(avg_e.shape)
print(avg_m.shape)
print(avg_b.shape)


# In[23]:

model.compile(optimizer='sgd', loss=[categorical_crossentropy2])
model.test_on_batch(x=[daughters[0], mother_momenta[0], mothers[0][1]], y=[mothers[0][0]])


# In[58]:

model.train_on_batch(x=[daughters[0], mother_momenta[0], mothers[0][1]], y=[mothers[0][0]])


# ### End Shower

# In[134]:

plt.plot(avg_e)
plt.ylim(0,1)
plt.xlim(0,50)
plt.show()


# In[56]:

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
    n_batches = len(out)
    padded_endings = []
    padded_mask = []
    for i in range(n_batches):
        padded_endings.append(tf.keras.preprocessing.sequence.pad_sequences(out[i], maxlen=maxlen, dtype='float32', padding='post', value=0))
        padded_mask.append(tf.keras.preprocessing.sequence.pad_sequences(data[i][1], maxlen=maxlen, dtype='bool', padding='post', value=True))
    
    endings = np.asarray(padded_endings)
    mask = np.asarray(padded_mask)
    return np.ma.average(np.ma.masked_array(endings, mask = mask), axis=(0,1))


# In[69]:

plt.plot(average_endings_data(endings), label='Pythia')
plt.plot(average_endings_output(endings_out, endings), label='JUNIPR')
plt.legend()
plt.show()


# ### Choose Parent

# In[142]:

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


# In[143]:

plt.plot(average_mothers_data(mothers), '.', label="Pythia")
plt.plot(np.mean(avg_m, axis=0), '.', label='JUNIPR')
plt.ylim(0,1)
plt.xlim(0,10)
plt.legend()
plt.show()


# ### Branching Function

# In[148]:

plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(1,2,3)))
plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(0,2,3)))
plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(0,1,3)))
plt.plot(np.mean(np.mean(avg_b, axis=0).reshape((10,10,10,10)), axis=(0,1,2)))
plt.show()


# In[19]:

branch_data = np.ma.average(np.ma.masked_array(discrete_splittings_training, mask = discrete_splittings_mask_training), axis=(0,1))


# In[20]:

branch_JUNIPR = np.ma.average(np.ma.masked_array(branch_outputs, mask = discrete_splittings_mask_training), axis=(0,1))


# In[21]:

z_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(1,2,3))
theta_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(0,2,3))
phi_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(0,1,3))
delta_JUNIPR = np.mean(branch_JUNIPR.reshape((10,10,10,10)), axis=(0,1,2))

z_JUNIPR = z_JUNIPR/np.sum(z_JUNIPR)
theta_JUNIPR = theta_JUNIPR/np.sum(theta_JUNIPR)
phi_JUNIPR = phi_JUNIPR/np.sum(phi_JUNIPR)
delta_JUNIPR = delta_JUNIPR/np.sum(delta_JUNIPR)


# In[22]:

z_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(1,2,3))
theta_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(0,2,3))
phi_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(0,1,3))
delta_data = np.mean(branch_data.reshape((10,10,10,10)), axis=(0,1,2))

z_data = z_data/np.sum(z_data)
theta_data = theta_data/np.sum(theta_data)
phi_data = phi_data/np.sum(phi_data)
delta_data = delta_data/np.sum(delta_data)


# In[23]:

plt.plot(z_JUNIPR)
plt.plot(z_data)
plt.ylim(0,max(z_data)*1.1)
plt.show()


# In[24]:

plt.plot(theta_JUNIPR)
plt.plot(theta_data)
plt.ylim(0,max(theta_data)*1.1)
plt.show()


# In[25]:

plt.plot(phi_JUNIPR)
plt.plot(phi_data)
plt.ylim(0, max(phi_data)*1.1)
plt.show()


# In[26]:

plt.plot(delta_JUNIPR)
plt.plot(delta_data)
plt.ylim(0, max(delta_data)*1.1)
plt.show()


# In[ ]:



