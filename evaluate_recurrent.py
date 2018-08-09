import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Masking
from keras.utils import to_categorical, multi_gpu_model
from keras.callbacks import Callback
from keras.optimizers import Adam
from random import shuffle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import glob

print('Loading saved model...')
model = load_model('../data/lr_aug_rnn/lr_aug_rnn_weights084.hdf5')
print(model.summary())

input_vecs = np.load('../data/length_40_rnn_processed_vec_test.npy')
print('Data loaded...')
#isd = np.load('data/all_isd.npy')
isd = [i < 20000 for i in range(40000)]
shuffled_indices = list(range(len(input_vecs)))
shuffle(shuffled_indices)
input_vecs = np.array([input_vecs[i] for i in shuffled_indices])
isd = np.array([isd[i] for i in shuffled_indices])

predictions = model.predict(input_vecs)
fpr, tpr, thresholds = roc_curve(to_categorical(isd)[:,0], predictions[:,0])
plt.figure(1)
plt.plot(tpr, 1-fpr)
plt.title('roc')
plt.savefig('/n/home03/tculp/jets/figures/stand_lr_aug_rnn_roc.png')
plt.figure(2)
plt.plot(tpr, tpr/np.sqrt(fpr))
plt.title('sic')
plt.savefig('/n/home03/tculp/jets/figures/stand_lr_aug_rnn_sic.png')
print('Area Under Curve: {}'.format(auc(fpr, tpr)))
