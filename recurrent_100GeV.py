import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Masking
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import Adam
from random import shuffle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import glob

class Checkpoint(Callback):
    def __init__(self, filepath, epochs_completed):
        super(Checkpoint, self).__init__()
        self.filepath = filepath
        self.epochs_completed = epochs_completed
        
    def on_epoch_end(self, epoch, logs=None):
        fn = self.filepath.format(epoch=epoch + 1 + self.epochs_completed)
        print('Saving {}'.format(fn))
        self.model.save(fn, overwrite=True)

TOTAL_EPOCHS = 100
n_epochs = 100
length = 40
batch = 6000
saves = sorted(glob.glob('../data/length_{}_batch_{}_rnn_100GeV/weights*.hdf5'.format(length, batch)))
model = None
if len(saves) != 0:
    print('Loading saved model...')
    model = load_model(saves[-1])
    n_epochs -= int(saves[-1][-7:-5])
    print('Retrieved {} ...'.format(saves[-1]))
else:
    model = Sequential()
    model.add(Masking(input_shape=(length, 5), mask_value=-999.))
    model.add(GRU(100, return_sequences=True))
    model.add(GRU(90, return_sequences=True))
    model.add(GRU(80, return_sequences=True))
    model.add(GRU(70, return_sequences=True))
    model.add(GRU(60, return_sequences=True))
    model.add(GRU(50, return_sequences=True))
    model.add(GRU(40, return_sequences=True))
    model.add(GRU(30, return_sequences=True))
    model.add(GRU(20, return_sequences=True))
    model.add(GRU(10, return_sequences=True))
    model.add(GRU(5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Model compiled...')
    
print(model.summary())

input_vecs = np.load('../data/length_40_rnn_processed_vec_100GeV.npy')
print('Data loaded...')
#isd = np.load('data/all_isd.npy')
isd = [i < 100000 for i in range(200000)]
shuffled_indices = list(range(len(input_vecs)))
shuffle(shuffled_indices)
input_vecs = np.array([input_vecs[i] for i in shuffled_indices])
isd = np.array([isd[i] for i in shuffled_indices])

checkpointer = Checkpoint('../data/length_{}_batch_{}_rnn_100GeV/weights'.format(length, batch) + '{epoch:03d}.hdf5', TOTAL_EPOCHS - n_epochs)
print('Fitting...')
model.fit(input_vecs[:160000], to_categorical(isd[:160000]), batch_size=batch, epochs=n_epochs, verbose=1,
          validation_data=(input_vecs[160000:180000], to_categorical(isd[160000:180000])), callbacks=[checkpointer])
print(model.evaluate(input_vecs[180000:], to_categorical(isd[180000:]), batch_size=500))

# making predictions to evaluate performance
predictions = model.predict(input_vecs[180000:])
print(len(predictions))
fpr, tpr, thresholds = roc_curve(to_categorical(isd[180000:])[:,0], predictions[:,0])
plt.figure(1)
plt.plot(tpr, 1-fpr)
plt.title('roc')
plt.savefig('/n/home03/tculp/jets/figures/rnn_roc_length_{}_batch_{}_100GeV.png'.format(length, batch))
plt.figure(2)
plt.plot(tpr, tpr/np.sqrt(fpr))
plt.title('sic')
plt.savefig('/n/home03/tculp/jets/figures/rnn_sic_length_{}_batch_{}_100GeV.png'.format(length, batch))
print('Area Under Curve: {}'.format(auc(fpr, tpr)))
