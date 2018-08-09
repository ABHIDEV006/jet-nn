import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import to_categorical, multi_gpu_model
from keras.callbacks import Callback
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
        
TOTAL_EPOCHS = 10
n_epochs = 10
saves = sorted(glob.glob('/n/regal/kovac_lab/tculp/jets/model_saves/100GeV_cnn_*.hdf5'))
model = None
if len(saves) != 0:
    print('loading saved model')
    model = load_model(saves[-1])
    n_epochs -= int(saves[-1][-7:-5])
else:
    model = Sequential()
    model.add(Conv2D(64, (8, 8), activation='relu', data_format='channels_last', input_shape=(33, 33,2)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.18))
    model.add(Conv2D(64, (4, 4), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.35))
    model.add(Conv2D(64, (4, 4), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('model compiled')

# loading data to feed to the neural net
# these images are 33x33
charge_images = np.load('/n/home03/tculp/jets/data/100GeV_processed_charge.npy')
pt_images = np.load('/n/home03/tculp/jets/data/100GeV_processed_pt.npy')
#isd = np.load('data/all_isd.npy')
zipped = np.stack([charge_images, pt_images], axis=3)
isd = [i < 100000 for i in range(200000)]
shuffled_indices = list(range(len(zipped)))
shuffle(shuffled_indices)
zipped = np.array([zipped[i] for i in shuffled_indices])
isd = np.array([isd[i] for i in shuffled_indices])

print(zipped.shape)

# create a callback which will save the model at each epoch in the event odyssey
# shuts me down
checkpointer = Checkpoint('/n/regal/kovac_lab/tculp/jets/model_saves/100GeV_cnn_{epoch:02d}.hdf5', TOTAL_EPOCHS - n_epochs)

# parallel model which will be able to run on multiple gpus
#parallel_model = multi_gpu_model(model, gpus=8)
#parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(zipped[:160000], to_categorical(isd[:160000]), batch_size=512, epochs=n_epochs, verbose=1, validation_data=(zipped[160000:180000], to_categorical(isd[160000:180000])), callbacks=[checkpointer])
print('Evaluating model...')
print(model.evaluate(zipped[180000:], to_categorical(isd[180000:]), batch_size=512))

# making predictions to evaluate performance
predictions = model.predict(zipped[180000:])
print(len(predictions))
fpr, tpr, thresholds = roc_curve(to_categorical(isd[180000:])[:,0], predictions[:,0])
plt.figure(1)
plt.plot(tpr, 1-fpr)
plt.title('roc')
plt.savefig('/n/home03/tculp/jets/figures/100GeV_cnn_roc.png')
plt.figure(2)
plt.plot(tpr, tpr/np.sqrt(fpr))
plt.title('sic')
plt.savefig('/n/home03/tculp/jets/figures/100GeV)cnn_sic.png')
print('Area Under Curve: {}'.format(auc(fpr, tpr)))

model.save('/n/home03/tculp/jets/data/100GeV_cnn.h5')
