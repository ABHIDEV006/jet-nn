import numpy as np
import matplotlib.pyplot as plt

#charge_images = np.load('data/charge_images.npy')
#pt_images = np.load('data/pt_images.npy')
#
#for i, imgs in enumerate(np.stack([charge_images, pt_images], axis=3)[:10]):
#    plt.title(i)
#    plt.imshow(imgs[:,:,0])
#    plt.colorbar()
#    plt.show()
#    plt.title(i)
#    plt.imshow(imgs[:,:,1])
#    plt.colorbar()
#    plt.show()

images = np.load('data/events/100GEV-downquark-K=0.3-K2=0-jetimage-seed7_33x33images_2chan.npz')['arr_0']
for i in range(10):
    plt.imshow(images[i][0])
    plt.colorbar()
    plt.show()
    plt.imshow(images[i][1])
    plt.colorbar()
    plt.show()

