import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys

charges = []
pts = []
jetpts = []
etas = []
phis = []
isd = []
for j in ['../data/d_jets_100GeV.out', '../data/u_jets_100GeV.out']:
    with open(j, 'r') as f:
        charge = []
        pt = []
        eta = []
        phi = []
        center_eta = 0
        center_phi = 0
        for l in f:
            if l[0] == 'D' or l[0] == 'U':
                if charge != []:
                    charges.append(np.array(charge))
                    pts.append(np.array(pt))
                    etas.append(np.array(eta))
                    phis.append(np.array(phi))
                    charge = []
                    pt = []
                    eta = []
                    phi = []
                pieces = l.split(' ')
                jetpts.append(float(pieces[-1]))
                center_eta = float(pieces[-3])
                center_phi = float(pieces[-2])
                if l[0] == 'D':
                    isd.append(True)
                else:
                    isd.append(False)
            else:
                pieces = l.split(' ')
                charge.append(float(pieces[-2]))
                pt.append(float(pieces[-1]))
                # centering the jet
                eta.append(float(pieces[0]) - center_eta)
                p = float(pieces[1]) - center_phi
                if p > 0.8:
                    p -= 6.28
                elif p < -0.8:
                    p += 6.28
                phi.append(p)
        charges.append(np.array(charge))
        pts.append(np.array(pt))
        etas.append(np.array(eta))
        phis.append(np.array(phi))

pixels = 33
r = 10 ** -5
k = 0.2

#setting up image arrays
pixel_borders = np.linspace(-0.4, 0.4, pixels + 1)
pt_images = [None] * len(isd)
charge_images = [None] * len(isd)

for i in range(len(etas)):
    pt_images[i],x,y = np.histogram2d(etas[i], phis[i], bins=pixel_borders, weights=pts[i])
    charge_images[i],x,y = np.histogram2d(etas[i], phis[i], bins=pixel_borders, weights=charges[i] * (pts[i] ** k))

pt_images = np.array(pt_images)
charge_images = np.array(charge_images)
isd = np.array(isd)

# making sure that i can catch the generation of nans. the try block below is
# working to the same effect.
skip = []
warnings.simplefilter("error", RuntimeWarning)

for j in range(len(isd)):
    try:
        # finishing up computing weighted jet charge now that i'm using a np array
        charge_images[j] *= 1/(jetpts[j] ** k)
        # normalizing
        pt_images[j] *= 1/sum(sum(pt_images[j]))
        # subtracting the mean
        pt_images[j] -= 1 / (pixels * pixels)
        # standardizing
        pt_images[j] *= 1/(np.std(pt_images[j]) + r)
    except RuntimeWarning:
        skip.append(j)
        continue

charge_images = np.delete(charge_images, skip, 0)
pt_images = np.delete(pt_images, skip, 0)
isd = np.delete(isd, skip, 0)

print(charge_images.shape)
print(pt_images.shape)
print(isd.shape)

np.save('../data/100GeV_processed_charge.npy', charge_images)
np.save('../data/100GeV_processed_pt.npy', pt_images)
np.save('../data/100GeV_processed_isd.npy', isd)
