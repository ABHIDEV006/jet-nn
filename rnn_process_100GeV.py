import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys

timesteps = 40

charges = []
pts = []
etas = []
phis = []
isd = []
dists = []
for j in ['../data/d_jets_100GeV.out', '../data/u_jets_100GeV.out']:
    with open(j, 'r') as f:
        charge = []
        pt = []
        eta = []
        phi = []
        dist = []
        center_eta = 0
        center_phi = 0
        for l in f:
            if l[0] == 'D' or l[0] == 'U':
                if charge != []:
                    charges.append(np.array(charge))
                    pts.append(np.array(pt))
                    etas.append(np.array(eta))
                    phis.append(np.array(phi))
                    dists.append(np.array(dist))
                    charge = []
                    pt = []
                    eta = []
                    phi = []
                    dist = []
                pieces = l.split(' ')
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
                eta.append(float(pieces[0]))
                phi.append(float(pieces[1]))
                # computing distance
                p = phi[-1] - center_phi
                if p > 3.14:
                    p -= 6.28
                elif p < -3.14:
                    p += 6.28
                dist.append(np.sqrt(p**2 + (center_eta - eta[-1])**2))
        charges.append(np.array(charge))
        pts.append(np.array(pt))
        etas.append(np.array(eta))
        phis.append(np.array(phi))
        dists.append(np.array(dist))

ret = [None] * len(pts)

# handling any divide by zeros
skip = []
warnings.simplefilter("error", RuntimeWarning)

for i, (e, ph, pt, c, d) in enumerate(zip(etas, phis, pts, charges, dists)):
  try:
    pt *= 1/sum(pt)
    pt -= 1 / len(pt)
    pt *= 1 / (np.std(pt) + 10**-5)
    a = [e, ph, pt, c, d]
    if len(e) < timesteps:
      mask = [-999] * (timesteps - len(e))
      for j in range(len(a)):
        a[j] = np.append(a[j], mask)
    ret[i] = np.array([a[0][:timesteps], a[1][:timesteps], a[2][:timesteps],
      a[3][:timesteps], a[4][:timesteps]]).transpose()
  except RuntimeWarning:
    skip.append(j)
    continue

ret = np.array(ret)
isd = np.array(isd)

print(ret.shape)
print(isd.shape)

np.save('../data/length_40_rnn_processed_vec_100GeV.npy', ret)
np.save('../data/rnn_processed_isd.npy', isd)
