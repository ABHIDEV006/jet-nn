import numpy as np
from jet_RGB_images import pixelate

charges = []
pts = []
etas = []
phis = []
isd = []
for j in ['d_jets.out', 'u_jets.out']:
    with open(j, 'r') as f:
        charge = []
        pt = []
        eta = []
        phi = []
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
                if l[0] == 'D':
                    isd.append(True)
                else:
                    isd.append(False)
            else:
                pieces = l.split(' ')
                charge.append(float(pieces[-2]))
                pt.append(float(pieces[-1]))
                # centering the jet
                eta.append(float(pieces[0]))
                phi.append(float(pieces[1]))
        charges.append(np.array(charge))
        pts.append(np.array(pt))
        etas.append(np.array(eta))
        phis.append(np.array(phi))

jets = zip(etas, phis, pts, charges)
images = [None] * len(etas)

for i, (e, ph, pt, c) in enumerate(jets):
    images[i] = pixelate(np.array([e, ph, pt, c]).transpose(), charge_image = True, K = 0.2, nb_chan = 2)

print(len(images))

np.save('data/patrick_images.npy', images)
