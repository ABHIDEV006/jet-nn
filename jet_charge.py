import re
import numpy as np
import matplotlib.pyplot as plt

dcharges = []
dpt = []
Dpt = []
ucharges = []
upt = []
Upt = []
D = False
with open('shower_energycut_2.out', 'r') as f:
    charge = []
    pt = []
    for l in f:
        if l[0] == 'D' or l[0] == 'U':
            if charge != []:
                if D:
                    dcharges.append(charge)
                    dpt.append(pt)
                else:
                    ucharges.append(charge)
                    upt.append(pt)
                charge = []
                pt = []
            jetpt = float(l.split(' ')[-1])
            if l[0] == 'D':
                D = True
                Dpt.append(jetpt)
            else:
                D = False
                Upt.append(jetpt)
        else:
            pieces = l.split(' ')
            charge.append(float(pieces[-2]))
            pt.append(float(pieces[-1]))
    if D:
        dcharges.append(charge)
        dpt.append(pt)
    else:
        ucharges.append(charge)
        upt.append(pt)

num_k = 10
k_array = np.linspace(0.1, 1, num_k)

djetcharges = [[None for i in range(len(dcharges))] for j in range(num_k)]
for i in range(len(dcharges)):
    charge = np.array(dcharges[i])
    pt = dpt[i]
    for j, k in enumerate(k_array):
        djetcharges[j][i] = np.dot(np.power(pt, k), charge) / (Dpt[i] ** k)

ujetcharges = [[None for i in range(len(ucharges))] for j in range(num_k)]
for i in range(len(ucharges)):
    charge = np.array(ucharges[i])
    pt = upt[i]
    for j, k in enumerate(np.linspace(0.1, 1, num_k)):
        ujetcharges[j][i] = np.dot(np.power(pt, k), charge) / (Upt[i] ** k)

b = 50
#plotting "normalized" pt-weighted dN/dQ
for i, k in enumerate(k_array):
    plt.figure(i)
    dhist = np.histogram(djetcharges[i], bins=b, range=(-1.5, 1.5), density=True)
    step = dhist[1][1] - dhist[1][0]
    midpoints = [c + step / 2 for c in dhist[1][:-1]]
    plt.plot(midpoints, dhist[0], label='d')

    uhist = np.histogram(ujetcharges[i], bins=b, range=(-1.5, 1.5), density=True)
    step = uhist[1][1] - uhist[1][0]
    midpoints = [c + step / 2 for c in uhist[1][:-1]]
    plt.plot(midpoints, uhist[0], label='u')

    plt.rc('text', usetex=True)
    plt.title(r'$\kappa = {}$'.format(k))
    plt.xlabel('jet charge')
    plt.ylabel('dN/dQ')
    plt.legend()
    plt.savefig('figures/jetcharge_k{}.png'.format(k))

for i, k in enumerate(k_array):
    dhist, edges = np.histogram(djetcharges[i], bins=b, range=(-1.5, 1.5))
    uhist, edges = np.histogram(ujetcharges[i], bins=b, range=(-1.5, 1.5))
    dtotal = sum(dhist)
    utotal = sum(uhist)
    dbehindcut = [dhist[0]]
    uaftercut = [utotal - uhist[0]]
    for j in range(1, len(dhist)):
        dbehindcut.append(dbehindcut[j-1] + dhist[j])
        uaftercut.append(uaftercut[j-1] - uhist[j])
    dbehindcut = np.array(dbehindcut)
    uaftercut = np.array(uaftercut)
    sens = dbehindcut / dtotal
    spec = uaftercut / utotal
    # this is how katie plots it
    plt.figure(20)
    plt.plot(sens, spec, label='k={}'.format(k))
    # this is the form used in the example i saw
    #plt.plot(1 - spec, sens)
    plt.figure(30)
    plt.plot(sens, sens/np.sqrt(1 - spec), label='k={}'.format(k))

plt.figure(20)
plt.title('roc')
plt.rc('text', usetex=True)
plt.xlabel(r'$\epsilon_s$, Down Quark Jet Identification')
plt.ylabel(r'$\epsilon_b$, Up Quark Jet Rejection')
plt.legend()
plt.savefig('figures/roc.png')
plt.figure(30)
plt.title('sic')
plt.rc('text', usetex=True)
plt.xlabel(r'$\epsilon_s$, Down Quark Jet Identification')
plt.ylabel(r'$\epsilon_s/\sqrt{1-\epsilon_b}$, Up Quark Jet Rejection')
plt.legend()
plt.savefig('figures/sic.png')

