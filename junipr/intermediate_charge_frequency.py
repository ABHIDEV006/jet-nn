import matplotlib
matplotlib.use('Agg')
from collections import Counter
from matplotlib import pyplot as plt
import sys
from numpy import std

if len(sys.argv) != 2:
  print('Input only a reclustered file path')
  sys.exit()

charges = []

with open(sys.argv[1], 'r') as f:
  for l in f:
    if l[0] == 'J':
      charges.append(int(l.split(' ')[-1]))
    elif l[0] != 'S':
      pieces = l.split(' ')
      charges.extend([float(pieces[5]), float(pieces[10])])

countsdict = dict(Counter(charges))
keys = list(countsdict.keys())
values = list(countsdict.values())

mx = max(keys)
mn = min(keys)
print('Maximum value {} with frequency {}'.format(mx, countsdict[mx]))
print('Minimum value {} with frequency {}'.format(mn, countsdict[mn]))
print('Standard deviation: {}'.format(std(charges)))

plt.bar(keys, values)
plt.savefig('chargefrequency.png')
