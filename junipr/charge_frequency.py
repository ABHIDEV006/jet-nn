import matplotlib
matplotlib.use('Agg')
from collections import Counter
from matplotlib import pyplot as plt
import sys

if len(sys.argv) != 2:
  print('Input only a reclustered file path')
  sys.exit()

charges = []

with open(sys.argv[1], 'r') as f:
  for l in f:
    if l[0] == 'J':
      charges.append(int(l.split(' ')[-1]))

countsdict = dict(Counter(charges))
keys = list(countsdict.keys())
values = list(countsdict.values())

plt.bar(keys, values)
plt.savefig('chargefrequency.png')
