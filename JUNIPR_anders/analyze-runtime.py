import numpy as np
import sys
import re

if len(sys.argv) != 2:
  print('Supply only path to stats file.')
  sys.exit(1)

fn = sys.argv[1]

epochs = []

with open(fn) as f:
  found_beginning = False
  epoch = 0
  for l in f:
    if l[:5] == 'Using':
      if found_beginning:
        epochs.append(epoch / 100)
        epoch = 0
      else:
        found_beginning = True
    elif found_beginning:
      m = re.search('took (.*) seconds', l)
      if m is not None:
        epoch += float(m.group(1))
  epochs.append(epoch / 100)

print(epochs)
