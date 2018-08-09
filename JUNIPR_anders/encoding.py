from numpy import prod

# all of the methods below assume that gs is in decreasing order. if it's not
# already, use the order method to sort it and the corresponding obs list.

def obs_to_i(gs, obs):
  # encode observables in obs into a single integer using the granularities gs
  i = obs[0]
  for ind in range(1, len(obs)):
    i = i * gs[ind] + obs[ind]
  return i

def get_single_ob(i, gs, index):
  # THIS IS SLOWER THAN get_all_obs IF you are going to loop over all indices in
  # gs. Use this ONLY for getting a single observable.
  # gets the observable corresponding to index in the array gs, where gs
  # contains granularities.
  # i is all the observables encoded into an integer
  return int(i // prod(gs[(index + 1):])) % gs[index]

def get_all_obs(i, gs):
  # gets all of the observables in i corresponding to the array gs, where gs
  # contains granularities.
  obs = [0] * len(gs)
  for (ind, g) in reversed(list(enumerate(gs))):
    (i, obs[ind]) = divmod(i, g)
  return obs

def order(gs, obs=None):
  # returns gs in reverse sorted order. if obs is given, then obs is sorted
  # according to the reverse ordering of gs (i.e. after sorting, elements in gs
  # will still correspond to the same elements of obs from before sorting)
  if obs is None:
    return sorted(gs, reverse=True)
  else:
    return [list(x) for x in zip(*sorted(list(zip(gs, obs)), reverse=True))]

