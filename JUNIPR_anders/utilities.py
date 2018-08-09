import tensorflow as tf
import numpy as np
import pickle
import os
from utilities_coordinates import *
from utilities_generator import * # dot producs and coordinate transformations
from JUNIPR_utilities import *
import sys

# the output of this function should always be checked for the number of batches
# actualy returned.
def load_data(data_path, n_events, batch_size, p_granularity, q_granularity, 
    split_p_q, save_dir='./saved_data', reload=False):
    data_basename = os.path.splitext(os.path.basename(data_path))[0]
    save_path = os.path.join(save_dir,
      '{}_N{}_BS{}_PG{}_QG{}{}.pickled'.format(data_basename, n_events, batch_size,
      p_granularity, q_granularity, '_split' if split_p_q else ''))
#    save_path = save_dir + '/'+ data_basename + '_N' + str(n_events) + '_BS' + str(batch_size) +'_G' + str(p_granularity * q_granularity) + '.pickled'
    if not reload and os.path.exists(save_path):
        print('Getting pickled data from ' + save_path)
        with open(save_path, 'rb') as f:
            daughters = pickle.load(f)
            endings = pickle.load(f)
            mothers = pickle.load(f)
            discrete_splittings = pickle.load(f)
            mother_momenta = pickle.load(f)
    else:
        print('Loading Jets')
        # Load data
        all_tree, all_state_ids, all_parent_ids, all_splittings = [np.asarray(lst) for lst in read_in_jets(data_path, range(n_events))]
        
        # Batch data
        n_batches = all_tree.shape[0] // batch_size
        tree_batches       = [all_tree[n*batch_size:(n+1)*batch_size] for n in range(n_batches)] 
        state_ids_batches  = [all_state_ids[n*batch_size:(n+1)*batch_size] for n in range(n_batches)]
        parent_ids_batches = [all_parent_ids[n*batch_size:(n+1)*batch_size] for n in range(n_batches)]
        splittings_batches = [all_splittings[n*batch_size:(n+1)*batch_size] for n in range(n_batches)]
        
        # Derived quantities from data
        daughters = [get_daughters(t, p, s) for t, p, s in zip(tree_batches, parent_ids_batches, splittings_batches)]
        endings = [get_endings(s) for s in splittings_batches]
        mothers = [get_mothers(t, s, p) for t, s, p in zip(tree_batches, state_ids_batches, parent_ids_batches)]
        discrete_splittings = [get_discrete_splittings(s, p_granularity, q_granularity, split_p_q=split_p_q)  for s in splittings_batches]
        if split_p_q:
          discrete_splittings = list(zip(*discrete_splittings))
        mother_momenta      = [get_mother_momenta(t, p) for t,p in zip(tree_batches, parent_ids_batches)]
        
        with open(save_path, 'wb') as f:
            for item in [daughters, endings, mothers, discrete_splittings, mother_momenta]:
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return [daughters, endings, mothers, discrete_splittings, mother_momenta]     

# Convert coordinates
def splitting_to_daughters(splitting, parent):
    '''Transform splitting into daughter momenta
    Input: splitting and parent in unshifted coordinates
    Output: daughters in shifted coordinates'''
    z, theta, phi, delta, q_hard = splitting
    p_E, p_th, p_phi, q_parent = parent
    
    d2_rel = [p_E*z, theta, phi] 
    d1_rel = [p_E*(1-z), delta, (phi+np.pi) % (2*np.pi)]
    
    d2_abs = shift_mom(relative_to_absolute_frame_ETP(d2_rel, parent))
    d1_abs = shift_mom(relative_to_absolute_frame_ETP(d1_rel, parent))
    
    # return most energetic daughter first
    return np.asarray([*d1_abs, q_hard, *d2_abs, q_parent - q_hard])

# Load data

def get_endings(splittings):
    '''Returns a one-hot representation of when the shower ends for a batch of jets, and a corresponding mask with 'True' where values have been padded (consistent with numpy.ma conventions). 
    Input: 
        List of splittings for a batch of jets. 
        Shape of splittings: [batch_size, shower_length for each jet (will be different for each jet), dim_split]
    Output: 
        [One-hot representation, mask]'''
    
    # Find length of each jet
    # Length=1 means one splitting occured
    length_of_jets = [len(s) for s in splittings] 
    # Convert to one_hot representation - this will be padded to the maximum length in the batch
    one_hot_rep = tf.keras.utils.to_categorical(length_of_jets) # [Jet, time-step]
    one_hot_rep = one_hot_rep.reshape(one_hot_rep.shape + (1,)) # Reshape to [Jet, time-step, 1]
    
    # Calculate mask
    mask = np.asarray([[[i>length] for i in range(max(length_of_jets)+1)] for j, length in enumerate(length_of_jets)]) # [Jet, Time-step, 1]
    return one_hot_rep, mask
    
def get_single_jet_daughters(tree, parent_ids, splittings):
    daughters_pad = [[0,0,0,0,0,0,0,0]]
    daughters = [splitting_to_daughters(s, tree[parent_ids[i]]) for i, s in enumerate(splittings)] 
    return np.concatenate((daughters_pad, daughters), axis=0)
    
def get_daughters(tree, parent_ids, splittings):
    '''Returns daughter momenta for each splitting for a batch of jets. 
        The returned array is padded with "-1"s to make a sqare array
    Input: 
        Three arrays corresponding to 
            - tree momenta
            - parent_ids
            - splittings
        for a batch of jets.
    Output:
        Daughters # [jet, time-step, dim_mom*2]
    '''
    daughters = np.asarray([get_single_jet_daughters(tree[i], parent_ids[i], splittings[i]) for i in range(len(tree))])
    daughters = tf.keras.preprocessing.sequence.pad_sequences(daughters, dtype='float32', padding='post', value=-1)
    return daughters
    
    
def get_mothers(all_trees, all_state_ids, all_parent_ids, max_number_of_mothers=100):
    '''Returns index of which mother split at a given time step. The index is energy ordered with the most energetic particle having index 0. 
        Also returns corresponding mask with 'True' where values have been padded (consistent with numpy.ma conventions). 
    Input: 
        Three arrays corresponding to 
            - tree momenta
            - ids of intermediate state particles for each timestep
            - parent_ids
        for a batch of jets.
    Output:
        mothers [jet, time-step, energy-ordered index 0-100], mask
    
    TODO: Change variable names to improve readability of code.
    '''
    
    all_choices = []
    for tree, state_ids, parent_ids in zip(all_trees, all_state_ids, all_parent_ids):
        # Sorting state by energy
        state_momenta = [[tree[j] for j in current_state_ids] for current_state_ids in state_ids] # [shower_length, state_length, dim_mom]
        sorted_indices = [np.asarray(current_state_momenta)[:,0].argsort()[::-1] for current_state_momenta in state_momenta] # [shower_length, state_length]
        state_sorted_ids = [np.asarray(current_state_ids)[current_sorted_indices] for current_state_ids, current_sorted_indices in zip(state_ids, sorted_indices)] # same
        
        # Output from Mother Network
        # choices := next parent's index in energy sorted intermediate state
        choices = [list(current_state_sorted_ids).index(parent_id) for parent_id, current_state_sorted_ids in zip(parent_ids, state_sorted_ids)] # [shower_length - 1]
        choices_one_hot = tf.keras.utils.to_categorical(choices, num_classes=max_number_of_mothers)
        choices_one_hot = np.append(choices_one_hot, [[-1]*max_number_of_mothers], axis=0)
        all_choices.append(choices_one_hot)
    mothers = tf.keras.preprocessing.sequence.pad_sequences(all_choices, dtype='int32', padding='post', value=-1)
    
    # Calculate mask
    mothers_mask = np.asarray([[[i<=t and one_hot_choice[0]>=0 for i in range(100)] for t, one_hot_choice in enumerate(choices)] for choices in mothers])
    
    return mothers, mothers_mask


def pad_and_mask_discrete_splittings(splittings, pad_value):
    discrete_splittings = [np.append(splitting, pad_value) for splitting in splittings] 
    discrete_splittings = tf.keras.preprocessing.sequence.pad_sequences(discrete_splittings, dtype='int32', padding='post', value=pad_value)
    discrete_splittings = np.expand_dims(discrete_splittings,-1)
    # Mask
    discrete_splittings_mask = np.asarray([[cell==pad_value for cell in splittings] for splittings in discrete_splittings])
    return discrete_splittings, discrete_splittings_mask

    
def get_discrete_splittings(splittings, p_granularity, q_granularity, split_p_q = False):  
    '''Returns a one_hot representation of the discreteized splitting and a corresponding mask for padded values'''
    discrete_splittings = [[splitting_to_i(cont_splitting, p_granularity, q_granularity, split_p_q) for cont_splitting in splitting] for splitting in splittings]
    if split_p_q:
      # wrangling array shape and order so that discrete_splittings looks like
      # [ [[i_p, ...], ....], [[i_q, ...], ...] ]
      discrete_splittings = list(zip(*[zip(*splitting) for splitting in discrete_splittings]))
      return [pad_and_mask_discrete_splittings(*ob) for ob in zip(discrete_splittings, [p_granularity**4, q_granularity])]
    else:
      return pad_and_mask_discrete_splittings(discrete_splittings, p_granularity**4 * q_granularity)
    
def get_discrete_splittings_old(splittings, granularity):  
    '''Returns a one_hot representation of the discreteized splitting and a corresponding mask for padded values'''
    discrete_splittings = [np.append(tf.keras.utils.to_categorical([splitting_to_i(cont_splitting, granularity) for cont_splitting in splittings[i]], num_classes=granularity**4), [[-1]*(granularity**4)], axis=0) for i in range(len(splittings))] 
    discrete_splittings = tf.keras.preprocessing.sequence.pad_sequences(discrete_splittings, dtype='int32', padding='post', value=-1)
    
    # Mask
    discrete_splittings_mask = np.asarray([[cell<0 for cell in splittings] for splittings in discrete_splittings])
    return discrete_splittings, discrete_splittings_mask
    
    
def get_mother_momenta(tree, parent_id):
    '''Returns the shifted momenta for the mother that split at each time step for a batch of jets. The returned array is padded with -1 to make a square matrix.'''
    mother_momenta = [np.append(np.asarray([[*shift_mom(tree[i][id][:-1]), tree[i][id][-1]] for id in parent]),[[-1]*4], axis=0) for i, parent in enumerate(parent_id)]
    return tf.keras.preprocessing.sequence.pad_sequences(mother_momenta, dtype='float32', padding='post', value=-1)        
