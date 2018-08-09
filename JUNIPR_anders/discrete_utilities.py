import os
import time
import copy
from collections import Counter

import numpy as np
import pandas as pd
#import theano
#import theano.tensor as T
import pickle

import multiprocessing
import itertools

from scipy.interpolate import RegularGridInterpolator

# Turn print statements on or off
debug = False


#def npsigmoid(x):
#    return 1 / (1 + np.exp(-x))


#def npsoftmax(x):
#    dim_x = len(x.shape)
#    num = np.exp(x - x.max(axis=dim_x-1, keepdims=True))
#    denom = num.sum(axis=dim_x-1, keepdims=True)
#    return num / denom


#def nprelu(x):
#    return np.maximum(x, 0)


#def phi_func(phi, lib=T):
#    '''Convert string to theano or numpy function'''
#    if lib not in [np, T]:
#        print('Invalid library. Must be T or np')
#    if phi == 'relu':
#        if lib == T:
#            return T.nnet.relu
#        elif lib == np:
#            return nprelu
#    elif phi == 'tanh':
#        if lib == T:
#            return T.tanh
#        elif lib == np:
#            return np.tanh
#    elif phi == 'sigmoid' or phi == 'sigm':
#        if lib == T:
#            return T.nnet.sigmoid
#        elif lib == np:
#            return npsigmoid
#    elif phi == 'linear':
#        return lambda x: x


#def param_init(rows=1, cols=None, planes=None):
#    '''Initialize network weights'''
#    if rows == 1 and cols is None:
#        return theano.shared(np.asscalar(np.random.randn(1).astype(theano.config.floatX)))
#    if cols is None:
#        sigma = np.sqrt(1.0 / rows)
#        return theano.shared(sigma*np.random.randn(rows).astype(theano.config.floatX))
#    if planes is None:
#        sigma = np.sqrt(2.0 / (rows + cols))
#        return theano.shared(sigma*np.random.randn(rows, cols).astype(theano.config.floatX))
#    sigma = np.sqrt(2.0 / (rows * cols + planes))
#    return theano.shared(sigma*np.random.randn(rows, cols, planes).astype(theano.config.floatX))


# For feature scaling
e_jet = 500.0 # JUST CHANGING THIS BY HAND AT THE MOMENT -- BAD
e_sub = 1.0
r_jet = np.pi / 2
r_sub = 0.1
e_shift = np.log(e_sub)
e_scale = np.log(e_jet) - e_shift
mom_theta_shift = np.log(e_sub * r_sub / e_jet)
mom_theta_scale = np.log(r_jet) - mom_theta_shift
phi_shift = 0
phi_scale = 2 * np.pi - phi_shift
z_shift = np.log(e_sub / e_jet)
z_scale = np.log(0.5) - z_shift 
split_theta_shift = np.log(r_sub / 2.0)
split_theta_scale = np.log(r_jet) - split_theta_shift 
delta_shift = np.log(e_sub * r_sub / e_jet)
delta_scale = np.log(r_jet / 2.0) - delta_shift


def shift_mom(mom):
    e, theta, phi = mom
    e2 = (np.log(np.clip(e, 1e-8, 1e8)) - e_shift) / e_scale
    theta2 = (np.log(np.clip(theta, 1e-8, 1e8)) - mom_theta_shift) / mom_theta_scale
    phi2 = (phi - phi_shift) / phi_scale
    return np.asarray([e2, theta2, phi2])
    
    
def unshift_mom(mom2):
    e2, theta2, phi2 = mom2
    e = np.exp(e2 * e_scale + e_shift)
    theta = np.exp(theta2 * mom_theta_scale + mom_theta_shift)
    phi = phi2 * phi_scale + phi_shift
    return np.asarray([e, theta, phi])
    
    
def shift_split(split):
    z, theta, phi, delta = split
    z2 = (np.log(np.clip(z, 1e-8, 1e8)) - z_shift) / z_scale
    theta2 = (np.log(np.clip(theta, 1e-8, 1e8)) - split_theta_shift) / split_theta_scale
    phi2 = (phi - phi_shift) / phi_scale
    delta2 = (np.log(np.clip(delta, 1e-8, 1e8)) - delta_shift) / delta_scale
    return [z2, theta2, phi2, delta2]


def shifted_splitting_to_ituple(shifted_splitting, granularity):
    z2, theta2, phi2, delta2 = shifted_splitting
    width = 1 / granularity
    i_z = int(np.clip(z2 / width, 0, granularity-1))
    i_theta = int(np.clip(theta2 / width, 0, granularity-1))
    i_phi = int(np.clip(phi2 / width, 0, granularity-1))
    i_delta = int(np.clip(delta2 / width, 0, granularity-1))
    ituple = [i_z, i_theta, i_phi, i_delta]
    return ituple


def ituple_to_i(ituple, granularity):
    n = granularity
    i_z, i_theta, i_phi, i_delta = ituple
    i = i_z * n**3 + i_theta * n**2 + i_phi * n + i_delta
    return i


def splitting_to_i(splitting, granularity):
    return ituple_to_i(shifted_splitting_to_ituple(shift_split(splitting), granularity), granularity)


def i_to_ituple(i, granularity):
    n = granularity
    i_z, r = np.divmod(i, n**3)
    i_theta, r = np.divmod(r, n**2)
    i_phi, r = np.divmod(r, n)
    i_delta = r
    ituple = [i_z, i_theta, i_phi, i_delta]
    return ituple


def ituple_to_shifted_splitting(ituple, granularity):
    i_z, i_theta, i_phi, i_delta = ituple
    width = 1 / granularity
    z2 = width * (i_z + 0.5)
    theta2 = width * (i_theta + 0.5)
    phi2 = width * (i_phi + 0.5)
    delta2 = width * (i_delta + 0.5)
    return z2, theta2, phi2, delta2


def unshift_split(split2):
    z2, theta2, phi2, delta2 = split2
    z = np.exp(z2 * z_scale + z_shift)
    theta = np.exp(theta2 * split_theta_scale + split_theta_shift)
    phi = phi2 * phi_scale + phi_shift
    delta = np.exp(delta2 * delta_scale + delta_shift)
    return [z, theta, phi, delta]


def i_to_splitting(i, granularity):
    return unshift_split(ituple_to_shifted_splitting(i_to_ituple(i, granularity), granularity))

def o_s_coordinates_and_values(o_s, granularity):
    ''' Find coordinates and value for every cell:
    Input: shifted_splitting [4], o_s [granularity**4], granularity[1]
    Output: [points, values] where points = [4, granularity]  and values = [granularity,granularity,granularity,granularity]
    Note: coordinates gives the value at the center of the cell. 
    '''
    z     = np.linspace(1/granularity/2, 1-1/granularity/2, granularity)
    theta = np.linspace(1/granularity/2, 1-1/granularity/2, granularity)
    phi   = np.linspace(1/granularity/2, 1-1/granularity/2, granularity)
    delta = np.linspace(1/granularity/2, 1-1/granularity/2, granularity)
    
    os = np.asarray(o_s).reshape(granularity,granularity,granularity,granularity)
    return [z,theta,phi,delta], os
    
def sample_points(shifted_splitting, granularity, fine_granularity):
    '''
    Find coordinates to new points to sample from around the shifted_splitting
    '''
    
    z, theta, phi, delta = shifted_splitting
    z_low, theta_low, phi_low, delta_low = [s-1/granularity/2 +1/granularity/fine_granularity/2 for s in shifted_splitting]
    z_high, theta_high, phi_high, delta_high = [s+1/granularity/2 -1/granularity/fine_granularity/2 for s in shifted_splitting]
    # low and high limits are the center values of the new fine grid. 
    # the 1/granularity/2 shifts from the center of the old granularity to the edge of the cell
    # the 1/granularity/fine_granularity/2 shifts from the edge of the cell to the center of the cells of fine_granularity
    
    zs     = np.linspace(z_low, z_high, fine_granularity)
    thetas = np.linspace(theta_low, theta_high, fine_granularity)
    phis   = np.linspace(phi_low, phi_high, fine_granularity)
    deltas = np.linspace(delta_low, delta_high, fine_granularity)
    
    sample_points = [(z,theta, phi, delta) for z in zs for theta in thetas for phi in phis for delta in deltas]
    return sample_points

def refine_sampling(shifted_splitting, o_s, granularity, fine_granularity, weighted = False):
    '''
    Sample in fin granularity grid around a splitting
    Input: 
    - shifted_splitting to be refined
    - o_s probabilities [granularity**4]
    - granularity
    - fine_granularity = new granularity around shifted_splitting within the old granularity
    Output:
    - unshifted_splitting
    '''
    
    # sample points and values from coarse sampling
    coordinates, values = o_s_coordinates_and_values(o_s, granularity) 
    
    # sample points for refined sampling
    new_sample_points = sample_points(shifted_splitting, granularity, fine_granularity) 
    
    # probabilities interpolated at new sample points
    interpolating_function = RegularGridInterpolator(coordinates, values, bounds_error=False, fill_value = None)
    
    new_probs = np.clip(interpolating_function(new_sample_points), 0, np.inf)
    norm = np.sum(new_probs)
    new_probs = new_probs/norm # normalize new probabilities
    
    # sample from new probabilities
    new_sampled_index = np.random.multinomial(n=1, pvals=new_probs).argmax() # this outputs integer => discretized splitting 
    shifted_sampled_splitting = new_sample_points[new_sampled_index]
    
    if weighted:
        return unshift_split(shifted_sampled_splitting), norm/(fine_granularity**4)
    else:
        return unshift_split(shifted_sampled_splitting)
'''
def get_batch_lists(data, max_batch_size, min_batch_size):
    # Filter out rare lengths
    lengths = [len(shower) for shower in data]
    data_DF = pd.DataFrame({'Length': lengths, 'Shower': list(data)})
    filtered_lengths = []
    counter_lengths = Counter(lengths)
    all_lengths = np.asarray(list(counter_lengths.keys()))
    np.random.shuffle(all_lengths)
    for ell in all_lengths:
        if counter_lengths[ell] >= min_batch_size:
            filtered_lengths.append(ell)
    # Build batch lists
    batch_lists = []
    for ell in filtered_lengths:
        jet_indices = data_DF[data_DF['Length'] == ell].index.values
        np.random.shuffle(jet_indices)
        uncut_size = len(jet_indices)
        if uncut_size <= max_batch_size:
            batch_lists.append(jet_indices)
        else:
            num_big_batches = uncut_size // max_batch_size
            for i in range(num_big_batches):
                batch_lists.append(jet_indices[i*max_batch_size : (i+1)*max_batch_size])
            if uncut_size - num_big_batches * max_batch_size >= min_batch_size:
                batch_lists.append(jet_indices[num_big_batches * max_batch_size : uncut_size])
    np.random.shuffle(batch_lists)
    return batch_lists


def get_batches(all_data, max_batch_size, min_batch_size):
    batch_lists = get_batch_lists(all_data[0], max_batch_size, min_batch_size)
    all_data_batches = []
    for data in all_data:
        data = np.asarray(data)
        data_batches = []
        for batch in batch_lists:
            data_batches.append(np.asarray(list(data[batch])))
        all_data_batches.append(data_batches)
    return all_data_batches
'''


def get_batches_as_lists(all_data, batch_size):
    num_data = len(all_data[0])
    batch_lists = np.arange(num_data)
    batch_lists = batch_lists[: (num_data // batch_size) * batch_size]
    #np.random.shuffle(batch_lists)
    batched_indices = batch_lists.reshape(num_data // batch_size, batch_size)
    all_data_batches = []
    for data in all_data:
        data_batches = [[data[index] for index in indices] for indices in batched_indices]
        all_data_batches.append(data_batches)
    return all_data_batches


def pad_batches(all_data_batches, dim_mom):
    batched_choices, batched_parents, batched_splittings, batched_daughters = all_data_batches
    padded_choices = [] # [num_batches, batch_length, shower_length]
    for batch in batched_choices:
        max_shower_length = max([len(shower) for shower in batch])
        padded = np.asarray([shower + [-1] * (max_shower_length - len(shower)) for shower in batch])
        padded_choices.append(padded)
    padded_parents = [] # [num_batches, batch_length, shower_length, dim_mom]
    for batch in batched_parents:
        max_shower_length = max([len(shower) for shower in batch])
        padded = np.asarray([shower + [[-1.] * dim_mom] * (max_shower_length - len(shower)) for shower in batch])
        padded_parents.append(padded)
    padded_splittings = [] # [num_batches, batch_length, shower_length]
    for batch in batched_splittings:
        max_shower_length = max([len(shower) for shower in batch])
        padded = np.asarray([shower + [-1] * (max_shower_length - len(shower)) for shower in batch])
        padded_splittings.append(padded)
    padded_daughters = [] # [num_batches, batch_length, shower_length, 2*dim_mom]
    padded_endings = [] # [num_batches, batch_length, shower_length]
    batched_masks = [] # [num_batches, batch_length, shower_length]
    for batch in batched_daughters:
        max_shower_length = max([len(shower) for shower in batch])
        padded = np.asarray([shower + [[-1.] * 2*dim_mom] * (max_shower_length - len(shower)) for shower in batch])
        padded_daughters.append(padded)
        endings = np.asarray([[0] * (len(shower) - 1) + [1] + [0] * (max_shower_length - len(shower)) for shower in batch])
        padded_endings.append(endings)
        masks = np.asarray([[1.] * len(shower) + [0.] * (max_shower_length - len(shower)) for shower in batch])
        batched_masks.append(masks)
    return padded_choices, padded_parents, padded_splittings, padded_daughters, padded_endings, batched_masks


def reserve_test_set(training_sets, frac=0.1):
    '''Input: list of training sets, each of which is a list of batches
    Output: list of testing sets; these batches have been removed from training sets'''
    test_sets = []
    for training_set in training_sets:
        test_set = []
        test_size = int(frac * len(training_set))
        for _ in range(test_size):
            test_set.append(training_set.pop())
        test_sets.append(test_set)
    return test_sets


def choose_sample_set(training_sets, frac=0.1):
    '''Sample a fraction of the training batches'''
    sample_sets = []
    for training_set in training_sets:
        sample_set = []
        sample_size = int(frac * len(training_set))
        for i in range(sample_size):
            sample_set.append(training_set[i])
        sample_sets.append(sample_set)
    return sample_sets


def list_to_string(p):
    p_string = ''
    for i in range(len(p)):
        p_string += str(p[i])
        if i < len(p)-1:
            p_string += ' '
    return p_string


def floatlist_to_string(num, prec=1):
    num_string = ''
    for i in range(len(num)):
        num_string += '{:.{}e}'.format(num[i], prec)
        if i < len(num)-1:
            num_string += ' '
    return num_string


def print_param_names(model, f):
    print('UN {}'.format(list_to_string(model.names[model.u_slice])), file=f, flush=True)
    print('EN {}'.format(list_to_string(model.names[model.e_slice])), file=f, flush=True)
    print('PN {}'.format(list_to_string(model.names[model.p_slice])), file=f, flush=True)
    print('SN {}'.format(list_to_string(model.names[model.s_slice])), file=f, flush=True)


def print_param_norms(model, f):
    u_norms, e_norms, p_norms, s_norms = model.param_norms()
    print('UV {}'.format(list_to_string(u_norms)), file=f, flush=True)
    print('EV {}'.format(list_to_string(e_norms)), file=f, flush=True)
    print('PV {}'.format(list_to_string(p_norms)), file=f, flush=True)
    print('SV {}'.format(list_to_string(s_norms)), file=f, flush=True)


def print_param_steps(model, batch, f):
    uni_e_steps, uni_p_steps, uni_s_steps, ful_e_steps, ful_p_steps, ful_s_steps = model.param_rel_step_sizes(batch)
    print('UES {}'.format(list_to_string(uni_e_steps)), file=f, flush=True)
    print('UPS {}'.format(list_to_string(uni_p_steps)), file=f, flush=True)
    print('USS {}'.format(list_to_string(uni_s_steps)), file=f, flush=True)
    print('FES {}'.format(list_to_string(ful_e_steps)), file=f, flush=True)
    print('FPS {}'.format(list_to_string(ful_p_steps)), file=f, flush=True)
    print('FSS {}'.format(list_to_string(ful_s_steps)), file=f, flush=True)


def print_progress(model, sample_set, test_set, f_log, f_loss, percent_string):
    
    try:
        start_time = print_progress.start_time
    except:
        print_progress.start_time = time.time()
        start_time = print_progress.start_time
    try:
        check_point = print_progress.check_point
    except:
        print_progress.check_point = time.time()
        check_point = print_progress.check_point
        
    print('ET {:.6e}'.format(time.time() - start_time), file=f_loss, flush=True)
        
    sample_loss = 0
    for batch in zip(*sample_set):
        sample_loss += model.loss(batch)
    sample_loss /= len(sample_set[0])
    test_loss = 0
    for batch in zip(*test_set):
        test_loss += model.loss(batch)
    test_loss /= len(test_set[0])
        
    print('SL {}'.format(floatlist_to_string(sample_loss, 6)), file=f_loss, flush=True)
    print('TL {}'.format(floatlist_to_string(test_loss, 6)), file=f_loss, flush=True)
    print('\t\t{}% complete '.format(percent_string) 
          + 'with training loss {:.5e} (testing loss {:.5e}) '.format(np.sum(sample_loss), np.sum(test_loss))
          + 'after {:.1e} more minutes'.format((time.time()-check_point)/60.0), file=f_log, flush=True)
    
    print_progress.check_point = time.time()
