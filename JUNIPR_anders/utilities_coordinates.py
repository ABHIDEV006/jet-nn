import numpy as np
import encoding

#####################
## Feature scaling ##
#####################

e_jet = 500.0
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


def ituple_to_i(ituple, p_granularity):
    n = p_granularity
    i_z, i_theta, i_phi, i_delta = ituple
    i = i_z * n**3 + i_theta * n**2 + i_phi * n + i_delta
    return i


def shift_q(q, q_granularity):
    return int(np.clip(q + (q_granularity // 2), 0, q_granularity - 1))


def unshift_q(i_q, q_granularity):
    return i_q - (q_granularity // 2)


def splitting_to_i(splitting, p_granularity, q_granularity, split_p_q = False):
    i_p = ituple_to_i(shifted_splitting_to_ituple(shift_split(splitting[:-1]), p_granularity), p_granularity)
    i_q = shift_q(splitting[-1], q_granularity)
    if split_p_q:
      return (i_p, i_q)
    else:
      return encoding.obs_to_i([p_granularity ** 4, q_granularity], [i_p, i_q])
    # the old method:
    #return ituple_to_i([*shifted_splitting_to_ituple(shift_split(splitting[:-1]), p_granularity), 
    #    int(np.clip(splitting[-1] + q_granularity, 0, q_granularity - 1))], p_granularity)


def i_to_ituple(i, p_granularity, q_granularity):
    i_p, i_q = encoding.get_all_obs(i, [p_granularity**4, q_granularity])
    n = p_granularity
    i_z, r = np.divmod(i_p, n**3)
    i_theta, r = np.divmod(r, n**2)
    i_phi, r = np.divmod(r, n)
    i_delta = r
    ituple = [i_z, i_theta, i_phi, i_delta, i_q]
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


def i_to_splitting(i, p_granularity, q_granularity):
    i_z, i_theta, i_phi, i_delta, i_q = i_to_ituple(i, p_granularity)
    return [*unshift_split(ituple_to_shifted_splitting([i_z, i_theta, i_phi, i_delta],
        p_granularity)), unshift_q(i_q, q_granularity)]
