import sys
import numpy as np

# change to 0 for purpose of ud jets
recluster_def = 0
SAMPLE_SIZE = 10**6

input_dir = sys.argv[1]
input_file = sys.argv[2]
input_path = input_dir + '/' + input_file
output_dir = sys.argv[3]
output_path = output_dir + '/final_' + input_file

r_jet = np.pi / 2

def list_to_string(p):
    p_string = ''
    for i in range(len(p)):
        p_string += str(p[i])
        if i < len(p)-1:
            p_string += ' '
    return p_string


def floatlist_to_string(p):
    p_string = ''
    for i in range(len(p)):
        p_string += '{:.5e}'.format(p[i])
        if i < len(p)-1:
            p_string += ' '
    return p_string


def dot3(p1,p2):
    return p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]


def norm(p):
    return np.sqrt(dot3(p,p))


def hat(p):
    return [p[i]/norm(p) for i in [0,1,2]]


def cos_theta(p1,p2):
    return dot3(p1,p2) / norm(p1) / norm(p2)


def angle(p1,p2):
    cos = cos_theta(p1,p2)
    if np.isclose(cos, 1., 1e-15, 1e-15):
        theta = 0.
    elif np.isclose(cos, -1., 1e-15, 1e-15):
        theta = np.pi
    else: 
        theta = np.arccos(cos)
    return theta


def cross(p1,p2):
    c0 = p1[1]*p2[2] - p1[2]*p2[1]
    c1 = p1[2]*p2[0] - p1[0]*p2[2]
    c2 = p1[0]*p2[1] - p1[1]*p2[0]
    return [c0,c1,c2]


def compute_phi(p, ref):
    
    if np.isclose(angle(p, ref), 0., 1e-15, 1e-15):
        return 0.0
    
    # Define x,y,z directions by a convention
    e_z = np.asarray(hat(ref))
    e_x = np.asarray(hat(cross([0,1,0], e_z)))
    e_y = np.asarray(cross(e_z, e_x))

    # Project p into xy plane
    p3 = np.asarray(p[:3])    
    p2 = p3 - e_z * dot3(e_z, p3)
    
    #Determine azimuthal angle in xy plane
    phi_y = angle(p2, e_y)
    phi = angle(p2, e_x)
    if phi_y > np.pi/2 :
        phi = 2*np.pi - phi
    if np.isnan(phi) :
        phi = 0.0
    return phi


def EPT(pXYZT, ref):
    e = pXYZT[3]
    theta = angle(pXYZT, ref)
    phi = compute_phi(pXYZT, ref)
    return [e, theta, phi]


def XYZ(pEPT, ref):
    e = pEPT[0]
    theta = pEPT[1]
    phi = pEPT[2]
    p_x = e * np.sin(theta) * np.cos(phi)
    p_y = e * np.sin(theta) * np.sin(phi)
    p_z = e * np.cos(theta)
    return [p_x, p_y, p_z]


jet_counter = 0
with open(input_path, 'r') as f1, open(output_path, 'w') as f2:
    for line in f1:
        words = line.split()
        
        if words[0] == 'J':
            jet_counter += 1
            if jet_counter > SAMPLE_SIZE:
                break
            if jet_counter % 10000 == 0:
                print('processing jet {}'.format(jet_counter))
                
            momenta = []
            splits = []
            charges = []
            momenta.append([float(words[i]) for i in [1,2,3,4]]) 
            charges.append(float(words[-1]))
            parents = {0: -1}
        
        elif words[0] == 'S': 
            n_splits = int(words[1])
            split_count = 0
        
        else:
            splitter = int(words[0])
            p_left = [float(words[i]) for i in [1,2,3,4]]
            q_left = float(words[5])
            p_right = [float(words[i]) for i in [6,7,8,9]]
            q_right = float(words[10])
            if p_left[3] > p_right[3]:
                momenta.append(p_left)
                charges.append(q_left)
                momenta.append(p_right)
                charges.append(q_right)
            else:
                momenta.append(p_right)
                charges.append(q_right)
                momenta.append(p_left)
                charges.append(q_left)
            parents[ len(momenta)-2 ] = splitter
            parents[ len(momenta)-1 ] = splitter
            cos = cos_theta(p_left, p_right)
            splits.append([split_count, splitter, cos, p_left[3], p_right[3], q_left, q_right])
            split_count += 1
            
            if split_count == n_splits:
                
                # Sort splittings by generalized-kt metric
                splits.sort(key=lambda x: min(x[3]**(2*recluster_def), x[4]**(2*recluster_def)) * (1-x[2]) / (1-np.cos(r_jet)), 
                            reverse=True) 
                
                # Make sure children do not split before parents
                already_split = [-1]
                i = 0
                j = 0
                while len(already_split) < n_splits:
                    if parents[ splits[i+j][1] ] in already_split:
                        if j != 0:
                            splits.insert(i, splits[i+j])
                            del splits[i+j+1]
                            j = 0
                        already_split.append( splits[i][1] )
                        i += 1
                    else:
                        j += 1
                
                # Store new ID's 
                new_name = {0:0}
                for i in range(n_splits):
                    new_name[ 2*splits[i][0]+1 ] = 2*i+1
                    new_name[ 2*splits[i][0]+2 ] = 2*i+2
                
                # Store tree_momenta, int_states, splitter_ids new order
                tree_momentaXYZT = []
                int_states = []
                splitter_ids = []
                tree_charges = []
                tree_momentaXYZT.append(momenta[0])
                tree_charges.append(charges[0])
                int_state = [0]
                int_states.append(list(int_state))
                for i in range(n_splits):
                    splitter_id = new_name[splits[i][1]]
                    splitter_ids.append(splitter_id)
                    int_state.remove(splitter_id)
                    tree_momentaXYZT.append(momenta[2*splits[i][0]+1])
                    tree_momentaXYZT.append(momenta[2*splits[i][0]+2])
                    tree_charges.append(charges[2*splits[i][0]+1])
                    tree_charges.append(charges[2*splits[i][0]+2])
                    int_state.append(2*i+1)
                    int_state.append(2*i+2)
                    int_states.append(list(int_state))  
                
                # Store jet's orientation (for ability to later convert back to XYZT)
                direc = np.asarray(tree_momentaXYZT[0][:3])
                direc /= np.sqrt(dot3(direc, direc))
                
                # Convert tree_momenta to energy, theta, phi (leave off mass for now)
                tree_momenta = []
                for (p, q) in zip(tree_momentaXYZT, tree_charges):
                    p_EPT = EPT(p, direc)
                    p_EPT.append(q)
                    tree_momenta.append(p_EPT)
                
                # Compute splitting infos
                splitting_infos = []
                count = 0
                for parent in splitter_ids:
                    p0 = tree_momentaXYZT[parent]
                    p1 = tree_momentaXYZT[2*count+1]
                    p2 = tree_momentaXYZT[2*count+2]
                    q1 = tree_charges[2*count+1]
                    q2 = tree_charges[2*count+2]
                    count += 1
                    if p1[3] > p2[3]:
                        z_soft = p2[3] / p0[3]
                        theta_soft = angle(p2, p0)
                        phi_soft = compute_phi(p2, p0)
                        theta_hard = angle(p1, p0)
                        q_hard = q1
                    else:
                        z_soft = p1[3] / p0[3]
                        theta_soft = angle(p1, p0)
                        phi_soft = compute_phi(p1, p0)
                        theta_hard = angle(p2, p0)
                        q_hard = q2
                    splitting_infos.append([z_soft, theta_soft, phi_soft, theta_hard, q_hard])
                
                # Print all jet info
                print('J {}'.format(jet_counter-1), file=f2)
                print('O {}'.format(floatlist_to_string(direc)), file=f2)
                for p in tree_momenta:
                    print('T {}'.format(floatlist_to_string(p)), file=f2)
                for state in int_states:
                    print('I {}'.format(list_to_string(state)), file=f2)
                for parent in splitter_ids:
                    print('P {}'.format(parent), file=f2)
                for info in splitting_infos:
                    print('S {}'.format(floatlist_to_string(info)), file=f2)
