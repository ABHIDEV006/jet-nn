import numpy as np

max_state_size=100

# Load jets for training or output_probabilities
def read_in_jets(data_path, jet_list, weighted=False):
        
    all_tree = []
    all_state_ids = []
    all_parent_ids = []
    all_splittings = []
    if weighted:
        all_weights = []
        
    # Read in data from input file
    jet_id = -1
    loading = False
    with open(data_path, 'r') as f:
        for line in f:
            words = line.split()
             
            # If 'J' then line starts new jet
            if words[0] == 'J':
                
                # Save info from previous jet, if necessary
                if loading:
                    if len(state_ids[:-1]) <= max_state_size:
                        all_tree.append(tree)
                        all_state_ids.append(state_ids)
                        all_parent_ids.append(parent_ids)
                        all_splittings.append(splittings)
                        if weighted:
                            all_weights.append(weights)
                    else:
                        print(' -- maximum shower length exceeded, jet ignored')
                    loading = False
                    
                # Check if done 
                if len(all_tree) == len(jet_list):
                    break
                    
                # Start loading next jet, if necessary
                jet_id += 1
                if jet_id in jet_list:
                    loading = True
                    tree = np.asarray([])
                    state_ids = []
                    parent_ids = []
                    splittings = []
                    if weighted:
                        weights = []
                    first_T = True
            
            # If 'O' then line gives jet orientation
            elif loading and words[0] == 'O':
                direc = [float(word) for word in words[1:]]
            
            # If 'T' then line gives a momentum in the tree: 
            # energy, theta, phi (wrt jet direction)
            elif loading and words[0] == 'T':
                p = [float(word) for word in words[1:]]
                if first_T:
                    first_T = False
                    tree = np.asarray([p])
                else:
                    tree = np.vstack([tree, p])
            
            # If 'I' then line gives an intermediate state (tree ids) of the jet
            elif loading and words[0] == 'I':
                ids = [int(word) for word in words[1:]]
                state_ids.append(ids[:])
            
            # If 'P' then line gives the parent (tree id) of a splitting
            elif loading and words[0] == 'P':
                parent_id = int(words[1])
                parent_ids.append(parent_id)
            
            # If 'S' then line gives splitting info: 
            # z, theta, phi, delta
            elif loading and words[0] == 'S':
                splitting = [float(word) for word in words[1:]]
                splittings.append(splitting) 
            
            elif loading and weighted and words[0] == 'W':
                weight = float(words[1])
                weights.append(weight)
    
    if weighted:
        return all_tree, all_state_ids, all_parent_ids, all_splittings, all_weights
    
    return all_tree, all_state_ids, all_parent_ids, all_splittings