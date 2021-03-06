# Patrick Komiske, MIT, 2017
#
# Contains several useful functions for creating/modifying jet images.

from utils import *
from time import clock
#from keras.preprocessing.image import ImageDataGenerator

# mapping from particle id to charge of the particle 
charge_map = {11: -1, -11:  1, 13: -1, -13:  1, 22:  0, -22:  0, 
              111:  0, -111:  0, 130:  0, -130:  0, 211:  1, -211: -1, 
              321:  1, -321: -1, 2112:  0, -2112:  0, 2212:  1, -2212: -1}


def pixelate(jet, charge_image = False, K = 0, K_two = 0, jet_center = [], img_size = 33, img_width = 0.8, nb_chan = 1, norm = True, 
             rap_i=0, phi_i=1, pT_i=2, centers = False):

    """ A function for creating a jet image from a list of particles.

    jet: an array containing the list of particles in a jet with each row 
         representing a particle and the columns being (rapidity, phi, pT, 
         pdgid), the latter not being necessary for a grayscale image.
    jet_center: the (y, phi) center of the jet. If None, compute the center
                via the pt weighted centroid.
    img_size: number of pixels along one dimension of the image.
    img_width: the image will be size img_width x img_width
    nb_chan: 1 - returns a grayscale image of total pt
             2 - returns a two-channel image with total pt and charge counts
             3 - returns a three-channel "RGB" image with charged pt, neutral
                 pt, and charge counts.
    rap_i: the column index of the jet corresponding to rapidity
    phi_i: the column index of the jet corresponding to azimuthal angle
    pT_i: the column index of the jet corresponding to pT
    """

    if nb_chan not in [1,2,3,4]:
        raise ValueError('Invalid number of channels for jet image.')

    # the image is (img_width x img_width) in size
    pix_width = img_width / img_size
    jet_image = np.zeros((nb_chan, img_size, img_size))

    raps = jet[:,rap_i]
    phis = jet[:,phi_i]
    pts  = jet[:,pT_i]

    # deal with split images
    ref_phi = phis[np.argmax(pts)] if len(jet_center) == 0 else jet_center[1]
    phis[phis - ref_phi >  img_width] -= 2 * np.pi
    phis[phis - ref_phi < -img_width] += 2 * np.pi  

    # get jet pt centroid index
    if len(jet_center) == 0:
        rap_avg = np.average(raps, weights = pts)
        phi_avg = np.average(phis, weights = pts)
    else:
        rap_avg = jet_center[0]
        phi_avg = jet_center[1]

    rap_pt_cent_index = np.ceil(rap_avg/pix_width - .5) - np.floor(img_size / 2)
    phi_pt_cent_index = np.ceil(phi_avg/pix_width - .5) - np.floor(img_size / 2)

    # center image and transition to indices
    rap_indices = np.ceil(raps/pix_width - .5) - rap_pt_cent_index
    phi_indices = np.ceil(phis/pix_width - .5) - phi_pt_cent_index

    # delete elements outside of range
    mask = np.ones(raps.shape).astype(bool)
    mask[rap_indices < 0] = False
    mask[phi_indices < 0] = False
    mask[rap_indices >= img_size] = False
    mask[phi_indices >= img_size] = False
    rap_indices = rap_indices[mask].astype(int)
    phi_indices = phi_indices[mask].astype(int)

    # Without Jet Charge
    if charge_image == False:
        # construct grayscale image
        if nb_chan == 1:
            for ph,y,pt in zip(phi_indices, rap_indices, pts[mask]):
                jet_image[0, ph, y] += pt
            num_pt_chans = 1

        # construct two-channel image
        elif nb_chan == 2:
            for ph,y,pt,label in zip(phi_indices, rap_indices, 
                                     pts[mask], jet[mask,3]):
                jet_image[0, ph, y] += pt
                if charge_map[label] != 0:
                    jet_image[1, ph, y] += 1
            num_pt_chans = 1

        # construct three-channel image
        elif nb_chan == 3:
            for ph,y,pt,label in zip(phi_indices, rap_indices, 
                                     pts[mask], jet[mask,3].astype(int)):
                if charge_map[label] == 0:
                    jet_image[1, ph, y] += pt
                else:
                    jet_image[0, ph, y] += pt
                    jet_image[2, ph, y] += 1
            num_pt_chans = 2

        # construct four-channel image
        elif nb_chan == 4:
            jet_pt = np.sum(pts)
            for ph,y,pt,label in zip(phi_indices, rap_indices, 
                                     pts[mask], jet[mask,3].astype(int)):
                if charge_map[label] == 0:
                    jet_image[1, ph, y] += pt
                else:
                    jet_image[0, ph, y] += pt
                    jet_image[2, ph, y] += 1
                jet_image[3, ph, y] += (charge_map[label] * pow(pt, K))/(pow(jet_pt,K))
            num_pt_chans = 2
    #Using Jet Charge
    if charge_image == True:
        if nb_chan == 2:
            jet_pt = np.sum(pts)
            for ph,y,pt,label in zip(phi_indices, rap_indices, 
                                     pts[mask], jet[mask,3].astype(int)):
                jet_image[0, ph, y] += pt
                jet_image[1, ph, y] += (label * pow(pt, K))/(pow(jet_pt,K))
            num_pt_chans = 1
        if nb_chan == 3:
            jet_pt = np.sum(pts)
            for ph,y,pt,label in zip(phi_indices, rap_indices,
                                     pts[mask], jet[mask,3].astype(int)):
                jet_image[0, ph, y] += pt
                jet_image[1, ph, y] += (charge_map[label] * pow(pt, K))/(pow(jet_pt,K))
                jet_image[2, ph, y] += (charge_map[label] * pow(pt, K_two))/(pow(jet_pt,K_two))
            num_pt_chans = 1            
        if nb_chan == 1:
            raise ValueError('Invalid number of channels for charged jet image, not implemented yet.')

    # L1-normalize the pt channels of the jet image
    if norm:
        normfactor = np.sum(jet_image[:num_pt_chans])
        if normfactor == 0:
            raise FloatingPointError('Image had no particles!')
            return np.zeros((nb_chan, img_size, img_size))
        else:
            jet_image[:num_pt_chans] = jet_image[:num_pt_chans]/normfactor

    if len(jet_center) == 0 and centers:
        return jet_image, (rap_avg, phi_avg)
    else:
        return jet_image

def upsample(images, factor = 1):

    """ A function for upsampling (A,B,N,N) images to (A,B, factor*N, factor*N) images, preserving norm
    
    images: the array of images.
    factor: the upsampling factor.
    """
    return images.repeat(factor, axis=3).repeat(factor, axis=2)/factor**2

def downsample(images, factor = 1):

    """ A function for downsampling (A,B,N,N) images to (A,B, N/factor, N/factor) images, preserving norm,
    using the naieve method of grabbing an element from that sub-block.
    
    images: the array of images.
    factor: the upsampling factor.
    """
    return images[:,:,::factor,::factor]*factor**2


def write_images_to_file(base_name, images, path = '../images', 
                         addendum = '_{0}x{0}images_{1}chan'):

    """ A function used for writing images to file as a numpy compressed
    array. Assumes that images has shape (nb_images, nb_chan, img_size, 
    img_size). 

    base_name: a string which should be descrptive of which images are being 
               saved. 
    images: the array of images.
    path: the directory path where the images should be placed.
    addendum: standard string to append to end of the base name. the default
              will include information about the size of the images and the
              number of channels they have.
    """

    ts = clock()
    fprint('Writing images for {} to file ... '.format(base_name))
    filename = os.path.join(path, (base_name + addendum).format(
                                    len(images[0][0]), len(images[0])))
    np.savez_compressed(filename, images)
    fprint('Done, in {:.3f} seconds.\n'.format(clock() - ts))


def load_images(gluon_img_files, quark_img_files, nb_gluons, nb_quarks, 
                img_size = 33, nb_chan = 1, path = '../images'):

    """ A function for loading in images files and returning them in a single 
    numpy array.

    gluon_img_files: a list of filenames containing the gluon images.
    quark_img_files: a list of filenames containing the quark images.
    nb_gluons: the total number of gluons.
    nb_quarks: the total number of quarks.
    """

    # allocate numpy array to hold all the jet images
    images = np.zeros((nb_gluons + nb_quarks, nb_chan, img_size, img_size))

    index = 0
    for gluon_file in parg(gluon_img_files):
        local_images = np.load(os.path.join(path, gluon_file))['arr_0']
        images[index:index+local_images.shape[0]] = local_images
        index += local_images.shape[0]

    for quark_file in parg(quark_img_files):
        local_images = np.load(os.path.join(path, quark_file))['arr_0']
        images[index:index+local_images.shape[0]] = local_images
        index += local_images.shape[0]

    return images


def make_labels(nb_gluons, nb_quarks, one_hot = True):

    """ Constructs class labels for quarks and gluons. Labels gluons as
    zero and quarks as one. Assumes that gluons come before quarks.

    nb_gluons: the number of gluons.
    nb_quarks: the number of quarks
    one_hot: if True, the labeling is one-hot encoded.
    """

    labels = np.concatenate((np.zeros(nb_gluons), np.ones(nb_quarks)))
    if one_hot:
        return to_categorical(labels, 2)
    else:
        return labels


def zero_center(*args, channels = [], copy = False):

    """ Subtracts the mean of arg[0,channels] from the other arguments.
    Assumes that the arguments are numpy arrays. The expected use case would
    be zero_center(X_train, X_val, X_test).

    channels: which channels to zero_center. The default will lead to all
              channels being affected.
    copy: if True, the arguments are unaffected. if False, the arguments
          themselves may be modified
    """

    assert len(args) > 0

    # treat channels properly
    if len(parg(channels)) == 0:
        channels = np.arange(args[0].shape[1])
    else:
        channels = parg(channels)

    # compute mean of the first argument
    mean = np.mean(args[0], axis = 0)

    # copy arguments if requested
    if copy:
        X = [np.copy(arg) for arg in args]
    else:
        X = args

    # iterate through arguments and channels
    for x in X:
        for chan in channels:

            # use broadcasting to do the heavy lifting here
            x[:,chan] -= mean[chan]

    return X


def standardize(*args, channels = [], copy = False, reg = 10**-6):

    """ Normalizes each argument by the standard deviation of the pixels in 
    arg[0]. The expected use case would be standardize(X_train, X_val, X_test).

    channels: which channels to zero_center. The default will lead to all
              channels being affected.
    copy: if True, the arguments are unaffected. if False, the arguments
          themselves may be modified
    reg: used to prevent divide by zero 
    """

    assert len(args) > 0

    # treat channels properly
    if len(parg(channels)) == 0:
        channels = np.arange(args[0].shape[1])
    else:
        channels = parg(channels)

    stds = np.std(args[0], axis = 0) + reg

    # copy arguments if requested
    if copy:
        X = [np.copy(arg) for arg in args]
    else:
        X = args

    # iterate through arguments and channels
    for x in X:
        for chan in channels:

            # use broadcasting to do the heavy lifting here
            x[:,chan] /= stds[chan]

    return X


def apply_jitter(images, Y = [], R = [], which = 'all', step = 1, shuffle = True):

    """ A function to apply various transformations to a set of images and
    return an expanded set of images containing the results of those operations,
    including the identity operation.

    Y: if present, an array that is assumed to pair with images (labels, etc.)
       and will be duplicated and returned along with the jittered images so
       as to maintain this pairing.  
    which: currently either 'all', 'reflect', 'translate' which determines 
           which type of operations are performed
    step: how many steps (measured by the manhattan metric) to translate by
    shuffle: whether or not to shuffle the results before returning
    """

    # assure proper usage
    assert which in ['all','reflect','translate'], 'Invalid jittering'

    # compute some sizes
    n_tot, n_chan = images.shape[0], images.shape[1]
    img_size = images.shape[2]
    n_trans = 2 * step * (step + 1)
    n_jitter = 3 if which == 'reflect' else \
                    (n_trans if which == 'translate' else n_trans + 3)

    # allocate memory in advance, this saves max memory usage by a factor of 2
    z = np.zeros((n_tot * (n_jitter + 1), n_chan, img_size, img_size))

    # set beginning chunk of memory equal to X_train
    z[:n_tot] = images

    # use index to keep track of which memory block we're indexing into
    index = n_tot
    if which == 'all' or which == 'reflect':
        z[n_tot:2*n_tot]   = images[:,:,::-1,:] # rap flip
        z[2*n_tot:3*n_tot] = images[:,:,:,::-1] # phi flip
        z[3*n_tot:4*n_tot] = images[:n_tot,:,::-1,::-1] # eta and phi flip
        index = 4 * n_tot
       # fprint('Jittered data by reflecting\n')

    if which == 'all' or which == 'translate':
        for i in range(-step, step + 1):
            for j in range(-step, step + 1):
                if 0 < abs(i) + abs(j) <= step:
                    if i >= 0:
                        start_ix = 0
                        end_ix = img_size - i
                        start_iz = i
                        end_iz = img_size
                    else:
                        start_ix = abs(i)
                        end_ix = img_size
                        start_iz = 0
                        end_iz = img_size - abs(i)
                    if j >= 0:
                        start_jx = 0
                        end_jx = img_size - j
                        start_jz = j
                        end_jz = img_size
                    else:
                        start_jx = abs(j)
                        end_jx = img_size
                        start_jz = 0
                        end_jz = img_size - abs(j)
                    z[index:index+n_tot,:,start_iz:end_iz,start_jz:end_jz] = \
                               images[:,:,start_ix:end_ix,start_jx:end_jx]
                    index += n_tot
                    #fprint('Jittered data by translating rap {}, phi {}\n'
                    #       .format(i,j))

    p = np.random.permutation(z.shape[0]) if shuffle \
                                                     else np.arange(z.shape[0])
    if len(Y) > 0:
        Y = np.concatenate([Y for i in range(n_jitter + 1)])
        if R == []:
            return z[p], Y[p]
        else:
            R = np.concatenate([R for i in range(n_jitter + 1)])
            return z[p], Y[p], R[p]
    else:
        #fprint('After jittering, images shape: {}\n'.format(z.shape))
        return z[p]

