import numpy as np

def normalize(data_path, destination_path, axis, ignore=[]):
    """
    a function to normalize each slice of a numpy array to 0-mean and 1-std where the slices correspond to all the measurements of the same variable

    Arguments:
        data_path: string -- contains the path to the npy file that contains the numpy array we would like to normalize
        destination_path: string -- contains the path to the location where we will store the normalized array
        axis: int -- the index corresponding the array dimension that describe the variables.
                    i.e. if my shape is (N, V, H, W) where V is the number of variables I have, then we would set axis = 1 to perform normalization on each variable independently
        ignore: list -- a list of indices that describes which variables should not be normalized

    """
    data_arr = np.load(data_path)
    axes = tuple([i for i in range(data_arr.ndim) if i is not axis])
    means = np.mean(data_arr, axis=axes)
    stds = np.std(data_arr, axis=axes)
    for mean, std, index in zip(means, stds, range(len(means))):
        if index in ignore:
            continue
        data_arr[:, index, :, :] -= mean
        data_arr[:, index, :, :] /= std

    np.save(destination_path, data_arr)