import netCDF4
import numpy as np
import os
import xarray as xr


def split(data_path, data_destination_path, labels_destination_path):
    """
    Combines all of the nc files in the given path, splits off the labels, then saves these two large files as npy files at the designated destination paths

    Arguments:
        - data_path: string -- where to find all the nc files to combine
        - data_destination_path: string -- the path where the resulting data npy file will be saved
        - labels_destination_path: string -- the path where the resulting label npy file will be saved

    """

    # create data and label numpy arrays to save
    data = []
    labels = []


    # iterate over all files in the data_path
    for filename in sorted(os.listdir(data_path)):
        file = os.path.join(data_path, filename)
        if os.path.isfile(file):
            # for each file present in the data_path, read it in as a netCDF file, then convert it into a numpy array
            x_dataset = xr.open_dataset(file)
            curr_data = []
            for data_var in x_dataset.data_vars:
                if data_var == 'LABELS':
                    curr_label = x_dataset.data_vars[data_var].to_numpy().squeeze()
                else:
                    curr_data.append(x_dataset.data_vars[data_var].to_numpy().squeeze())
            curr_data = np.stack(curr_data, axis=0)
            data.append(curr_data)
            labels.append(curr_label)

    data = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)

    np.save(data_destination_path, data)
    np.save(labels_destination_path, labels)







    