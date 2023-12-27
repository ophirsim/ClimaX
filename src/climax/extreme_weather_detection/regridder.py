import xesmf as xe
import numpy as np


def regrid(input_dims, output_dims, algorithm):
    """
    """

    grid_in = {"lat": np.linspace(-90, 90, input_dims[0]), "lon": np.linspace(-180, 180, input_dims[1])}
    grid_out = {"lat": np.linspace(-90, 90, output_dims[0]), "lon": np.linspace(-180, 180, output_dims[1])}
    regridder = xe.Regridder(grid_in, grid_out, algorithm)
    return regridder