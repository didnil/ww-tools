import numpy as np


def reorder_geometry(layer):
    """
    Sets geometry last in columns

    Parameters
    ----------
    layer : GeoDataFrame
        The dataframe which to reorder.
    """
    layer = layer.iloc[
        :,
        np.r_[
            : np.nonzero(layer.columns == "geometry")[0][0],
            np.nonzero(layer.columns == "geometry")[0][0] + 1: layer.columns.shape[0],
            np.nonzero(layer.columns == "geometry")[0][0],
        ],
    ]
    return layer
