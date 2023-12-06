from pathlib import Path
import geopandas as gpd
import rasterio
import numpy as np
import rasterio.mask


# Tilldelar polygon-datasetet med de samplade värdena
def set_sampled_values(df, sampled_values):
    """
    Assign the calculated statistics to the DataFrame

    Parameters
    -----------
    df : GeoDataFrame
        DataFrame with polygons
    sampled_values : dict
        Dictionary with indices and values to assign


    """
    for stat in ["max", "min", "average"]:
        df.loc[:, stat] = df.index.map(
            {key: val[stat] for key, val in sampled_values.items()}
        )
    return df


def zonal_statistics(polygons, raster_data, statistics=["max", "min", "average"]):
    """
    Calculate raster statistics within polygons

    Parameters
    -----------
    polygons : GeoDataFrame
        DataFrame with polygons
    raster_data : Raster
        Raster with the values to be sampled. The values are assumed to be within layer 1
    statistics : list
        List with the statistics to calculate. Currently max, min and average are supported

    """
    original_columns = list(polygons.columns)
    update_values = dict()
    for idx in polygons.index:
        filtered_raster, filtered_raster_transform = rasterio.mask.mask(
            raster_data, [polygons.loc[idx, "geometry"]], crop=True,
            all_touched=True
        )
        filtered_raster_meta = raster_data.meta
        filtered_values = filtered_raster[filtered_raster > raster_data.meta["nodata"]]
        update_values[idx] = {
            "max": np.max(filtered_values),
            "min": np.min(filtered_values),
            "average": np.average(filtered_values),
        }

    polygons = set_sampled_values(polygons, update_values)

    return polygons.loc[:, original_columns + statistics]


# %% [markdown]
# ## Koden nedan kan användas om man vill spara de klippa lagren också
#
#     filtered_raster_meta.update({"driver": "GPKG",
#                                  "height": filtered_raster.shape[1],
#                                  "width": filtered_raster.shape[2],
#                                  "transform": filtered_raster_transform})
# ```

# %%
