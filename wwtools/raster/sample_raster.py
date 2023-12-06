from pathlib import Path
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask


# Tilldelar polygon-datasetet med de samplade värdena
def set_sampled_values(df, sampled_values):
    for stat in ["max", "min", "average"]:
        df.loc[:, stat] = df.index.map(
            {key: val[stat] for key, val in sampled_values.items()}
        )
    return df


def zonal_statistics(polygons, elev_data):  # , statistics=['max', 'min', 'average']):
    # Loopa igenom alla polygoner och välja ut raster-värdena inom varje polygon
    # sedan filtreras väredna ut som inte är nodata
    # varefter statistik beräknas

    update_values = dict()
    for idx in polygons.index:
        filtered_raster, filtered_raster_transform = mask(
            elev_data, [polygons.loc[idx, "geometry"]], crop=True
        )
        filtered_raster_meta = elev_data.meta
        filtered_values = filtered_raster[filtered_raster > elev_data.meta["nodata"]]
        update_values[idx] = {
            "max": np.max(filtered_values),
            "min": np.min(filtered_values),
            "average": np.average(filtered_values),
        }
        # print(f"Polygon: {idx}\nMax: {np.max(filtered_values)}\nMin: {np.min(filtered_values)}\n-----------------------\n")

    polygons = set_sampled_values(polygons, update_values)
    return polygons

# %% [markdown]
# ## Koden nedan kan användas om man vill spara de klippa lagren också
#
# ```python
# for idx in polygons.index:
#     filtered_raster, filtered_raster_transform = mask(elev_data, [polygons.loc[idx, 'geometry']], crop=True)
#     filtered_raster_meta = elev_data.meta
#     filtered_values = filtered_raster[filtered_raster>elev_data.meta['nodata']]
#     print(f"Polygon: {idx}\nMax: {np.max(filtered_values)}\nMin: {np.min(filtered_values)}\n-----------------------\n")
#
#     filtered_raster_meta.update({"driver": "GPKG",
#                                  "height": filtered_raster.shape[1],
#                                  "width": filtered_raster.shape[2],
#                                  "transform": filtered_raster_transform})
# ```

# %%
