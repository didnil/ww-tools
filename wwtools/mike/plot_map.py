# Plot Mike data.

import pandas as pd
import geopandas as gpd
from mikeio1d.res1d import QueryDataReach, QueryDataNode
from shapely import Point, LineString

def create_result_map(res1d, attribute, resType, objType, crs):


    """
    Create Geopandas DataFrame from Res1D file.

    ...

    Parameters
    ----------
    res1d : variable
        Res1D-file
    attribute : str
        Attribute to visualize. Fore example Pressure, Flow
    resType : str
        Which statistic to plot. Currently supporting maximum and minimum
    crs : str
        Coordinate reference system, for example epsg:3016
    objType : str
        Whether to create GeoDataFrame for pipes or nodes

    """
    toGdf = dict()
    gdf = True
    if objType.upper() in ['PIPE', 'REACH']:
        coordinates = list()
        for reach in res1d.data.Reaches:
            reachCoord = list()
            for coord in reach.DigiPoints:
                reachCoord.append(Point(coord.X, coord.Y))
            toGdf[reach.Id] = {'geometry': LineString(reachCoord)}
        
        for k in toGdf.keys():
            if resType == 'min':
                toGdf[k].update({attribute: res1d.read(QueryDataReach(attribute, k)).min().iloc[0]})
            if resType == 'max':
                toGdf[k].update({attribute: res1d.read(QueryDataReach(attribute, k)).max().iloc[0]})
            else:
                print('Result type is not supported')
                gdf = False
    
    elif objType.upper() in ['NODE']:
        for n in res1d.data.Nodes:
            toGdf[n.ID] = {'geometry': Point(n.XCoordinate, n.YCoordinate)}
        for k in toGdf.keys():
            if resType == 'min':
                toGdf[k].update({attribute: res1d.read(QueryDataNode(attribute, k)).min().iloc[0]})
            elif resType == 'max':
                toGdf[k].update({attribute: res1d.read(QueryDataNode(attribute, k)).max().iloc[0]})
            else:
                print('Result type is not supported')
                gdf = False
    if gdf:
        return gpd.GeoDataFrame(pd.DataFrame.from_dict(toGdf, orient='index'), geometry='geometry', crs=crs)
        gdf = gpd.GeoDataFrame(pd.DataFrame.from_dict(toGdf, orient='index'), geometry='geometry', crs=crs)
    else:
        return None
    # return gdf