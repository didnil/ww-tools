import copy
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, LineString
import networkx as nx
from shapely import force_2d


def convert3dTo2d(dfTmp, crs="epsg:3016"):
    """
    Converts all geometries in GeoDataFrame to 2D

    Parameters
    -----------
    dfTmp : GeoDataFrame
        GeoDataFrame with geometries
    crs : str
        Example: 'epsg:3016'
    """

    df = dfTmp.copy(deep=True)
    geom2d = list()
    for geom in df.geometry:
        geom2d.append(shapely.ops.transform(lambda x, y, z=None: (x, y), geom))
    df.loc[:, "geometry"] = geom2d
    df = df.set_geometry("geometry", crs=crs)
    return df


def interpolate_zvalues(df_tmp, _invert_levels):
    """
    Interpolate z-values for vertices between start and stop based on columns

    Parameters
    ----------
    df_tmp : GeoDataFrame
        GeoDataFrame with columns for upstream level and downstream level
    _invert_levels : Tuple
        Tuple with names of columns fo upstream and downstream level
    """

    df = df_tmp.copy(deep=True)
    _uplvl, _dwlvl = _invert_levels
    df = df.assign(length_c=df.geometry.length)
    # return df
    lines_with_z = list()
    for uplvl, dwlvl, geom in (
        df.loc[:, [_uplvl, _dwlvl, "geometry"]].to_records(
            index=False)
    ):
        slope = (uplvl - dwlvl) / geom.length
        points_with_z = list()
        points_with_z.append(Point((geom.coords[0][0],
                                    geom.coords[0][1], uplvl)))
        for previous_idx, coord in enumerate(geom.coords[1:]):
            i_distance = Point((coord[0], coord[1])).distance(
                force_2d(points_with_z[previous_idx])
            )
            points_with_z.append(
                Point(
                    (
                        coord[0],
                        coord[1],
                        points_with_z[previous_idx].z - i_distance * slope,
                    )
                )
            )
        lines_with_z.append(LineString(points_with_z))
    df.geometry = lines_with_z
    return df


def create_node_names(dfTmp, prefix):
    """
    Creates from node id and to node id

    Parameters
    ----------
    dfTmp : GeoDataFrame
        GeoDataFrame with pipes.
    prefix : str
        Prefix before numbered node.
    """

    df = dfTmp.copy(deep=True)
    df = df.assign(lid=np.arange(df.shape[0]))
    tolerance = 2
    df = df.assign(
        fpid_c=df.loc[:, "geometry"].apply(
            lambda geom: (
                np.round(geom.coords[0][0], tolerance),
                np.round(geom.coords[0][1], tolerance),
            )
        )
    )
    df = df.assign(
        tpid_c=df.loc[:, "geometry"].apply(
            lambda geom: (
                np.round(geom.coords[-1][0], tolerance),
                np.round(geom.coords[-1][1], tolerance),
            )
        )
    )
    uniqueCoords = np.unique(
        np.concatenate([df.loc[:, "fpid_c"].values, 
                        df.loc[:, "tpid_c"].values])
    )
    nNodes = np.unique(uniqueCoords).shape[0]
    uniqueNodeNames = np.core.defchararray.add(
        [prefix] * nNodes, np.vectorize(str)(np.arange(nNodes) + 1)
    )
    nodeNamesDict = dict(zip(uniqueCoords, uniqueNodeNames))
    df = df.assign(fpid=df.loc[:, "fpid_c"].map(nodeNamesDict))
    df = df.assign(tpid=df.loc[:, "tpid_c"].map(nodeNamesDict))
    df = df.assign(muid=df.fpid + "_" + df.tpid)
    df.drop(columns=["fpid_c", "tpid_c"], axis=1, inplace=True)
    return df


def create_points(
    df,
    crs,
    idCol="lid",
    fpidCol="fpid",
    tpidCol="tpid",
    invert_levels=False,
    zCols=False,
):
    """
    Creates nodes from pipes with FromNodeId and ToNodeId

    Parameters
    ----------
    df : GeoDataFrame
        GeoDataFrame with pipes.
    crs : str
        Example: 'epsg:3016'
    idCol : str
        Name of column with ID
    fpidCol : str
        Name of column with FromNodeId
    tpidCol : str
        Name of column with FromNodeId
    invert_levels : tuple, optional
        Tuple with upstream and downstream level
    zCols : str
        Name of column with z-levels
    """

    # The option of passing zCols should be removed.
    # Instead, pass the dataframe to the function
    # interpolate_zvalues first

    if np.all(df.geometry.has_z):
        has_z = True
    else:
        has_z = False

    edgelist = list()
    nodeAttributes = dict()
    npv1, npv2 = invert_levels
    for row_ in df.iterrows():
        # print(row_)
        row = row_[1]
        if invert_levels:
            edgelist.append(
                (
                    row[fpidCol],
                    row[tpidCol],
                    {"id": row[idCol], "npv1": row[npv1], "npv2": row[npv2]},
                )
            )
        else:
            edgelist.append((row[fpidCol], row[tpidCol], {"id": row[idCol]}))
        if zCols:
            upLvl, dwLvl = zCols
            nodeAttributes[row[fpidCol]] = {
                "coord": row["geometry"].coords[0][:2] + (row[upLvl],)
            }
            nodeAttributes[row[tpidCol]] = {
                "coord": row["geometry"].coords[-1][:2] + (row[dwLvl],)
            }
        else:
            nodeAttributes[row[fpidCol]] = {"coord":
                                            row["geometry"].coords[0]}
            nodeAttributes[row[tpidCol]] = {"coord":
                                            row["geometry"].coords[-1]}

    # G = nx.from_edgelist(edgelist)
    G = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    nx.set_node_attributes(G, nodeAttributes)
    # checkConnection = nx.connected_components(G)
    # checkConnection = nx.connected_components(nx.Graph(G))
    if len(list(nx.connected_components(nx.Graph(G)))) > 1:
        print("Alla ledningar sitter inte ihop")
    else:
        print("Ledningarna sitter ihop")
    # return G
    for u in G.nodes():
        levels_list = list()
        for iu, iv in G.in_edges(u):
            levels_list.append(nx.get_edge_attributes(G, "npv2")[(iu, iv)])
        for ou, ov in G.out_edges(u):
            levels_list.append(nx.get_edge_attributes(G, "npv1")[(ou, ov)])

        if np.any([~np.isnan(val) for val in levels_list]):
            lowest_level = np.nanmin(levels_list)
            nx.set_node_attributes(G, {u: {"invert_level": lowest_level}})
        else:
            nx.set_node_attributes(G, {u: {"invert_level": np.nan}})

    if invert_levels:
        to_merge = ["invert_level", "coord"]
    else:
        to_merge = ["coord"]

    df_list = list()
    for column in to_merge:
        if column == "coord":
            if has_z:
                df_list.append(
                    gpd.GeoDataFrame(
                        pd.DataFrame.from_dict(
                            nx.get_node_attributes(G, column),
                            orient="index",
                            columns=["x", "y", "z"],
                        ).apply(lambda row: Point(
                            row.x,
                            row.y,
                            row.z), axis=1),
                        geometry=0,
                        crs=crs,
                    ).rename_geometry("geometry")
                )
            else:
                df_list.append(
                    gpd.GeoDataFrame(
                        pd.DataFrame.from_dict(
                            nx.get_node_attributes(G, column),
                            orient="index",
                            columns=["x", "y"],
                        ).apply(lambda row: Point(row.x, row.y), axis=1),
                        geometry=0,
                        crs=crs,
                    ).rename_geometry("geometry")
                )
        else:
            df_list.append(
                pd.DataFrame.from_dict(
                    nx.get_node_attributes(G, column), columns=[column], orient="index"
                )
            )
    points = gpd.GeoDataFrame(pd.concat(df_list, axis=1))
    points.index.name = "id"
    return points


def splitLine(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return False, LineString(line)
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[: i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            try:
                return [
                    LineString(coords[:i] + [(cp.x, cp.y, cp.z)]),
                    LineString([(cp.x, cp.y, cp.z)] + coords[i:]),
                ]
            except:
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:]),
                ]


def splitLinesEqualDistance(df, distance, crs):
    splitLines = list()
    for row in df.iterrows():
        row = row[1]
        rid = row.name
        unsplitLine = row.geometry
        lineLength = unsplitLine.length
        remainingLine = copy.deepcopy(unsplitLine)
        newLine = False
        for i in range(10, int(np.ceil(lineLength) + 10), distance):
            newLine, remainingLine = splitLine(remainingLine, distance)
            if newLine:
                splitLines.append({"idx": rid, "geometry": newLine})
            else:
                break
        if not newLine:
            splitLines.append({"idx": rid, "geometry": remainingLine})
        else:
            pass
    # return splitLines
    newDf = gpd.GeoDataFrame(splitLines, geometry="geometry", crs=crs)
    newDf.set_index("idx", inplace=True)
    return newDf.join(df.drop("geometry", axis=1), how="left")
    return gpd.GeoDataFrame(splitLines, geometry="geometry", crs=crs)
    # return splitLines# gpd.GeoDataFrame(geometry=splitLines, crs=crs)


def connectNodes(pipes, nodes, pipe_id, pipe_geom, node_id, crs):
    # ['EntityID', 'geometry']
    conNode = dict()
    for ei, p in pipes.loc[:, [pipe_id, pipe_geom]].to_records(index=False):
        startPoint = Point(p.coords[0][:2])
        endPoint = Point(p.coords[-1][:2])
        nearestFpid = gpd.sjoin_nearest(
            gpd.GeoDataFrame(geometry=[startPoint], index=[0], crs=crs),
            nodes,
            max_distance=0.2,
        )
        nearestTpid = gpd.sjoin_nearest(
            gpd.GeoDataFrame(geometry=[endPoint], index=[0], crs=crs),
            nodes,
            max_distance=0.2,
        )

        conNode[ei] = {}
        if nearestFpid.shape[0] == 1:
            conNode[ei].update({"fpid": nearestFpid.loc[0, node_id]})
        else:
            conNode[ei].update({"fpid": "saknas"})
            # pass

        if nearestTpid.shape[0] == 1:
            conNode[ei].update({"tpid": nearestTpid.loc[0, node_id]})
        else:
            conNode[ei].update({"tpid": "saknas"})

    pipes.loc[:, "fpid"] = pipes.loc[:, pipe_id].map(lambda x: conNode[x]["fpid"])
    pipes.loc[:, "tpid"] = pipes.loc[:, pipe_id].map(lambda x: conNode[x]["tpid"])
    pipes = pipes.assign(muid=pipes.fpid + "_" + pipes.tpid)

    return pipes


def addExtraNodes(pipes, nodes, pipe_id, pipe_geom, col, pidx, crs):
    """
    FUNGERAR INGET BRA JUST NU
    col = kolumnnan där saknas-information finns
    pidx = nod-index att
    """

    # Detta måste på något sätt läggas till

    # nodesToAdd = np.unique(np.concatenate([pipe_s.loc[pipe_s['fpid'] == 'saknas', 'geometry'].apply(lambda g: g.coords[0][:2]).to_numpy(), pipe_s.loc[pipe_s['tpid'] == 'saknas', 'geometry'].apply(lambda g: g.coords[-1][:2]).to_numpy()]))
    # nodesToAddGdf = gpd.GeoDataFrame(data={'EntityID': [f'sxp{i}' for i in range(nodesToAdd.shape[0])]}, geometry=[Point(coords) for coords in nodesToAdd], crs='epsg:3016')
    # fpidToAdd = addExtraNodes(pipe_s, nodesToAddGdf, 'fpid', 0)
    # tpidToAdd = addExtraNodes(pipe_s, nodesToAddGdf, 'tpid', -1)
    # pipe_s.loc[pipe_s['EntityID'].isin(fpidToAdd.keys()), 'fpid'] = pipe_s.loc[pipe_s['EntityID'].isin(fpidToAdd.keys()), 'EntityID'].map(fpidToAdd)
    # pipe_s.loc[pipe_s['EntityID'].isin(tpidToAdd.keys()), 'tpid'] = pipe_s.loc[pipe_s['EntityID'].isin(tpidToAdd.keys()), 'EntityID'].map(tpidToAdd)

    extraNodes = dict()

    for ei, p in pipes.loc[pipes[col] == "saknas", [pipe_id, pipe_geom]].to_records(
        index=False
    ):
        slPnt = Point(p.coords[pidx][:2])
        # endPoint = Point(p.coords[-1][:2])
        nrstPnt = gpd.sjoin_nearest(
            gpd.GeoDataFrame(geometry=[slPnt], index=[0], crs=crs),
            nodes,
            max_distance=0.2,
        )

        extraNodes[ei] = {}
        if nrstPnt.shape[0] == 1:
            extraNodes[ei] = nrstPnt.loc[0, "EntityID"]
        else:
            extraNodes[ei] = "saknas"

    return extraNodes
