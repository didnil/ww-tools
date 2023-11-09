# Version 12. 2023-04-16. Lagt till möjlighet att välja scenario
# Version 13. 2023-04-23. Lagt in så att man kan välja figurstorlek# %%
# Uppdatering 2023-09-22. Lagt in så att man kan styra teckenstorlek
# på legend, ax-text och xticklabelsize
# Framtidsförbättring.
# sqlite-databasen behövs inte. Nivåerna går att hämta så här:
# results['nulage'].nodes['DANDE559'].GroundLevel
#
# Lägg till så att alla beräknignspunkter används. Går att söka ut, har jag lärt mig nu.
#
# Diameter går inte att hämta från res1d så databasen behövs fortfarande

import locale
from pathlib import Path
import numpy as np

# import sqlite3
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import geopandas as gpd
from mikeio1d.res1d import (
    Res1D,
    QueryDataReach,
    QueryData,
    QueryDataNode,
    mike1d_quantities,
)
import pyperclip as pc

# class PlotProfile:
#     def __init__(self):
#         self


def createNetwork(dfTmp, fpid="fromnodeid", tpid="tonodeid"):  # , diameter='diameter'):
    df = dfTmp.copy(deep=True)
    G = nx.Graph()
    # attributes = dict()
    for row in df.iterrows():
        rowCols = row[1]
        G.add_edge(
            rowCols[fpid], rowCols[tpid], id=rowCols.name
        )  # , diameter=diameter)
    return G


# Taget från https://stackoverflow.com/questions/32531117/getting-subgraph-of-nodes-between-two-nodes
def selectPipes(G, flags):
    start, end = flags[0], flags[-1]
    paths_between_generator = nx.all_simple_edge_paths(G, source=start, target=end)

    allPaths = list()
    for path in paths_between_generator:
        nodes = list()
        for edge in path:
            nodes += list(edge)
        allPaths.append(set(nodes))

    for path in allPaths:
        if set(flags).issubset(path):
            SG = G.subgraph(path)
            break

    DiSG = nx.traversal.bfs_tree(SG, end).reverse()

    for edge in DiSG.edges:
        u, v = edge
        try:
            DiSG[u][v]["id"] = nx.get_edge_attributes(SG, "id")[edge]
        except:
            DiSG[u][v]["id"] = nx.get_edge_attributes(SG, "id")[edge[::-1]]

    pipesInOrder = list()
    for u, v, attr in DiSG.edges(nx.topological_sort(DiSG), data=True):
        pipesInOrder.append(attr["id"])

    pc.copy("MUID IN ('" + "', '".join(pipesInOrder) + "')")

    nodesInOrder = list(nx.topological_sort(DiSG))
    return nodesInOrder, pipesInOrder


def createObjectData(
    pipes, nodes, nodeNamesInOrder, pipeNamesInOrder, groundLevelID="groundlevel"
):
    selectedPipes = pipes.loc[pipeNamesInOrder]
    selectedPipes.loc[:, "endChainage"] = selectedPipes.length.cumsum()
    startChainage = np.roll(selectedPipes.length.values, 1)
    startChainage[0] = 0
    selectedPipes = selectedPipes.assign(startChainage=startChainage.cumsum())

    pipeData = dict()
    # nodeData = {nodeNamesInOrder[0]: {'Chainage': 0, 'GroundLevel': nodes.loc[nodeNamesInOrder[0], groundLevelID]}}
    nodeData = {
        nodeNamesInOrder[0]: {
            "Chainage": 0,
            "InvertLevel": np.min(
                selectedPipes.loc[pipeNamesInOrder[0], ["uplevel", "dwlevel"]].iloc[0]
            ),
            "GroundLevel": nodes.loc[nodeNamesInOrder[0], groundLevelID],
        }
    }
    for idx, uplvl, dwlvl, diam, length, sc, ec, fnid, tnid in (
        selectedPipes.reset_index()
        .loc[
            :,
            [
                "muid",
                "uplevel",
                "dwlevel",
                "diameter",
                "geometric_l",
                "startChainage",
                "endChainage",
                "fromnodeid",
                "tonodeid",
            ],
        ]
        .to_records(index=False)
    ):
        if tnid in nodeData.keys():
            print("Hej")
            pipeData[idx] = {
                "UpLevel": dwlvl,
                "UpLevelTop": dwlvl + diam,
                "DwLevel": uplvl,
                "DwLevelTop": uplvl + diam,
                "Length": length,
                "StartChainage": sc,
                "EndChainage": ec,
                "Diameter": diam,
            }
            nodeData[fnid] = {
                "Chainage": ec,
                "InvertLevel": np.min(
                    [pipeData[idx]["UpLevel"], pipeData[idx]["DwLevel"]]
                ),
                "GroundLevel": nodes.loc[fnid, groundLevelID],
            }
        else:
            pipeData[idx] = {
                "UpLevel": uplvl,
                "UpLevelTop": uplvl + diam,
                "DwLevel": dwlvl,
                "DwLevelTop": dwlvl + diam,
                "Length": length,
                "StartChainage": sc,
                "EndChainage": ec,
                "Diameter": diam,
            }
            nodeData[tnid] = {
                "Chainage": ec,
                "InvertLevel": np.min(
                    [pipeData[idx]["UpLevel"], pipeData[idx]["DwLevel"]]
                ),
                "GroundLevel": nodes.loc[tnid, groundLevelID],
            }

    startDiam = pipeData[pipeNamesInOrder[0]]["Diameter"]
    endDiam = pipeData[pipeNamesInOrder[-1]]["Diameter"]

    return nodeData, pipeData, startDiam, endDiam


# Fixat så att den hämtar start- och slutvärdet från ledningen
# Kan inte hämta värden där emellan för jag kan inte
# identifiera var den ska ligga (hämtar man det första värdet blir det fel)
# Jag ska prova att hämta värdet i mitten typ. Fast det blir nog inget bra
# heller.

# I stället


def plotProfile(
    pipes,
    nodes,
    results,
    flags,
    fpid="fromnodeid",
    tpid="tonodeid",
    groundLevelID="groundlevel",
    plotGround=True,
    legendloc=False,
    labels=False,
    relativeLabels=False,
    absoluteLabels=False,
    filename=False,
    includeSteps=False,
    includeSpecificNodes=False,
    title=False,
    legendOutside=False,
    paddingForLegend=1.5,
    ncols=1,
    figsize=False,
    plotNodeNames=True,
    selectedPath=False,
    legendFontsize=9,
    labelFontsize=9,
    tickFontsize=9,
):
    G = createNetwork(pipes, fpid=fpid, tpid=tpid)

    # nodeNamesInOrder, pipeNamesInOrder = selectPipes(G, flags)
    # Om det strular så kommentera bort if-satsen och behåll bara nodeNamesInOrder
    if selectedPath == False:
        nodeNamesInOrder, pipeNamesInOrder = selectPipes(G, flags)
    else:
        nodeNamesInOrder, pipeNamesInOrder = selectedPath

    nodeData, pipeData, startDiam, endDiam = createObjectData(
        pipes, nodes, nodeNamesInOrder, pipeNamesInOrder, groundLevelID=groundLevelID
    )
    mm2Inch = 1 / 25.4
    if figsize:
        figWidth, figHeight = figsize
    else:
        figHeight = (297 / 2.8) * mm2Inch
        figWidth = (210 - 25 * 2) * 0.88 * mm2Inch
    fig, ax = plt.subplots(figsize=(figWidth, figHeight))  # , layout='constrained')

    ax.tick_params(axis="both", which="major", labelsize=tickFontsize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)

    # fig.tight_layout()

    distance = list()
    invLevels = list()
    topLevels = list()

    for pidx in pipeNamesInOrder:
        distance.append(pipeData[pidx]["StartChainage"])
        distance.append(pipeData[pidx]["EndChainage"])

        invLevels.append(pipeData[pidx]["UpLevel"])
        invLevels.append(pipeData[pidx]["DwLevel"])

        topLevels.append(pipeData[pidx]["UpLevelTop"])
        topLevels.append(pipeData[pidx]["DwLevelTop"])

    nodeChainage = list()
    groundLevel = list()
    for nidx in nodeNamesInOrder:
        nodeChainage.append(nodeData[nidx]["Chainage"])
        groundLevel.append(nodeData[nidx]["GroundLevel"])

    resultsList = list()
    for key in results.keys():
        resultChainage = list()
        resultWaterLevel = list()
        for pipeName in pipeNamesInOrder:
            pipeResultChainage = list()
            pipeResultWaterLevel = list()
            for chainage in np.arange(0, pipeData[pipeName]["Length"], 5):
                pipeResultChainage.append(
                    pipeData[pipeName]["StartChainage"] + chainage
                )
                pipeResultWaterLevel.append(
                    results[key]
                    .read(QueryDataReach("WaterLevel", pipeName, chainage))
                    .max(axis=0)
                    .iloc[0]
                )
            pipeResultChainage.append(
                pipeData[pipeName]["StartChainage"] + pipeData[pipeName]["Length"]
            )
            pipeResultWaterLevel.append(
                results[key]
                .read(
                    QueryDataReach("WaterLevel", pipeName, pipeData[pipeName]["Length"])
                )
                .max(axis=0)
                .iloc[0]
            )

            values, indices = np.unique(pipeResultWaterLevel, return_index=True)
            indices = sorted(indices)
            # print(indices)
            # print('---')
            # indices[-1] = -1
            indices = [0, -1]

            pipeResultChainage = np.asarray(pipeResultChainage)
            pipeResultWaterLevel = np.asarray(pipeResultWaterLevel)

            resultChainage += list(pipeResultChainage[indices])
            resultWaterLevel += list(pipeResultWaterLevel[indices])
        resultsList.append([resultChainage, resultWaterLevel])

    ax.plot(distance, invLevels, c="grey", linewidth=1)
    ax.plot(distance, topLevels, c="grey", linewidth=1)
    if plotGround:
        ax.plot(nodeChainage, groundLevel, c="g", linewidth=1, label="Marknivå")

    if not labels:
        labels = len(resultsList) * [False]

    colors = iter([plt.cm.Dark2(i) for i in range(30)])
    linestyles = ["dashed", (0, (3, 1, 1, 1, 1, 1)), "dashdot", "dotted"]
    linewidths = [1.5, 1.3, 1.1, 1.3]
    markevery = int(len(resultsList[0][0]) / 10)
    markers = ["d", "o", "x", ">"]
    for i, (result, label) in enumerate(zip(resultsList, labels)):
        # print(i % len(markers))
        # ax.plot(result[0], result[1], markers[i % len(markers)], ms=4, color=next(colors), linestyle=linestyles[i % len(linestyles)], linewidth=linewidths[i % len(linewidths)], label = label, markevery = (markevery + i))
        ax.plot(
            result[0],
            result[1],
            color=next(colors),
            linestyle=linestyles[i % len(linestyles)],
            linewidth=linewidths[i % len(linewidths)],
            label=label,
        )
    allNodeChainage = [nodeData[node]["Chainage"] for node in nodeData]
    for chainage in allNodeChainage:
        ax.axvline(chainage, linewidth=0.2, c="grey", ls=":")

    ymin, ymax = ax.get_ylim()
    distLabels = (ymax - ymin) * 0.01
    nodeLabelText = list()

    if includeSteps:
        nodesToPlot = nodeNamesInOrder
        textRotation = 90
        nodesToPlot = np.asarray(nodeNamesInOrder)[
            np.arange(0, len(nodeNamesInOrder) - 1, includeSteps)
        ]
        nodesToPlot = np.concatenate([nodesToPlot, np.asarray([nodeNamesInOrder[-1]])])
    else:
        nodesToPlot = np.asarray([nodeNamesInOrder[0], nodeNamesInOrder[-1]])
        textRotation = 0

    if includeSpecificNodes:
        if not isinstance(includeSpecificNodes, list):
            includeSpecificNodes = [includeSpecificNodes]
        else:
            pass
        specificNodesToAdd = [
            includeNode
            for includeNode in includeSpecificNodes
            if includeNode not in nodesToPlot
        ]
        for specificNode in specificNodesToAdd:
            nidx = nodeNamesInOrder.index(specificNode)
            nodesToPlot = nodesToPlot[
                (nodesToPlot != nodeNamesInOrder[nidx - 1])
                & (nodesToPlot != nodeNamesInOrder[nidx + 1])
            ]

        nodesToPlot = np.concatenate([nodesToPlot, np.asarray(specificNodesToAdd)])
        nodesToPlot = sorted(nodesToPlot, key=nodeNamesInOrder.index)

    for node in nodesToPlot:
        if plotNodeNames:
            ax.add_line(
                Line2D(
                    [nodeData[node]["Chainage"], nodeData[node]["Chainage"]],
                    [
                        nodeData[node]["InvertLevel"] - distLabels,
                        nodeData[node]["InvertLevel"] - distLabels * 1.2,
                    ],
                    color="k",
                )
            )
            nodeLabelText.append(
                ax.text(
                    nodeData[node]["Chainage"],
                    nodeData[node]["InvertLevel"] - distLabels * 1.2,
                    node,
                    horizontalalignment="center",
                    verticalalignment="top",
                    rotation=textRotation,
                )
            )
        else:
            nodeLabelText.append(
                ax.text(
                    nodeData[node]["Chainage"],
                    nodeData[node]["InvertLevel"] - distLabels * 1.2,
                    node,
                    horizontalalignment="center",
                    verticalalignment="top",
                    rotation=textRotation,
                    alpha=0,
                )
            )

    # Fixa storlek på graf
    inv = ax.transData.inverted()
    r = fig.canvas.get_renderer()

    bbStart = nodeLabelText[0].get_window_extent(renderer=r)
    bbEnd = nodeLabelText[-1].get_window_extent(renderer=r)

    bbStart = inv.transform(bbStart)
    bbEnd = inv.transform(bbEnd)

    # bb är en boundary box där den första listan
    # motsvarar nedre vänstra hörnet
    # och den andra över högra hörnet

    horizontalPadding = np.max(allNodeChainage) * 0.01

    newXLimLeft = bbStart[0][0] - horizontalPadding
    newXLimRight = bbEnd[1][0] + horizontalPadding

    ax.set_xlim(newXLimLeft, newXLimRight)

    ymin, ymax = ax.get_ylim()

    verticalPadding = (ymax - ymin) * 0.07
    newYmin = bbEnd[0][1] - verticalPadding
    ax.set_ylim(newYmin, ymax)
    # ----------------------------------
    # ----------------------------------
    # ----------------------------------
    if relativeLabels:
        if not isinstance(relativeLabels, list):
            relativeLabels = [relativeLabels]
        ymin, ymax = ax.get_ylim()
        padding = (ymax - ymin) * 0.01
        xmin, xmax = ax.get_xlim()
        horizontalPadding = (xmax - xmin) * 0.01
        for labelToAdd in relativeLabels:
            if isinstance(labelToAdd[2], str):
                nodeLabel, distanceToShift, yLoc, textValue = labelToAdd
                xcoord = nodeData[nodeLabel]["Chainage"] + distanceToShift
                ax.axvline(xcoord, color="red", linewidth=0.4)
                if yLoc == "lower":
                    shiftFactor = 0.25
                elif yLoc == "middle":
                    shiftFactor = 0.5
                elif yLoc == "upper":
                    shiftFactor = 0.75
                else:
                    print("Positionen saknas")
                    return None
                ycoord = ymin + (ymax - ymin) * shiftFactor
                ax.text(
                    xcoord + horizontalPadding, ycoord, textValue
                )  # , horizontalalignment = 'left')#, verticalalignment='bottom')

            else:
                nodeLabel, distanceToShift, ycoord, textValue = labelToAdd
                xcoord = nodeData[nodeLabel]["Chainage"] + distanceToShift
                ax.scatter(xcoord, ycoord, marker="+")
                ax.text(
                    xcoord,
                    ycoord + padding,
                    textValue,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

    if absoluteLabels:
        if not isinstance(absoluteLabels, list):
            absoluteLabels = [absoluteLabels]
        ymin, ymax = ax.get_ylim()
        padding = (ymax - ymin) * 0.01
        xmin, xmax = ax.get_xlim()
        horizontalPadding = (xmax - xmin) * 0.01

        for labelToAdd in absoluteLabels:
            if isinstance(labelToAdd[1], str):
                xcoord, yLoc, textValue = labelToAdd
                ax.axvline(xcoord, color="red", linewidth=0.4)
                if yLoc == "lower":
                    shiftFactor = 0.25
                elif yLoc == "middle":
                    shiftFactor = 0.5
                elif yLoc == "upper":
                    shiftFactor = 0.75
                else:
                    print("Positionen saknas")
                    return None
                ycoord = ymin + (ymax - ymin) * shiftFactor
                ax.text(
                    xcoord + horizontalPadding, ycoord, textValue
                )  # , horizontalalignment = 'left')#, verticalalignment='bottom')

            else:
                xcoord, ycoord, textValue = labelToAdd
                ax.scatter(xcoord, ycoord, marker="+")
                ax.text(
                    xcoord,
                    ycoord + padding,
                    textValue,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

    ax.set_xlabel("Avstånd", fontsize=labelFontsize)
    ax.set_ylabel("Meter över havet", fontsize=labelFontsize)
    fig.tight_layout()
    if legendOutside:
        legendBox = fig.legend(fontsize=9, loc="outside lower center", ncols=ncols)
        inv = ax.transData.inverted()
        lbb = legendBox.get_window_extent(renderer=r)
        lbb = inv.transform(lbb)
        fbb = fig.get_window_extent(renderer=r)
        fbb = inv.transform(fbb)
        legendHeight = lbb[1][1] - lbb[0][1]
        figureHeight = fbb[1][1] - fbb[0][1]
        increaseFactor = legendHeight / figureHeight
        ## Jämför ylim och figHeight för att beräkna padding för legend
        ## Ändra paddingForLegend för att legenden ska hamna
        ## rätt i figuren
        newFigHeight = figHeight * (increaseFactor * paddingForLegend + 1)
        fig.set_figheight(newFigHeight, forward=True)
        fig.subplots_adjust(bottom=(increaseFactor * paddingForLegend))
        # fig.subplots_adjust(bottom=0.5)
    elif legendloc:
        legendBox = ax.legend(fontsize=legendFontsize, loc=legendloc, ncols=ncols)
    else:
        ax.legend(fontsize=legendFontsize, ncols=ncols)  # , loc=legendloc)
    if title:
        ax.set_title(title)
    else:
        pass
    # fig.tight_layout()
    if filename:
        if isinstance(filename, str):
            filename = Path(filename)
        if filename.suffix.upper() == ".SVG":
            try:
                fig.savefig(filename, backend="Cairo", transparent=True, dpi=100)
            except:
                print("Installera pycairo för att figuren ska se bra ut i Word")
                fig.savefig(filename, transparent=True, dpi=100)
        else:
            fig.savefig(filename, transparent=True, dpi=100)


def filterScenario(dfTmp, selectedScenario, databaseFilename, crs="epsg:3016"):
    df = dfTmp.copy(deep=True)
    definedCrs = "3016"
    if selectedScenario.upper() != "BASE":
        scenarios = gpd.read_file(
            databaseFilename, layer="m_ScenarioManagementScenario", driver="GeoPackage"
        )

        alternatives = gpd.read_file(
            databaseFilename,
            layer="m_ScenarioManagementAlternative",
            driver="GeoPackage",
        )
        alternatives = alternatives.astype({"altid": int})

        scenario = scenarios.loc[scenarios["name"] == selectedScenario].to_dict(
            orient="records"
        )[0]
        bottomAlternatives = list(map(int, scenario["alternatives"].split(";")))
        sortedAlternatives = list()
        sortedAlternatives += bottomAlternatives
        i = 0
        for val in bottomAlternatives:
            status = True
            currentAlternative = val
            while status == True:
                upperAlternative = alternatives.loc[
                    alternatives["altid"] == currentAlternative, "parent"
                ].iloc[0]
                if upperAlternative not in sortedAlternatives:
                    if upperAlternative == 0:
                        status = False
                    else:
                        sortedAlternatives.append(upperAlternative)
                else:
                    pass
                currentAlternative = upperAlternative
        sortedAlternatives += [0]
        df = (
            df.sort_values(
                by="altid",
                key=lambda column: column.map(
                    {value: index for index, value in enumerate(sortedAlternatives)}
                ),
            )
            .reset_index()
            .drop_duplicates(subset="muid")
        )
    else:
        df = df.loc[df["altid"] == 0]
    return df


def readData(
    databaseFilename,
    uplevel="uplevel",
    uplevel_c="uplevel_c",
    dwlevel="dwlevel",
    dwlevel_c="dwlevel_c",
    selectedScenario="Base",
    **kwargs
):
    pipes = gpd.read_file(databaseFilename, layer="msm_link", driver="GeoPackage")
    pipes = filterScenario(pipes, selectedScenario, databaseFilename)
    nodes = gpd.read_file(databaseFilename, layer="msm_node", driver="GeoPackage")
    nodes = filterScenario(nodes, selectedScenario, databaseFilename)
    pipes.set_index("muid", inplace=True)
    pipes.loc[pipes.loc[:, uplevel].isnull(), uplevel] = pipes.loc[
        pipes.loc[:, uplevel].isnull(), uplevel_c
    ]
    pipes.loc[pipes.loc[:, dwlevel].isnull(), dwlevel] = pipes.loc[
        pipes.loc[:, dwlevel].isnull(), dwlevel_c
    ]
    pipes.loc[:, "geometric_l"] = pipes.length
    nodes.set_index("muid", inplace=True)

    results = dict()
    for key in kwargs.keys():
        results[key] = Res1D(kwargs[key])
    return pipes, nodes, results
