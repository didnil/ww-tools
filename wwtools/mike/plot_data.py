import pandas as pd
import numpy as np
import locale
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
from mikeio1d.res1d import Res1D, QueryDataReach, QueryDataNode
from sklearn.metrics import auc
from pathlib import Path
import datetime


locale.setlocale(locale.LC_ALL, '')
plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams['font.family']='Arial'
plt.rcParams['font.size']='12'
plt.rcParams['svg.fonttype'] = 'none'


# ## Förslag till förbättring
# - Skapa en bättre metod, tvinga en plot för varje. Kräver mer av användaren, men blir mer flexibelt
# - Spara alla data i en dataframe men med en multiindex heading för att separera de olika mätpunkterna


def calcMeanFlow(res, pipe, start=False, end=False):
    if (start is not False) and (end is not False):
        flow = res.read(QueryDataReach('Flow', pipe)).loc[start:end]
    elif (start is not False):
        flow = res.read(QueryDataReach('Flow', pipe)).loc[start:]
    elif (end is not False):
        flow = res.read(QueryDataReach('Flow', pipe)).loc[:end]
    else:
        flow = res.read(QueryDataReach('Flow', pipe))
    dt = (flow.index[-1] - flow.index[0]).total_seconds()
    vol = auc(flow.index.view(np.int64) // 1e9, flow.iloc[:, 0])
    meanFlow = vol / dt
    print(f"Medelflödet är {meanFlow:.1f} l/s")


def selectDataReach(objects, items, result, start=False, end=False, chainage=None):
    resultItems = list()
    for objVal in objects:
        # resultDf = pd.DataFrame(index=result._time_index)
        resultDf = pd.DataFrame(index=result.time_index)
        if (start is not False) and (end is not False):
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataReach(itemVal, objVal, chainage)).loc[start:end]], axis=1, join='inner')
        elif (start is not False):
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataReach(itemVal, objVal, chainage)).loc[start:]], axis=1, join='inner')
        elif (end is not False):
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataReach(itemVal, objVal, chainage)).loc[:end]], axis=1, join='inner')
        else:
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataReach(itemVal, objVal, chainage))], axis=1, join='inner')
        resultDf.columns = pd.MultiIndex.from_arrays([[objVal] * len(items), items])
        resultItems.append(resultDf)
    resultItems = pd.concat(resultItems, axis=1)
    return resultItems


def selectDataNode(objects, items, result, start=False, end=False):
    resultItems = list()
    for objVal in objects:
        resultDf = pd.DataFrame(index=result.time_index)
        if (start is not False) and (end is not False):
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataNode(itemVal, objVal)).loc[start:end]], axis=1, join='inner')
        elif (start is not False):
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataNode(itemVal, objVal)).loc[start:]], axis=1, join='inner')
        elif (end is not False):
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataNode(itemVal, objVal)).loc[:end]], axis=1, join='inner')
        else:
            for itemVal in items:
                resultDf = pd.concat([resultDf, result.read(QueryDataNode(itemVal, objVal))], axis=1, join='inner')
        resultDf.columns = pd.MultiIndex.from_arrays([[objVal] * len(items), items])
        resultItems.append(resultDf)
    resultItems = pd.concat(resultItems, axis=1)
    return resultItems


def selectData(objects, items, result, objType='reach', model='wd', chainage=0, nSplit = 3, title=False, start=False, end=False, shift = False, legend = False, mapp = '.', namn = False):
    """
    Plotta flödesdata
    """

    if not isinstance(objects, list):
        objects = [objects]
    else:
        pass

    if not isinstance(items, list):
        items = [items]
    else:
        pass

    mm2inch = 1 / 25.4
    
    if model == 'wd':
        chainage = None
    else:
        pass

    reachTypes = ['pipe', 'reach']
    nodeTypes = ['node']
    if objType in reachTypes:
        resultItems = selectDataReach(objects, items, result, start, end)
    elif objType in nodeTypes:
        resultItems = selectDataNode(objects, items, result, start, end)
    else:
        print(f'Objekttypen saknas. De typer som stöds är {reachTypes + nodeTypes}')
        # resultItems.append(resultDf)
    return resultItems


def plotData(resultItems, axes=False):
    # Skapa dict axTranslation i början i stället för i slutet.
    # Borde spara några rader kod.
    mm2Inch = (1/25.4)
    colors = iter([plt.cm.Dark2(i) for i in range(30)])
    items = list()
    for objVal in resultItems.values():
        items += list(objVal.columns)
    items = list(set(items))

    fig = plt.figure(figsize=((210 - 25 * 2) * .8 * mm2Inch, (297 - 25 * 2) * .4 * mm2Inch))
    if not axes:
        ax = fig.add_subplot(111)
        plotAxes = {items[0]: ax}
        for item in items[1:]:
            plotAxes[item] = ax.twinx()
        
        for item, ax in plotAxes.items():
            ax.set_ylabel(item)

    else:
        uniqueAxes = np.unique(list(axes.values()))
        firstAx = uniqueAxes[0]
        axesToTwin = [ax for ax in uniqueAxes if ax !=firstAx]
        axDict = {firstAx: fig.add_subplot(111)}
        for ax in axesToTwin:
            axDict[ax] = axDict[firstAx].twinx()
        
        plotAxes = dict()
        for item in items:
            plotAxes[item] = axDict[axes[item]]
        
        standardLabels = {key: '_'.join(val) for key, val in {val: [key for key in axes.keys() if axes[key] == val] for val in uniqueAxes}.items()}
        for ax, item in standardLabels.items():
            axDict[ax].set_ylabel(item)

    for objName, objVal in resultItems.items():
        for itemVal in objVal.columns:
            plotAxes[itemVal].plot(objVal.index, objVal.loc[:, itemVal],
                                    label=f"{objName}: {itemVal}", color=next(colors))

    fig.legend()
    return fig, plotAxes


def plotDataWrapper(objects, items, result, objType='reach', model='wd', chainage=0, axes=False, start=False, end=False):
    resultItems = selectData(objects, items, result, objType=objType, model=model, chainage=chainage, start=start, end=end)
    fig, ax = plotData(resultItems, axes=axes)
    return fig, ax


def plotTS(df, item, colors, ax, labels=False, linestyle=None):
    if isinstance(labels, list):
        pass
    else:
        labels = [labels]
    for i, location in enumerate(np.unique(df.columns.get_level_values(0))):
        # if labels:
        #     ax.plot(df.loc[:, (location, item)], label=labels[i], color=next(colors))
        # else:
        #     ax.plot(df.loc[:, (location, item)], label=f'{location}: {item}', color=next(colors))

        if isinstance(colors, list):
            ax.plot(df.loc[:, (location, item)], label=f'{location}: {item}', color=colors[i], linestyle=linestyle)
        elif isinstance(colors, str):
            ax.plot(df.loc[:, (location, item)], label=f'{location}: {item}', color=colors, linestyle=linestyle)
        elif iter(colors):
            ax.plot(df.loc[:, (location, item)], label=f'{location}: {item}', color=next(colors), linestyle=linestyle)
        else:
            ax.plot(df.loc[:, (location, item)], label=f'{location}: {item}', linestyle=linestyle)# , color=colors)

    return ax


def createPlot(interval=6, figsize=(5, 3.9), rotation=45, formatter='%d | %H:%M'):
    fig, ax = plt.subplots(figsize=figsize)
    xfmt = mdates.DateFormatter(formatter)
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    [label.set_rotation(45) for label in ax.get_xticklabels()]
    return fig, ax