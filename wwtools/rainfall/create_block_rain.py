from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from mikeio.eum import ItemInfo, EUMType, EUMUnit  # , DataValueType
from mikecore.DfsFile import DataValueType


def dahlstrom(Tr, Tc):
    """Dahlströms formel (2010)
    Tr anges i månader
    Tc anges i minuter
    """
    if (
        # (type(Tc) == int)
        isinstance(Tc, int)
        or (isinstance(Tc, float))
        or (isinstance(Tc, np.float64))
        or (isinstance(Tc, np.int32))
        or (isinstance(Tc, np.int64))
    ):
        i = 190 * np.power(Tr, 1 / 3) * np.log(Tc) / np.power(Tc, 0.98) + 2
    else:
        i = np.zeros(len(Tc))
        if Tc[0] == 0:
            i[1:] = (
                190 * np.power(Tr, 1 / 3) * np.log(Tc[1:]) / np.power(Tc[1:], 0.98) + 2
            )
        else:
            i[:] = 190 * np.power(Tr, 1 / 3) * np.log(Tc) / np.power(Tc, 0.98) + 2
    return i


def dahlstrom_depth(Tr, Tc):
    """Dahlströms formel (2010)
    Tr anges i månader
    Tc anges i minuter
    """
    intensity = dahlstrom(Tr, Tc)
    rainDepth = intensity * (Tc * 60) / 10000  # mm
    return rainDepth


def create_rain_df(dfTmp, name, path, rainDepthCol="mm", dateCol="Dattid"):
    df = dfTmp.copy(deep=True)
    df = dfTmp.copy(deep=True)
    # parameter = EUMType.Rainfall_Depth
    # enhet = EUMUnit.millimeter
    df = df.loc[:, [dateCol, rainDepthCol]]
    df.set_index(dateCol, inplace=True)
    iteminfo = ItemInfo(
        name,
        itemtype=EUMType.Rainfall_Depth,
        unit=EUMUnit.millimeter,
        data_value_type=DataValueType.StepAccumulated,
    )
    df.to_dfs0(path, items=[iteminfo])


def create_variable_block_rains(durations, returnTime):
    """
    durations anges som lista och i minuter
    returnTime anges i år
    """
    if isinstance(durations, np.ndarray):
        pass
    elif isinstance(durations, list):
        durations = np.array(durations)
    else:
        durations = np.array([durations])
    depths = dahlstrom_depth(returnTime * 12, durations)
    baseTime = datetime(year=1900 + returnTime, month=1, day=1, hour=12, minute=0)
    precipDepth = np.zeros(4 * durations.shape[0] + 2)
    precipTime = np.zeros(4 * durations.shape[0] + 2, dtype="datetime64[s]")
    precipTime[0] = baseTime - timedelta(hours=12)
    for i, duration in enumerate(durations):
        startTime = baseTime + (timedelta(days=(2 * i)))
        endTime = startTime + timedelta(minutes=int(duration))
        precipDepth[1 + i * 4: 1 + ((i + 1) * 4)] = [0, 0, depths[i], 0]
        precipTime[1 + i * 4: 1 + int((i + 1) * 4)] = [
            startTime - timedelta(minutes=60),
            startTime,
            endTime,
            endTime + timedelta(minutes=30),
        ]
    lastTime = precipTime[-2].tolist() + timedelta(days=3)
    precipTime[-1] = datetime(lastTime.year, lastTime.month, lastTime.day, 0, 0)
    blockRains = pd.DataFrame({"Dattid": precipTime, "Rain (mm)": precipDepth})
    return blockRains


def create_variable_return_time(duration, returnTimes):
    returnTimes *= 12
    depths = dahlstrom_depth(returnTimes, duration)
    precipDepth = np.zeros(4 * returnTimes.shape[0] + 1)
    precipTime = np.zeros(4 * returnTimes.shape[0] + 1, dtype="datetime64[s]")

    for i, returnTime in enumerate(returnTimes):
        yr, mnth = [int(val) for val in np.divmod(returnTime, 12)]
        mnth = np.max([0, mnth - 1])
        startTime = datetime(year=1900 + yr, month=1 + mnth, day=1, hour=12, minute=0)
        endTime = startTime + timedelta(minutes=int(duration))
        precipDepth[i * 4: ((i + 1) * 4)] = [0, 0, depths[i], 0]
        precipTime[i * 4: int((i + 1) * 4)] = [
            startTime - timedelta(minutes=60),
            startTime,
            endTime,
            endTime + timedelta(minutes=30),
        ]

    lastTime = precipTime[-2].tolist() + timedelta(days=3)
    precipTime[-1] = datetime(lastTime.year, lastTime.month, lastTime.day, 0, 0)
    blockRains = pd.DataFrame({"Dattid": precipTime, "Rain (mm)": precipDepth})
    return blockRains
