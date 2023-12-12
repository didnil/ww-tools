import numpy as np
import pandas as pd
import geopandas as gpd

def convert_to_dim(dfTmp, _inner, _outer):
    """
    Conver inner and outer diameter to one dimension column

    parameters
    ----------
    dfTmp : GeoDataFrame
        DataFrame with pipe objects with attributes
    _inner : str
        Name of column with inner diameter
    _outer : str
        Name of column with outer diameter
    """

    df = dfTmp.copy(deep=True)
    df.replace({'INND': {0: np.nan}, 'YTTD': {0: np.nan}}, inplace=True)
    existing_columns = list(df.columns)

    mask = df[['INND', 'YTTD']].notnull().any(axis=1)
    df.loc[mask, 'DIM'] = df.loc[mask, ['INND', 'YTTD']].min(axis=1)
    df.loc[~mask, 'DIM'] = np.nan

    new_columns = [val for val in existing_columns if val not in [_inner, _outer]]
    new_columns.append('DIM')
    return df.loc[:, new_columns]


def groupby_multi(dfTmp, by, columns):
    """
    Group data based on columns and check missing values

    parameters
    -----------
    dfTmp : GeoDataFrame
        DataFrame with pipe objects with attributes
    by : str
        Value to group on
    columns : list
        Which columns to calculate missing values for

    """

    df = dfTmp.copy(deep=True)
    df.loc[:, columns] = df.loc[:, columns].notnull().astype(int)
    dfList = list()
    for col in columns:
        _df = df.groupby(by=[by, col]).aggregate(func={col: 'count', 'LENGTH': 'sum'})
        _df = _df.assign(LENGTH_KM=(_df['LENGTH'] / 1000).round(1))
        _df = _df.join(_df.groupby(level=0)['LENGTH'].sum(), rsuffix='_TOT')
        _df = _df.assign(LÄNGD_PROC=np.round((_df['LENGTH'] / _df['LENGTH_TOT']) * 100, 2))
        _df.drop(columns=['LENGTH_TOT'], inplace=True)
        dfList.append(_df)
    mi = pd.MultiIndex.from_arrays(np.concatenate([[np.repeat(columns, 4)], [np.tile(['ANTAL', 'LÄNGD', 'LÄNGD_KM', 'LÄNGD_PROC'], len(columns))]]))
    dfGrp = pd.concat(dfList, axis=1)
    dfGrp.columns = mi
    dfGrp.index = dfGrp.index.set_levels(dfGrp.index.levels[1].map({0: 'Saknas', 1: 'Existerande'}), level=1)
    return dfGrp


def renameValues(dfTmp, current, new, yr=False):
    """
    Rename a material name to another used in ltp analysis.
    Material must be named RMAT
    Construction year must be named ANLAR

    Parameters
    -----------
    dfTmp : GeoDataFrame
        DataFrame with pipe objects with attributes
    current : str
        Current name
    new : str
        New name
    yr : int | list
        If name differs based on year, these are the delimiters
    
    """
    df = dfTmp.copy(deep = True)
    if yr:
        if not isinstance(yr, list):
            yr = [yr]
        
        df.loc[(df.RMAT == current) & (df.ANLAR <str(yr[0])), 'RMAT'] =  f'{new} <{yr[0]}'        
        
        for i in range(0, len(yr)-1):
            df.loc[(df.RMAT == current) & ((df.ANLAR >= str(yr[i])) & (df.ANLAR < str(yr[i+1]))), 'RMAT'] =  f'{new} {yr[i]}-{str(yr[i+1]-1)[-2:]}'
        
        df.loc[(df.RMAT == current) & (df.ANLAR >=str(yr[-1])), 'RMAT'] =  f'{new} >={yr[-1]}'
    else:
        # print('Då')
        df.replace({'RMAT': {current: new}}, inplace = True)

    return df


def prepareForLtp(dfTmp, constructionYear="ANLAR"):
    """
    Rename all material names to the ones used in LTP analysis

    parameters
    -----------
    dfTmp : GeoDataFrame
        DataFrame with pipe objects with attributes
    constructionYear : str
        Name of column with constructionYear

    """

    df = dfTmp.copy(deep=True)
    dfWater = df.loc[df["system"].str.upper() == "WATER"]
    dfWaste = df.loc[df["system"].str.upper() == "WASTE"]
    dfStorm = df.loc[df["system"].str.upper() == "STORM"]

    dfWater.loc[dfWater["RMAT"].isnull(), "RMAT"] = "Övrigt/Okänt A"
    translator = {
        "PE": ["PEH", "PEM", "PEL"],
        "PLAST": ["PP"],
        "Gråjärn": ["GJJ", "GJUTJ", "JÄRN", "Gråjärn"],
        "Segjärn": ["SEGJ", "SGJ", "Segjärn"],
        "PVC": ["PVC"],
        "Övrigt/Okänt A": ["CU", "LERA", "BTG", "STÅL", "GAP", "ETERNI", "GALV", '---', 'GALV'],
    }
    for key in translator.keys():
        if key == "Gråjärn":
            for val in translator[key]:
                dfWater.loc[dfWater[constructionYear].notnull()] = renameValues(
                    dfWater.loc[dfWater[constructionYear].notnull()], val, key, 1950
                )
                dfWater.loc[dfWater[constructionYear].isnull()] = renameValues(
                    dfWater.loc[dfWater[constructionYear].isnull()], val, key
                )
        elif key == "PVC":
            for val in translator[key]:
                dfWater.loc[dfWater[constructionYear].notnull()] = renameValues(
                    dfWater.loc[dfWater[constructionYear].notnull()], val, key, 1970
                )
                dfWater.loc[dfWater[constructionYear].isnull()] = renameValues(
                    dfWater.loc[dfWater[constructionYear].isnull()], val, key
                )
        elif key == "Segjärn":
            for val in translator[key]:
                dfWater.loc[dfWater[constructionYear].notnull()] = renameValues(
                    dfWater.loc[dfWater[constructionYear].notnull()], val, key, 1980
                )
                dfWater.loc[dfWater[constructionYear].isnull()] = renameValues(
                    dfWater.loc[dfWater[constructionYear].isnull()], val, key
                )
        else:
            for val in translator[key]:
                dfWater = renameValues(dfWater, val, key)

    dfWaste.loc[dfWaste["RMAT"].isnull(), "RMAT"] = "S-Övrigt/Okänt A"
    translator = {
        "S-Betong": ["BTG", "S-Betong"],
        "S-Plast": [
            "PVC",
            "PLAST",
            "PP",
            "PEM",
            "PE",
            "PEH",
            "PEL",
            "PRAGMA",
            "S-Plast",
        ],
        "S-Övrigt/Okänt A": [
            "GAP",
            "GJJ",
            "GJUTJ",
            "LERA",
            "STÅL",
            "PLÅT",
            "SEGJ",
            "SGJ",
            "FLE",
            "ÖVRIG",
            "JÄRN",
            " ",
            "GLAS",
            "GLASFIBER",
            "OKÄ",
            'AsbestBTG',
            '---',
            np.nan,
        ],
        "S-Övrigt/Okänt B": [
            "PVC_tryck",
            "SEGJ_tryck",
            "GJJ_tryck",
            "AsbestBTG_tryck",
            "PE_tryck",
            "---_tryck"
        ]
    }

    for key in translator.keys():
        if key == "S-Betong":
            for val in translator[key]:
                dfWaste.loc[dfWaste[constructionYear].notnull()] = renameValues(
                    dfWaste.loc[dfWaste[constructionYear].notnull()],
                    val,
                    key,
                    [1950, 1970],
                )
                dfWaste.loc[dfWaste[constructionYear].isnull()] = renameValues(
                    dfWaste.loc[dfWaste[constructionYear].isnull()], val, key
                )
        else:
            for val in translator[key]:
                dfWaste = renameValues(dfWaste, val, key)

    dfStorm.loc[dfStorm["RMAT"].isnull(), "RMAT"] = "D-Övrigt/Okänt A"
    translator = {
        "D-Betong": ["BTG", "D-Betong"],
        "D-Plast": ["PVC", "PLAST", "PP", "PEM", "PE", "PEH", "PEL", "PRAGMA"],
        "D-Övrigt/Okänt A": [
            "GAP",
            "GJJ",
            "LERA",
            "STÅL",
            "PLÅT",
            "SEGJ",
            "SGJ",
            " ",
            "GLAS",
            "OKÄ",
            '---',
            np.nan,
        ],
    }
    for key in translator.keys():
        if key == "D-Betong":
            for val in translator[key]:
                dfStorm.loc[dfStorm[constructionYear].notnull()] = renameValues(
                    dfStorm.loc[dfStorm[constructionYear].notnull()], val, key, [1950]
                )
                dfStorm.loc[dfStorm[constructionYear].isnull()] = renameValues(
                    dfStorm.loc[dfStorm[constructionYear].isnull()], val, key
                )
        else:
            for val in translator[key]:
                dfStorm = renameValues(dfStorm, val, key)
    df = pd.concat([dfWater, dfWaste, dfStorm], axis=0)

    return df


def prepEditColumn(dfTmp):
    """
    Create FTYP column and construction decade

    parameters
    ----------
    dfTmp : GeoDataFrame
        DataFrame with pipe objects with attributes
    """
    # Jag tror att denna funktion kan tas bort. Det är nog en rest.
    # Den används, men kan nog lätt bytas ut. Flera steg
    # kan dessutom tas bort.

    df = dfTmp.copy(deep=True)
    df.loc[:, 'FTYP'] = df['system'].replace({'water': 'V', 'waste': 'S', 'storm': 'D'})
    df = df.assign(period=pd.arrays.IntervalArray.from_tuples(df.loc[:, ['start', 'end']].to_records(index=False), closed='left'))
    df = df.assign(ANLAR=pd.to_datetime(df.period.apply(lambda row: row.left).dt.year, format='%Y'))
    df.drop(columns=['start', 'end'], inplace=True)
    return df


def distributeMissingYear_wrapper(dfTmp, mapDfTmp, system, material='RMAT'):
    """
    Distribute material when year is missing.

    parameters
    ----------
    dfTmp : GeoDataFrame
        DataFrame with pipe objects with attributes
    mapDfTmo : DataFrame
        Svenskt vattens material and year distribution for LTP analysis
    system : str
        Which system to do the analysis for. The distribution varies for
        water, wastewater and stormwater
    material : str
        Name of column with material
    
    """

    mapDf = mapDfTmp.loc[mapDfTmp['system']==system]
    woMaterial = dfTmp.loc[(dfTmp['system']==system) & (dfTmp['RMAT'].str.upper().str.contains('OKÄNT'))]
    wMaterial = dfTmp.loc[(dfTmp['system']==system) & ~(dfTmp['RMAT'].str.upper().str.contains('OKÄNT'))]
    #return wMaterial

    # mapDf contains the distribution of material and year from Svenskt vatten
    # Here the the material per year in meters is recalculated to the part of the total
    # length
    mapDf = pd.merge(mapDf, mapDf.groupby(by=['Material'])['part'].sum().reset_index(), left_on='Material', right_on='Material', suffixes=[None, '_sum'])
    mapDf_material = mapDf.assign(part=mapDf['part'] / mapDf['part_sum'])

    # Calculate the total length of each material
    wMaterial = wMaterial.groupby(by=material)['LENGTH'].sum().reset_index()
    
    # Join the data from the municipality with the data form Svenskt vatten
    # When the data is joined, the length of the material for the municipality
    # is multiplied with the part from Svenskt vattens distribution
    wMaterialEdit = pd.merge(mapDf_material, wMaterial, how='inner', left_on='Material', right_on='RMAT')
    wMaterialEdit = wMaterialEdit.assign(LENGTH=wMaterialEdit['LENGTH'] * wMaterialEdit['part'])    

    # Here the lenght of the pipes where both material is missing is calculated with the corresponding
    # distribution from Svenskt vatten.
    woMaterialEdit = mapDf.assign(LENGTH=woMaterial['LENGTH'].sum() * mapDf['part'])

    materialEdit = pd.concat([wMaterialEdit, woMaterialEdit], axis=0)
    materialEdit = materialEdit.assign(**{material: materialEdit['Material']})
    materialEdit.drop(columns=['Material', 'type', 'part', 'part_sum'], inplace=True)

    return materialEdit


# fill_value={'water': 'Övrigt/Okänt A', 'waste': 'S-Övrigt/Okänt A', 'storm': 'D-Övrigt/Okänt A'},
def distributeMissingYear(dfTmp, mapDfTmp, material='RMAT'):
    """
    In this function is system type is sent to the function for distribute missing year

    parameters
    ----------
    dfTmp : GeoDataFrame
        Dataframe with pipe objects and attributes
    mapDfTmp : DataFrame
        Distribution from Svenskt vatten
    material : str
        Name of column with material data
    
    """
    df = dfTmp.copy(deep=True)
    mapDf = mapDfTmp.copy(deep=True)
    dfList = list()
    for system in ['water', 'waste', 'storm']:
        dfList.append(distributeMissingYear_wrapper(df.loc[df['system']==system], mapDf.loc[mapDf['system']==system], system, material='RMAT'))
    # return dfList
    df = pd.concat(dfList, axis=0)
    return prepEditColumn(df)


def appendToIndex(bns, start, end):
    """
    Create interval index 

    parameters
    ----------
    bns : IntervalINdex
        Intervalindex for all years in style 1910 - 1919, 1920 - 1929 and so on
    start : int
        start year
    end : int
        end year

    """

    s = pd.to_datetime(start, format='%Y')
    e = pd.to_datetime(end, format='%Y') + pd.offsets.YearEnd()
    return pd.IntervalIndex(np.concatenate([np.asarray(bns), [pd.Interval(s, e, closed='left')]]))
    # return 


def prepareForExport(df):
    """
    Prepare data for export to match layout if excel sheet for LTP analyses

    parameters
    ----------
    df : DataFrame
    """
    dfGrp = df.groupby(by = ['RMAT', 'period']).agg({'LENGTH': ['sum']}).unstack(
        level=0, fill_value=0).droplevel([0, 1], axis=1) 
    dfGrp.index = dfGrp.index.left.year
    return dfGrp