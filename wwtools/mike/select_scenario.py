import qgis
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsPoint,
    QgsPointXY,
    QgsLineString,
    QgsGeometry,
    QgsDataSourceUri
)
from qgis.utils import iface

import numpy as np

import copy

def filter_scenario(db_name, feature_type, selectedScenario, crs="epsg:3016", model_type='wd'):
    
    project = QgsProject.instance()
    gpkg_layer = QgsVectorLayer(db_name, "layer_name", "ogr")

    if selectedScenario.upper() != "BASE":

        geom_column = "geometry"
        uri = f"dbname='{db_name}' table='m_ScenarioManagementScenario' {geom_column} sql="
        scenarios = QgsVectorLayer(uri, 'm_ScenarioManagementScenario', "spatialite")

        uri = f"dbname='{db_name}' table='m_ScenarioManagementScenario' {geom_column} sql="
        scenarios = QgsVectorLayer(uri, 'm_ScenarioManagementScenario', "spatialite")

        uri = f"dbname='{db_name}' table='m_ScenarioManagementAlternative' {geom_column} sql="
        alternatives = QgsVectorLayer(uri, 'm_ScenarioManagementAlternative', "spatialite")
        
        if not scenarios.isValid():
            print("Failed to load layer:", scenarios.errorString())
            exit()
        
        if not alternatives.isValid():
            print("Failed to load layer:", alternatives.errorString())
            exit()
        # alternatives = alternatives.astype({"altid": int})
        for feature in scenarios.getFeatures():
            if feature["name"] == selectedScenario:
                fields = [field.name() for field in feature.fields()]
                scenario = dict(zip(fields, feature.attributes()))
                break
            else:
                continue

        bottomAlternatives = list(map(int, scenario["alternatives"].split(";")))
        sortedAlternatives = list()
        sortedAlternatives += bottomAlternatives
        # i = 0
        for val in bottomAlternatives:
            status = True
            currentAlternative = val
            while status == True:
                filter_string = f"altid = '{currentAlternative}'"
                alternatives.setSubsetString(filter_string)
                upperAlternative = list(alternatives.getFeatures())[0]["parent"]
                alternatives.setSubsetString("")
                if upperAlternative not in sortedAlternatives:
                    if upperAlternative == 0:
                        status = False
                    else:
                        sortedAlternatives.append(upperAlternative)
                else:
                    pass
                currentAlternative = upperAlternative
        sortedAlternatives += [0]

        if feature_type.upper() in ['PIPE', 'LEDNING', 'LINK', 'LINKS', 'PIPES']:
            if model_type == 'wd':
                object_table_name = 'mw_Pipe'
                uri = f"dbname='{db_name}' table='mw_Pipe' {geom_column} sql="
                objects = QgsVectorLayer(uri, 'm_ScenarioManagementScenario', "spatialite")
            else:
                pass
        
        # objects.getFeatures

        # sql_query = f"SELECT DISTINCT MUID FROM {object_table_name}"
        # results = objects.dataProvider().executeQuery(sql_query)
        idx_names = list()
        for feature in objects.getFeatures():
            idx_names.append(feature['MUID'])
        
        unique_idx = np.unique(idx_names)
        
        filter_object_string = str()

        for idx in unique_idx:
            filter_string = f"MUID = '{idx}'"
            objects.setSubsetString(filter_string)
            for altid in sortedAlternatives:
                filter_string = f"MUID = '{idx}' AND altid = {altid}"
                objects.setSubsetString(filter_string)
                if len(list(objects.getFeatures())) > 0:
                    filter_object_string += f"(MUID = '{idx}' AND altid = {altid}) OR "
        filter_object_string = filter_object_string.rstrip(' OR ')
        
        objects.setSubsetString("")
        # objects.setSubsetString(filter_object_string)

        # QgsProject.instance().addMapLayer(objects)

        # uri = f"dbname='{db_name}' table='{object_table_name}' {geom_column} sql={filter_object_string}"
        # filtered_layer = QgsVectorLayer(uri, object_table_name, "spatialite")
        # QgsProject.instance().addMapLayer(filtered_layer)


        uri = QgsDataSourceUri()
        uri.setDatabase(db_name)
        schema = ''
        table = object_table_name
        geom_column = 'Geometry'
        uri.setDataSource(schema, table, geom_column)
        # uri.setSql(filter_object_string)
        display_name = f"{object_table_name}_{selectedScenario}"
        vlayer = QgsVectorLayer(uri.uri(), display_name, 'spatialite')

        vlayer.setSubsetString(filter_object_string)

        QgsProject.instance().addMapLayer(vlayer)





# db_name = R"C:\Users\SEDINI\Documents\SEDINI_Arbetsmapp\30028970_Faluns_Vattenmodell\modell\l176_Falun.sqlite"
filter_scenario(db_name, 'pipe', "Nollmodell_2", crs="epsg:3016", model_type='wd')

#"exec(Path('C:/Users/SEDINI/Documents/Programmering/ww-tools/wwtools/mike/select_scenario.py').read_text())"