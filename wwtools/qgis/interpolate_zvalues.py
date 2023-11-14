import qgis
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsPoint,
    QgsPointXY,
    QgsLineString,
    QgsGeometry,
)
from qgis.utils import iface


def duplicate_layer():
    """
    Duplicates selected and then makes the edits on duplicated layer.
    """

    layer = qgis.utils.iface.activeLayer()
    crs = layer.crs().authid().split(":")[1]
    memory_layer = QgsVectorLayer(
        f"LineString?crs=epsg:{crs}", "Layer_with_z_values", "memory"
    )
    features = [feature for feature in layer.getFeatures()]
    memory_layer_data = memory_layer.dataProvider()
    attributes = layer.dataProvider().fields().toList()
    memory_layer_data.addAttributes(attributes)
    memory_layer.updateFields()
    memory_layer_data.addFeatures(features)
    QgsProject.instance().addMapLayer(memory_layer)
    return memory_layer


def interpolate_zvalues(_uplvl, _dwlvl):
    """
    Sets z-values based on interpolation between uplevel- and downlevel.

    Parameters
    ----------
    _uplvl : str
        Name of column with upstream level
    _dwlvl : str
        Name of column with downstream level
    """

    layer = duplicate_layer()
    layer.startEditing()
    # Loopa inte igenom sista linjen utan lägg bara till
    # dwLvl längst ned.
    for feature in layer.getFeatures():
        points_with_z = list()
        geom = feature.geometry()
        uplvl, dwlvl = feature[_uplvl], feature[_dwlvl]
        slope = (uplvl - dwlvl) / geom.length()
        points_with_z.append(
            QgsPoint(list(geom.vertices())[0].x(), list(geom.vertices())[0].y(), uplvl)
        )
        accumulated_length = 0
        for p0, p1 in zip(list(geom.vertices())[:-1], list(geom.vertices())[1:]):
            p0 = QgsPointXY(p0.x(), p0.y())
            p1 = QgsPointXY(p1.x(), p1.y())
            current_lineString = QgsLineString([p0, p1])
            current_length = current_lineString.length()
            accumulated_length += current_length
            points_with_z.append(
                QgsPoint(p1.x(), p1.y(), uplvl - accumulated_length * slope)
            )
        new_geom = QgsLineString(points_with_z)
        layer.dataProvider().changeGeometryValues(
            {feature.id(): QgsGeometry.fromPolyline(new_geom)}
        )
    iface.vectorLayerTools().stopEditing(layer, False)


interpolate_zvalues("npv1", "npv2")
