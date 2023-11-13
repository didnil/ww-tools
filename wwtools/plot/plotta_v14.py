# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 13:03:47 2021

@author: SEDINI
"""
# Inspiration tagits från
# https://stackoverflow.com/questions/65395378/how-to-update-a-plot-in-pyqtgraph
# Det mesta i hur plotten skapas är taget från ovan

import sys
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from datetime import datetime as dt
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import partial
from sklearn import metrics
from PyQt6 import QtCore

from PyQt6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QCheckBox,
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QDateTimeEdit,
)


class MyApp(QMainWindow):
    def __init__(self, *args):  # , **kwargs):
        super().__init__()
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.central_layout = QVBoxLayout()

        self.tab1.setLayout(self.central_layout)
        self.tabs.addTab(self.tab1, "AGP")

        self.plot_boxes_layout = QHBoxLayout()
        self.boxes_layout = QVBoxLayout()
        self.plot_widget = pg.GraphicsLayoutWidget()

    def plotData(self, **kwargs):
        series = kwargs["series"]
        if isinstance(series, list):
            pass
        else:
            series = [series]

        self.data = list()

        for i in range(len(series)):
            self.data.append(dict())
            for name, df in series[i].items():
                for col in df.columns:
                    xdata = np.asarray([d.timestamp() for d in df.index.to_list()])
                    ydata = np.asarray(df.loc[:, col].to_list())
                    seriesName = f"{name}-{col}"
                    self.data[i][seriesName] = [xdata, ydata]

        self.num = sum([len(d.keys()) for d in self.data])

        # Måste lägga till unika färger!
        self.colors = [
            (c[0] * 255, c[1] * 255, c[2] * 255)
            for c in [plt.cm.tab10(i) for i in range(10)]
        ]
        self.colors.append((60, 120, 180))
        self.colors = self.colors + [
            (c[0] * 255, c[1] * 255, c[2] * 255)
            for c in [plt.cm.tab10(i) for i in range(10)]
        ]
        self.check_boxes = [
            QCheckBox(f"{name}") for d in self.data for name in d.keys()
        ]
        self.plot_data = list()

        self.central_layout.addWidget(QLabel("plot window"))
        self.central_layout.addLayout(self.plot_boxes_layout)
        self.plot_boxes_layout.addWidget(self.plot_widget)
        self.plot_boxes_layout.addLayout(self.boxes_layout)

        i = 0
        for axis in range(len(self.data)):
            for sn in self.data[axis]:
                self.boxes_layout.addWidget(self.check_boxes[i])
                self.check_boxes[i].stateChanged.connect(
                    partial(self.box_changed, i, axis)
                )
                self.plot_data.append(
                    pg.PlotDataItem(
                        self.data[axis][sn][0],
                        self.data[axis][sn][1],
                        pen=self.colors[i],
                    )
                )
                i += 1

        seriesNames = [ts for axis in self.data for ts in axis.keys()]

        # Här finns referens på hur x-axeln data hämtas:
        # https://stackoverflow.com/questions/42262550/python-in-pyqtgraph-how-to-obtain-the-axes-range-of-a-plotwidget
        self.dropDownSeries = QComboBox()
        self.dropDownSeries.addItems(seriesNames)
        self.boxes_layout.addWidget(self.dropDownSeries)

        self.timeSeriesUnits = QComboBox()
        # self.timeSeriesUnits.addItems(['m3/s', 'l/s', 'm3/h','l/m'])
        self.timeSeriesUnits.addItems(["seconds", "minutes", "hours"])
        self.boxes_layout.addWidget(self.timeSeriesUnits)
        self.seriesType = QComboBox()
        self.seriesType.addItems(["auc", "sum"])
        self.boxes_layout.addWidget(self.seriesType)

        self.xRangeBox = QHBoxLayout()

        self.x0range = QDateTimeEdit()
        self.x1range = QDateTimeEdit()

        self.x0range.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.x1range.setDisplayFormat("yyyy-MM-dd HH:mm:ss")

        self.xRangeBox.addWidget(self.x0range)
        self.xRangeBox.addWidget(self.x1range)
        self.boxes_layout.addLayout(self.xRangeBox)
        self.updateTimeButton = QPushButton("Update time", self)
        self.boxes_layout.addWidget(self.updateTimeButton)
        self.updateTimeButton.clicked.connect(self.set_xrange)

        self.calculateButton = QPushButton("Calculate", self)
        self.boxes_layout.addWidget(self.calculateButton)
        self.presentResultBox = QLabel(
            alignment=QtCore.Qt.AlignmentFlag.AlignTop
            | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.boxes_layout.addWidget(self.presentResultBox)

        self.calculateButton.clicked.connect(self.calc_auc)

        xValues = [
            xVal for axis in self.data for ts in axis.keys() for xVal in axis[ts][0]
        ]  # for v in val[0]]
        xMin = np.nanmin(xValues)
        xMax = np.nanmax(xValues)
        self.ax = list()
        self.add_ax()

        self.x0range.setDateTime(dt.fromtimestamp(xMin))
        self.x1range.setDateTime(dt.fromtimestamp(xMax))
        self.ax[0].setRange(xRange=[xMin, xMax])
        self.ax[0].sigXRangeChanged.connect(
            self.update_xrange_boxes
        )  # setRange(xRange=[xMin, xMax])

    def set_xrange(self):
        # self.ax[0].setRange(xRange=[xMin, xMax])
        # print(self.x0range.dateTime().toSecsSinceEpoch())
        newX0 = self.x0range.dateTime().toSecsSinceEpoch()
        newX1 = self.x1range.dateTime().toSecsSinceEpoch()
        self.ax[0].setRange(xRange=[newX0, newX1])  # , yRange=[0,0])

        # for row in range(len(self.ax)):
        #     x_range, y_range = self.ax[row].getViewBox().viewRange()
        #     self.ax[row].setRange(yRange=y_range)#, yRange=[0,0])

    def update_xrange_boxes(self):
        xRanges, yRanges = self.ax[0].getViewBox().viewRange()
        x0, x1 = xRanges

        self.x0range.setDateTime(dt.fromtimestamp(x0))
        self.x1range.setDateTime(dt.fromtimestamp(x1))

    def calc_auc(self):
        x0, x1 = self.ax[0].getAxis("bottom").range
        self.dropDownSeries

        for i in range(len(self.data)):
            if self.dropDownSeries.currentText() in self.data[i].keys():
                x, y = self.data[i][self.dropDownSeries.currentText()]
                mask = np.nonzero((x >= x0) & (x <= x1))[0]
                if self.seriesType.currentText() == "auc":
                    if self.timeSeriesUnits.currentText() == "seconds":
                        auc_ = metrics.auc(x[mask], y[mask])
                    elif self.timeSeriesUnits.currentText() == "minutes":
                        auc_ = metrics.auc(x[mask] / 60, y[mask])
                    elif self.timeSeriesUnits.currentText() == "hours":
                        auc_ = metrics.auc(x[mask] / 3600, y[mask])
                    self.presentResultBox.setText(f"{auc_:.1f}")
                    break
                elif self.seriesType.currentText() == "sum":
                    self.presentResultBox.setText(f"{np.sum(y[mask]):.1f}")

    def add_ax(self):
        idx = 0
        for row, series in enumerate(self.data):
            self.ax.append(self.plot_widget.addPlot(row=row, col=0))
            self.ax[row].setAxisItems({"bottom": pg.DateAxisItem(orientation="bottom")})
            self.ax[row].setMouseEnabled(x=True, y=False)
            self.ax[row].setAspectLocked(lock=False)
            self.ax[row].enableAutoRange(axis="y", enable=True)
            self.ax[row].setAutoVisible(y=True)
            if row != 0:
                self.ax[row].setXLink(self.ax[0])

    def box_changed(self, idx, axis):
        if self.check_boxes[idx].isChecked():
            self.ax[axis].addItem(self.plot_data[idx])
        else:
            self.ax[axis].removeItem(self.plot_data[idx])

    # =============================================================================
    #     def mouseMoved(evt):
    #       mousePoint = p.vb.mapSceneToView(evt[0])
    # =============================================================================

    # =============================================================================
    #     def onMouseMoved(p, point):
    #         p = self.plot.plotItem.vb.mapSceneToView(point)
    #         self.statusBar().showMessage("{}-{}".format(p.x(), p.y()))
    # =============================================================================

    def slidingMean(y, n):
        y_rm = np.zeros(len(y))
        for j, v in enumerate(y):
            if j >= n & (j + n) <= len(y):
                y_rm[j] = np.mean(np.concatenate((y[j - n : j], y[j : j + n])))
            elif j < n:
                y_rm[j] = np.mean(np.concatenate((y[:j], y[j : (2 * n - j)])))
            elif (j + n) > len(y):
                y_rm[j] = np.mean(
                    np.concatenate((y[j - n : j], y[j : j + (len(y) - j)]))
                )
        return y_rm


def readData(dbPath):
    con = sqlite3.connect(dbPath)
    dfDict = dict()
    res = con.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for name in res.fetchall():
        if name[0] not in ["meas_flow", "meas_level"]:
            sqlQuery = f"""
                SELECT *
                FROM "{name[0]}"
                """
            df = pd.read_sql(sqlQuery, con=con)
            df.loc[:, "Datetime"] = pd.to_datetime(
                df["Datetime"], format="%Y-%m-%d %H:%M:%S"
            )
            df.set_index("Datetime", inplace=True)
            dfDict[name[0]] = df
    con.close()
    return dfDict


def filterData(dfDict, toPlot):
    selectedDfDict = dict()
    if toPlot.upper() in ["FLOW", "FLÖDE", "l/m"]:
        for name, df in dfDict.items():
            selectedCols = df.columns.str.upper().str.contains(
                "FLÖDE", regex=True
            ) | df.columns.str.upper().str.contains("L/M", regex=True)
            selectedDfDict[name] = df.loc[:, selectedCols]
    elif toPlot.upper() in [
        "PRESSURE",
        "LEVEL",
        "TRYCK",
        "NIVÅ",
    ]:
        for name, df in dfDict.items():
            if name == "29 ingas väg":
                continue
            selectedCols = df.columns.str.upper().str.contains(
                "TRYCK", regex=True
            ) | df.columns.str.upper().str.contains("NIVÅ", regex=True)
            if any(selectedCols):
                selectedDfDict[name] = df.loc[:, selectedCols]
            else:
                continue
    return selectedDfDict


# dfDict = readData(Path(r"C:\Users\SEDINI\Documents\SEDINI_Arbetsmapp\diverse\lisa\plotta_data\measurementData.sqlite"))

# Stenungsund
dfDict = readData(
    Path(
        R"C:\Users\SEDINI\Documents\Arbetsmapp\Stenungsund_30063116\measurement\measurements.sqlite"
    )
)
# feather_dir = Path(R"C:\Users\SEDINI\Documents\SEDINI_Arbetsmapp\Stenungsund\arbetsmaterial\mätning\feather")

# # Falun

# dfDict = readData(Path(r"C:\Users\SEDINI\Documents\SEDINI_Arbetsmapp\30028970_Faluns_Vattenmodell\arbetsmaterial\input_data.sqlite"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()

    # ## Stenungsund tryckmätning
    # vbp160 = pd.read_feather(feather_dir / 'vbp160.feather')
    # vbp460 = pd.read_feather(feather_dir / 'vbp460.feather')
    # flow_meas = pd.read_feather(feather_dir / 'flow_meas_sep-oct.feather')
    # window.plotData(series=[{'flow': flow_meas}, {'vbp160': vbp160, 'vbp460': vbp460}])

    ## Stöten
    # stoten_dir = Path(R"C:\Users\SEDINI\Documents\SEDINI_Arbetsmapp\Stoten_XX")
    # flow = pd.read_feather(stoten_dir / 'arbetsmaterial' / 'flow_data.feather')
    # pres = pd.read_feather(stoten_dir / 'arbetsmaterial' / 'pressure_data.feather')
    # window.plotData(series=[{'flow': flow}, {'pressure': pres}])

    # # Stenungsund
    dfDict1 = filterData(dfDict, "flow")
    dfDict2 = filterData(dfDict, "pressure")
    window.plotData(series=[dfDict1, dfDict2])

    # # Falun
    # dfDict1 = filterData(dfDict, 'flow')
    # dfDict2 = filterData(dfDict, 'pressure')
    # window.plotData(series=[dfDict1, dfDict2])

    # window.plotData(series=[dfDict1])
    window.show()
    sys.exit(app.exec())
