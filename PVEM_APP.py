import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QGridLayout, QMessageBox
from pvmodel_class import pvmodel
import serial
import struct
import time
import threading

""" This class is responsible for creating the graphs and plotting the curves """
class Canvas(FigureCanvas):

    # Initialize and create the graph
    def __init__(self, parent, title, xlabel, ylabel):
        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi = 100)
        self.ax.grid()
        self.ax.set_title(title, color='#3d3d3d')
        self.ax.set_xlabel(xlabel, color='#3d3d3d')
        self.ax.set_ylabel(ylabel, color='#3d3d3d')
        super().__init__(self.fig)
        self.setParent(parent)

    # Plot the curve given to the class
    def plotCurve(self, x, y, title, xlabel, ylabel, xlim, ylim):
        self.fig.canvas.ax.clear() 
        self.ax.plot(x,y)
        self.ax.grid()
        self.ax.set_title(title,color='#3d3d3d')
        self.ax.set_xlabel(xlabel,color='#3d3d3d')
        self.ax.set_ylabel(ylabel,color='#3d3d3d')
        self.ax.set_xlim([0, xlim])
        self.ax.set_ylim([0, ylim])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # Plot only a point
    def plotPoint(self, x, y, type):
        self.point, = self.ax.plot(x, y, type)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # Plot the first point
    def plotDummyPoint(self):
        self.point, = self.ax.plot(0, 0, alpha = 0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # Remove the last plotted point
    def removePoint(self):
        self.point.remove()

    # Clear the graph
    def clearAxis(self):
        self.fig.canvas.ax.clear()

""" This class is resposible for creating the user interface """
class PVEM(QMainWindow):
  
    def __init__(self):
        super().__init__()
        
        layout = QGridLayout()
        self.setWindowTitle("PVEM V1.0")

        # Create the user interface and its components
        labelSTC = QtWidgets.QLabel(self)
        labelSTC.setStyleSheet("font-size: 16px; margin-bottom: 0.5em;"
                                + "text-decoration: underline;")
        labelSTC.setText("Parameters at Standard Test Conditions (STC):" 
                          + "1000 W/m\N{SUPERSCRIPT TWO}, 25 \N{DEGREE SIGN}C")
        layout.addWidget(labelSTC, 0, 0, 2, 3)

        labelVoc = QtWidgets.QLabel(self)
        labelVoc.setText("Open Circuit Voltage (Voc)")
        layout.addWidget(labelVoc, 2, 0)
        self.editVoc = QtWidgets.QLineEdit(self)
        self.editVoc.setMaximumWidth(75)
        layout.addWidget(self.editVoc, 2, 1)
        unitVoc = QtWidgets.QLabel(self)
        unitVoc.setText("V")
        layout.addWidget(unitVoc, 2, 2)

        labelVmp = QtWidgets.QLabel(self)
        labelVmp.setText("Maximum Power Voltage (Vmp)")
        layout.addWidget(labelVmp, 3 , 0)
        self.editVmp = QtWidgets.QLineEdit(self)
        self.editVmp.setMaximumWidth(75)
        layout.addWidget(self.editVmp, 3, 1)
        unitVmp = QtWidgets.QLabel(self)
        unitVmp.setText("V")
        layout.addWidget(unitVmp, 3, 2)

        labelIsc = QtWidgets.QLabel(self)
        labelIsc.setText("Short Circuit Current (Isc)")
        layout.addWidget(labelIsc, 4, 0 )
        self.editIsc = QtWidgets.QLineEdit(self)
        self.editIsc.setMaximumWidth(75)
        layout.addWidget(self.editIsc, 4, 1)
        unitIsc = QtWidgets.QLabel(self)
        unitIsc.setText("A")
        layout.addWidget(unitIsc, 4, 2)

        labelImp = QtWidgets.QLabel(self)
        labelImp.setText("Maximum Power Current (Imp)")
        layout.addWidget(labelImp, 5, 0)
        self.editImp = QtWidgets.QLineEdit(self)
        self.editImp.setMaximumWidth(75)
        layout.addWidget(self.editImp, 5, 1)
        unitImp = QtWidgets.QLabel(self)
        unitImp.setText("A")
        layout.addWidget(unitImp, 5, 2)

        labelKv = QtWidgets.QLabel(self)
        labelKv.setText("Temperature Coefficient of Voc (Beta)")
        layout.addWidget(labelKv, 6, 0)
        self.editKv = QtWidgets.QLineEdit(self)
        self.editKv.setMaximumWidth(75)
        layout.addWidget(self.editKv, 6, 1)
        unitKv = QtWidgets.QLabel(self)
        unitKv.setText("%")
        layout.addWidget(unitKv, 6, 2)

        labelKi = QtWidgets.QLabel(self)
        labelKi.setText("Temperature Coefficient of Isc (Alpha)")
        layout.addWidget(labelKi, 7, 0)
        self.editKi = QtWidgets.QLineEdit(self)
        self.editKi.setMaximumWidth(75)
        layout.addWidget(self.editKi, 7, 1)
        unitKi = QtWidgets.QLabel(self)
        unitKi.setText("%")
        layout.addWidget(unitKi, 7, 2)

        labelNs = QtWidgets.QLabel(self)
        labelNs.setText("Cells in Series")
        layout.addWidget(labelNs, 8, 0)
        self.editNs = QtWidgets.QLineEdit(self)
        self.editNs.setMaximumWidth(75)
        layout.addWidget(self.editNs, 8, 1)
        unitNs = QtWidgets.QLabel(self)
        unitNs.setText("-")
        layout.addWidget(unitNs, 8, 2)

        labelStrings = QtWidgets.QLabel(self)
        labelStrings.setText("No of Cell Arrays")
        layout.addWidget(labelStrings, 9, 0)
        self.editStrings = QtWidgets.QLineEdit(self)
        self.editStrings.setMaximumWidth(75)
        layout.addWidget(self.editStrings, 9, 1)
        unitStrings = QtWidgets.QLabel(self)
        unitStrings.setText("-")
        layout.addWidget(unitStrings, 9, 2)

        labelAmbient = QtWidgets.QLabel(self)
        labelAmbient.setText("Emulating Conditions")
        labelAmbient.setStyleSheet("font-size: 16px; margin-bottom: 0.5em;"
                                    + "text-decoration: underline")
        layout.addWidget(labelAmbient, 10, 0, 2, 3)

        labelG = QtWidgets.QLabel(self)
        labelG.setText("Solar Irradiation (G)")
        layout.addWidget(labelG, 12, 0)
        self.editG = QtWidgets.QLineEdit(self)
        self.editG.setMaximumWidth(75)
        layout.addWidget(self.editG, 12, 1)
        unitG = QtWidgets.QLabel(self)
        unitG.setText("W/m\N{SUPERSCRIPT TWO}")
        layout.addWidget(unitG, 12, 2)

        labelT = QtWidgets.QLabel(self)
        labelT.setText("Cell Temperature (T)")
        layout.addWidget(labelT, 13, 0)
        self.editT = QtWidgets.QLineEdit(self)
        self.editT.setMaximumWidth(75)
        layout.addWidget(self.editT, 13, 1)
        unitT = QtWidgets.QLabel(self)
        unitT.setText("\N{DEGREE SIGN}C")
        layout.addWidget(unitT, 13, 2)

        labelOptions = QtWidgets.QLabel(self)
        labelOptions.setText("Options")
        labelOptions.setStyleSheet("font-size: 16px; margin-bottom: 0.5em;"
                                    + "text-decoration: underline")
        layout.addWidget(labelOptions, 14, 0, 2, 3)

        self.chxFile = QtWidgets.QCheckBox(self)
        self.chxFile.setChecked(False)
        layout.addWidget(self.chxFile, 16, 1)
        labelSaveFile = QtWidgets.QLabel(self)
        labelSaveFile.setText("Save File?")
        layout.addWidget(labelSaveFile, 16, 0)

        labelFile = QtWidgets.QLabel(self)
        labelFile.setText("Filename")
        layout.addWidget(labelFile, 17, 0)
        self.editFilename = QtWidgets.QLineEdit(self)
        self.editFilename.setMaximumWidth(250)
        layout.addWidget(self.editFilename, 17, 1, 1 , 2)

        self.btnCreateCurve  = QtWidgets.QPushButton(self)
        self.btnCreateCurve.setText("PLOT CURVES")
        self.btnCreateCurve.clicked.connect(self.CreateCurves)
        self.btnCreateCurve.setStyleSheet("background-color: #444445;"
                                           + "color: white; font-size: 16px;" 
                                           + "font-weight: bold;")  
        layout.addWidget(self.btnCreateCurve, 18, 0, 1, 3)
        
        self.btnEmulate  = QtWidgets.QPushButton(self)
        self.btnEmulate.setText("START EMULATION")
        self.btnEmulate.setStyleSheet("background-color: #21b523; color: white;"
                                       + "font-size: 16px; font-weight: bold;")  
        self.btnEmulate.clicked.connect(self.Emulate)
        layout.addWidget(self.btnEmulate, 18, 3, 1, 12)
        self.btnEmulate.setEnabled(False)
        self.bEmulate = 0
        
        self.ivPlot = Canvas(self,"Current versus voltage (I-V Curve)","Voltage [V]"
                             ,"Current [A]")
        layout.addWidget(self.ivPlot, 1, 3, 17, 6)

        self.pvPlot = Canvas(self,"Power verus voltage (P-V Curve)","Voltage [V]"
                             ,"Power [W]")
        layout.addWidget(self.pvPlot, 1, 9, 17, 6)

        self.labelVocSim = QtWidgets.QLabel(self)
        self.labelVocSim.setText("Voc: 0.00 V")
        layout.addWidget(self.labelVocSim, 0, 4)
        self.labelVocSim.setStyleSheet("font-size: 14px;")

        self.labelIscSim = QtWidgets.QLabel(self)
        self.labelIscSim.setText("Isc: 0.00 A")
        layout.addWidget(self.labelIscSim, 0, 6)  
        self.labelIscSim.setStyleSheet("font-size: 14px;")      

        self.labelVmpSim = QtWidgets.QLabel(self)
        self.labelVmpSim.setText("Vmp: 0.00 V")
        layout.addWidget(self.labelVmpSim, 0, 8)
        self.labelVmpSim.setStyleSheet("font-size: 14px;") 

        self.labelImpSim = QtWidgets.QLabel(self)
        self.labelImpSim.setText("Imp: 0.00 A")
        layout.addWidget(self.labelImpSim, 0, 10)
        self.labelImpSim.setStyleSheet("font-size: 14px;") 

        self.labelPmpSim = QtWidgets.QLabel(self)
        self.labelPmpSim.setText("Pmp: 0.00 W")
        layout.addWidget(self.labelPmpSim, 0, 12)    
        self.labelPmpSim.setStyleSheet("font-size: 14px;")     

        widget = QWidget(self)
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.show()

        screen = app.primaryScreen()
        rect = screen.availableGeometry()
        self.setFixedWidth(int(np.trunc(rect.width() / 1.25)))
        self.setFixedHeight(int(np.trunc(rect.height() / 2)))

    """ This function captures user input and calls
        class for parameter extraction """        
    def CreateCurves(self):
        
        # Capture user input
        Isc_stc = float(self.editIsc.text())
        Voc_stc = float(self.editVoc.text())
        Vmp_stc = float(self.editVmp.text())
        Imp_stc = float(self.editImp.text())
        ki      = float(self.editKi.text())
        kv      = float(self.editKv.text())
        N_series = float(self.editNs.text())
        N_strings = float(self.editStrings.text())

        # STC Conditions
        G_stc   = 1000
        T_stc   = 25

        # Ambient Conditions
        G       = float(self.editG.text())
        T       = float(self.editT.text())

        # Call pvmodel_class to perform parameter extraction
        model = pvmodel(Isc_stc, Voc_stc, Vmp_stc, Imp_stc, ki, kv, N_series, 
                        N_strings, G_stc, T_stc, G, T)

        # Extract solution from created model
        self.curve, self.Voc, self.Isc, self.Vmp, self.Imp = model.generate_model()
        voltage = self.curve[0,:]
        current = self.curve[1,:]

        # Get the lookup table operating points
        self.opPoints = model.operating_points()

        # Plot the curves on the user interface
        self.ivPlot.plotCurve(voltage, current, "Current versus voltage (I-V Curve)", 
                              "Voltage [V]", "Current [A]", max(voltage) * 1.1,
                              max(current) * 1.1)
        self.pvPlot.plotCurve(voltage, current * voltage, "Power verus voltage (P-V Curve)", 
                              "Voltage [V]", "Power [W]", max(voltage) * 1.1, 
                              self.Vmp * self.Imp * 1.1)
        self.pvPlot.plotPoint(self.Vmp, self.Vmp * self.Imp, 'rx')
        self.ivPlot.plotPoint(self.Vmp, self.Imp, 'rx')
        self.ivPlot.plotDummyPoint()
        self.pvPlot.plotDummyPoint()

        # Allow emulation to start
        self.btnEmulate.setEnabled(True)

        # Display solution on user interface
        lblVoc = "Voc: " + str(float("{0:.2f}".format(self.Voc))) + " V"
        lblIsc= "Isc: " + str(float("{0:.2f}".format(self.Isc))) + " A"
        lblVmp = "Vmp: " + str(float("{0:.2f}".format(self.Vmp))) + " V"
        lblImp = "Imp: " + str(float("{0:.2f}".format(self.Imp))) + " A"
        lblPmp = "Pmp: " + str(float("{0:.2f}".format(self.Vmp*self.Imp))) + " W"
        self.labelVocSim.setText(lblVoc)
        self.labelIscSim.setText(lblIsc)
        self.labelVmpSim.setText(lblVmp)
        self.labelImpSim.setText(lblImp)
        self.labelPmpSim.setText(lblPmp)

    """ This function is responsible for sending the data to the controller and
        reading the operating points from the controller """
    def Emulate(self):

        # If emulation was not in process start emulation
        if self.bEmulate == 0:
            self.EmulationValues = list()
            self.bEmulate = 1
            self.btnEmulate.setText("STOP EMULATION")
            self.btnEmulate.setStyleSheet("background-color: red; color: white;"
                                          + "font-size: 16px; font-weight: bold;")  
            self.btnCreateCurve.setEnabled(False)

            try:
                # Start the emulation and set reading interval to 250ms
                self.emulation=Emulation(0.25,self)
            except:
                pass

        # If emulation was in process stop emulation
        else:
            self.bEmulate = 0
            self.btnEmulate.setText("START EMULATION")
            self.btnEmulate.setStyleSheet("background-color: #21b523; color: white;"
                                          + "font-size: 16px; font-weight: bold;")  
            self.btnCreateCurve.setEnabled(True)

            # Stop the emulation and save the data if save file option was selected
            try:
                self.emulation.stop()
                if(self.chxFile.isChecked() == True):
                    name = self.editFilename.text()
                    np.savetxt("D:/OneDrive - NORTH-WEST UNIVERSITY/BENG_2022/EERI474" 
                               + "- Final Year Project/Solution/PV Model/THIS WORKS/"
                               + "Simulations/"+name+".csv", self.EmulationValues,
                               delimiter=",")
                    name = self.editFilename.text() +"_curve"
                    np.savetxt("D:/OneDrive - NORTH-WEST UNIVERSITY/BENG_2022/EERI474"
                               + " - Final Year Project/Solution/PV Model/THIS WORKS/"
                               + "Simulations/"+name+".csv", list(map(list, zip(*self.curve))),
                               delimiter=",")
                    name = self.editFilename.text() + "_op_points"
                    np.savetxt("D:/OneDrive - NORTH-WEST UNIVERSITY/BENG_2022/EERI474"
                                + " - Final Year Project/Solution/PV Model/THIS WORKS/"
                                + "Simulations/"+name+".csv", self.opPoints, 
                                delimiter=",")
            except:
                pass

""" This class allows for the emulation process to continuously run """
class Emulation:

    """ Initialize the class """
    def __init__(self, increment,PVEMClass):
        self.EmulationValues = PVEMClass.EmulationValues
        self.bEmulate = PVEMClass.bEmulate
        self.btnEmulate = PVEMClass.btnEmulate
        self.opPoints = PVEMClass.opPoints
        self.parentWidget = PVEMClass
        self.ivPlot = PVEMClass.ivPlot
        self.pvPlot = PVEMClass.pvPlot
        self.next_t = time.time()
        self.done=False
        self.increment = increment
        self.bDataSent = 1
        self.ser = serial.Serial()
        self.ser.baudrate = 115200
        self.ser.port = 'COM3'
        self.ser.timeout = 0.2
        self.ErrorCount = 0
        self.disconnectFlag = 0
        self.emulate_count = 0
        self.voltage = 0
        self.current = 0

        # Establish serial connection with PVEM
        try:
            self.ser.open()
            self.bSerial = 1
        except:
            self.bSerial = 0
            QMessageBox.about(self.parentWidget, "Communication Error",
                              "No PVEM found")
            self.ser.close()
            self.parentWidget.Emulate()

        if(self.bSerial == 1):
            self.sendData()
            if(self.bDataSent == 1):
                self.ser.flushInput()
                self.ser.flushOutput()
                self._run()

    # Continuous loop
    def _run(self):
        self.emulateIteration()
        self.next_t += self.increment
        if not self.done:
            threading.Timer(self.next_t - time.time(), self._run).start()
    
    # Stop the loop
    def stop(self):
        if self.bSerial == 1:
            self.ser.close()
        self.done=True
        
    # For each iteration of the loop
    def emulateIteration(self):
        self.emulate_count += 1

        # Read the serial data
        try:
            bytesToRead = self.ser.inWaiting()
            float_data = self.ser.readline(bytesToRead) 
            float_data = float_data.decode('utf-8')
            float_split = float_data.split('#')
            self.voltage += float(float_split[0])
            self.current += float(float_split[1])
            self.ErrorCount = 0
            self.disconnectFlag = 0

        # If serial data cannot be read the PVEM was disconnecte
        except:
            voltage = 0
            current = 0
            self.ErrorCount += 1
            self.disconnectFlag = 1

        # Display the operating point
        if( self.emulate_count == 1):
            self.voltage = self.voltage / (self.emulate_count+1)
            self.current = self.current / (self.emulate_count+1)
            self.ivPlot.removePoint()
            self.ivPlot.plotPoint(self.voltage, self.current, 'ro')      
            self.EmulationValues.append([self.voltage, self.current])
            self.pvPlot.removePoint()
            self.pvPlot.plotPoint(self.voltage, self.voltage * self.current,
                                  'ro')
            self.emulate_count = 0

        # If something went wrong
        if(self.disconnectFlag == 1 and self.ErrorCount > 50):
            self.ser.close()
            self.parentWidget.Emulate()

    """ Send the lookup table data to the PV emulator """
    def sendData(self):
        self.opPoints = [round(val, 6) for val in self.opPoints]
        self.opPoints = np.insert(self.opPoints, len(self.opPoints), 1111)
        self.ser.write(struct.pack('%sf' % len(self.opPoints), 
                       *np.asarray(self.opPoints)))
        total = 0
        while (total < len(self.opPoints) + 1) and (self.bDataSent == 1):
            try:
                float_data = self.ser.read(4) # Refer IEEE 754
                serial_data = struct.unpack('f', float_data)
            except:
                self.bDataSent = 0

            total = total + 1
        if(self.bDataSent == 1):
            pass
        else:
            QMessageBox.about(self.parentWidget, "Communication Error", 
                              "A data transmission error occurred, please retry.")
            self.ser.close()
            self.parentWidget.Emulate()

if __name__ == '__main__': 
    app = QApplication(sys.argv)
    w = PVEM()
    sys.exit(app.exec_())