import os, sys
import datetime

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import QTimer
# from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import soundfile as sf

from mainWindow00 import Ui_MainWindow
from audio_classifier_library import audioProcess, classifierTrainer, audioClassifier


class ConsoleOutput:
    def __init__(self, target_widget, max_lines):
        self.target_widget = target_widget
        self.max_lines = max_lines
        self.line_count = 0

    def write(self, text):
        self.target_widget.insertPlainText(text)

        self.line_count += text.count('\n')

        if self.line_count > self.max_lines:
            cursor = self.target_widget.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            self.line_count -= 1

        self.target_widget.ensureCursorVisible()

    def flush(self):
        pass

class StdErrOutput:
    def __init__(self, target_widget, max_lines):
        self.target_widget = target_widget
        self.max_lines = max_lines
        self.line_count = 0

    def write(self, text):
        self.target_widget.setTextColor("red")
        self.target_widget.insertPlainText(text)

        self.line_count += text.count('\n')

        if self.line_count > self.max_lines:
            cursor = self.target_widget.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            self.line_count -= 1

        self.target_widget.ensureCursorVisible()
        self.target_widget.setTextColor("black")

    def flush(self):
        pass


class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        console_output = ConsoleOutput(self.textEditConsole, max_lines=1000)
        sys.stdout = console_output
        stderr_output = StdErrOutput(self.textEditStdError, max_lines=1000)
        sys.stderr = stderr_output

        self.pushButtonExit.clicked.connect(self.close)

        # record tab
        self.initPlotUI()
        self.AP = audioProcess(0, 'test.wav')
        self.timerRecord = QTimer(self)
        self.timerRecord.timeout.connect(self.updateTimerRecord)
        self.timerTest = QTimer(self)
        self.timerTest.timeout.connect(self.updateTimerTest)

        self.pushButtonRefreshSources.clicked.connect(self.regreshSourceList)
        self.pushButtonWorkingFolderSelect.clicked.connect(self.selectWorkingFolder)
        self.pushButtonStartRecord.clicked.connect(self.startRecording)

        # train tab
        self.initDataPlotUI()
        self.CT = classifierTrainer()
        self.pushButtonUpdateClasses.clicked.connect(self.updateClassList)
        self.listWidgetDataClasses.currentItemChanged.connect(self.updateDataFileList)
        self.listWidgetDataFiles.currentItemChanged.connect(self.updateDataGraph)
        self.listWidgetDataFiles.clicked.connect(self.updateDataGraph)
        self.pushButtonStartTrain.clicked.connect(self.startTrain)

        # test tab
        self.initTestPlotUI()
        self.AC = audioClassifier()
        self.pushButtonRefreshTestSources.clicked.connect(self.refreshTestSources)
        self.pushButtonUpdateModels.clicked.connect(self.updateModelList)
        self.pushButtonStartClassificatiion.clicked.connect(self.startClassification)


    # subs for audio record =================================================

    def initPlotUI(self):
        layout = self.horizontalLayoutRecordGraph

        # Create the Matplotlib figure and canvas
        self.figure1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.figure1)
        layout.addWidget(self.canvas1)
        self.figure2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvas(self.figure2)
        layout.addWidget(self.canvas2)

    def drawGraph(self, given_data):
        # Clear the plot
        self.ax1.clear()
        self.ax2.clear()

        # Plot waveform
        self.ax1.plot(given_data)
        self.ax1.set_title("Waveform")
        self.ax1.set_xlabel("Sample")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.set_ylim(-2,2)

        # Plot waveform
        # self.ax2.plot(given_data)
        self.specgram_plot = self.ax2.specgram(given_data.flatten(), Fs=44100)
        self.ax2.set_title("Spectrogram of Captured Audio")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Frequency (Hz)")

        # # Adjust plot layout
        # self.figure1.tight_layout()
        # self.figure2.tight_layout()

        self.canvas1.draw()
        self.canvas2.draw()

    def updateTimerRecord(self):
        self.timerRecord.stop()
        current_date_time = datetime.datetime.now()
        recording_index = current_date_time.strftime("%Y%m%d_%H%M%S")
        file_path = f'{self.lineEditWorkingFolder.text()}/{self.lineEditClass.text()}/recording_{recording_index}.wav'
        audio_data = self.AP.captureAudioByIndex(self.input_index, file_path, 
                                                 samplingRate=int(self.comboBoxSamplingRate.currentText()), 
                                                 targetDuration=float(self.lineEditTargetDuration.text()), 
                                                 inputLevel=float(self.comboBoxGain.currentText()))
        self.drawGraph(audio_data)

        self.record_repeat += 1
        if int(self.lineEditRecordRepeat.text()) == 0 or int(self.lineEditRecordRepeat.text()) > self.record_repeat:
            self.timerRecord.start(100)
        else:
            print('Record end...')
            self.pushButtonStartRecord.setText('Start record')

    def startRecording(self):
        if self.pushButtonStartRecord.text() == "Start record":
            input_item = self.listWidgetAudioSource.currentItem()
            self.input_index = self.listWidgetAudioSource.row(input_item)

            if self.input_index>=0:
                class_folder = f'{self.lineEditWorkingFolder.text()}/{self.lineEditClass.text()}'
                os.makedirs(class_folder, exist_ok=True)
                # self.timerRecord.start(float(self.lineEditTargetDuration.text())+1)
                self.record_repeat = 0
                self.timerRecord.start(100)
                self.pushButtonStartRecord.setText('Stop record')
        else:
            self.timerRecord.stop()
            self.pushButtonStartRecord.setText('Start record')

    def selectWorkingFolder(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontUseNativeDialog

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder_path:
            self.lineEditWorkingFolder.setText(folder_path)

    def regreshSourceList(self):
        self.listWidgetAudioSource.clear()
        input_device_list = self.AP.getInputDeviceList()
        # print(input_device_list[0])
        self.listWidgetAudioSource.addItems(input_device_list)


    # subs for train ======================================================

    def initDataPlotUI(self):
        layout = self.horizontalLayoutTrainGraph

        # Create the Matplotlib figure and canvas
        self.figure3, self.ax3 = plt.subplots()
        self.canvas3 = FigureCanvas(self.figure3)
        layout.addWidget(self.canvas3)
        self.figure4, self.ax4 = plt.subplots()
        self.canvas4 = FigureCanvas(self.figure4)
        layout.addWidget(self.canvas4)


    def updateClassList(self):
        def get_subfolders(directory_path):
            subfolders = sorted([f.path.split('/')[-1] for f in os.scandir(directory_path) if f.is_dir()])
            return subfolders
        
        # Specify the directory path for which you want to get the subfolders
        directory_path = self.lineEditWorkingFolder.text()
        subfolders = get_subfolders(directory_path)

        self.listWidgetDataClasses.clear()
        self.listWidgetDataClasses.addItems(subfolders)

    def updateDataFileList(self):
        self.listWidgetDataFiles.clear()
        folder_path = f'{self.lineEditWorkingFolder.text()}/{self.listWidgetDataClasses.currentItem().text()}'
        file_list = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        self.listWidgetDataFiles.addItems(file_list)

    def updateDataGraph(self):
        if len(self.listWidgetDataClasses.selectedIndexes())==0 or len(self.listWidgetDataFiles.selectedIndexes())==0:
            return

        file_path = f'{self.lineEditWorkingFolder.text()}/'
        file_path += f'{self.listWidgetDataClasses.currentItem().text()}/'
        file_path += f'{self.listWidgetDataFiles.currentItem().text()}'
        audio_data, sampling_rate = sf.read(file_path)
        # Clear the plot
        self.ax3.clear()
        self.ax4.clear()

        # Plot waveform
        self.ax3.plot(audio_data)
        self.ax3.set_title("Waveform")
        self.ax3.set_xlabel("Sample")
        self.ax3.set_ylabel("Amplitude")
        self.ax3.set_ylim(-2,2)

        # Plot waveform
        self.specgram_plot = self.ax4.specgram(audio_data.flatten(), Fs=44100)
        self.ax4.set_title("Spectrogram of Selected Audio File")
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Frequency (Hz)")

        self.canvas3.draw()
        self.canvas4.draw()

    def startTrain(self):
        data_root = self.lineEditWorkingFolder.text()
        model_path = f'{self.lineEditWorkingFolder.text()}/{self.lineEditModelFileName.text()}'
        SAMPLE_RATE = int(self.lineEditTrainSamplingRate.text())
        TARGET_DURATION = float(self.lineEditTargetDuration.text())
        BATCH_SIZE = int(self.lineEditTrainBatches.text())
        EPOCHS = int(self.lineEditTrainEpochs.text())

        myTrainer = classifierTrainer(sample_rate=SAMPLE_RATE, audio_duration=TARGET_DURATION)
        myTrainer.buildModel(data_root, model_path, EPOCHS, BATCH_SIZE)


    # subs for test ======================================================

    def initTestPlotUI(self):
        layout = self.horizontalLayoutTestGraph

        # Create the Matplotlib figure and canvas
        self.figure5, self.ax5 = plt.subplots()
        self.canvas5 = FigureCanvas(self.figure5)
        layout.addWidget(self.canvas5)
        self.figure6, self.ax6 = plt.subplots()
        self.canvas6 = FigureCanvas(self.figure6)
        layout.addWidget(self.canvas6)

    def refreshTestSources(self):
        self.listWidgetAudioTestSource.clear()
        input_device_list = self.AP.getInputDeviceList()
        # print(input_device_list[0])
        self.listWidgetAudioTestSource.addItems(input_device_list)

    def updateModelList(self):
        self.comboBoxModel.clear()
        folder_path = f'{self.lineEditWorkingFolder.text()}'
        file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".h5")])
        self.comboBoxModel.addItems(file_list)

    def startClassification(self):
        if self.pushButtonStartClassificatiion.text() == 'Start classification':
            input_item = self.listWidgetAudioTestSource.currentItem()
            self.input_index = self.listWidgetAudioTestSource.row(input_item)
            if self.input_index>=0:
                self.pushButtonStartClassificatiion.setText('End classification')
                model_path = f'{self.lineEditWorkingFolder.text()}/{self.comboBoxModel.currentText()}'
                self.AC.loadModel(model_path)

                self.listWidgetTestAllClasses.clear()
                class_list = sorted(self.AC.loadClasses(model_path))
                self.listWidgetTestAllClasses.addItems(class_list)

                self.timerTest.start(100)
        else:
            self.pushButtonStartClassificatiion.setText('Start classification')
            self.timerTest.stop()


        # try:
        #     while True:
        #         self.AP.captureAudio()
        #         detected_class = self.AC.classify_audio_clip(file_path)
        #         print(f"Detected class: {detected_class}")

        # except KeyboardInterrupt:
        #     print("\nKeyboard interrupt detected. Exiting...")

    def updateTimerTest(self):
        self.timerTest.stop()
        file_path = f'{self.lineEditWorkingFolder.text()}/temp.wav'
        audio_data = self.AP.captureAudioByIndex(self.input_index, file_path, 
                                                 samplingRate=int(self.comboBoxTestSamplingRate.currentText()), 
                                                 targetDuration=float(self.lineEditTestTargetDuration.text()), 
                                                 inputLevel=float(self.comboBoxTestGain.currentText()))
        self.drawTestGraph(audio_data)
        detected_class = self.AC.classify_audio_clip(file_path)
        self.textEditTestResults.setText(detected_class)
        self.timerTest.start(100)

    def drawTestGraph(self, given_data):
        # Clear the plot
        self.ax5.clear()
        self.ax6.clear()

        # Plot waveform
        self.ax5.plot(given_data)
        self.ax5.set_title("Waveform")
        self.ax5.set_xlabel("Sample")
        self.ax5.set_ylabel("Amplitude")
        self.ax5.set_ylim(-2,2)

        # Plot waveform
        # self.ax2.plot(given_data)
        self.specgram_plot = self.ax6.specgram(given_data.flatten(), Fs=44100)
        self.ax6.set_title("Spectrogram of Captured Audio")
        self.ax6.set_xlabel("Time (s)")
        self.ax6.set_ylabel("Frequency (Hz)")

        self.canvas5.draw()
        self.canvas6.draw()


app = QtWidgets.QApplication()
window = MainWindow()
window.show()
app.exec() 
