from PyQt5.QtCore import (Qt)
from PyQt5.QtGui import (QColor, QPalette, QStandardItemModel, QStandardItem)
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGraphicsView, QGroupBox, QSizePolicy,
                               QPushButton, QComboBox, QCheckBox, QLabel, QSpinBox, QLineEdit
                              )
from mypackage.table_parts import Model, View

class UIMainWindow():
    def __init__(self, parent):
        self.imageMenu(parent)
        self.imageProcessingMenu(parent)
        self.drawImageMenu(parent)
        self.imageListMenu(parent)
        self.labelMenu(parent)
        self.landmarkMenu(parent)
        self.labelDataMenu(parent)

    def imageMenu(self, parent):
        btn_open = QPushButton("Open File")
        btn_open.clicked.connect(parent.openFile)
        self.file_edit = QLineEdit()
        self.cb_batchProcessing = QCheckBox()
        self.cb_batchProcessing.stateChanged.connect(parent.checkBatchBox)
        self.label_imgWidth = QLabel("(width)")
        self.label_imgHeihgt = QLabel("(height)")
        self.sb_maxImgWidth = QSpinBox()
        self.sb_maxImgWidth.setMaximum(3840)
        self.sb_maxImgWidth.setValue(1080)
        self.sb_maxImgHeihgt = QSpinBox()
        self.sb_maxImgHeihgt.setMaximum(2160)
        self.sb_maxImgHeihgt.setValue(720)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(btn_open)
        hbox1.addWidget(self.file_edit)
        hbox1.addWidget(QLabel("List:"))
        hbox1.addWidget(self.cb_batchProcessing)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Image Size:"))
        hbox2.addWidget(self.label_imgWidth)
        hbox2.addWidget(QLabel("x"))
        hbox2.addWidget(self.label_imgHeihgt)
        hbox2.addStretch(0)
        hbox2.addWidget(QLabel("Max Window:"))
        hbox2.addWidget(self.sb_maxImgWidth)
        hbox2.addWidget(QLabel("x"))
        hbox2.addWidget(self.sb_maxImgHeihgt)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.grbox_imageMenu = QGroupBox("Image Menu")
        self.grbox_imageMenu.setLayout(vbox)

    def imageProcessingMenu(self, parent):
        btn_ipRun = QPushButton("Run")
        btn_ipRun.clicked.connect(parent.selectImageProcessing)
        self.cmb_IP = QComboBox()
        IPs = ["Grayscale", "Flip(Horizon)", "Translation","Sobel(X)","Sobel(Y)", "Laplacian", "Canny"]
        for IP in IPs:
            self.cmb_IP.addItem(IP)
        self.cb_keepImage = QCheckBox("Keep")
        self.cb_keepImage.stateChanged.connect(parent.keepImage)
        btn_saveImage = QPushButton("Save")
        btn_saveImage.clicked.connect(parent.saveImage)
        btn_initImage = QPushButton("Clear")
        btn_initImage.clicked.connect(parent.clearItems)
        hbox = QHBoxLayout()
        hbox.addWidget(self.cmb_IP)
        hbox.addWidget(btn_ipRun)
        hbox.addWidget(self.cb_keepImage)
        hbox.addWidget(btn_saveImage)
        hbox.addWidget(btn_initImage)
        hbox.addStretch(0)
        self.grbox_IP = QGroupBox("Image Processing")
        self.grbox_IP.setLayout(hbox)

    def drawImageMenu(self, parent):
        self.picView = QGraphicsView()
        self.picView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.picView.customContextMenuRequested.connect(parent.contextMenue)
        self.picView.installEventFilter(parent)

    def imageListMenu(self, parent):
        self.imagelistView = View(parent)
        self.imagelistModel = QStandardItemModel(parent)
        self.imagelistModel.setHorizontalHeaderItem(0, QStandardItem('FilePath'))
        self.imagelistView.setModel(self.imagelistModel)
        self.imagelistView.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.imagelistView.clicked.connect(parent.setFilenameFromList)

    def labelMenu(self, parent):
        self.cmb_ID = QComboBox()
        values = ["0:None","1:RightEye","2:LeftEye", "3:Nose"]
        for value in values:
            self.cmb_ID.addItem(value)
        self.sb_radius = QSpinBox()
        self.sb_radius.setValue(5)
        btn_color = QPushButton("Color")
        btn_color.clicked.connect(parent.selectColor)
        self.label_color = QLabel("●")
        pal = self.label_color.palette()
        pal.setColor(QPalette.Foreground, QColor("lime"))
        self.label_color.setPalette(pal)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("ID:"))
        hbox1.addWidget(self.cmb_ID)
        hbox1.addStretch(0)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Point:"))
        hbox2.addWidget(self.label_color)
        hbox2.addWidget(btn_color)
        hbox2.addWidget(QLabel("Size"))
        hbox2.addWidget(self.sb_radius)
        hbox2.addStretch(0)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.grbox_labelMenu = QGroupBox("Label Menu")
        self.grbox_labelMenu.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.grbox_labelMenu.setLayout(vbox)

    def landmarkMenu(self, parent):
        self.sb_inputWidth = QSpinBox()
        self.sb_inputWidth.setMaximum(1080)
        self.sb_inputWidth.setValue(90)
        # self.sb_inputWidth.setEnabled(False)
        self.sb_inputHeight = QSpinBox()
        self.sb_inputHeight.setMaximum(720)
        self.sb_inputHeight.setValue(100)
        # self.sb_inputHeight.setEnabled(False)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Input Size:"))
        hbox1.addWidget(self.sb_inputWidth)
        hbox1.addWidget(QLabel("x"))
        hbox1.addWidget(self.sb_inputHeight)
        hbox1.addStretch(0)
        btn_modelPath = QPushButton("Model Path")
        btn_modelPath.clicked.connect(parent.selectModelPath)
        self.le_modelPath = QLineEdit()
        self.le_modelPath.setText("model/mdl_ep1000")
        hbox2 = QHBoxLayout()
        hbox2.addWidget(btn_modelPath)
        hbox2.addWidget(self.le_modelPath)
        btn_landmark = QPushButton("Image Run")
        btn_landmark.clicked.connect(parent.detectLandmarks)
        btn_landmark_batch = QPushButton("Batch Run")
        btn_landmark_batch.clicked.connect(parent.startBatchProcess)

        self.le_csvFileName = QLineEdit("default.csv")
        hbox = QHBoxLayout()
        hbox.addWidget(btn_landmark)
        hbox.addWidget(btn_landmark_batch)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox)
        vbox.addWidget(self.le_csvFileName)
        self.grbox_landmark = QGroupBox("Landmark Detection")
        self.grbox_landmark.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.grbox_landmark.setLayout(vbox)

    def labelDataMenu(self, parent):
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(parent.loadFile)
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(parent.saveFile)
        self.tableView = View(parent)
        self.model = Model(parent)
        self.model.setHeaders(['No.', 'ID', 'POS(X)', 'POS(Y)'])
        self.tableView.setModel(self.model)
        self.tableView.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.tableView.setMaximumWidth(215)
        self.tableView.setColumnWidth(0, 20) # have to load the tableview before setting size
        self.tableView.setColumnWidth(1, 20)
        self.tableView.setColumnWidth(2, 50)
        self.tableView.setColumnWidth(3, 50)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(btn_load)
        hbox1.addWidget(btn_save)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addWidget(self.tableView)
        self.grbox_labelDataMenu = QGroupBox("Label Data")
        self.grbox_labelDataMenu.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.grbox_labelDataMenu.setLayout(vbox)
