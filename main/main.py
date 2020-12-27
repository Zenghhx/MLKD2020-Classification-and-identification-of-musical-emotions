import sys
from PySide2.QtWidgets import (QMainWindow, QLineEdit, QPushButton, QApplication,
    QVBoxLayout, QHBoxLayout, QDialog, QTextBrowser, QRadioButton, QWidget, QTabWidget, 
    QCheckBox, QComboBox)
import knn_classifier
import numpy
import librosa
import Extract_feature
from ssGMM import ss_GaussianMixtureModels

emotion_dict = {
    1:"伤感",
    2:"抒情",
    3:"庄严",
    4:"欢快",
    5:"激情",
    6:"安静"
}

class Form(QMainWindow):

    def __init__(self, path_label, path_unlab, path_test,   parent=None):
        super(Form, self).__init__(parent)
        self.path_label = path_label
        self.path_unlab = path_unlab
        self.path_test  = path_test
        # Create widgets
        self.setWindowTitle("Music Emotion Classifier")
        self.setMinimumSize(480, 480)
        self.widget = QWidget()
        
        layout = QVBoxLayout()

        #train
        widget1 = QWidget()
        layout1 = QHBoxLayout()
        self.button_train = QPushButton("Train the model Now!")
        self.checkButtom = QCheckBox("Use unlabeled data")
        self.text_train = QTextBrowser(self)
        self.text_train.setFixedHeight(50)
        self.text_train.setText("Haven't been trained.")
        layout1.addWidget(self.button_train)
        layout1.addWidget(self.checkButtom)
        widget1.setLayout(layout1)

        layout.addWidget(widget1)
        layout.addWidget(self.text_train)

        #tab
        self.tab = QTabWidget()
        self.subwidget1 = QWidget()
        self.subwidget2 = QWidget()

        self.tab.addTab(self.subwidget1, "Find Similar Music")
        self.tab.addTab(self.subwidget2, "Get Music with giving emotion")

        #tab1
        self.sublayout1 = QVBoxLayout()

        self.edit_music_name = QLineEdit("The music name you want to query")
        self.edit_neibors    = QLineEdit("The number of neibors you want to query")
        self.button_start_query   = QPushButton("Start Query Now!")
        self.text_query = QTextBrowser(self)

        self.sublayout1.addWidget(self.edit_music_name)
        self.sublayout1.addWidget(self.edit_neibors)
        self.sublayout1.addWidget(self.button_start_query)
        self.sublayout1.addWidget(self.text_query)

        self.subwidget1.setLayout(self.sublayout1)

        #tab2
        self.sublayout2 = QVBoxLayout()

        self.edit_emotion = QComboBox(self)
        self.edit_emotion.addItems(['伤感', '抒情', '庄严', '欢快', '激情', '安静'])
        self.edit_numbers = QLineEdit("How many pieces of music you want")
        self.buttom_start_find = QPushButton("Start Query Now!")
        self.text_find = QTextBrowser(self)

        self.sublayout2.addWidget(self.edit_emotion)
        self.sublayout2.addWidget(self.edit_numbers)
        self.sublayout2.addWidget(self.buttom_start_find)
        self.sublayout2.addWidget(self.text_find)
        
        self.subwidget2.setLayout(self.sublayout2)
        
        
        layout.addWidget(self.tab)


        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)



        # Add button signal to greetings slot
        self.button_train.clicked.connect(self.train)
        self.button_start_query.clicked.connect(self.query)
        self.buttom_start_find.clicked.connect(self.find_music)

        self.trained = False
        self.clf = None
        self.labeled_size = 0
        self.unlabel_size = 0

        self.train_data = None
        self.train_label= None
        self.labeled_name = None
        self.unlabel_name = None

    #查找输入音乐的k近邻
    def query(self):
        self.text_query.setText("querying...")
        QApplication.processEvents()
        if self.trained == False:
            self.text_query.setText("Model haven't been trained! Please train the model first.")
            return

        #读取要查几个邻居
        k = int(self.edit_neibors.text())

        #读取输入歌曲的名字
        
        name = self.edit_music_name.text()
        try:
            name = int(name)
            test_data = numpy.load(path_test + '/feature_test.npy')
            data_query = test_data[name]
            cout =  "Test Music Name: \n"
            cout += self.labeled_name[name + self.labeled_size][1][:-4] + '-%d'%self.labeled_name[name + self.labeled_size][0]
            cout += "\nMost similar music: \n" #此处应输出歌曲名，暂时先输出编号

        except:
            self.text_query.setText("Processing the music, please wait a minute...\nIt may show no response, please ignore it.")
            QApplication.processEvents()
            if name[-4:] == ".mp3":
                path = path_test + "/"+name
            else:
                name = name + '.mp3'
                path = path_test + "/"+name
            song, _ = librosa.load(path)
            data_query = Extract_feature.get_all_features(song)
            cout =  "Test Music Name: \n"
            cout += name
            cout += "\nMost similar music: \n" #此处应输出歌曲名，暂时先输出编号

        #查询k近邻
        neibors, distance = self.clf.get_neibors(data_query, k)
        neibors_label = self.train_label[neibors]

        #计算分类成分
        neibors = neibors[0]
        neibors_label = neibors_label[0]
        distance = distance[0]
        labels_dict = {}
        sum_distance = numpy.sum( numpy.exp( -distance ) )
        for idx in range(k):
            lb = neibors_label[idx]
            if lb in labels_dict.keys():
                labels_dict[lb] += numpy.exp( - distance[idx])/sum_distance
            else:
                labels_dict[lb] = numpy.exp( - distance[idx])/sum_distance

        for idx in neibors:
            if idx < self.labeled_size:
                cout += self.labeled_name[idx][1][:-4] + '-' + str(self.labeled_name[idx][0])
                cout += "\n"
            else:
                cout += self.unlabel_name[idx - self.labeled_size][1][:-4] + '-' \
                    + str(self.unlabel_name[idx - self.labeled_size][0])
                cout += "\n"

        cout += "Emotion Analysis: \n"
        for lb in labels_dict.keys():
            cout += emotion_dict[lb]
            cout += "\t"
            cout += "%f"%labels_dict[lb]
            cout += "\n"

        self.text_query.setText(cout)

    #使用有标签样本训练kNN
    def train(self):
        if self.checkButtom.isChecked():
            self.text_train.setText("Training...")
            QApplication.processEvents()
            labeled_data  = numpy.load(self.path_label + "/feature_train.npy")
            labeled_label = numpy.load(self.path_label + "/label_train.npy").astype(numpy.int)
            unlabel_data  = numpy.load(self.path_unlab + "/feature.npy")
            unlabel_label  = numpy.zeros((unlabel_data.shape[0]))

            self.labeled_size = labeled_data.shape[0]
            self.unlabel_size = unlabel_data.shape[0]

            #用GMM训练，给无标签样本做伪标签
            [unlabel_label, _, _] = ss_GaussianMixtureModels(labeled_data,labeled_label,unlabel_data,0.7,1.0,1000,True)

            self.train_data  = numpy.concatenate((labeled_data, unlabel_data), axis = 0)
            self.train_label = numpy.concatenate((labeled_label, unlabel_label), axis = 0)
            self.labeled_name = numpy.load(self.path_label + '/name.npy', allow_pickle=True).item()
            self.unlabel_name = numpy.load(self.path_unlab + '/name.npy', allow_pickle=True).item()

            #用包含伪标签的数据训练kNN，同下。
            #初始化分类器
            self.clf = knn_classifier.knn_clf()
            self.trained = True

            #训练分类器
            self.clf.fit(self.train_data, self.train_label)

            self.text_train.setText("Train Finished! All data has been used.")
            return
        self.text_train.setText("Training...")
        QApplication.processEvents()
        #读训练数据
        self.train_data = numpy.load(self.path_label + "/feature_train.npy")
        self.train_label = numpy.load(self.path_label + "/label_train.npy").astype(numpy.int)
        self.labeled_name = numpy.load(self.path_label + '/name.npy', allow_pickle=True).item()
        self.labeled_size = self.train_data.shape[0]


        #初始化分类器
        self.clf = knn_classifier.knn_clf()
        self.trained = True

        #训练分类器
        self.clf.fit(self.train_data, self.train_label)

        self.text_train.setText("Train Finished! Only labeled data has been used.")

    #查找对应情绪的音乐，随机输出数据集中的k个结果
    def find_music(self):
        self.text_find.setText("Finding...")
        QApplication.processEvents()
        if self.trained == False:
            self.text_find.setText("Model haven't been trained! Please train the model first.")
            QApplication.processEvents()
            return
        emotion = self.edit_emotion.currentText()
        k = int(self.edit_numbers.text())
        for lb in emotion_dict.keys():
            if emotion_dict[lb] == emotion:
                break
        match_list = []
        labels = []
        for idx in range(self.train_data.shape[0]):
            if self.train_label[idx] == lb:
                match_list.append(idx)
                labels.append(self.train_label[idx])
        cout = "Music found: \n"
        if k > len(match_list):
            k = len(match_list)
            cout += "Didn't find enough music. Show all of them.\n"
        match_list = numpy.random.choice(match_list, k)
        
        for idx in match_list:
            if idx < self.labeled_size:
                cout += self.labeled_name[idx][1][:-4] + '-' + str(self.labeled_name[idx][0])
                cout += "\n"
            else:
                cout += self.unlabel_name[idx - self.labeled_size][1][:-4] + '-' \
                    + str(self.unlabel_name[idx - self.labeled_size][0])
                cout += "\n"
        self.text_find.setText(cout)


if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    arg_ls = sys.argv
    path_label = "./train_labeled"
    path_unlab = "./train_unlabel"
    path_test  = "./test"
    if "-path-label" in arg_ls:
        for i in range(len(arg_ls)):
            if arg_ls[i] == "-path-label":
                break
        path_label = arg_ls[i+1]
    if "-path-unlabel" in arg_ls:
        for i in range(len(arg_ls)):
            if arg_ls[i] == "-path-unlabel":
                break
        path_unlab = arg_ls[i+1]
    if "-path-test" in arg_ls:
        for i in range(len(arg_ls)):
            if arg_ls[i] == "-path-test":
                break
        path_test = arg_ls[i+1]
        print(path_test)
    # Create and show the form
    form = Form(path_label, path_unlab, path_test)
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())