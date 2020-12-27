import numpy
import metric_learn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class knn_clf():
    def __init__(self):
        # Hyper param
        self.k = 2
        self.metric = metric_learn.LSML_Supervised()
        self.metric_mtx = None

        # kNN classifier
        self.classifier = None

        #standarScaler
        self.standar = StandardScaler()
        
    def fit(self, data, label):
        #标准化
        data = self.standar.fit_transform(data)
        #度量学习
        self.metric.fit(data, label)
        self.metric_mtx = self.metric.get_mahalanobis_matrix()

        #构造、训练分类器
        self.classifier = KNeighborsClassifier(n_neighbors = self.k, weights='distance', algorithm='brute', metric='mahalanobis', metric_params={'VI' : self.metric_mtx})
        self.classifier.fit(data, label)

    def get_neibors(self, data, k):
        data = self.standar.transform(data.reshape(1,-1), copy = True)
        distance, neibors = self.classifier.kneighbors(data, n_neighbors = k)
        return neibors, distance



