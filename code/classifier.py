from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import OneHotEncoder

def classifier(train_data, train_label, test_data, test_label, classifiertype,k=3):
    clf_dic = {
        'PA': PassiveAggressiveClassifier(max_iter=1000,random_state=True),
        'Perceptron': Perceptron(max_iter=1000,eta0=0.1,random_state=0),
        'SGD': SGDClassifier(max_iter=1000),
        'KNN1': KNeighborsClassifier(n_neighbors=k,weights='distance'),
        'KNN2': KNeighborsClassifier(n_neighbors=k),
        'Guass': GaussianProcessClassifier(),
        'GuassNB': GaussianNB(),
        'SVC':SVC(),
        'LinSVC':LinearSVC(),
        'NuSVC': NuSVC(),
        'Tree': DecisionTreeClassifier(),
        'MLP': MLPClassifier(max_iter=500),
        'semi-LP':LabelPropagation(kernel='knn',max_iter=100000)
    }

    if classifiertype not in clf_dic.keys():
        print("No such type of classifier!")
        return None,None
    if classifiertype[:5]=="semi-":
        rng = np.random.RandomState(int(time.time()))
        random_unlabeled_points = rng.rand(len(train_label)) < 0.3
        train_label = np.copy(train_label)
        train_label[random_unlabeled_points] = -1
    clf = clf_dic[classifiertype]
    clf.fit(train_data,train_label)
    acc = clf.score(test_data,test_label)
    matrix = confusion_matrix(test_label,clf.predict(test_data))
    return acc,matrix

def LGC(train_data, train_label, test_data, test_label, max_iter=100000,alpha=0.99,sigma=0.1):
    label_in = np.concatenate((OneHotEncoder(sparse=False).fit_transform(train_label.reshape(-1,1)),np.zeros((len(test_label),6))))

    data_in = np.concatenate((train_data,test_data))

    # dis_matrix = cdist(data_in,data_in,'chebyshev')#mahalanobis
    # rbf = lambda x,sigma:np.exp((-x)/(2*(np.power(sigma,2))))
    # vfunc = np.vectorize(rbf)
    # W = vfunc(dis_matrix/100,sigma)
    # np.fill_diagonal(W,0)

    W = Construct_Adjacency_Graph(data_in,k=3)

    sum_lines = np.sum(W,axis=1)
    D = np.diag(sum_lines)

    D = fractional_matrix_power(np.linalg.inv(D),0.5)
    S = np.dot(np.dot(D,W),D)

    F = np.dot(S,label_in)*alpha + (1-alpha)*label_in
    for epoch in range(max_iter):
        F = np.dot(S,F)*alpha + (1-alpha)*label_in

    label_result = np.zeros_like(F)
    label_result[np.arange(len(F)),F.argmax(axis=1)]=1
    label_result = label_result * np.arange(6)
    label_result = np.sum(label_result,axis=1)

    acc = np.sum((label_result[len(train_label):]==test_label))/len(test_label)
    return acc,None

def Construct_Adjacency_Graph(data:np.ndarray, method = "kNN", **param):
    if method == "kNN":
        if len(param) == 0:
            k = 7
        else:
            k = param['k'] + 1
        Adjacency_Graph = np.zeros(( data.shape[0], data.shape[0] ))
        for i in range( data.shape[0] ):
            Adjacency_Graph[i] = np.linalg.norm(data[i]-data,axis=-1)
            k_near_idx = np.argpartition(Adjacency_Graph[i], range(k))[:k]
            k_far_idx  = np.argpartition(Adjacency_Graph[i], range(k))[k:]
            Adjacency_Graph[i,k_near_idx] = np.exp(-Adjacency_Graph[i,k_near_idx]**2 / 100)
            Adjacency_Graph[i,k_far_idx]  = 0
            Adjacency_Graph[i, i] = 0
        return Adjacency_Graph