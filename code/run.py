import numpy
import classifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from GMM import ss_GaussianMixtureModels

data = numpy.load('./data/multi/numpy/feature.npy')
label = numpy.load('./data/multi/numpy/label.npy')
label = label.astype(numpy.int)-1
name = numpy.load('./data/multi/labeled/name.npy')

""" 
    # data = PCA(n_components=3).fit_transform(data_)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter3D(data[:,0], data[:,1], data[:,2],c=data[:,0])
    # plt.show()
    # train_data,train_label,test_data,test_label = data_split(data,label)
 """

train_data = []
train_label = []
test_data = []
test_label = []
rate = 0.8

index = numpy.arange(len(data))
numpy.random.shuffle(index)

data = StandardScaler().fit_transform(data)
train_data = data[index][:int(rate*len(data))]
train_label = label[index][:int(rate*len(data))]
test_data = data[index][int(rate*len(data)):]
test_label = label[index][int(rate*len(data)):]

## 有监督
# clf_dic = ['KNN1','KNN2','PA','Perceptron','SGD','Guass','GuassNB','SVC','LinSVC','NuSVC','Tree','MLP']
# for clf in clf_dic:
#     acc,matrix = classifier.classifier(train_data,train_label,test_data,test_label,clf)
#     print("Classifier "+clf+" :%f"%acc)
#     print(matrix)

##半监督
#标签传播LP
# acc,matrix = classifier.classifier(train_data,train_label,test_data,test_label,'semi-LP')
# print("Classifier "+clf+" :%f"%acc)
# print(matrix)
#局部一致性LGC
# acc,_ = classifier.LGC(train_data,train_label,test_data,test_label)
# print("Classifier LGC:%f"%acc)
#混合高斯
# acc,_ = ss_GaussianMixtureModels(train_data,train_label,test_data,test_label,0.7,1.0,1000,True)
# print("Classifier GMM:%f"%acc)
#S3VM

#MR
