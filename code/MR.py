import numpy
from sklearn.preprocessing import StandardScaler

class _ManifoldRegularization_binary:
    def __init__(self, Adjacency_method = "kNN", 
        Meature = lambda dataX, dataY:numpy.linalg.norm(dataX - dataY, axis = -1),
        kernel  = lambda dataX, dataY:numpy.linalg.norm(dataX - dataY, axis = -1)**2, 
        **param):
        self.stander = StandardScaler()
        self.Adjacency_method = Adjacency_method
        self.param  = param
        self.alphastar = None
        self.K = None
        self.Meature = Meature
        self.kernel = kernel
        if Adjacency_method == 'kNN':
            self.k = 7
            self.GammaA = 0.7
            self.GammaI = 0.3
            if 'k' in param:
                self.k = param['k']
            if 'GammaA' in param:
                self.GammaA = param['GammaA']
            if 'GammaI' in param:
                self.GammaI = param['GammaI']


    def Adjacency_Graph(self, data:numpy.ndarray, method = "kNN" ):
        if method == "kNN":
            Adjacency_Graph = numpy.zeros(( data.shape[0], data.shape[0] ))
            for i in range( data.shape[0] ):
                Adjacency_Graph[i] = self.Meature(data[i], data)
                k_near_idx = numpy.argpartition(Adjacency_Graph[i], range(self.k))[:self.k]
                k_far_idx  = numpy.argpartition(Adjacency_Graph[i], range(self.k))[self.k:]
                Adjacency_Graph[i,k_near_idx] = numpy.exp(-Adjacency_Graph[i,k_near_idx]**2 / 32)
                Adjacency_Graph[i,k_far_idx]  = 0
                Adjacency_Graph[i, i] = 0
        return Adjacency_Graph

    # def kernel(self, dataX, dataY ):
    #     return numpy.linalg.norm(dataX - dataY, axis = -1)**2
    #     # return numpy.exp( -numpy.linalg.norm(dataX - dataY, axis = -1)**2/32 )

    def Gram_Matrix(self, data):
        K = numpy.zeros( (data.shape[0], data.shape[0]) )
        for i in range( data.shape[0] ):
            K[i] = self.kernel(data[i],data)
        return K

    def Laplasian_Graph(self, Adjacency_Graph):
        D = numpy.diag( numpy.sum(Adjacency_Graph,axis=1) )
        Laplasian = D - Adjacency_Graph
        return Laplasian

    def fit(self, data, label):
        J = numpy.diag( numpy.append( numpy.ones((l)), numpy.zeros((u),dtype=numpy.float64) ) )
        self.K = self.Gram_Matrix(data)
        W = self.Adjacency_Graph(data, method = self.Adjacency_method)
        L = self.Laplasian_Graph(W)

        self.alphastar = numpy.dot( J, self.K ) + self.GammaA*l*numpy.identity(l+u) + self.GammaI*l/(u+l)/(u+l)*numpy.dot(L,self.K)
        self.alphastar = numpy.dot( numpy.linalg.pinv(self.alphastar), label )

    def predict(self, data):
        pred = numpy.zeros((data.shape[0]))
        for  i in range(l+u):
            pred[i] = numpy.sign(numpy.dot( self.alphastar.T, self.K[:,i] ))
        return pred

"""class ManifoldRegularization:
    def __init__(self, Adjacency_method = "kNN", 
        Meature = lambda dataX, dataY:numpy.linalg.norm(dataX - dataY, axis = -1),
        kernel  = lambda dataX, dataY:numpy.linalg.norm(dataX - dataY, axis = -1)**2, 
        param):
        self.stander = StandardScaler()
        self.Adjacency_method = Adjacency_method
        self.param  = param
        self.Meature = Meature
        self.kernel = kernel
        self.binary_classifier = []
        self.labels = []
    def fit(self, data, label):
        data = self.stander.fit_transform(data)
        self.labels = numpy.unique(label)
        if len(labels) > 3:
            binary_0 = ManifoldRegularization(
                Adjacency_method = self.Adjacency_method,
                Meature = self.Meature,
                kernel = self.kernel,
                self.param
            )
            split_num = numpy.median(self.labels)
            label_0 = numpy.copy(label)
            label_0[label>=split] = 1
            label_0[label<split and label!=0] = -1
            binary_0.fit(data, label_0)
            self.binary_classifier.append(binary_0)

            binary_1 = ManifoldRegularization(
                Adjacency_method = self.Adjacency_method,
                Meature = self.Meature,
                kernel = self.kernel,
                self.param
            )
            data1  = data[label>=split or label==0]
            label1 = label[label>=split or label==0]
            binary_1.fit(data1, label1)
            self.binary_classifier.append(binary_1)

            if len(labels) > 4:
                binary_2 = ManifoldRegularization(
                    Adjacency_method = self.Adjacency_method,
                    Meature = self.Meature,
                    kernel = self.kernel,
                    self.param
                )
                data2  = data[label<split or label==0]
                label2 = label[label<split or label==0]
                binary_2.fit(data2, label2)
                self.binary_classifier.append(binary_2)
        if len(label) ==3:
            binary = ManifoldRegularization(
                Adjacency_method = self.Adjacency_method,
                Meature = self.Meature,
                kernel = self.kernel,
                self.param
            )
            label0 = numpy.copy(label)
            label0[label]
            binary_0.fit(data1, label1)
            self.binary_classifier.append(binary_1)
"""
class ManifoldRegularization:
    def __init__(self, Adjacency_method = "kNN", 
        Meature = lambda dataX, dataY:numpy.linalg.norm(dataX - dataY, axis = -1),
        kernel  = lambda dataX, dataY:numpy.linalg.norm(dataX - dataY, axis = -1)**2, 
        **param):
        self.stander = StandardScaler()
        self.Adjacency_method = Adjacency_method
        self.param  = param
        self.Meature = Meature
        self.kernel = kernel
        self.binary_classifier = []
        self.labels = []
    def fit(self, data, label):
        data = self.stander.fit_transform(data)
        self.labels  =numpy.unique(label)
        for i in range(0, len(self.labels) - 2 ):
            self.binary_classifier.append(
                _ManifoldRegularization_binary(
                    Adjacency_method = self.Adjacency_method,
                    Meature = self.Meature,
                    kernel = self.kernel,
                    k = self.param['k'],
                    GammaA = self.param['GammaA'],
                    GammaI = self.param['GammaI']
                )
            )
            label_i = numpy.copy(label)
            label_i[label>=i+2] = 1
            label_i[label==0] == 100
            label_i[label<i+2] = -1
            label_i[label==100] = 0
            self.binary_classifier[i].fit(data, label_i)
    def predict(self, data, label = None):
        data = self.stander.transform(data)
        pred = numpy.zeros( ( len(self.labels) - 2, data.shape[0]) )
        for i in range(0, len(self.labels) - 2 ):
            pred[i,:] = self.binary_classifier[i].predict(data)
        pred = pred
        pred[pred == -1] = 0
        pred[pred == 1 ] = 1
        pred = numpy.sum(pred, axis=0)
        pred = pred + 1
        if type(label) == type(None):
            return pred
        acc = numpy.sum(pred == label)/label.shape[0]
        return pred, acc

if __name__ == "__main__":
    data  = numpy.load('dataset/feature.npy')
    stander = StandardScaler()
    data  = stander.fit_transform(data)
    label = numpy.load('dataset/label.npy').astype(numpy.float64)

    # label[label<=3] = 1
    # label[label>3]  = 2

    idx = numpy.random.permutation(data.shape[0]) 
    data = data[idx]
    label = label[idx] 
    """data_size = 1000
        data = numpy.random.normal(size=(2*data_size,5))
        data[:data_size] = data[:data_size]/numpy.sqrt(numpy.sum(data[:data_size]**2,axis=1)).reshape(data_size,1) \
            * numpy.random.normal(10,0.5,size=(data_size,1))
        data[data_size:] = data[data_size:]/numpy.sqrt(numpy.sum(data[data_size:]**2,axis=1)).reshape(data_size,1) \
            * numpy.random.normal(7,0.5,size=(data_size,1))
        label = numpy.append( numpy.ones((data_size)), 2*numpy.ones((data_size)) )

        idx = numpy.random.permutation(2*data_size) 
        data = data[idx]
        label = label[idx] 
        """


    l = int(numpy.floor(data.shape[0] * 0.8))
    u = data.shape[0] - l

    Y = numpy.zeros_like(label)
    Y[u:] = label[u:]
    
    rbf = lambda dataX, dataY:numpy.exp( -numpy.linalg.norm(dataX - dataY, axis = -1)**2/30 )
    laplace = lambda dataX, dataY:numpy.exp( -numpy.linalg.norm(dataX - dataY, axis = -1)/4 )
    linear = lambda dataX, dataY:numpy.dot( dataX.reshape(1,-1), dataY.T )**1
    poly  = lambda dataX, dataY:0.1 * numpy.dot( dataX.reshape(1,-1), dataY.T )**2
    sigmoid = lambda dataX, dataY: numpy.tanh( numpy.dot( dataX.reshape(1,-1), dataY.T ) * 10 + 0 )

    classifier = ManifoldRegularization(Adjacency_method = "kNN", 
        Meature = rbf,
        kernel = poly,
        k = 800, GammaA = 0.8, GammaI = 0.2)

    classifier.fit(data, Y)

    pred = classifier.predict(data)

    acc = numpy.sum( pred[:u]==label[:u] )/(u)
    acc_s = numpy.sum( pred[u:]==label[u:] )/(l)
    # W = Construct_Adjacency_Graph(data, k = 7)


    # print( Meature(data,data[1]) )
    pass
