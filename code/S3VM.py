from s3vm_ import S3VM_SGD
from sklearn.preprocessing import StandardScaler
import numpy as np

def extract_data(data, label, l1, l2):
    data1 = [data[i] for i in range(0, len(label)) if int(label[i]) == l1]
    data2 = [data[i] for i in range(0, len(label)) if int(label[i]) == l2]
    len1 = len(data1)
    len2 = len(data2)
    label1 = np.zeros(len1 + len2)
    label1[:len1] = 1
    label1[len1:] = -1
    data_new = np.array(data1 + data2)
    return data_new, label1


class S3VM:
    def __init__(self):
        self.s3vm16 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm15 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm14 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm13 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm12 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm26 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm36 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm46 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm56 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm23 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm34 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm35 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm24 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm25 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)
        self.s3vm45 = S3VM_SGD(knn=5, eta0=1, alpha=0.001, buffer_size=50)

    def fit(self, train_data, train_label, train_nolabel):
        data12, label12 = extract_data(train_data, train_label, 1, 2)
        self.s3vm12.fit(data12, label12, train_nolabel)
        data13, label13 = extract_data(train_data, train_label, 1, 3)
        self.s3vm13.fit(data13, label13, train_nolabel)
        data14, label14 = extract_data(train_data, train_label, 1, 4)
        self.s3vm14.fit(data14, label14, train_nolabel)
        data15, label15 = extract_data(train_data, train_label, 1, 5)
        self.s3vm15.fit(data15, label15, train_nolabel)
        data16, label16 = extract_data(train_data, train_label, 1, 6)
        self.s3vm16.fit(data16, label16, train_nolabel)
        data23, label23 = extract_data(train_data, train_label, 2, 3)
        self.s3vm23.fit(data23, label23, train_nolabel)
        data24, label24 = extract_data(train_data, train_label, 2, 4)
        self.s3vm24.fit(data24, label24, train_nolabel)
        data25, label25 = extract_data(train_data, train_label, 2, 5)
        self.s3vm25.fit(data25, label25, train_nolabel)
        data26, label26 = extract_data(train_data, train_label, 2, 6)
        self.s3vm26.fit(data26, label26, train_nolabel)
        data34, label34 = extract_data(train_data, train_label, 3, 4)
        self.s3vm34.fit(data34, label34, train_nolabel)
        data35, label35 = extract_data(train_data, train_label, 3, 5)
        self.s3vm35.fit(data35, label35, train_nolabel)
        data36, label36 = extract_data(train_data, train_label, 3, 6)
        self.s3vm36.fit(data36, label36, train_nolabel)
        data45, label45 = extract_data(train_data, train_label, 4, 5)
        self.s3vm45.fit(data45, label45, train_nolabel)
        data46, label46 = extract_data(train_data, train_label, 4, 6)
        self.s3vm46.fit(data46, label46, train_nolabel)
        data56, label56 = extract_data(train_data, train_label, 5, 6)
        self.s3vm56.fit(data56, label56, train_nolabel)

    def predict(self, test_data):
        result = np.zeros((6, 6))
        result[0, 1] = self.s3vm12.predict(test_data)[0]
        result[0, 2] = self.s3vm13.predict(test_data)[0]
        result[0, 3] = self.s3vm14.predict(test_data)[0]
        result[0, 4] = self.s3vm15.predict(test_data)[0]
        result[0, 5] = self.s3vm16.predict(test_data)[0]
        result[1, 2] = self.s3vm23.predict(test_data)[0]
        result[1, 3] = self.s3vm24.predict(test_data)[0]
        result[1, 4] = self.s3vm25.predict(test_data)[0]
        result[1, 5] = self.s3vm26.predict(test_data)[0]
        result[2, 3] = self.s3vm34.predict(test_data)[0]
        result[2, 4] = self.s3vm35.predict(test_data)[0]
        result[2, 5] = self.s3vm36.predict(test_data)[0]
        result[3, 4] = self.s3vm45.predict(test_data)[0]
        result[3, 5] = self.s3vm46.predict(test_data)[0]
        result[4, 5] = self.s3vm56.predict(test_data)[0]
        result = result - result.T
        result = np.sum(result, axis = 1)
        pred = np.argmax(result) + 1
        return pred

    def score(self, test_data, test_label):
        error_cnt = 0
        for i in range(0, test_data.shape[0]):
            pred = self.predict(np.array([test_data[i, :]]))
            if pred != int(test_label[i]):
                error_cnt+=1

        return 1 - error_cnt / len(test_label)

if __name__ == '__main__':
    data = np.load(r'./data/multi/numpy/feature.npy')
    label = np.load(r'./data/multi/numpy/label.npy')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    rate = 0.8
    index = np.arange(len(data))
    np.random.shuffle(index)
    data = StandardScaler().fit_transform(data)
    train_data = data[index][:int(rate * len(data))]
    train_label = label[index][:int(rate * len(data))]
    test_data = data[index][int(rate * len(data)):]
    test_label = label[index][int(rate * len(data)):]

    s3vm = S3VM()
    s3vm.fit(train_data, train_label, test_data)
    acc = s3vm.score(test_data, test_label)
    print(acc)
