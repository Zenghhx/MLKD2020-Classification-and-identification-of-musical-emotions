import os,glob
import librosa
import numpy
import Extract_feature

print("----loading data----")


flag = 0 #1--有标签 0--无标签

if flag:
    path = r"./data/multi/splited/*.mp3"

    # data = []
    feature_train = []
    feature_test = []
    label_train = []
    label_test = []
    name = {}
    i = 0
    rate = 0.8
    connector = '-'
    file_path = glob.glob(path)
    numpy.random.shuffle(file_path)
    for song_path in file_path:
        song_name = os.path.basename(song_path)
        song, _ = librosa.load(song_path)
        if i < int(rate*len(file_path)):
            label_train.append(song_name.split("-")[2])
            # data.append(song)
            feature_train.append(Extract_feature.get_all_features(song))
            name[i] = [int(song_name.split("-")[1]),connector.join(song_name.split("-")[3:])]
        else:
            label_test.append(song_name.split("-")[2])
            # data.append(song)
            feature_test.append(Extract_feature.get_all_features(song))
            name[i] = [int(song_name.split("-")[1]),connector.join(song_name.split("-")[3:])]
        i+=1
        if i % 20==0:
            print('load_process = %d'%i)

    # data = numpy.array(data)
    label_train = numpy.array(label_train)
    feature_train = numpy.array(feature_train)
    label_test = numpy.array(label_test)
    feature_test = numpy.array(feature_test)

    # numpy.save('./data/multi/labeled/data.npy', data)
    numpy.save('./data/multi/labeled/label_train.npy', label_train)
    numpy.save('./data/multi/labeled/label_test.npy', label_test)
    numpy.save('./data/multi/labeled/feature_train.npy', feature_train)
    numpy.save('./data/multi/labeled/feature_test.npy', feature_test)
    numpy.save('./data/multi/labeled/name.npy',name)
else:
    path = r"./data/multi/splited_unlabelled/*.mp3"

    feature_unlabel = []
    name = {}
    i=0
    connector = '-'

    file_path = glob.glob(path)
    for song_path in file_path:
        song_name = os.path.basename(song_path)
        song,_ = librosa.load(song_path)
        feature_unlabel.append(Extract_feature.get_all_features(song))
        name[i]=[int(song_name.split("-")[0]),connector.join(song_name.split("-")[1:])]
        i+=1

    feature_unlabel = numpy.array(feature_unlabel)

    numpy.save('./data/multi/unlabelled/feature_unlabel.npy', feature_unlabel)
    numpy.save('./data/multi/unlabelled/name_unlabel.npy',name)

