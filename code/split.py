from pydub import AudioSegment
from eyed3 import mp3
import os,glob
import math

DURATION = 30
flag = 0 #1--有标签 0--无标签

if flag:
    path = r"./data/multi/.../*.mp3"

    for song_path in glob.glob(path):
        song_label = os.path.basename(song_path)
        [index,emotion,song_name] = song_label.split("-")
        # if int(index) <= 22:
        #     continue
        song = AudioSegment.from_mp3(song_path)
        song_ = mp3.Mp3AudioFile(song_path)
        song_length = song_.info.time_secs
        pieces = math.floor(song_length/DURATION)
        for i in range(pieces):
            piece = song[i*DURATION*1000:(i+1)*DURATION*1000]
            piece_path = r'./data/multi/splited/'+index+'-'+str(i+1)+'-'+emotion+'-'+song_name
            piece.export(piece_path,format="mp3",bitrate="320k")
        print("%d/%d"%(int(index),len(glob.glob(path))))
else:
    path = r"./data/multi/.../*.mp3"

    for song_path in glob.glob(path):
        song_label = os.path.basename(song_path)
        song = AudioSegment.from_mp3(song_path)
        song_ = mp3.Mp3AudioFile(song_path)
        song_length = song_.info.time_secs
        pieces = math.floor(song_length/DURATION)
        for i in range(pieces):
            piece = song[i*DURATION*1000:(i+1)*DURATION*1000]
            piece_path = r'./data/multi/splited_unlabelled/'+str(i+1)+'-'+song_label
            piece.export(piece_path,format="mp3",bitrate="320k")
