import librosa
import librosa.display
import numpy as np
import os
from scipy.fftpack import fft
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def get_fft(data):
    wlen = 1024  # 每帧信号的帧数
    inc = 256  # 帧移
    data_fft = abs(librosa.stft(data, n_fft=1024, hop_length=inc, win_length=wlen))
    return data_fft


def framing(signal):
    signal_len = len(signal)
    wlen = 1024  # 每帧信号的帧数
    inc = 256  # 帧移
    nf = int(np.ceil((1.0 * signal_len - wlen + inc) / inc))  # 计算帧数nf
    pad_len = int((nf - 1) * inc + wlen)
    zeros = np.zeros(pad_len - signal_len)  # 不够的点用0填补
    pad_signal = np.concatenate((signal, zeros))
    indices = np.tile(np.arange(0, wlen), (nf, 1))+np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]  # 最终得到帧数*帧长形式的信号数据
    return frames


def get_spectual_variability(data_fft):   #频谱变化度,就是频谱求方差,输入的data是一个音乐片段
    spectual_variability = np.std(data_fft, axis=1, ddof=1)
    return np.mean(spectual_variability), np.std(spectual_variability, ddof=1)


def get_peak(fft_data):   #找到频谱上的峰值,输入为音乐片段分帧并进行傅里叶变换后得到的矩阵
    peak = fft_data.argmax(axis=1)
    return np.mean(peak), np.std(peak, ddof=1)


def get_RMS(fft_data):   #获得短时能量均方值(?)
    RMS = np.std(fft_data, axis=1, ddof=1)
    return np.mean(RMS), np.std(RMS, ddof=1)


def get_fraction_of_low_energy(fft_data):   #获得以某帧为中心100帧的低能量帧比例
    RMS = np.std(fft_data, axis=1, ddof=1)
    length = RMS.shape[0]
    fole = []
    for i in range(50, length-50):
        d = RMS[i-50:i+50]
        energy_mean = np.mean(d)
        fole.append(np.sum(d<energy_mean))
    return np.mean(fole), np.std(fole, ddof=1)


def get_zero_crossing(data):   #获得短时过零率,输入为一段音频
    length = data.shape[1]
    zero_crossing = np.sum(librosa.zero_crossings(data, pad=False),axis=1)/length
    return np.mean(zero_crossing), np.std(zero_crossing, ddof=1)


def get_spectral_centroid(data):   #输入是一段音频信号
    index = np.array(range(len(data[0])))
    m = data * index
    spectral_centroid = np.sum(m, axis=1) / (np.sum(data, axis=1)+(np.sum(data, axis=1)==0))
    return np.mean(spectral_centroid), np.std(spectral_centroid, ddof=1)


def get_MFCC(data): #获得Mel倒谱系数
    mfccs = librosa.feature.mfcc(data, n_mfcc=13)
    return np.mean(mfccs, 1), np.std(mfccs, axis=1, ddof=1)


def get_LPC(frames):
    row = frames.shape[0]
    LPC = []
    for i in range(0, row):
        if frames[i,:].any() == 0:
            continue
        plp = librosa.lpc(frames[i,:], 10)
        LPC.append(plp[1:])
    LPC = np.array(LPC)
    return np.mean(LPC, 0), np.std(LPC, axis=0, ddof=1)


def get_flux(energy):
    row = energy.shape[0]
    flux = []
    energy_norm = normalize(energy,axis=1)
    for i in range(0, row-1):
        flux1 = 0
        flux1 += np.sum((energy_norm[i] - energy_norm[i + 1]) ** 2)
        flux.append(flux1)
    flux = np.array(flux)
    return np.mean(flux, 0), np.std(flux, axis=0, ddof=1)


def get_attenuation_point(energy):
    row = energy.shape[0]
    attenuation_point = []
    for j in range(0, row):
        energy_sum = np.sum(energy[j, :])
        tmp = 0
        for i in range(len(energy[j, :])):
            tmp += energy[j, i]
            if tmp >= 0.85 * energy_sum:
                attenuation_point.append(i)
                break
    attenuation_point = np.array(attenuation_point)
    return np.mean(attenuation_point, 0), np.std(attenuation_point, axis=0, ddof=1)


def get_compactness(fft_data):
    row = fft_data.shape[0]
    compactness1 = []
    for j in range(0, row):
        compactness = 0
        for i in range(1, len(fft_data[j, :]) - 1):
            if fft_data[j,i+1] == 0:
                i = i + 3
                continue
            compactness += 20 * abs(np.log10(fft_data[j, i]) - (
                        np.log10(fft_data[j, i - 1]) + np.log10(fft_data[j, i]) + np.log10(fft_data[j, i + 1]) / 3))
        compactness1.append(compactness)
    return np.mean(compactness1, 0), np.std(compactness1, axis=0, ddof=1)

def get_beat_related_features(data):
    RMS = []
    index = 0
    while index + 512 < len(data):
        tmp = data[index:index + 512]
        RMS.append(np.sqrt(np.sum(tmp ** 2) / 512))
        index = index + 512
    # plt.plot(RMS)
    # plt.show()
    RMS_fft = abs(fft(RMS)[1:]) ** 2
    # plt.plot(RMS_fft[1:])
    # plt.show()
    ftr = []
    ftr.append(np.sum(RMS_fft))   #节拍强度和
    ftr.append(np.argmax(RMS_fft))   #最强节拍
    ftr.append(np.max(RMS_fft) / np.sum(RMS_fft))   #最强节拍的强度

    return ftr

def get_all_features(data):
    feature = []
    frames = framing(data)

    fft_data = abs(fft(frames))[:, :513]
    energy = fft_data ** 2

    feature = feature + list(get_spectual_variability(fft_data))#2
    
    feature = feature + list(get_peak(fft_data))#4

    feature = feature + list(get_RMS(fft_data))#6
    
    feature = feature + list(get_fraction_of_low_energy(fft_data))#8

    feature = feature + list(get_zero_crossing(frames))#10

    feature = feature + list(get_spectral_centroid(fft_data))#12

    feature1 = get_MFCC(data)
    feature = feature + list(feature1[0])#25
    feature = feature + list(feature1[1])#38

    feature1 = get_LPC(frames)
    feature = feature + list(feature1[0])#48
    feature = feature + list(feature1[1])#58

    feature = feature + list(get_flux(energy))#60

    feature = feature + list(get_attenuation_point(energy))#62

    feature = feature + list(get_compactness(fft_data))#64

    feature = feature + get_beat_related_features(data)#67

    feature = np.array(feature)
    return feature