from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm_notebook as tqdm
import scipy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sr = 300000
    wave = np.sin(np.linspace(0, 2 * np.pi, 90000000))
    f, t, Zxx = scipy.signal.stft(wave, fs=sr, nperseg=sr)
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    # wave.plot()
    plt.savefig('sth.png')

    wav_dir = '/media/tomoya/SSD-PGU3/research/asd/USV_Data'
    sr = 300000
    for wav_path in tqdm(sorted(list(Path(wav_dir).iterdir()))):
        wav, _ = librosa.load(str(wav_path), sr=sr)
        f, t, Zxx = scipy.signal.stft(wave, fs=sr, nperseg=sr)
        plt.pcolormesh(t, f, np.abs(Zxx))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        # wave.plot()
        plt.savefig('sth.png')
