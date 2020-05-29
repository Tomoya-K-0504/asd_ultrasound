from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
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

    wav_dir = Path('../input/USV_Data')
    output_path = Path('../output') / 'stft'
    output_path.mkdir(exist_ok=True, parents=True)
    sr = 300000
    for wav_path in tqdm(sorted(list(wav_dir.iterdir()))):
        wav, _ = librosa.load(str(wav_path), sr=sr)
        f, t, Zxx = scipy.signal.stft(wave, fs=sr, nperseg=sr)
        plt.pcolormesh(t, f, np.abs(Zxx))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.savefig(output_path / 'stft' / f'{wav_path.name}.png')
