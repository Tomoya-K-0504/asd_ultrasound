from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    wav_dir = Path('../input/USV_Data')
    processed_dir = Path('../input/processed')
    output_path = Path('../output') / 'stft_maxpool'
    output_path.mkdir(exist_ok=True, parents=True)
    processed_dir.mkdir(exist_ok=True, parents=True)

    sr = 300000
    n_fft = sr // 100
    win_length = sr // 100
    hop_length = sr // 100

    for wav_path in tqdm(sorted(list(wav_dir.iterdir()))):
        wav, _ = librosa.load(str(wav_path), sr=sr)
        y = wav.astype(float)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        spect, phase = librosa.magphase(D)
        spect = librosa.power_to_db(spect, ref=np.max)
        print(spect.shape)

        np.save(processed_dir / f'{wav_path.name}.npy', np.abs(spect))
        # plt.pcolormesh(list(range(spect.shape[1])), list(range(spect.shape[1])), np.abs(spect))
        # plt.title('STFT Magnitude')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        plt.imshow(spect)
        plt.savefig(output_path / f'{wav_path.name}.png')

        exit()