from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


def cut_pad_wave(wave, sr, one_audio_sec):
    const_length = sr * one_audio_sec
    if wave.shape[0] > const_length:
        wave = wave[:const_length]
    elif wave.shape[0] < const_length:
        n_pad = (const_length - wave.shape[0]) // 2 + 1
        wave = np.pad(wave[:const_length], n_pad)[:const_length]
    assert wave.reshape((1, -1)).shape[0] == const_length
    return wave.reshape((1, -1))


def preprocess(wav_dir: str):
    sr = 300000
    audio_length = 300
    length = 10
    overlap = 5

    len_sections = (audio_length - length) // overlap + 1
    index_list = list(range(1, len(df) * len_sections + 1))
    np.random.shuffle(index_list)
    count = 0

    wav_out_dir = Path(wav_dir).parent / 'split_wav'
    wav_out_dir.mkdir(exist_ok=True)

    wav_list = Path(wav_dir).iterdir()
    for wav_path in tqdm(wav_list):
        wav, _ = librosa.load(str(wav_path), sr=sr)
        wav = cut_pad_wave(wav)
        wav_split_list = split_audio(wav)
        for wav_split in wav_split_list:
            file_name = f'sth.wav'
            librosa.output.write_wav(str(wav_out_dir / file_name), wav_split, sr)

        break

    return split_wav_path_list


if __name__ == '__main__':
    preprocess()