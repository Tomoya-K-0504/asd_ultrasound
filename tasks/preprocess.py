import argparse
import itertools
import json
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def expt_args(parser):
    prep_parser = parser.add_argument_group("ASD ultrasound")
    prep_parser.add_argument('--pool-size', default=5, type=int)
    prep_parser.add_argument('--win-length', default=300, type=int)
    prep_parser.add_argument('--win-stride', default=150, type=int)

    return parser


def preprocess(expt_conf):
    """
    Spectrogram (300 x 3000) -> Split -> Spectrogram (300 x 200) x 18
    :return:
    """
    input_dir = Path(__file__).parent.resolve() / 'input'
    output_dir = Path(__file__).parent.resolve() / 'input' / 'split'
    output_dir.mkdir(exist_ok=True)
    label_df = pd.read_excel(input_dir / 'P08_12_USV_15qDup_MouseInfo.xlsx')
    label_kind = {'pat': 1, 'WT': 0}
    subj2label = dict(zip(label_df['MouseID'], label_df['Genotype']))

    spect_list = sorted((input_dir / 'processed').iterdir())

    for spect_path in tqdm(spect_list):
        spect = np.load(spect_path)
        label = subj2label[spect_path.name.split('_')[0]]

        n_frames = (spect.shape[1] - expt_conf['win_length']) // expt_conf['win_stride'] + 1
        for i_frame in n_frames:
            s_idx = i_frame * expt_conf['win_stride']
            frame = spect[:, s_idx:s_idx + expt_conf['win_length']]
            file_name = f"{label}_{spect_path.name.split('.')[0]}_{i_frame}.npy"
            np.save(frame, output_dir / file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(parser).parse_args())
    preprocess(expt_conf)
