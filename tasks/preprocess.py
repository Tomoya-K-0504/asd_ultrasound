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

import mlflow
import numpy as np
import pandas as pd
from aggregate import aggregate
from joblib import Parallel, delayed
from ml.src.dataset import CSVDataSet
from ml.tasks.base_experiment import typical_train, base_expt_args, typical_experiment
from preprocess import preprocess
from scipy.stats import stats
from ml.utils.notify_slack import notify_slack
from ml.utils.utils import dump_dict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def expt_args(parser):
    parser = base_expt_args(parser)
    prep_parser = parser.add_argument_group("ASD ultrasound")
    prep_parser.add_argument('--pool-size', default=5, type=int)
    prep_parser.add_argument('--split-size', default=100, type=int)

    return parser


def preprocess():
    """
    Spectrogram(1500 x 30000) -> avg pool (5 x 1) -> Spectrogram (300 x 30000) -> Spectrogram (300 x 300) x 100

    :return:
    """
    pass


if __name__ == '__main__':
    preprocess()