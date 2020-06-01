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
from joblib import Parallel, delayed
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import typical_train, base_expt_args, typical_experiment
from scipy.stats import stats
from ml.utils.notify_slack import notify_slack
from ml.utils.utils import dump_dict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

LABEL2INT = {'pat': 1, 'WT': 0}


def expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("ASD ultrasound Experiment arguments")
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--mlflow', action='store_true')
    expt_parser.add_argument('--notify-slack', action='store_true')

    return parser


def label_func(path):
    return LABEL2INT[path[1]]


def load_func(path):
    return np.load(path[0])


def set_data_paths(expt_conf):
    data_dir = Path(__file__).resolve().parents[1] / 'input' / 'processed'
    label_df = pd.read_excel(data_dir.parent / 'P08_12_USV_15qDup_MouseInfo.xlsx')
    manifest_df = pd.read_csv(data_dir.parent / 'manifest.csv')
    label_df['label'] = label_df['Genotype'].apply(lambda x: LABEL2INT[x])
    label_df['group'] = 0

    for label in LABEL2INT.values():
        n_subjects = label_df[label_df['label'] == label].shape[0]
        label_df[label_df['label'] == label, 'group'] = [j % expt_conf['n_splits'] for j in range(n_subjects)]

    manifest_df['mouse_id'] = manifest_df['file_path'].apply(lambda x: x.split('_')[1])
    groups = manifest_df.merge(right=label_df, how='left', left_on='mouse_id', right_on='MouseID')['group']

    manifest_df = manifest_df.drop('mouse_id', axis=1)
    # This split rate has no effect if you specify group k-fold. Train and val set will be combined on CV
    manifest_df.iloc[:int(len(manifest_df) // 2)].to_csv(data_dir / 'train_data.csv', index=False, header=None)
    manifest_df.iloc[int(len(manifest_df) // 2):].to_csv(data_dir / 'val_data.csv', index=False, header=None)
    expt_conf['train_path'] = str(data_dir / 'train_data.csv')
    expt_conf['val_path'] = str(data_dir / 'val_data.csv')
    
    return expt_conf, groups


def main(expt_conf, hyperparameters, typical_train_func):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')
    expt_dir = Path(__file__).resolve().parents[1] / 'output' / expt_conf['expt_id']

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    expt_conf['class_names'] = [0, 1]
    metrics_names = {'train': ['loss', 'uar'],
                     'val': ['loss', 'uar'],
                     'test': ['loss', 'uar']}

    expt_conf['sample_rate'] = 44100

    expt_conf, groups = set_data_paths(expt_conf)

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])
    dataset_cls = ManifestWaveDataSet
    process_func = None

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"

        with mlflow.start_run():
            result_series, val_pred, _ = typical_train_func(expt_conf, load_func, label_func, process_func, dataset_cls,
                                                            groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            # mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    # For debugging
    if expt_conf['n_parallel'] == 1:
        result_pred_list = [experiment(pattern, deepcopy(expt_conf)) for pattern in patterns]
    else:
        expt_conf['n_jobs'] = 0
        result_pred_list = Parallel(n_jobs=expt_conf['n_parallel'], verbose=0)(
            [delayed(experiment)(pattern, deepcopy(expt_conf)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = patterns
    result_list = np.array([result for result, pred in result_pred_list])
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)
    pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())

    val_results.to_csv(expt_dir / 'val_results.csv', index=False)
    print(f"Devel results saved into {expt_dir / 'val_results.csv'}")
    for (_, _), pattern in zip(result_pred_list, patterns):
        pattern_name = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        dump_dict(expt_dir / f'{pattern_name}.txt', expt_conf)

    # Train with train + devel dataset
    if expt_conf['test']:
        best_trial_idx = val_results['uar'].argmax()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]
        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        train_df = pd.read_csv(expt_conf['train_path']).iloc[:, :-1]

        metrics, pred_dict_list, experimentor = typical_experiment(expt_conf, load_func, label_func, process_func,
                                                                   dataset_cls, groups)

        if expt_conf['return_prob']:
            ensemble_pred = np.argmax(np.array([pred_dict['test'] for pred_dict in pred_dict_list]).sum(axis=0), axis=1)
        else:
            ensemble_pred = stats.mode(np.array([pred_dict['test'] for pred_dict in pred_dict_list]), axis=0)[0][0]
        _, test_labels = load_func(expt_conf['test_path'])
        uar = balanced_accuracy_score(test_labels, ensemble_pred)
        print(f'{uar:.05f}')
        print(f'Confusion matrix: \n{confusion_matrix(test_labels, ensemble_pred)}')
        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}_{uar:.04f}.csv"
        pd.DataFrame(ensemble_pred).to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")

        result_file_name = f"results_{expt_conf['model_type']}_{expt_conf['target']}_{expt_conf['test_data_kind']}.csv"
        with open(expt_dir.parent / result_file_name, 'a') as f:
            f.write(f"{expt_conf['n_splits']},{expt_conf['feature']},{val_results['uar'].max()},{uar}\n")

    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    hyperparameters = {
        'lr': [1e-4],
        'batch_size': [2],
        'model_type': ['panns'],
        'checkpoint_path': ['../Cnn14.pth'],
        'epoch_rate': [1.0],
        'sample_balance': ['same'],
    }

    main(expt_conf, hyperparameters, typical_train)

    if not expt_conf['mlflow']:
        import shutil

        shutil.rmtree('mlruns')

    if expt_conf['notify_slack']:
        cfg = dict(
            body=f"Finished experiment {expt_conf['expt_id']}: \n" +
                 "Notion ticket: https://www.notion.so/DCASE-2020-010ca4ceda0f49828d2ee81b77b8e1a4",
            webhook_url='https://hooks.slack.com/services/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'
        )
        notify_slack(cfg)
