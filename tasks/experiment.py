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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

AVAILABLE_TARGET = ['da', 'dr']
# CHOICES = list(itertools.combinations(list(FEATURE_COLUMNS.keys()), r)) for r in range(len(FEATURE_COLUMNS.keys()))]
# FEATURE_PATTERNS = []
# np.array(FEATURE_COLUMNS.keys())[list(choice)] for choice in
# [FEATURE_PATTERNS.extend(list())]
FEATURE_PATTERNS = ['env1_zcm5_pim5']
DATA_KIND = ['daily', 'simulator']


def expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Elderly Experiment arguments")
    expt_parser.add_argument('--target', default='dr', choices=AVAILABLE_TARGET)
    expt_parser.add_argument('--feature', default='env')
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--train-data-kind', default='daily', choices=DATA_KIND)
    expt_parser.add_argument('--test-data-kind', default='daily', choices=DATA_KIND)
    expt_parser.add_argument('--drop-day-edge', action='store_true')
    expt_parser.add_argument('--n-test-users', default=2, type=int)
    expt_parser.add_argument('--hyperparameters', default='')
    expt_parser.add_argument('--expt-type', default='stat')
    expt_parser.add_argument('--mlflow', action='store_true')


    return parser


def label_func(row):
    return row.values[-1]


def load_func(path):
    df = pd.read_csv(path, header=None).values
    return df[:, :-1], df[:, -1]


def set_process_func(model_type, mean_, std_):
    def z_score(x):
        return (x - mean_) / std_

    def no_process(x):
        return x

    if model_type in ['nn', 'svm', 'knn']:
        return z_score
    else:
        return no_process


def set_data_paths(expt_dir, expt_conf, test_user_ids) -> Dict:
    data_dir = Path(__file__).resolve().parents[1]
    data_df = preprocess(data_dir, expt_conf['train_data_kind'], expt_conf['target'], expt_conf['feature'],
                         expt_conf['drop_day_edge'])
    train_data_df = data_df[~data_df['sub_id'].isin(test_user_ids)].drop('sub_id', axis=1)
    # This split rate has no effect if you specify group k-fold. Train and val set will be combined on CV
    train_data_df.iloc[:int(len(train_data_df) // 2)].to_csv(expt_dir / 'train_data.csv', index=False, header=None)
    train_data_df.iloc[int(len(train_data_df) // 2):].to_csv(expt_dir / 'val_data.csv', index=False, header=None)
    expt_conf['train_path'] = str(expt_dir / 'train_data.csv')
    expt_conf['val_path'] = str(expt_dir / 'val_data.csv')

    data_df = preprocess(data_dir, expt_conf['test_data_kind'], expt_conf['target'], expt_conf['feature'],
                         expt_conf['drop_day_edge'])
    test_df = data_df[data_df['sub_id'].isin(test_user_ids)].drop('sub_id', axis=1)
    test_df.to_csv(expt_dir / 'test_data.csv', index=False, header=None)
    expt_conf['test_path'] = str(expt_dir / 'test_data.csv')

    return expt_conf


def get_cv_groups(expt_conf, test_user_ids: List[int]):
    data_dir = Path(__file__).resolve().parents[1]
    data_df = preprocess(data_dir, expt_conf['train_data_kind'], expt_conf['target'], expt_conf['feature'], expt_conf['drop_day_edge'])
    data_df = data_df[~data_df['sub_id'].isin(test_user_ids)]

    subjects = data_df[['sub_id']].drop_duplicates()
    subjects['group'] = [j % expt_conf['n_splits'] for j in range(len(subjects))]
    subjects = subjects.set_index('sub_id')
    groups = data_df['sub_id'].apply(lambda x: subjects.loc[x, 'group'])
    return groups


def dump_dict(path, dict_):
    with open(path, 'w') as f:
        json.dump(dict_, f, indent=4)


def main(expt_conf, expt_dir, hyperparameters, typical_train_func, test_user_ids):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    expt_conf['class_names'] = [0, 1]
    if expt_conf['train_manager'] == 'nn':
        metrics_names = {'train': ['loss', 'uar'],
                         'val': ['loss', 'uar'],
                         'test': ['loss', 'uar']}
    else:
        metrics_names = {'train': ['uar'],
                         'val': ['uar'],
                         'test': ['uar']}

    dataset_cls = CSVDataSet
    expt_conf = set_data_paths(expt_dir, expt_conf, test_user_ids)

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    groups = get_cv_groups(expt_conf, test_user_ids)

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        # TODO cv時はmean と stdをtrainとvalの分割後に求める必要がある
        train_df = pd.read_csv(expt_conf['train_path']).iloc[:, :-1]
        process_func = set_process_func(expt_conf['model_type'], train_df.mean().values, train_df.std().values)

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
        process_func = set_process_func(expt_conf['model_type'], train_df.mean().values, train_df.std().values)

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

        # features = []
        # [features.extend(FEATURE_COLUMNS[feature]) for feature in expt_conf['feature'].split('_')]
        # print(getattr(experimentor.train_manager_cls, 'model_manager'))
        # importances = experimentor.train_manager_cls.model_manager.model.get_feature_importances(features)
        # importances.to_csv(expt_dir / 'feature_importances.csv', index=False)

    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if expt_conf['hyperparameters']:
        with open(f"{expt_conf['hyperparameters']}.txt", 'r') as f:
            hyperparameters = json.load(f)
    elif 'nn' in expt_conf['model_type'] and expt_conf['model_type'] != 'knn':
        hyperparameters = {
            'lr': [0.001, 0.0001],
            'nn_hidden_nodes': [[50], [50, 50], [50, 50, 50]],
            'sample_balance': ['same'],
        }
    elif expt_conf['model_type'] == 'lightgbm':
        hyperparameters = {
            'lr': [0.001, 0.0001],
            'n_leaves': [6, 16],
            'min_data_in_leaf': [50, 100],
            'max_depth': [3, 6],
            'n_estimators': [200],
            'subsample': [0.7],
            'feature_fraction': [0.7],
            'max_bin': [100],
            'reg_alpha': [1.0],
            'reg_lambda': [1.0],
        }
    elif expt_conf['model_type'] == 'rf':
        hyperparameters = {
            'max_depth': [3, 6],
            'n_estimators': [50, 200],
            'subsample': [0.6, 0.9],
        }
    elif expt_conf['model_type'] == 'svm':
        hyperparameters = {
            'C': [0.001, 0.01, 0.1, 1.0],
            'class_weight': ['balanced'],
        }
    else:
        hyperparameters = {
            'lr': [0.01],
        }
    dump_dict(f"{expt_conf['model_type']}.txt", hyperparameters)

    tmp_dir = Path(__file__).resolve().parents[1] / 'daily'
    test_users = [csv.name.replace('_life_dms.csv', '') for csv in Path(tmp_dir).iterdir() if csv.name.startswith('p')]
    # np.random.shuffle(test_users)
    test_users.sort()
    test_user_id_patterns = np.array(test_users).reshape((-1, expt_conf['n_test_users']))

    test_folder_name = f"ntest-{expt_conf['n_test_users']}_{expt_conf['expt_type']}"

    for test_user_id_pattern in test_user_id_patterns:
        expt_conf['expt_id'] = f"{expt_conf['expt_type']}_{expt_conf['model_type']}_{expt_conf['target']}_test-{expt_conf['test_data_kind']}_{expt_conf['feature']}"
        pj_dir = Path(__file__).resolve().parents[1]
        expt_dir = pj_dir / 'output' / test_folder_name / '_'.join(test_user_id_pattern) / f"{expt_conf['expt_id']}"
        expt_dir.mkdir(exist_ok=True, parents=True)
        main(expt_conf, expt_dir, hyperparameters, typical_train, test_user_id_pattern)
        # break

    aggregate(expt_conf)

    if not expt_conf['mlflow']:
        import shutil

        shutil.rmtree('mlruns')
