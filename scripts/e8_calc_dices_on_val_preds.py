import argparse
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import List
import csv
import sys

from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

from umcglib.utils import read_yaml_to_dict, print_, print_stats_np


################################  README  ######################################
# NEW - This script will loaded validation data for a model and perform sliding
# window predictions with it.
################################################################################


def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for splitting training, validation and the test set.')

    parser.add_argument(
        '-td',
        '--train_dir',
        type=str,
        required=True,
        help='Path to train directory.',
    )

    parser.add_argument(
        '-pp',
        '--post_processing',
        type=str,
        required=False,
        default='',
        help='The kind of post_processing revers also revers to the name of the csv file',
    )

    args = parser.parse_args()

    print(args)
    return args


################################################################################


def get_mean_dice_and_dices_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
):
    n_decimals = 3
    dices = []
    for c in range(1, num_classes):
        y_class_true = (y_true == c) * 1.
        y_class_pred = (y_pred == c) * 1.

        nom = np.sum(y_class_pred * y_class_true)
        den = np.sum(y_class_pred) + np.sum(y_class_true)

        dc = (2 * nom + 1e-7) / (den + 1e-7)
        dices.append(dc)

        mean_dice = sum(dices) / len(dices)

    dices = [float(np.round(v, n_decimals)) for v in dices]
    return float(round(mean_dice, n_decimals)), dices


def read_config(config_path, verbatim=True):
    
    configs = read_yaml_to_dict(config_path)
    
    configs['window_size'] = tuple(configs['window_size'])
    configs['seed'] = None if not configs['use_seed'] else configs['seed']
    
    if verbatim:
        print_("\n Configs")
        for key in configs:
            print_(f"{key}: {configs[key]} \t\t{type(configs[key])}")
        print_()
    return configs



def make_performance_log_file(fname: str):
    fpath = fname
    if os.path.exists(fpath):
        return

    headers = ['val_idx', 'mean_dices', 'fem_car1', 'tib_car2', 'pat_car3', 'fem4', 'tib5', 'pat6']
    
    file = open(fpath, 'w')
    writer = csv.writer(file, delimiter=';')
    writer.writerow(headers)
    file.close()


def append_row_to_log(row: list, fname: str):
    
    fpath = os.path.join(VAL_DIR_Y_PRED, fname)

    file = open(fpath, 'a')
    writer = csv.writer(file, delimiter=';')
    writer.writerow(row)
    file.close()


################################################################################


if __name__ == '__main__':
    args = parse_input_args()

    # model parameters
    m_configs = read_config(os.path.join(args.train_dir, 'train_config.yml'))
    train_dir = os.path.join(args.train_dir, f"fold{m_configs['fold_num']}")
    print(train_dir)
    print(args.post_processing)
    VAL_DIR_X_TRUTH = os.path.join(train_dir, 'val_preds')
    VAL_DIR_Y_PRED  = os.path.join(train_dir, 'val_preds', args.post_processing)
    
    
    print(VAL_DIR_X_TRUTH)
    print(VAL_DIR_Y_PRED)

    os.makedirs(VAL_DIR_X_TRUTH, exist_ok=True)
    os.makedirs(VAL_DIR_Y_PRED, exist_ok=True)

    fname  = f"{VAL_DIR_Y_PRED}val_scores_{args.post_processing}.csv"
    print(fname)
    make_performance_log_file(f'val_scores_{args.post_processing}.csv')

    
    # load validation indexes
    val_set_key = f"val_set{m_configs['train_set_key'][-1]}"
    indexes_dict = read_yaml_to_dict(m_configs['indexes_path'])
    val_idxs = indexes_dict[val_set_key]
    print_(f"> Validation idxs: {val_idxs}\n\twith length: {len(val_idxs)}")

    # loop over validation data
    for val_idx in val_idxs:

        print_(f"\n>Calculating dices on validation image: {val_idx}...")

        y_true_fpath = os.path.join(VAL_DIR_X_TRUTH, f"idx{val_idx}_y_true.nii.gz")
        y_pred_fpath = os.path.join(VAL_DIR_Y_PRED, f"idx{val_idx}_y_pred.nii.gz")
        
        if not os.path.exists(y_pred_fpath):
            print(f">{y_pred_fpath} does not exist yet. Run the e7_sliding_window_pred_on_model.py script first.")
            print(f">{y_pred_fpath} does not exist yet. Run the e7_sliding_window_pred_on_model.py script first. or use post processing to get these images (e12 script)")
            continue
        if not os.path.exists(y_true_fpath):
            print(f">{y_true_fpath} does not exist yet. Run the e7_sliding_window_pred_on_model.py script first.")
            continue

        # Read Nifti's 
        y_true_s = sitk.ReadImage(y_true_fpath, sitk.sitkFloat32)
        y_pred_s = sitk.ReadImage(y_pred_fpath, sitk.sitkFloat32)

        # Convert to numpy
        y_true_n = sitk.GetArrayFromImage(y_true_s).T
        y_pred_n = sitk.GetArrayFromImage(y_pred_s).T

        mean_dice, dices = get_mean_dice_and_dices_np(
            y_true=y_true_n,
            y_pred=y_pred_n,
            num_classes=7
        )

        print(f"\t>Mean dice: {round(mean_dice, 3)} and dices: {dices}")
        append_row_to_log(row = [val_idx, mean_dice] + dices, fname = f'val_scores_{args.post_processing}.csv',)

    print(" -- DONE --")