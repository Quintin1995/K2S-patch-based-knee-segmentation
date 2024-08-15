import argparse
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import csv

from tensorflow.keras.models import load_model

from umcglib.utils import read_yaml_to_dict, set_gpu, print_, print_stats_np
from umcglib.predict import predict_sliding
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.compat.v1 import disable_eager_execution


################################  README  ######################################
# NEW - This script will loaded validation data for a model and perform sliding
# window predictions with it.


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
        '-mf',
        '--model_fname',
        type=str,
        required=True,
        help='Name of the model .h5 file to be loaded for the validation set.',
    )

    parser.add_argument(
        '-exp',
        '--experiment',
        type=str,
        required=False,
        default='',
        help='Name of experiment when using different prediction methods (e.g. padding/weight_map/..)',
    )

    parser.add_argument(
        '-test',
        dest     = 'test_set',
        action   = 'store_true',
        default  = False,
        help     = 'In case test set for final submission. Leave empty in case no test set. Any value will activate.',
    )

    args = parser.parse_args()

    print(args)
    return args

################################################################################


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


def write_to_file(val_idx, X_true_n, Y_true_n, Y_pred_n) -> None:
    
    print_stats_np(X_true_n, "\nX_true_n")
    print_stats_np(Y_true_n, "Y_true_n")
    print_stats_np(Y_pred_n, "Y_pred_s")

    X_true_s = sitk.GetImageFromArray(X_true_n)
    Y_true_s = sitk.GetImageFromArray(Y_true_n)
    Y_pred_s = sitk.GetImageFromArray(Y_pred_n)

    fname_X_true_s = os.path.join(VAL_DIR, f"idx{val_idx}_x_true.nii.gz")
    fname_Y_true_s = os.path.join(VAL_DIR, f"idx{val_idx}_y_true.nii.gz")
    fname_Y_pred_s = os.path.join(VAL_DIR, f"idx{val_idx}_y_pred.nii.gz")

    sitk.WriteImage(X_true_s, fname_X_true_s)
    sitk.WriteImage(Y_true_s, fname_Y_true_s)
    sitk.WriteImage(Y_pred_s, fname_Y_pred_s)

    print_(f"> Wrote to: {fname_X_true_s}")
    print_(f"> Wrote to: {fname_Y_true_s}")
    print_(f"> Wrote to: {fname_Y_pred_s}")
    

def write_to_file_test(val_idx, X_true_n, Y_pred_n) -> None:
    
    print_stats_np(X_true_n, "\nX_true_n")
    print_stats_np(Y_pred_n, "Y_pred_s")

    X_true_s = sitk.GetImageFromArray(X_true_n)
    Y_pred_s = sitk.GetImageFromArray(Y_pred_n)

    fname_X_true_s = os.path.join(VAL_DIR, f"idx{val_idx}_x_true.nii.gz")
    fname_Y_pred_s = os.path.join(VAL_DIR, f"idx{val_idx}_y_pred.nii.gz")

    sitk.WriteImage(X_true_s, fname_X_true_s)
    sitk.WriteImage(Y_pred_s, fname_Y_pred_s)

    print_(f"> Wrote to: {fname_X_true_s}")
    print_(f"> Wrote to: {fname_Y_pred_s}")


def make_performance_log_file(fname: str = "val_scores.csv"):
    fpath = os.path.join(VAL_DIR, fname)
    if os.path.exists(fpath):
        return

    headers = ['val_idx', 'mean_dices', 'fem_car1', 'tib_car2', 'pat_car3', 'fem4', 'tib5', 'pat6']
    
    file = open(fpath, 'w')
    writer = csv.writer(file, delimiter=';')
    writer.writerow(headers)
    file.close()


def append_row_to_log(row: list, fname: str = 'val_scores.csv',):
    
    fpath = os.path.join(VAL_DIR, fname)

    for idx, val in enumerate(row):
        if type(val) == float:
            row[idx] = round(val, 4)

    file = open(fpath, 'a')
    writer = csv.writer(file, delimiter=';')
    writer.writerow(row)
    file.close()


def focus_map(shape: tuple):
    aranges = [np.arange(d) - (d-1)/2 for d in shape]
    X, Y, Z = np.meshgrid(*aranges)
    distances = np.sqrt((X/shape[0])**2 + (Y/shape[1])**2 + (Z/shape[2])**2) + 0.01
    distances = distances / np.amax(distances)
    focus = 1 - np.sqrt(distances)
    focus -= np.amin(focus)
    return focus / np.amax(focus)


################################################################################
DEBUG = False


if __name__ == '__main__':
    disable_eager_execution()
    set_gpu(gpu_idx=1)

    # script input params
    args = parse_input_args()

    # model parameters
    m_configs = read_config(os.path.join(args.train_dir, 'train_config.yml'))
    train_dir = os.path.join(args.train_dir, f"fold{m_configs['fold_num']}")

    if args.test_set:
        VAL_DIR = os.path.join(train_dir, f"test_preds{args.experiment}/")
    else:
        VAL_DIR = os.path.join(train_dir, f"val_preds{args.experiment}/")

    os.makedirs(VAL_DIR, exist_ok=True)
    
    # load validation indexes
    val_set_key = f"val_set{m_configs['train_set_key'][-1]}"
    indexes_dict = read_yaml_to_dict(m_configs['indexes_path'])
    if args.test_set:
        val_idxs = np.arange(50)
    else:
        val_idxs = indexes_dict[val_set_key]

    print_(f"> Validation idxs: {val_idxs}\n\twith length: {len(val_idxs)}")

    # Get all the filesnames 
    if args.test_set:
        test_files = [l.strip() for l in open(m_configs["test_path_list"])]
    else:
        train_files = [l.strip() for l in open(m_configs["train_path_list"])]
        label_files = [l.strip() for l in open(m_configs["label_path_list"])]

    # Load the model
    model = load_model(
        filepath = os.path.join(train_dir, "models", args.model_fname),
        compile  = False,
        custom_objects={'InstanceNormalization': InstanceNormalization},
    )

    val_idxs = val_idxs[::-1]
    # loop over validation data
    for val_idx in val_idxs:

        fpath = Path(os.path.join(VAL_DIR, f"idx{val_idx}_x_true.nii.gz"))
        if fpath.is_file():
            print_(f"> {fpath} is already processed")
            continue

        print_(f">Processing img: {val_idx}...")

        if args.test_set:
            X_true_n = np.load(test_files[val_idx])
        else:
            X_true_n = np.load(train_files[val_idx])
            Y_true_n = np.load(label_files[val_idx])

        Y_pred_n = predict_sliding(
            input_n        = X_true_n,
            models         = [model],
            window_size    = m_configs['window_size'],
            window_norm    = m_configs['normalization'],
            n_steps        = (10, 10, 12),
            n_out_channels = m_configs['n_classes'],
            do_mirror_pred = True,
            padding        = True,
        )
            # weight_map = focus_map(m_configs['window_size'])

        # round the input image for less memory consumption
        X_true_n = np.round(X_true_n * 1000)
        # Y_pred_n[:,:,:,:,6] = np.multiply(Y_pred_n[:,:,:,:,6],1.5)
        # Collapse to a the softmax prediction to a single class
        Y_pred_n = np.argmax(Y_pred_n, axis=-1).astype(np.float32).squeeze()

        if args.test_set:
            write_to_file_test(val_idx, X_true_n, Y_pred_n)
        else:
            write_to_file(val_idx, X_true_n, Y_true_n, Y_pred_n)

    print(" -- DONE --")