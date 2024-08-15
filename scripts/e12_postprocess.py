import argparse
import os
import glob
import SimpleITK as sitk
from multiprocessing import set_start_method
import numpy as np
from scipy import ndimage
import time
import sys


from umcglib.utils import print_stats_np, read_yaml_to_dict, set_gpu, print_, apply_parallel

################################  README  ######################################
# NEW - This script will perform post processing on full resolution segmentation
# predictions. First run the sliding window predict script to get validation set
# predictions in the train directory.
################################################################################


def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for splitting training, validation and the test set.')

    parser.add_argument(
        '-td',
        '--train_dir',
        type=str,
        required=True,
        help='Relative path to train directory',
    )

    parser.add_argument(
            '-fn',
            '--fold_num',
            type=int,
            required=True,
            help='Path to conig file with hyper parameters for a training run.',
        )

    parser.add_argument(
            '-pp',
            '--post_processing',
            type=str,
            required=False,
            default = '',
            help='name of post processing, in case of testing multiple',
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
    
    # read
    configs = read_yaml_to_dict(config_path)
    
    # fix some params
    configs['window_size'] = tuple(configs['window_size'])
    configs['seed'] = None if not configs['use_seed'] else configs['seed']
    
    # print
    if verbatim:
        print_("\n Configs")
        for key in configs:
            print_(f"{key}: {configs[key]} \t\t{type(configs[key])}")
        print_()
    return configs


################################################################################


def get_val_idx(path):
    path = path.split('/')[-1]
    path = path.split('_')[0]
    path = path[3:]
    return int(path)


def postprocess_img(
    path: str,
    train_dir: str,
    post_processing: str,
    perc_of_max: float=0.6,
    dilation = False,
    is_test_set = False, 
):
    
    t = time.time()

    # val_idx are the patient id's
    val_idx = get_val_idx(path)

    if is_test_set:
        cur_dir = os.path.join(train_dir, "test_preds") if not DEBUG else os.path.join(train_dir, "temp")
    else:
        cur_dir = os.path.join(train_dir, "val_preds") if not DEBUG else os.path.join(train_dir, "temp")

    n_classes = 6

    y_pred = sitk.ReadImage(path, sitk.sitkFloat32)
    y_pred = sitk.GetArrayFromImage(y_pred).T
    processed = np.zeros_like(y_pred, dtype=np.float32)

    for class_idx in range(1, n_classes+1):

        y_pred_copy = np.copy(y_pred)
        
        # Set values of voxels not beloning to the target class to zero.
        y_pred_copy[y_pred != class_idx] = 0.0

        # Perform component analysis
        labeled, n_comps = ndimage.label(y_pred_copy, np.ones((3, 3,3),dtype='int32'))
        if DEBUG:
            print(f">Number of components found: {n_comps} for class: {class_idx}")
        
        # Get the sizes of the detected components
        sizes = ndimage.sum(y_pred_copy, labeled, range(n_comps + 1))
        
        # Remove the components that do not satisfy the perc_of_max criterion.
        mask = sizes > (np.multiply(max(sizes), perc_of_max))
        
        if dilation:
            if class_idx == 6:
                struct2 = ndimage.generate_binary_structure(3,2)
                print(type(np.clip(mask[labeled],0,1).astype('float64')))
                dilation = ndimage.binary_dilation(np.clip(mask[labeled],0,1).astype('float64'), structure=struct2)
                class_img = np.multiply(dilation,class_idx).astype('int32')
            else:
                class_img = np.multiply(mask[labeled],class_idx).astype('int32')
                class_img[class_img == 9] = 3
        else:
            class_img = np.multiply(mask[labeled], class_idx).astype('int32')

        # Add the processed image of the current class to the processed image.
        processed += class_img

        # Save the processed image to file:
        temp_fname = os.path.join(cur_dir, post_processing, f"idx{val_idx}_y_pred.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(processed.squeeze().T), temp_fname)
        print(f">Wrote post processed image to: {temp_fname} - time: {round(time.time() - t, 1)}s  - pid: {os.getpid()}")

    return processed


################################################################################


def main(args):
    print_("\n\n\n\n-- Start of postprocessing --")

    configs = read_config(os.path.join(args.train_dir, 'train_config.yml'))
    
    train_dir = os.path.join(args.train_dir, f'fold{args.fold_num}')

    if args.test_set:
        val_dir = os.path.join(train_dir, 'test_preds')
        output_dir = os.path.join(train_dir, "test_preds", args.post_processing)
    else:
        val_dir   = os.path.join(train_dir, 'val_preds')
        output_dir = os.path.join(train_dir, "val_preds", args.post_processing)

    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(val_dir):
        print(f"Make sure this model has validation predictions in the folder: {val_dir}")
        quit()

    y_pred_paths = glob.glob(os.path.join(val_dir, f"idx*_y_pred.nii.gz"))
    print_(f">Number of validation predictions found: {len(y_pred_paths)}")

    processed = apply_parallel(
        item_list       = y_pred_paths,
        function        = postprocess_img,
        num_workers     = 8 if not DEBUG else 4,
        train_dir       = train_dir,
        post_processing = args.post_processing,
        is_test_set     = args.test_set
    )
    
    print_(" -- End   of postprocessing --")


################################################################################


DEBUG = False

if __name__ == '__main__':
    set_start_method("spawn")
    args = parse_input_args()
    main(args)