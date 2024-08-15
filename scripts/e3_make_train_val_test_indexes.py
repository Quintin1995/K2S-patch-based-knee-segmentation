import argparse
import random
import os

from umcglib.utils import print_, dump_dict_to_yaml


################################  README  ######################################
# NEW - This script will create indexes for the training, validation and test
# sets. Create a yaml file with 10 sets of indexes for train, val and test. So 
# that they can be used for 10 fold cross validation when needed. The filename
# should contain an integer, which is the seed used to generate the indexes. The
# filename also indicates the type of split used.
# Output structure:
#   trainSet0: [indexes]
#   valSet0:   [indexes]
#   testSet0:  [indexes]
#   trainSet1: [indexes]
#   valSet1:   [indexes]
#   testSet1:  [indexes]
# etc...


################################ PARSER ########################################


def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for splitting training, validation and the test set.')

    parser.add_argument('-nf',
                        '--num_folds',
                        type=int,
                        default=5,
                        help='The number of folds in total. This amount of index sets will be created.')

    parser.add_argument('-p',
                        '--path_to_paths_file',
                        type=str,
                        default="data/kspace_paths.txt",
                        help='Path to the .txt file containting paths to the nifti files.')

    parser.add_argument('-s',
                        '--split',
                        type=str,
                        default="80/10/10",
                        help='Train/validation/test split in percentages.')

    args = parser.parse_args()

    # split the given split string.
    args.p_train = int(args.split.split('/')[0])
    args.p_val = int(args.split.split('/')[1])
    args.p_test = int(args.split.split('/')[2])

    assert args.p_train + args.p_val + args.p_test == 100, "The train, val, test split to sum to 100%."
    print_("\n\nMaking Train - Validation - Test indexes based.")
    print_(args)
    return args


################################################################################
SEED = 1234

if __name__ == '__main__':
    args = parse_input_args()

    os.makedirs('data/path_lists/', exist_ok=True)

    # Read the amount of observations/subjects in the data.
    paths_ksps = [l.strip() for l in open(args.path_to_paths_file)]
    # paths_segs = [l.strip() for l in open("data/seg_paths.txt")]
    num_obs = len(paths_ksps)
    # assert num_obs == len(paths_segs), "Number of paths in kspace path list file and corresponding segmentation path list file are not equal."
    print_(f"Number of observations in {args.path_to_paths_file}: {len(paths_ksps)}")

    # Create cutoff points for training, validation and test set.
    train_cutoff = int(args.p_train/100 * num_obs)
    val_cutoff   = int(args.p_val/100 * num_obs) + train_cutoff
    test_cutoff  = int(args.p_test/100 * num_obs) + val_cutoff
    print_(f"\ncutoffs: {train_cutoff}, {val_cutoff}, {test_cutoff}")

    # Create dict that will hold all the data
    data_dict = {}
    data_dict["init_seed"] = SEED
    data_dict["split"] = args.split
        
    # loop over the amount of folds, that many sets will be created in a yaml file.
    for set_idx in range(args.num_folds):

        # Set new seed first
        random.seed(SEED + set_idx)

        # shuffle the indexes 
        indexes = list(range(num_obs))
        random.shuffle(indexes)

        train_idxs = indexes[:train_cutoff]
        val_idxs   = indexes[train_cutoff:val_cutoff]
        test_idxs  = indexes[val_cutoff:test_cutoff]
        
        data_dict[f"train_set{set_idx}"] = train_idxs
        data_dict[f"val_set{set_idx}"]   = val_idxs
        data_dict[f"test_set{set_idx}"]  = test_idxs
        
    for key in data_dict:
        if type(data_dict[key]) == list:
            print_(f"{key}: num elements={len(data_dict[key])}")

    dump_dict_to_yaml(data_dict, "data/path_lists", filename=f"train_val_test_idxs_n{num_obs}_test_set", verbose=False)