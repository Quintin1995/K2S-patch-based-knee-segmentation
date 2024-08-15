import h5py, os, gzip
import numpy as np
from multiprocessing import set_start_method
import glob
from typing import List
import argparse
import traceback
import logging

from umcglib.utils import print_, apply_parallel



################################  README  ######################################
# NEW - This script is used for extracting the compressed (.gz) files and 
# obtaining the .h5 file per patient. This .h5 has the segmentation and 
# kspace in it. Both are extracted and stored into a new the k2s repository as
# numpy arrays for speed later in the process.

def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training a Reconstruction model')

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help="Use this if the code should be run in debug mode. So paramters will be scaled down so that it runs much faster.")

    args = parser.parse_args()
    print_(f"args:\n {args}")
    return args


################################################################################


def get_patient_num(path: str):
    fname = path.split('/')[-1]
    fname = fname.split('-')[-1]
    return fname.split('.')[0]


def logs_to_file(log_list: List[str]):
    fname_log = os.path.join(SAVE_TO_DIR, "logs.txt")
    log_textfile = open(fname_log, "w")
    for path1 in log_list:
        log_textfile.write(path1 + "\n")
    log_textfile.close()


################################################################################


ROOT_H5_GZ  = r'/scratch/p290820/MICCAI/test_set/'
SAVE_TO_DIR = r'/data/pca-rad/datasets/miccai_2022/K2S_MICCAI2022_GRP/test/data/TBrecon1/test/untarred'
SAVE_TO_DIR = r'/scratch/p290820/MICCAI/test_set_numpy/'


if __name__ == '__main__':

    # For multiprocessing
    set_start_method("spawn")

    args = parse_input_args()
    DEBUG = True if args.debug else False
    N_CPUS = 4 if DEBUG else 10

    fpaths = glob.glob(os.path.join(ROOT_H5_GZ, f"{'TBrecon*.h5.gz'}"))
    logs = []
    for idx, inp_path in enumerate(fpaths):

        try:
            pat_num = get_patient_num(inp_path)
            print_(f"\nProcessing patient idx: {idx+1}/{len(fpaths)}\n\twith patient num: {pat_num} - \t Input path: {inp_path}")
            patient_path = os.path.join(SAVE_TO_DIR, f"pat{pat_num}/")

            if not os.path.isdir(patient_path):
                os.makedirs(patient_path, exist_ok=True)

                unzipped = gzip.open(inp_path,'rb')
                h5_file = h5py.File(unzipped,'r') # Keys: kspace, kspace_info, seg, seg_info

                ksp_n = h5_file['kspace'][()]                   #np.complex64
                # seg_n = h5_file['seg'][()].astype(np.float32)   #np.float32

                fname_ksp = os.path.join(patient_path, f"pat{pat_num}_ksp.npy")
                # fname_seg = os.path.join(patient_path, f"pat{pat_num}_seg.npy")
                np.save(fname_ksp, ksp_n)
                # np.save(fname_seg, seg_n)

                print_(f"Wrote to: {fname_ksp}")
                # print_(f"Wrote to: {fname_seg}")
            else:
                print_(f"{patient_path}  already exists.")
        except Exception as e:
            print(f"\n\n\nFILE: {inp_path} COULD NOT BE READ. \n\n\n")
            logs.append(f"File {inp_path} resulted in an error.")
            logging.error(traceback.format_exc())
            # logs_to_file(traceback.format_exc())
            continue

    print_("--- DONE ---")