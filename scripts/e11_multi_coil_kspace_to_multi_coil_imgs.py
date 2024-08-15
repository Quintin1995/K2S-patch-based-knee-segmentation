import glob
import os
import numpy as np

from umcglib.kspace import kspace_to_img
from umcglib.utils import apply_parallel
from umcglib.utils import print_stats_np, read_yaml_to_dict, set_gpu, print_


################################  README  ######################################
# NEW - This script will read in multi-coil kspace data and make an image of 
# each coil and write it to the scratch directory on Peregrine.
################################################################################


def get_patient_id(path):
    path = path.split('/')[-1]
    path = path.split('.')[-2]
    path = path.split('_')[-2]
    path = path[3:]
    return int(path)    # get patient number


###############################################################################
DEBUG = True

KSPACES_PATH = f'/data/pca-rad/datasets/miccai_2022/K2S_MICCAI2022_GRP/train/data/TBrecon1/train/untarred'
SAVE_TO_DIR  = f'/scratch/p290820/MICCAI/multi_coil_images/'


if __name__ == '__main__':

    ksps_paths = glob.glob(os.path.join(KSPACES_PATH, "pat*", "pat*_ksp.npy"))
    ksps_paths.reverse()
    for img_idx, fname in enumerate(ksps_paths):
        
        pat_id = get_patient_id(fname)
        print_(f"Processing patient: {pat_id} with idx: {img_idx}")

        if False not in [os.path.isfile(os.path.join(SAVE_TO_DIR, f"pat{pat_id}", f"coil_{idx}_kspace.npy")) for idx in range(18)]:
            print_(f"All coils of Patient: {pat_id} already processed.")
            continue

        mco_ksp = np.load(fname)  # Multi-COil K-SPace
        mco_ksp = np.transpose(mco_ksp, (2, 0, 1, 3))

        coil_imgs = apply_parallel(
            item_list   = mco_ksp,
            function    = kspace_to_img,
            num_workers = 12 if not DEBUG else 4
        )
        coil_imgs = np.stack(coil_imgs, axis=0)

        # remove first and last two slices(z-dim). And reverse z axis.
        coil_imgs = coil_imgs[:,:,:,2:-2]
        coil_imgs = coil_imgs[:,:,:,::-1]

        for c_idx in range(coil_imgs.shape[0]):
            pat_dir = os.path.join(SAVE_TO_DIR, f"pat{pat_id}")
            os.makedirs(pat_dir, exist_ok=True)

            p = os.path.join(SAVE_TO_DIR, f"pat{pat_id}", f"coil_{c_idx}_kspace.npy")
            np.save(p, coil_imgs[c_idx].astype(np.float16))
            print_(f"Saved kspace coil to: {p}")

            