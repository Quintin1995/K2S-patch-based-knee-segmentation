import glob
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import h5py
import gzip


def get_patient_num(path):
    path = path.split('/')[-1]
    path = path.split('.')[0]
    path = path.split('_')[0]
    path = path[3:]
    return path


def get_idx_from_path(path):
    path = path.split('/')[-1]
    path = path.split('.')[0]
    path = path.split('_')[0]
    idx = path[3:]
    return int(idx)


################################################################################


# Load predictions and convert from .nii.gz to np array. 
pred_dir = '../../qvlohuizen/k2s_umcg/train_output/0_weighted_dice/fold0/test_preds'        #stefan
pred_dir = 'train_output/0_weighted_dice/fold0/test_preds'                                  #quintin

y_pred_paths = glob.glob(os.path.join(pred_dir, f"idx*_y_pred.nii.gz"))
_ = [print(x) for x in y_pred_paths]
print(f"> Found num files: {len(y_pred_paths)}\n")

# load indexes from X:\qvlohuizen\k2s_umcg\data\kspace_paths_test_set
test_path_list = os.path.join('data', 'rss_recon_paths_test_set.txt')
test_files = [l.strip() for l in open(test_path_list)]

for i, img_path in enumerate(y_pred_paths):

    # Find the index of the patient in the filepath
    print(f"\n>Found y_pred_path post processed: {img_path}")
    relevant_idx = get_idx_from_path(img_path)   # indexes 0-50
    print(f"idx found from path: {relevant_idx}")

    # Based on the found index we can find the patient ID.
    pat_id = get_patient_num(test_files[relevant_idx])

    # Reat the image as sitk and convert to np array
    y_pred = sitk.ReadImage(img_path, sitk.sitkFloat32)
    y_pred = sitk.GetArrayFromImage(y_pred)
    print(f"Detected y_pred shape: {y_pred.shape}")
    
    # Put the array in a .h5 file and compress with gzip.
    new_fpath_h5 = os.path.join("temp", "4sub1", f"TBrecon-01-02-{pat_id}.h5")
    h5f = h5py.File(new_fpath_h5, 'w')
    h5f.create_dataset('seg', data=y_pred)
    h5f.close()
    print(f"> Wrote to {new_fpath_h5}")

    # if i == 2:
    #     break

# Check the orientation of a train segmenation image file and a slice of the prediction.
seg_fname = '/data/pca-rad/datasets/miccai_2022/K2S_MICCAI2022_GRP/train/data/TBrecon1/train/untarred/pat00008/pat00008_seg.npy'
seg_train = np.load(seg_fname)
print(seg_train.shape)

fig, axes = plt.subplots(2,2)
fig.suptitle('Images', fontsize=16)

TITLES = ['seg_train reference','image prediction']
axes[0,0].imshow(y_pred[:,:,135],cmap='gray', vmax=6, vmin=0)
axes[0,0].set_axis_off()
axes[0,1].imshow(seg_train[:,:,135],cmap='gray', vmax=6, vmin=0)
axes[0,1].set_axis_off()

axes[1,0].imshow(y_pred[:,:,135],cmap='gray', vmax=6, vmin=0)
axes[1,0].set_axis_off()
axes[1,1].imshow(seg_train[:,:,135],cmap='gray', vmax=6, vmin=0)
axes[1,1].set_axis_off()

plt.savefig(os.path.join("temp", '4sub1', "y_pred_as_array.png"), dpi=300)

print("-- done --")
