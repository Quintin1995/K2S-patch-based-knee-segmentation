import os
import glob
import h5py
import gzip
import matplotlib.pyplot as plt

reverse_dir = os.path.join('temp', '4sub1', 'reverse_test')


files = glob.glob(os.path.join(reverse_dir, "*.h5.gz"))

_ = [print(x) for x in files]


for fpath in files:
    unzipped = gzip.open(fpath,'rb')
    h5_file = h5py.File(unzipped,'r') # Keys: kspace, kspace_info, seg, seg_info

    pred_seg_np = h5_file['pred_seg'][()]
    print(pred_seg_np.shape)


    fig, axes = plt.subplots(2,2)
    fig.suptitle('Images', fontsize=16)

    TITLES = ['seg_train reference','image prediction']
    axes[0,0].imshow(pred_seg_np[:,:,135],cmap='gray', vmax=6, vmin=0)
    axes[0,0].set_axis_off()
    axes[0,1].imshow(pred_seg_np[:,:,135],cmap='gray', vmax=6, vmin=0)
    axes[0,1].set_axis_off()

    axes[1,0].imshow(pred_seg_np[:,:,135],cmap='gray', vmax=6, vmin=0)
    axes[1,0].set_axis_off()
    axes[1,1].imshow(pred_seg_np[:,:,135],cmap='gray', vmax=6, vmin=0)
    axes[1,1].set_axis_off()

    plt.savefig(os.path.join("temp", '4sub1', "y_pred_reverse.png"), dpi=300)

    break