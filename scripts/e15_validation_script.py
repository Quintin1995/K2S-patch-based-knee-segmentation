import h5py
import numpy as np
import gzip
import matplotlib.pyplot as plt
import scipy.ndimage
import os

#Loading a given segmentation from the dataset, as a test case
path_ = '/data/TBrecon1/miccai_challenge/train/untarred/TBrecon-01-02-00019.h5.gz'
path_ = 'temp/4sub1/TBrecon-01-02-00729.h5.gz'

with gzip.open(path_, 'rb') as f:
    with h5py.File(f, "r") as f_:
        seg_    = list(f_['seg'])
    seg_gt = np.array(seg_)

#Doing a binary erosion for each of the 6 segmentation classes of the dataset, to provide a general test example
#for what a predicted mask will look like
seg_pred = np.zeros(seg_gt.shape)
for tissue_ in range(1,7):
    eroded_tissue = scipy.ndimage.binary_erosion(seg_gt == tissue_).astype(int)
    seg_pred[eroded_tissue != 0] = tissue_

# Load a training segmentation to calculate the dice fore that. To check orientation etc
train_seg_fpath = "/data/pca-rad/datasets/miccai_2022/K2S_MICCAI2022_GRP/train/data/TBrecon1/train/untarred/pat00007/pat00007_seg.npy"
seg_pred = np.load(train_seg_fpath)


"""
Visualizing the ground truth and eroded masks. We'll use the eroded one as a proxy for what a predicted segmentation
mask will look like. Predicted masks should be thresholded and converted completely to integers that correspond to 
the different tissues.
"""
fig = plt.figure(figsize = (15,5))
plt.subplot(1,3,1)
plt.imshow(seg_gt[:,:,100],cmap = 'rainbow',vmin=0, vmax=6)
plt.axis('off')
plt.title('GT (Original)')

plt.subplot(1,3,2)
plt.imshow(seg_pred[:,:,100],cmap = 'rainbow',vmin=0, vmax=6)
plt.axis('off')
plt.title('Different training segmentation')

plt.subplot(1,3,3)
plt.imshow(seg_gt[:,:,100]-seg_pred[:,:,100],cmap = 'rainbow',vmin=0, vmax=6)
plt.axis('off')
plt.title('Difference')

p = os.path.join("temp", '4sub1', "validation_script2.png")
plt.savefig(p, dpi=300)
print(f">Wrote to {p}")


#Calculate DSC for a single tissue between 3D predicted and ground truth masks
def calculate_dsc(pred,gt,tissue_index):
    pred_tissue  = (pred == tissue_index).astype(int)
    gt_tissue    = (gt == tissue_index).astype(int)
    
    return 2*(np.sum((pred_tissue + gt_tissue) == 2))/(np.sum(pred_tissue) + np.sum(gt_tissue)), np.sum(gt_tissue)

"""
Calculate a weighted DSC across the 6 tissues, index correspondence is as follows:
1: femoral cartilage
2: tibial cartilage
3: patellar cartilage
4: femur
5: tibia
6: patella

DSC is calculated for each of these tissues and combined to generate a weighted DSC that ranges from [0,1].
Each individual tissue DSC is divided by the total number of pixels that tissue had in the ground truth segmentation
and summed, such that each tissue's contribution to the weighted DSC is inversely proportional to the total number of
pixels it had in the ground truth segmentation. The summed weighted DSC is then scaled such that it will range from
[0,1].
"""
def calculate_weighted_dsc(pred,gt):
    scaling_multiplier = 0
    weighted_dsc       = 0

    for tissue_ in range(1,7):
        curr_dsc, n_tissue_pixels = calculate_dsc(pred,gt,tissue_)
        weighted_dsc += curr_dsc/n_tissue_pixels
        scaling_multiplier += 1/n_tissue_pixels

    weighted_dsc = weighted_dsc*(1/scaling_multiplier)
    return weighted_dsc

#Calculate weighted DSC
wdsc = calculate_weighted_dsc(seg_pred,seg_gt)
print(f">Weighted dice: {wdsc}")