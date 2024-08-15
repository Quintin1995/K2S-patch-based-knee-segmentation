import numpy as np
from os import listdir
from scipy.fftpack import fftshift, ifftshift,  ifftn
from umcglib.utils import apply_parallel


################################  README  ######################################
# NEW - This script reconstructs images from kspace using root sum of squares method. 
# It loops over the untarred folder, reads the kspace (npy), reconstruct coil images (parrallel), 
# applies root sum of squares and saves the recon images in the corresponding folder (npy).


###############################################################################
def kspace_to_img(kspace_single_coil: np.ndarray):
    
    # zero padding of k-space to 512,512,200 (similar to segmentation)
    kspace_padded = np.pad(np.squeeze(kspace_single_coil[:,:,:]),[(128,128),(128,128),(0,0)],'constant')
    # reconstruct coil image using fft shift over x and y axis (contains np.roll function)
    coil_image = np.abs(fftshift(fftshift(ifftn(ifftshift(kspace_padded)),axes=0),axes=1))
    return coil_image


def img_from_ksp(
    kspace: np.ndarray
):
    kspace = np.transpose(kspace, (2, 0, 1, 3))
    # calculete coil images 
    coil_images = apply_parallel(kspace, kspace_to_img, 12)
    # apply root sum of squares
    image = np.round(np.sqrt(np.sum(np.power(coil_images,2),0)))
    # remove outer edges (similar to segmentation) 
    image = image[:,:,2:-2]
    # mirrors along the z-axis (=x-axis in niffti of ITK_snap) 
    image = image[:,:,::-1]
    return image


########################
KSPACES_PATH = f'/data/pca-rad/datasets/miccai_2022/K2S_MICCAI2022_GRP/train/data/TBrecon1/train/untarred'    # for the train set
KSPACES_PATH = f'/scratch/p290820/MICCAI/test_set_numpy'                                                      # test set
PATIENT_IDS = [f for f in listdir(KSPACES_PATH)]

if __name__ == '__main__':
    for file_idx in range(50):          # 300 for train set and 50 for test set
        image_recon = []
        kspace = []
        patient_id = []

        patient_id = PATIENT_IDS[file_idx]
        kspace = np.load(f'{KSPACES_PATH}/{patient_id}/{patient_id}_ksp.npy')
        image_recon = img_from_ksp(kspace)

        np.save(f'{KSPACES_PATH}/{patient_id}/{patient_id}_rss_recon.npy', image_recon)
        
        print('done:',file_idx,'  ',f'{KSPACES_PATH}/{patient_id}/{patient_id}_rss_recon.npy')

