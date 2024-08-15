from multiprocessing import set_start_method
import numpy as np
from umcglib.utils import apply_parallel
from os import listdir
import matplotlib.pyplot as plt

################################  README  ######################################
# NEW - This script calculates the histograms of the labels in the segmentation/
# prediction. It applies parrallel loading and histogram calculation on 
# 0: background; 1: femoral cartilage; 2: tibial cartilage; 
# 3: patellar cartilage; 4: femur; 5: tibia; 6: patella

###############################################################################

def calc_hist_per_patient(file,mypath):        
    # file is the patient folder
    # mypath is the directory to the patient folders
     patient_id = file
     print(patient_id)
     seg = np.load(f'{mypath}/{patient_id}/{patient_id}_seg.npy').astype(int)
     background = np.sum(seg == 0)
     femoral_cartilage = np.sum(seg == 1)
     tibial_cartilage = np.sum(seg == 2)
     patellar_cartilage = np.sum(seg == 3)
     femur = np.sum(seg == 4)
     tibia = np.sum(seg == 5)
     patella = np.sum(seg == 6)
     output = [background,femoral_cartilage,tibial_cartilage,patellar_cartilage,femur,tibia,patella]
     return output

#####################################################
DATA_DIR = f'/data/pca-rad/datasets/miccai_2022/K2S_MICCAI2022_GRP/train/data/TBrecon1/train/untarred'
HIST_SAVE_PATH = f'temp/'
NCPU = 12

if __name__ == '__main__':
    # create a list of all patient's paths
     files = [f for f in listdir(DATA_DIR)]

     # For multiprocessing
     set_start_method("spawn")

     # calculate the amount of voxels per label in each patient
     counted_voxels_list = apply_parallel(files, calc_hist_per_patient, NCPU, mypath = DATA_DIR)
     print('>>>>>>>>> Calculation is done')
     counted_voxels = np.stack(counted_voxels_list, axis=0)

    # make a dict of the amount of voxels per label per patient
     dict = {
          'background': counted_voxels[:,0],
          'femoral_cartilage': counted_voxels[:,1],
          'tibial_cartilage': counted_voxels[:,2],
          'patellar_cartilage': counted_voxels[:,3],
          'femur': counted_voxels[:,4],
          'tibia': counted_voxels[:,5],
          'patella': counted_voxels[:,6]
        }

    # save a histogram per label

    # calculate 
     median_label_values = np.median(counted_voxels,axis=0)
     weights = np.divide(1,np.divide(median_label_values,np.sum(median_label_values)))
     weights_norm = np.divide(weights,np.sum(weights))
     print(median_label_values)   
     for idx,keys in enumerate(dict):
          plt.figure()
          data = dict[f'{keys}']
          plt.hist(data)
          plt.title(f"historgram of {keys}")
          plt.savefig(f"{HIST_SAVE_PATH}{keys}.png", dpi=300)
          weight = weights_norm[idx]
          print('done  ',keys,' weight:',weight)