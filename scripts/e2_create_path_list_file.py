from umcglib.utils import print_
import glob
import os


################################  README  ######################################
# NEW - Use this script to create a text file with paths to segmentations and
# kspaces in it.
################################################################################


def extract_id(path):
    path = path.split('/')[-1]
    path = path.split('.')[0]
    path = path.split('_')[0]
    path = path[3:]
    return int(path)


def convert_path_to_scratch_path(path):
    id = extract_id(path)
    new_path = "/scratch/p290820/MICCAI/multi_coil_images"
    new_path = os.path.join(new_path, f"pat{id}")
    return new_path


################################################################################


# DATA_DIR      = r'/data/pca-rad/datasets/miccai_2022/K2S_MICCAI2022_GRP/train/data/TBrecon1/train/untarred'     # global path    -old
DATA_DIR      = r'/scratch/p290820/MICCAI/test_set_numpy'                                                                       #-new
SAVE_LOCATION = r'data/'        #relative path

if __name__ == '__main__':

    ksp_paths = glob.glob(os.path.join(DATA_DIR, "pat*/pat*_ksp.npy"))
    # seg_paths = glob.glob(os.path.join(f"{DATA_DIR}", "pat*/pat*_seg.npy"))
    # rss_paths = glob.glob(os.path.join(f"{DATA_DIR}", "pat*/pat*_rss_recon.npy"))
    # rss_u_paths = glob.glob(os.path.join(f"{DATA_DIR}", "pat*/pat*_rss_recon_usampled.npy"))
    # rss_u_paths = glob.glob(os.path.join(f"{DATA_DIR}", "pat*/pat*_rss_recon_usampled.npy"))
    # mcis_paths = glob.glob(os.path.join(f"{DATA_DIR}", "pat*/pat*_ksp.npy"))

    print_(f"Num paths in kspace (test set) paths: {len(ksp_paths)}")
    # print_(f"Num paths in seg       paths: {len(seg_paths)}")
    # print_(f"Num paths in rss recon paths: {len(rss_paths)}")
    # print_(f"Num paths in rss recon paths: {len(rss_u_paths)}")
    # print_(f"Num paths in kspace    paths: {len(mcis_paths)}")
    
    ksp_paths = []
    # seg_paths = []
    rss_paths = []
    # rss_u_paths = []
    # mcis_paths = []
    
    # mcis_path_list = list(os.listdir(f"/scratch/p290820/MICCAI/multi_coil_images/"))   #Multi-Coil-Image-Space images path list.

    for pat_dir in os.listdir(DATA_DIR):

        ksp_path_list   = glob.glob(f"{os.path.join(DATA_DIR, pat_dir)}/pat*_ksp.npy")
        # seg_path_list   = glob.glob(f"{os.path.join(DATA_DIR, pat_dir)}/pat*_seg.npy")
        rss_path_list   = glob.glob(f"{os.path.join(DATA_DIR, pat_dir)}/pat*_rss_recon.npy")
        # rss_u_path_list = glob.glob(f"{os.path.join(DATA_DIR, pat_dir)}/pat*_rss_recon_usampled.npy")
        # mcis_path_list  = glob.glob(f"{os.path.join(DATA_DIR, pat_dir)}/pat*_ksp.npy")


        ksp_paths.append(ksp_path_list[0])
        # seg_paths.append(seg_path_list[0])
        rss_paths.append(rss_path_list[0])
        # rss_u_paths.append(rss_u_path_list[0])
        # mcis_paths.append(convert_path_to_scratch_path(mcis_path_list[0]))  # this is a bit hacky, but yeah...

        # Sanity check
        # if len(ksp_path_list) != 1 or len(seg_path_list) != 1 or len(rss_path_list) != 1 or len(rss_u_path_list) != 1:
        if len(ksp_path_list) != 1:
            
            print(ksp_path_list)
            # print(seg_path_list)
            # print(rss_path_list)
            # print(rss_u_path_list)

            print(len(ksp_path_list))
            # print(len(seg_path_list))
            # print(len(rss_path_list))
            # print(len(rss_u_path_list))
            
            print_(f"{pat_dir} IS INCOMPLETE")
            print(ksp_path_list)
            # print(seg_path_list)
            # print(rss_path_list)
            # print(rss_u_path_list)

            print("EXITING WITH ERROR!")
            exit()

    fname_ksp = os.path.join(SAVE_LOCATION, "kspace_paths_test_set.txt")
    ksp_textfile = open(fname_ksp, "w")
    for path1 in ksp_paths:
        ksp_textfile.write(path1 + "\n")
    ksp_textfile.close()
    print_(f"Wrote path text file (ksp) to: {fname_ksp}")

    # fname_seg = os.path.join(SAVE_LOCATION, "seg_paths.txt")
    # seg_textfile = open(fname_seg, "w")
    # for path2 in seg_paths:
    #     seg_textfile.write(path2 + "\n")
    # seg_textfile.close()
    # print_(f"Wrote path text file (seg) to: {fname_seg}")
    
    fname_rss = os.path.join(SAVE_LOCATION, "rss_recon_paths_test_set.txt")
    rss_textfile = open(fname_rss, "w")
    for path3 in rss_paths:
        rss_textfile.write(path3 + "\n")
    rss_textfile.close()
    print_(f"Wrote path text file (rss recon) to: {fname_rss}")

    # fname_rss_u = os.path.join(SAVE_LOCATION, "rss_recon_usampled_paths.txt")
    # rss_u_textfile = open(fname_rss_u, "w")
    # for path4 in rss_u_paths:
    #     rss_u_textfile.write(path4 + "\n")
    # rss_u_textfile.close()
    # print_(f"Wrote path text file (rss recon) to: {fname_rss_u}")


    # fname_mcis = os.path.join(SAVE_LOCATION, "multi_coil_image_space_paths.txt")
    # mcis_textfile = open(fname_mcis, "w")
    # for path5 in mcis_paths:
    #     mcis_textfile.write(path5 + "\n")
    # mcis_textfile.close()
    # print_(f"Wrote path text file (multi_coil_image_spaces path list file) to: {fname_mcis}")


    print("--- DONE ---")