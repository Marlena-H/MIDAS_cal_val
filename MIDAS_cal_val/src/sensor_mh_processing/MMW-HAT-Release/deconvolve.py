"""
Created on Thu Mar  6 10:23:58 2025

@author: Marlena Holloway
"""

import time
import os
import sys
import numpy as np
from natsort import natsorted
from pathlib import Path

#Import all required paths and functions

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("../../..")

from path_definitions import path_fun, radar_path_definitions, plotting_choices, retrack_and_mf_choices_dreamrf, rewrite
from mmw_functions import get_bin_data, sort_dist_issues, deconv_waveform, apply_manual_cleaning_deconv_version
from universal_functions import find_pp, apply_retracking, plot_waveforms

py_dir = add_path("../../../src/sensor_srcs/MMW-HAT-Release/")
print(py_dir)

plot_deconv_wfs, plot_retracked_wfs, plot_z_scopes, generate_gps_maps, generate_gps_and_radar_maps = plotting_choices()
sensor, retrack, n = retrack_and_mf_choices_dreamrf()
n = 1

rewrite = rewrite()

#Generate list of files to process

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)
#print(path)
files = os.listdir(path + data_folder)
files = [f for f in files if f.endswith('.bin')]
files = natsorted(files)

def main_for_deconvolve(save_as):
    save_as, save_as_2, dist = sort_dist_issues(save_as)

    imgs = get_bin_data(path, path_ssd, path_save_loc, data_folder, save_as_2, [("Azimuth", "Range")], 16)
    imgs = np.log10(imgs)
    imgs = np.mean(imgs, axis=1)
    
    x_axis = np.linspace(0, dist, len(imgs[0]))
    
    imgs_added = np.zeros(np.shape(imgs)[0], dtype=bool)

    with open(f"{path}/MIDAS_cal_val/src/sensor_mh_processing/MMW-HAT-Release/ref_wf.txt", "r") as f:
        ref = [list(map(float, line.split())) for line in f]

    model = np.array(ref).reshape(57,)
    img_deconv = np.apply_along_axis(deconv_waveform, 1, imgs, model=model)

    pp_wf = np.apply_along_axis(find_pp, 1, img_deconv)
    r_wf = np.apply_along_axis(apply_retracking, 1, img_deconv, retrack=retrack.lower(), dist=dist)
    
    imgs_added, r_wf = apply_manual_cleaning_deconv_version(imgs_added, img_deconv, r_wf, pp_wf, dist)
    imgs_filtered = imgs[imgs_added]

    for seq, Magnitude in enumerate(imgs):
        surf_r = r_wf[seq]
        if plot_deconv_wfs:
            plot_waveforms(path, path_ssd, f"{path_save_loc}/Iteration2", save_as, x_axis, Magnitude, surf_r, seq, [x_axis[0], dist, 6, 9])
        
    if len(imgs_filtered) > 0:
        imgs_filtered = np.mean(imgs_filtered, axis=0)
        with open(f"{path}/{path_save_loc}/Iteration/normalizer/normalizer_{save_as[:-4]}.txt", "w") as ref_f:
            for item in imgs_filtered:
                ref_f.write(f"{item}\n")
    if len(imgs_filtered) == 0:
        with open(f"{path}/MIDAS_cal_val/src/sensor_mh_processing/MMW-HAT-Release/ref_wf.txt", "r") as f:
            imgs_filtered = [list(map(float, line.split())) for line in f]
            print("No Empty Waveforms Found - The Default Reference Deconvolution Function Will Be Applied For This File")
        with open(f"{path}/{path_save_loc}/Iteration/normalizer/normalizer_{save_as[:-4]}.txt", "w") as ref_f:
            for item in imgs_filtered:
                ref_f.write(f"{item[0]}\n")

def main():
    print("--- Started: deconvolve.py ---\n")
    start_time_1 = time.time()
    for save_as in files:
        start_time = time.time()
        my_file = Path(f"{path}/{path_save_loc}/Iteration/z_scope_array/retracked_z_scope{save_as[:-4]}.txt")
        print(save_as)
        try:
            if rewrite == False:
                if my_file.is_file():
                    print("Path exists")
                else:
                    main_for_deconvolve(save_as)
            else:
                main_for_deconvolve(save_as)
        except ValueError:
            print("ValueError")
        except UnboundLocalError:
            print("Unbound Local Error")
        print("--- %s seconds ---\n" % (time.time() - start_time))

    print(f"--- Total Run Time: {round((time.time() - start_time_1)/60, 1)} minutes ---\n")

    print("--- Finished: deconvolve.py ---\n")

if __name__ == "__main__":
    main()
