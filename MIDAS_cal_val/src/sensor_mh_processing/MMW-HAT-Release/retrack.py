"""
Created on Wed Apr 30 08:16:42 2025

@author: Marlena Holloway
"""

import os
import sys
import numpy as np
from natsort import natsorted
import time
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
from mmw_functions import get_bin_data, sort_dist_issues, deconv_waveform, apply_manual_cleaning
from universal_functions import mean_filter, find_pp, apply_retracking, plot_waveforms

plot_deconv_wfs, plot_retracked_wfs, plot_z_scopes, generate_gps_maps, generate_gps_and_radar_maps = plotting_choices()
sensor, retrack, n = retrack_and_mf_choices_dreamrf()

rewrite = rewrite()

#Generate list of files to process

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)

files = os.listdir(path + data_folder)
files = [f for f in files if f.endswith('.bin')]
files = natsorted(files)

def main_for_retrack(save_as):
    save_as, save_as_2, dist = sort_dist_issues(save_as)

    imgs = get_bin_data(path, path_ssd, path_save_loc, data_folder, save_as_2, [("Azimuth", "Range")], 16)
    imgs = np.log10(imgs)
    imgs = np.mean(imgs, axis=1)
    
    x_axis = np.linspace(0, dist, len(imgs[0]))
    
    '''
    with open(f"{path}/{path_save_loc[:-1]}1/Iteration/normalizer/normalizer_{save_as[:-4]}.txt", 'r') as ref_f:
        ref = [list(map(float, line.split())) for line in ref_f]'''
    
    with open(f"{path}/MIDAS_cal_val/src/sensor_mh_processing/MMW-HAT-Release/ref_wf.txt", "r") as f:
        ref = [list(map(float, line.split())) for line in f]
        
    model = np.array(ref).reshape(57,)
    img_deconv = np.apply_along_axis(deconv_waveform, 1, imgs, model=model)
        
    img_mean_filtered = mean_filter(img_deconv, n)

    if len(img_mean_filtered) <= n:
        pp_wf = []
        r_wf = []

    else:
        pp_wf = np.apply_along_axis(find_pp, 1, img_mean_filtered)
        r_wf = np.apply_along_axis(apply_retracking, 1, img_mean_filtered, retrack=retrack.lower(), dist=dist)
        r_wf = apply_manual_cleaning(img_mean_filtered, imgs, r_wf, pp_wf, dist)
    
    for seq, Magnitude in enumerate(img_mean_filtered):
        surf_r = r_wf[seq]
        if plot_retracked_wfs:
            plot_waveforms(path, path_ssd, f"{path_save_loc}/Iteration2", save_as, x_axis, Magnitude, surf_r, seq, [x_axis[0], dist, -0.05, 1])
        
    with open(f"{path}/{path_save_loc}/Iteration2/z_scope_array/retracked_z_scope{save_as_2[:-4]}.txt", "w") as f:
        for item in r_wf:
            f.write(f"{item}\n")
    with open(f"{path}/{path_save_loc}/Iteration2/pp_array/retracked_z_scope{save_as[:-4]}.txt", "w") as pp_f:
        for item in pp_wf:
            pp_f.write(f"{item}\n")
            

def main():
    print("--- Started: retrack.py ---\n")
    start_time_1 = time.time()
    for save_as in files:
        start_time = time.time()
        my_file = Path(f"{path}/{path_save_loc}/Iteration2/z_scope_array/retracked_z_scope{save_as[:-4]}.txt")
        print(save_as)
        try:
            if rewrite == False:
                if my_file.is_file():
                    print("Path exists")
                else:
                    main_for_retrack(save_as)
            else:
                main_for_retrack(save_as)
        except ValueError:
            print("Value Error")
        except UnboundLocalError:
            print("Unbound Local Error")
        print("--- %s seconds ---\n" % (time.time() - start_time))
    
    print(f"--- Total Run Time: {round((time.time() - start_time_1)/60, 1)} minutes ---\n")
    
    print("--- Finished: retrack.py ---\n")

if __name__ == '__main__':
    main()