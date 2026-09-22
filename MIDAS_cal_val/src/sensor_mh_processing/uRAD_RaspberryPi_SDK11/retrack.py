#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:01:26 2025

@author: kftg4
"""

import os
from natsort import natsorted
import time
import sys
from pathlib import Path

#Import all required paths and functions

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("../../..")

from path_definitions import path_fun, radar_path_definitions, plotting_choices, retrack_and_mf_choices_urad, rewrite
from universal_functions import mean_filter, apply_retracking, plot_waveforms
from urad_functions import get_IQ_data

plot_deconv_wfs, plot_retracked_wfs, plot_z_scopes, generate_gps_maps, generate_gps_and_radar_maps = plotting_choices()
sensor, retrack, n = retrack_and_mf_choices_urad()
n = 1
rewrite = rewrite()

#Generate list of files to process

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)

files = os.listdir(path + data_folder)
files = [f for f in files if f.endswith('.txt')]
files = natsorted(files)

#Run main processing

def main_for_retrack(save_as):
    '''
    Process and Plot IQ data

    Adjusted from GUI Code

    Parameters
    ----------
    save_as : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    params, imgs, x_axis, max_distance, all_times, PP = get_IQ_data(save_as, path, data_folder)

    imgs = mean_filter(imgs, n)
    
    r_wf = []
    for seq, Magnitude in enumerate(imgs):
        surf_r = apply_retracking(Magnitude, retrack.lower(), max_distance)
        r_wf += [surf_r]
        
        if plot_retracked_wfs:
            plot_waveforms(path, path_ssd, path_save_loc, save_as, x_axis, Magnitude, surf_r, seq, [x_axis[0], 12, -0.05, 0.2])

    with open(f"{path}/{path_save_loc}/z_scope_array/retracked_z_scope{save_as[:-4]}.txt", 'w') as f:
        for i in r_wf:
            f.write(f"{i}\n")
    with open(f"{path}/{path_save_loc}/timestamps/timestamps_{save_as[:-4]}.txt", 'w') as f:
        for i in all_times:
            f.write(f"{i}\n")
    with open(f"{path}/{path_save_loc}/pp_array/pp_{save_as[:-4]}.txt", 'w') as f:
        for i in PP:
            f.write(f"{i}\n")

def main():
    print("--- Started: retrack.py ---\n")
    start_time_1 = time.time()
    for save_as in files:
        start_time = time.time()
        my_file = Path(f"{path}/{path_save_loc}/z_scope_array/retracked_z_scope{save_as[:-4]}.txt")
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
            print("ValueError")
        except UnboundLocalError:
            print("Unbound Local Error")

        print("--- %s seconds ---\n" % (time.time() - start_time))

    print(f"--- Total Run Time: {round((time.time() - start_time_1)/60, 1)} minutes ---\n")

    print("--- Finished: retrack.py ---\n")

if __name__ == '__main__':
    main()

