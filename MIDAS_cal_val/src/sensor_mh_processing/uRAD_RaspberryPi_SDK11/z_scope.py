#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 10:14:51 2025

@author: kftg4
"""
    
import os
import numpy as np
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
from universal_functions import mean_filter, plotting_z_scopes
from urad_functions import get_IQ_data

plot_deconv_wfs, plot_retracked_wfs, plot_z_scopes, generate_gps_maps, generate_gps_and_radar_maps = plotting_choices()
sensor, retrack, n = retrack_and_mf_choices_urad()

rewrite = rewrite()

#Generate list of files to process

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)

files = os.listdir(path + data_folder)
files = [f for f in files if f.endswith('.txt')]
files = natsorted(files)

#Run main processing
        
def main_for_z_scope(save_as):
    '''
    Parameters
    ----------
    save_as : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''    
    params, imgs, x_axis, max_distance, times, PP = get_IQ_data(save_as, path, data_folder)
    
    img_mean_filtered = mean_filter(imgs, n)
    imgs = np.rot90(img_mean_filtered, k=3)
    imgs = np.flip(imgs, axis=1)
    
    if len(imgs) != 0:
        with open(f"{path}/{path_save_loc}/z_scope_array/retracked_z_scope{save_as[:-4]}.txt", 'r') as f:
            r = [list(map(float, line.split())) for line in f]
        r_wf = []
        for i in r:
            r_wf += [i[0]]
            
        with open(f"{path}/{path_save_loc}/pp_array/pp_{save_as[:-4]}.txt", 'r') as f:
            pp = [list(map(float, line.split())) for line in f]
        PP = []
        for i in pp:
            PP += [i[0]]
            
    if plot_z_scopes:
        plotting_z_scopes(path, path_ssd, path_save_loc, save_as, times, imgs, sensor, n, max_distance, r_wf, PP, [-0.05, 12, 0, 0.2])
    
    return imgs, r_wf, max_distance
#Run processing for all files

def main():
    print("--- Started: z_scope.py ---\n")
    start_time_1 = time.time()
    for save_as in files:
        print(save_as)
        start_time = time.time()
        my_file = Path(f"{path}/{path_save_loc}/z_scope/ZSCOPE_{save_as[:-4]}.png")
        try:
            if rewrite == False:
                if my_file.is_file():
                    print("Path exists")
                else:
                    main_for_z_scope(save_as)
            else:
                main_for_z_scope(save_as)
        except ValueError:
            print("ValueError")
        except UnboundLocalError:
            print("Unbound Local Error")
        print("--- %s seconds ---\n" % (time.time() - start_time))

    print(f"--- Total Run Time: {round((time.time() - start_time_1)/60, 1)} minutes ---\n")

    print("--- Finished: z_scope.py ---\n")

if __name__ == '__main__':
    main()
