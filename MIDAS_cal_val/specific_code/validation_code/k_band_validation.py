#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:10:39 2026

@author: kftg4
"""

import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("../../../MIDAS_cal_val/src/sensor_mh_processing/")

import uRAD_RaspberryPi_SDK11.z_scope as k_z_scope

py_dir = add_path("../../../MIDAS_cal_val/")

from path_definitions import path_fun, radar_path_definitions, plotting_choices, py_choices, retrack_and_mf_choices_urad, rewrite
from stats_functions import  plot_calibration_figure, get_cal_data
from cal_val_functions import airborne_validation_plot, est_v_true_height

run_deconv, run_good_deconv, run_retrack, run_z_scope = py_choices()

sensor, retrack, n = retrack_and_mf_choices_urad()

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)
print(path, data_folder)
files = ['u_rad_data_20260408_180033.txt', 'u_rad_data_20260410_104447.txt', 'u_rad_data_20260410_105301.txt', 'u_rad_data_20260410_105440.txt', 'u_rad_data_20260410_110453.txt', 'u_rad_data_20260410_111141.txt', 'u_rad_data_20260410_112140.txt', 'u_rad_data_20260410_113004.txt']

heights = [(i * 4)+1 for i in [16, 14, 12, 10, 8, 6, 4, 2]]

cutoffs = [[1000,8500], [500,-1], [1,5700], [1250,9500], [1,7500], [300,8750], 
           [500,7500], [1,8000]]       
lims = [[[1850,3270],[5550,6840]], [[2090,3380],[5730,6720]], [[670,1980],[3900,5020]], 
        [[2200,3720],[5860,7450]], [[2800,4780],[6360,7050]], [[2350,4310],[6240,7750]], 
        [[2200,3510],[5440,6500]], [[1950,3510],[5700,7200]]]

def main():
    big_imgs = []
    big_ranges = []
    big_output = []
    for i, save_as in enumerate(files):
        imgs, r_wf, dist = k_z_scope.main_for_z_scope(save_as)
        
        big_imgs += [imgs]
        big_ranges += [r_wf]
    
        big_output += [airborne_validation_plot(False, i, None, heights[i], imgs, big_ranges[i], cutoffs[i], lims[i], [0.5, 5, 5], np.linspace(200, 400, 50)/100)]

    return heights, big_imgs, big_ranges, big_output

if __name__ == '__main__':
    heights, big_imgs, big_ranges, big_output = main()
    
    fig, axes = plt.subplots(nrows=len(files), ncols=2, figsize=(10, 5 * len(files)), constrained_layout=True)
    
    estimated = []
    for i, imgs in enumerate(big_imgs):
        output = [airborne_validation_plot(True, i, axes, heights[i], imgs, big_ranges[i], cutoffs[i], lims[i], [0.5, 5, 5], np.linspace(200, 400, 50)/100)]
        estimated += [output[0][0]]
        
    plt.tight_layout()
    plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/KBandBigPlotUncorr.png", dpi=300)
    plt.show()
    
    est_v_true_height(heights, estimated)
