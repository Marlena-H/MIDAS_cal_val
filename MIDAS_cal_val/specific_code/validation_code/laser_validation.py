#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:39:45 2026

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

py_dir = add_path("../../../MIDAS_cal_val/")

from path_definitions import path_fun, mini_laser_path_definitions, plotting_choices, py_choices, rewrite
from stats_functions import  plot_calibration_figure, get_cal_data
from cal_val_functions import airborne_validation_plot, est_v_true_height
from laser_functions import read_data

path, data_folder, path_ssd, path_save_loc = mini_laser_path_definitions()

files = ['mini_laser_data_20260511_113355.txt', 'mini_laser_data_20260511_112849.txt', 'mini_laser_data_20260511_112506.txt', 'mini_laser_data_20260511_111804.txt', 'mini_laser_data_20260511_111400.txt', 'mini_laser_data_20260511_111016.txt', 'mini_laser_data_20260511_110540.txt', 'mini_laser_data_20260511_110104.txt', 'mini_laser_data_20260511_105421.txt', 'mini_laser_data_20260511_105030.txt', 'mini_laser_data_20260511_104554.txt', 'mini_laser_data_20260511_104205.txt', 'mini_laser_data_20260511_103310.txt', 'mini_laser_data_20260511_102836.txt']
heights = [(i * 4)+1 for i in range(1, 15)]
heights.reverse()
cutoffs = [[130,400], [300,570], [50,350], [0,268], [20,201], [10,268], [30,334], 
           [30,314], [50,268], [50,268], [100,402], [50,335], [100,401], [50,370], 
           [50,370]]       
lims = [[[73,129],[187,249]], [[74,126],[197,248]], [[100,158],[218,273]], 
        [[79,130],[198,253]], [[70,133]], [[89,150],[220,-1]], [[91,160],[222,282]],
        [[95,153],[207,265]], [[61,127],[191,-1]], [[92,165]], [[103,184],[237,-1]], 
        [[92,147],[221,278]], [[107,153],[227,280]], [[124,194],[275,311]]]

def main():
    big_ranges = []
    big_output = []
    for i, save_as in enumerate(files):
        
        df = read_data(path, data_folder, save_as)
    
        big_ranges += [np.array(df["range"])]
        
        big_output += [airborne_validation_plot(False, i, None, heights[i], None, big_ranges[i], cutoffs[i], lims[i], [0, 5, 5], np.linspace(50, 150, 50)/100)]
        
    return heights, big_ranges, big_output

if __name__ == '__main__':
    heights, big_ranges, big_output = main()    

    fig, axes = plt.subplots(nrows=len(files), ncols=2, figsize=(10, 5 * len(files)), constrained_layout=True)

    estimated = []
    for i, r_wf in enumerate(big_ranges):
        
        output = [airborne_validation_plot(True, i, axes, heights[i], None, big_ranges[i], cutoffs[i], lims[i], [0, 5, 5], np.linspace(50, 150, 50)/100)]
        estimated += [output[0][0]]
        
    plt.tight_layout()
    plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/LaserBigPlotUncorr.png", dpi=300)
    plt.show()
    
    est_v_true_height(heights, estimated)