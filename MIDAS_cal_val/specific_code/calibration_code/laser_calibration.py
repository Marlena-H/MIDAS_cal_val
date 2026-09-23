#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:57:56 2026

@author: kftg4
"""
import os
import sys
import numpy as np

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("../../../MIDAS_cal_val/")
print(py_dir)

from path_definitions import path_fun, mini_laser_path_definitions, plotting_choices, py_choices, rewrite
from stats_functions import  plot_calibration_figure, get_cal_data

run_deconv, run_good_deconv, run_retrack, run_z_scope = py_choices()

path, data_folder, path_ssd, path_save_loc = mini_laser_path_definitions()

def main():
    trues = np.array([3.507, 3.247, 3.094, 2.992, 2.793, 2.654, 2.469, 2.311, 2.198, 2.038, 1.826, 1.654, 1.497, 1.399])
    files = ['mini_laser_data_20260306_115613.txt', 'mini_laser_data_20260306_115822.txt', 'mini_laser_data_20260306_115939.txt', 'mini_laser_data_20260306_120041.txt', 'mini_laser_data_20260306_120126.txt', 'mini_laser_data_20260306_121846.txt', 'mini_laser_data_20260306_121929.txt', 'mini_laser_data_20260306_122007.txt', 'mini_laser_data_20260306_122049.txt', 'mini_laser_data_20260306_122127.txt', 'mini_laser_data_20260306_122207.txt', 'mini_laser_data_20260306_122240.txt', 'mini_laser_data_20260306_122324.txt', 'mini_laser_data_20260306_122357.txt']
    
    trues, means, ranges = get_cal_data(path, path_save_loc, data_folder, files, trues)
    
    return trues, means, ranges
    
if __name__ == '__main__':
    trues, means, ranges = main()
    plot_calibration_figure(trues, means, ranges, [0, 4.5])