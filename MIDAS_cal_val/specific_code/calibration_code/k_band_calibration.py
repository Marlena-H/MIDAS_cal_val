#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:02:55 2026

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

from path_definitions import path_fun, radar_path_definitions, plotting_choices, py_choices, retrack_and_mf_choices_urad, rewrite
from stats_functions import  plot_calibration_figure, get_cal_data

run_deconv, run_good_deconv, run_retrack, run_z_scope = py_choices()

sensor, retrack, n = retrack_and_mf_choices_urad()

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)

def main():
    trues = np.array([1.656, 1.656, 1.805, 1.947, 2.097, 2.821, 2.672, 3.479, 4.324, 5.186, 6.044, 2.426, 3.182, 3.670, 5.641, 7.649])
    files = ['u_rad_data_20260107_160849.txt', 'u_rad_data_20260107_160939.txt', 'u_rad_data_20260107_161038.txt', 'u_rad_data_20260107_161514.txt', 'u_rad_data_20260107_161607.txt', 'u_rad_data_20260107_161648.txt', 'u_rad_data_20260107_162004.txt', 'u_rad_data_20260107_162148.txt', 'u_rad_data_20260107_162314.txt', 'u_rad_data_20260107_162516.txt', 'u_rad_data_20260107_162724.txt', 'u_rad_data_20260108_114953.txt', 'u_rad_data_20260108_115128.txt', 'u_rad_data_20260108_115357.txt', 'u_rad_data_20260108_115508.txt', 'u_rad_data_20260108_115614.txt']
    
    trues, means, ranges = get_cal_data(path, path_save_loc, data_folder, files, trues)
    
    return trues, means, ranges

if __name__ == '__main__':
    trues, means, ranges = main()

    plot_calibration_figure(trues, means, ranges, [0, 10])