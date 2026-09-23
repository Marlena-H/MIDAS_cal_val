#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:19:52 2026

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

from path_definitions import path_fun, radar_path_definitions, plotting_choices, py_choices, retrack_and_mf_choices_dreamrf, rewrite
from stats_functions import  plot_calibration_figure, get_cal_data

run_deconv, run_good_deconv, run_retrack, run_z_scope = py_choices()

sensor, retrack, n = retrack_and_mf_choices_dreamrf()

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)

def main():
    trues = np.array([88.7, 93.0, 98.6, 102.8, 107.1, 111.6, 116.8, 121.2, 127.2, 131.7, 42.8, 48.8, 52.6, 56.5, 61.2, 70, 76.0, 83.9])/100
    files = ['mmw_spi_20250730_121357_985.bin', 'mmw_spi_20250730_121554_274.bin', 'mmw_spi_20250730_121741_302.bin', 'mmw_spi_20250730_121924_362.bin', 'mmw_spi_20250730_122106_665.bin', 'mmw_spi_20250730_122251_242.bin', 'mmw_spi_20250730_122422_038.bin', 'mmw_spi_20250730_122602_034.bin', 'mmw_spi_20250730_122759_164.bin', 'mmw_spi_20250730_122950_857.bin', 'mmw_spi_20250730_123347_361.bin', 'mmw_spi_20250730_123542_868.bin', 'mmw_spi_20250730_123719_516.bin', 'mmw_spi_20250730_123851_252.bin', 'mmw_spi_20250730_134121_844.bin', 'mmw_spi_20250730_134250_558.bin', 'mmw_spi_20250730_134406_744.bin', 'mmw_spi_20250730_134554_195.bin']
    
    trues, means, ranges = get_cal_data(path, f"{path_save_loc}/Iteration2", data_folder, files, trues)
    
    return trues, means, ranges

if __name__ == '__main__':
    trues, means, ranges = main()

    plot_calibration_figure(trues, means, ranges, [0, 2])
