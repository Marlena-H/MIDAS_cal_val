#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:30:39 2026

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

py_dir = add_path("../../../MIDAS_cal_val/src/sensor_mh_processing/MMW-HAT-Release/")

import z_scope

py_dir = add_path("../../../MIDAS_cal_val/")

from path_definitions import path_fun, radar_path_definitions, plotting_choices, py_choices, retrack_and_mf_choices_dreamrf, rewrite
from stats_functions import  plot_calibration_figure, get_cal_data
from cal_val_functions import airborne_validation_plot, est_v_true_height

run_deconv, run_good_deconv, run_retrack, run_z_scope = py_choices()

sensor, retrack, n = retrack_and_mf_choices_dreamrf()

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)

files = ['mmw_spi_2m_20260511_113402_466.bin', 'mmw_spi_2m_20260511_112856_901.bin', 'mmw_spi_2m_20260511_112513_914.bin', 'mmw_spi_2m_20260511_111811_812.bin', 'mmw_spi_2m_20260511_111408_247.bin', 'mmw_spi_2m_20260511_111023_715.bin', 'mmw_spi_2m_20260511_110547_780.bin', 'mmw_spi_2m_20260511_110111_832.bin', 'mmw_spi_2m_20260511_105457_287.bin', 'mmw_spi_2m_20260511_105037_733.bin', 'mmw_spi_2m_20260511_104602_221.bin', 'mmw_spi_2m_20260511_104213_202.bin', 'mmw_spi_2m_20260511_103318_053.bin', 'mmw_spi_2m_20260511_103011_522.bin']

heights = [(i * 4)+1 for i in range(1, 15)]
heights.reverse()

cutoffs = [[1,1255], [850,1817], [350,1100], [120,870], [150,633], [150,945],
           [200,1070], [260,1010], [1,665], [300,1152], [400,1360], [300,1043], 
           [550,1270], [1,331]]   
lims = [[[610,810],[950,1180]], [[330,510],[700,900]], [[94,280],[490,660]], 
        [[92,260],[460,650]], [[100,303]], [[120,320],[520,740]], 
        [[140,360],[560,770]], [[85,300],[450,660]], [[60,260],[460,-1]], 
        [[110,335],[570,780]],[[230,500],[640,870]], [[110,290],[480,688]], 
        [[107,280],[490,668]], [[135,270]]]

#Import Data

def main():
    big_imgs = []
    big_ranges = []
    big_output = []
    for i, save_as in enumerate(files):
        
        imgs, r_wf, dist = z_scope.main_for_z_scope(save_as)
        
        big_imgs += [imgs]
        big_ranges += [r_wf]
        
        big_output += [airborne_validation_plot(False, i, None, heights[i], imgs, big_ranges[i], cutoffs[i], lims[i], [0.2, 2, 2], np.linspace(0, 150, 50)/100)]

    return heights, big_imgs, big_ranges, big_output


if __name__ == '__main__':
    heights, big_imgs, big_ranges, big_output = main()

    fig, axes = plt.subplots(nrows=len(files), ncols=2, figsize=(10, 5 * len(files)), constrained_layout=True)

    estimated = []
    for i, imgs in enumerate(big_imgs):
        
        output = [airborne_validation_plot(True, i, axes, heights[i], imgs, big_ranges[i], cutoffs[i], lims[i], [0.2, 2, 2], np.linspace(0, 150, 50)/100)]
        print(estimated)
        estimated += [output[0][0]]
        
    plt.tight_layout()
    plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/VBandBigPlotUncorr.png", dpi=300)
    plt.show()

    est_v_true_height(heights, estimated)