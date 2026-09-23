"""
Created on Wed Apr 30 08:16:42 2025

@author: Marlena Holloway
"""

import os
import sys
import numpy as np
import time
from pathlib import Path
from natsort import natsorted
#Import all required paths and functions

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("../../..")

from path_definitions import path_fun, radar_path_definitions, plotting_choices, py_choices, retrack_and_mf_choices_dreamrf, rewrite
from mmw_functions import get_bin_data, sort_dist_issues, deconv_waveform
from universal_functions import plotting_z_scopes, mean_filter
run_deconv, run_good_deconv, run_retrack, run_z_scope = py_choices()

py_dir = add_path("../../../src/sensor_srcs/MMW-HAT-Release/")

plot_deconv_wfs, plot_retracked_wfs, plot_z_scopes, generate_gps_maps, generate_gps_and_radar_maps = plotting_choices()
sensor, retrack, n = retrack_and_mf_choices_dreamrf()

rewrite = rewrite()

path, data_folder, path_ssd, path_save_loc = radar_path_definitions(sensor, retrack, n)

#Generate list of files to process

files = os.listdir(path + data_folder)
files = [f for f in files if f.endswith('.bin')]
files = ['mmw_spi_20250730_121357_985.bin', 'mmw_spi_20250730_121554_274.bin', 'mmw_spi_20250730_121741_302.bin', 'mmw_spi_20250730_121924_362.bin', 'mmw_spi_20250730_122106_665.bin', 'mmw_spi_20250730_122251_242.bin', 'mmw_spi_20250730_122422_038.bin', 'mmw_spi_20250730_122602_034.bin', 'mmw_spi_20250730_122759_164.bin', 'mmw_spi_20250730_122950_857.bin', 'mmw_spi_20250730_123347_361.bin', 'mmw_spi_20250730_123542_868.bin', 'mmw_spi_20250730_123719_516.bin', 'mmw_spi_20250730_123851_252.bin', 'mmw_spi_20250730_134121_844.bin', 'mmw_spi_20250730_134250_558.bin', 'mmw_spi_20250730_134406_744.bin', 'mmw_spi_20250730_134554_195.bin']
files = ['mmw_spi_2m_20260511_113402_466.bin', 'mmw_spi_2m_20260511_112856_901.bin', 'mmw_spi_2m_20260511_112513_914.bin', 'mmw_spi_2m_20260511_111811_812.bin', 'mmw_spi_2m_20260511_111408_247.bin', 'mmw_spi_2m_20260511_111023_715.bin', 'mmw_spi_2m_20260511_110547_780.bin', 'mmw_spi_2m_20260511_110111_832.bin', 'mmw_spi_2m_20260511_105457_287.bin', 'mmw_spi_2m_20260511_105037_733.bin', 'mmw_spi_2m_20260511_104602_221.bin', 'mmw_spi_2m_20260511_104213_202.bin', 'mmw_spi_2m_20260511_103318_053.bin', 'mmw_spi_2m_20260511_103011_522.bin']

files = natsorted(files)

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

    save_as, save_as_2, dist = sort_dist_issues(save_as)

    imgs = get_bin_data(path, path_ssd, path_save_loc, data_folder, save_as_2, [("Azimuth", "Range")], 16)
    imgs = np.log10(imgs)
    imgs = np.mean(imgs, axis=1)
    
    if run_good_deconv:
        try:
            with open(f"{path}/{path_save_loc[:-1]}1/Iteration/normalizer/normalizer_{save_as[:-4]}.txt", 'r') as ref_f:
                ref = [list(map(float, line.split())) for line in ref_f]
        except:
            with open(f"{path}/{path_save_loc[:-1]}1/Iteration/normalizer/normalizer_{save_as_2[:-4]}.txt", 'r') as ref_f:
                ref = [list(map(float, line.split())) for line in ref_f]
                
    with open(f"{path}/MIDAS_cal_val/src/sensor_mh_processing/MMW-HAT-Release/ref_wf.txt", "r") as f:
        ref = [list(map(float, line.split())) for line in f]
        
    model = np.array(ref).reshape(57,)
    img_deconv = np.apply_along_axis(deconv_waveform, 1, imgs, model=model)

    if not run_good_deconv and not run_deconv:
        img_deconv = imgs

    img_mean_filtered = mean_filter(img_deconv, n)

    imgs = np.rot90(img_mean_filtered, k=3)
    imgs = np.flip(imgs, axis=1)
    
    if len(img_deconv) != 0:
        try:
            with open(f"{path}/{path_save_loc}/Iteration2/z_scope_array/retracked_z_scope{save_as[:-4]}.txt", 'r') as f:
                r = [list(map(float, line.split())) for line in f]
        except:
            with open(f"{path}/{path_save_loc}/Iteration2/z_scope_array/retracked_z_scope{save_as_2[:-4]}.txt", 'r') as f:
                r = [list(map(float, line.split())) for line in f]
        try:
            with open(f"{path}/{path_save_loc}/Iteration2/pp_array/retracked_z_scope{save_as[:-4]}.txt", 'r') as f:
                pp = [list(map(float, line.split())) for line in f]
        except:
            with open(f"{path}/{path_save_loc}/Iteration2/pp_array/retracked_z_scope{save_as_2[:-4]}.txt", 'r') as f:
                pp = [list(map(float, line.split())) for line in f] 
        r_wf = []
        for i in r:
            r_wf += [i[0]]
        PP = []
        for i in pp:
            PP += [i[0]]            
        with open(f"{path}{data_folder}/{save_as_2[:-4]}.txt") as f:
            times = []
            for line in f:
                times += [line.split()]
    
    if plot_z_scopes:
        plotting_z_scopes(path, path_ssd, f"{path_save_loc}/Iteration2", save_as, times, imgs, sensor, n, dist, r_wf, PP, [0.05, 1.78, 0, 1])
        
    return imgs, r_wf, PP

def main():
    print("--- Started: z_scope.py ---\n")
    start_time_1 = time.time()
    for save_as in files:
        start_time = time.time()
        my_file = Path(f"{path}/{path_save_loc}/Iteration2/z_scope/ZSCOPE_{save_as[:-4]}.png")
        print(save_as)
        try:
            if rewrite == False:
                if my_file.is_file():
                    print("Path exists")
                else:
                    main_for_z_scope(save_as)
            else:
                main_for_z_scope(save_as)
        except ValueError:
            print("Value Error")
        except UnboundLocalError:
            print("Unbound Local Error")
        print("--- %s seconds ---\n" % (time.time() - start_time))

    print(f"--- Total Run Time: {round((time.time() - start_time_1)/60, 1)} minutes ---\n")

    print("--- Finished: z_scope.py ---\n")

if __name__ == '__main__':
    main()

