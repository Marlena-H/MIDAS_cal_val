#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:21:57 2026

@author: kftg4
"""

import os
import sys
import json
import numpy as np

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("src/sensor_srcs/MMW-HAT-Release/")

from utility.helper import find_setting_in_directory
from utility.mmw_cube_proc_v0 import CubeProcessor

def get_bin_data(path, path_ssd, path_save_loc, data_folder, save_as, plots, num_angle_bins):
    '''
    

    Parameters
    ----------
    path : str
        path describing the location of the GitHub repo.
    path_ssd : str
        path describing the location of the ssd (if using)
    path_save_loc : str
        path describing the location where data is saved
    data_folder : str
        location of where sensor data that needs to be processed can be found
    save_as : str
        filename
    plots : array
        toggle which plots to plot
    num_angle_bins : int
        number of angle bins

    Returns
    -------
    imgs : array
        2D array of power with range and time

    '''
    
    #Apply method from DreamRF code
    
    cfg_dir = f"{path}/MIDAS_cal_val/src/sensor_srcs/MMW-HAT-Release/radar_config/config_3rx_2m"
    proc_save_dir = f"{path}/{path_save_loc}/Iteration2/z_scope/"
    proc_save_dir_ssd = f"{path_ssd}/{path_save_loc}/Iteration2/z_scope/"
    
    imgs = []
    setting_fn = find_setting_in_directory(cfg_dir)
    with open(setting_fn, "r") as file:
        setting = json.load(file)
    mmw_proc = CubeProcessor(setting, num_azimuth_bin=num_angle_bins, num_elevation_bin=num_angle_bins)

    directory = os.path.dirname(proc_save_dir)
    if directory:  # Check if a directory is specified in the path
        os.makedirs(directory, exist_ok=True)

    # Open the file in binary read mode
    with open(f"{path}{data_folder}/{str(save_as)}", "rb") as file:
        while True:
            version_bytes = file.read(4)
            if not version_bytes:
                break
            version = int.from_bytes(version_bytes, byteorder="little", signed=False)  # Read the first N bytes
            if version == 0:
                seq = int.from_bytes(file.read(4), byteorder="little", signed=False)
                data_len = int.from_bytes(file.read(4), byteorder="little", signed=False)
                mmw_data = file.read(data_len)

                try:
                    mmw_proc.process_raw_data(mmw_data)
                except ValueError:
                    pass
                
                axis_0_name, axis_1_name = plots[0]
                img = mmw_proc.vis_2d(axis_0_name, axis_1_name)

                imgs += [img]
                
    return imgs

def sort_dist_issues(save_as):
    '''
    Sort issues with different distance parameters.

    '''
    save_as_2 = save_as
    dist = save_as[8:-24]

    if len(dist) == 0:
        dist = 2
    else:
        dist = dist[:-1]
        save_as = save_as[:7] + save_as[10:-4] + ".bin"

    #print(save_as)
    dist = int(dist)

    if dist == 2:
        dist = 2*(57/64)
    if dist == 5:
        dist = 5*(57/64)

    return save_as, save_as_2, dist

def deconv_waveform(wf_og, model):
    '''
    Deconvolves waveforms to remove dependance of power on range.

    Parameters
    ----------
    wf_og : numpy array
        Input waveform with dependance on range
    model : numpy array
        Model used to deconvolve, this can be generic or specific to a file or waveform

    Returns
    -------
    wf : numpy array
        Deconvolved waveform aligned with its mean at 0.1

    '''
    wf = (wf_og - model.flatten())
    wf = wf - min(wf)

    return wf

def apply_manual_cleaning_deconv_version(imgs_added, img_deconv, r_wf, pp_wf, dist):
    '''
    Apply manual cleaning for 1st deconvolution iteration. Generate imgs_added (array containing waveforms with no surface in range)

    '''
    for i in range(len(img_deconv)):
        wf = img_deconv[i]
        surf_r = r_wf[i]
        PP = pp_wf[i]
        
        if surf_r is np.nan and PP < 2:
            imgs_added[i] = True
        else:
            if surf_r > dist:
                r_wf[i] = np.nan
                imgs_added[i] = True
            elif surf_r < 0.15 and PP > 2 and np.nanmax(wf) > 0.25:
                r_wf[i] = 0
            elif surf_r < 0.15 and np.nanmax(wf) < 0.35:
                r_wf[i] = np.nan
                imgs_added[i] = True
            elif  PP < 2 and np.nanmax(wf) < 0.35:
                r_wf[i] = np.nan
                imgs_added[i] = True
            elif PP < 2 or np.nanmax(wf) < 0.5:
                r_wf[i] = np.nan
            else:
                pass

    return imgs_added, r_wf

def apply_manual_cleaning(img_mean_filtered, imgs, r_wf, pp_wf, dist):
    '''
    Apply manual cleaning for 2ND deconvolution iteration and retracking. 

    '''
    for i in range(len(img_mean_filtered)):
        wf = img_mean_filtered[i]
        img = imgs[i]
        surf_r = r_wf[i]
        PP = pp_wf[i]

        if surf_r > dist:
            r_wf[i] = np.nan
            
        elif surf_r < 0.15 and PP > 2 and np.nanmax(wf) > 0.25:
            r_wf[i] = 0
            
        elif surf_r < 0.15 and np.nanmax(wf) < 0.35:
            r_wf[i] = np.nan
            
        elif  PP < 2 and np.nanmax(wf) < 0.35:
            r_wf[i] = np.nan
            
        elif PP < 2 or np.nanmax(wf) < 0.5:
            r_wf[i] = np.nan
            
        else:
            pass
        
    return r_wf