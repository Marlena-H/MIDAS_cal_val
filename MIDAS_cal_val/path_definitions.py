#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:17:58 2025

@author: kftg4
"""

import os
import sys

def path_fun(data_folder):
    '''
    Defines location of GitHub folder and name of data folder. The data folder should be found inside the path folder e.g.

    path = "/home/kftg4/Documents/Code/Software/Code/Drone/RPi Copy"
    data_folder = "/innsbruck_data"

    So data_folder location must be "/home/kftg4/Documents/Code/Software/Code/Drone/RPi Copy/innsbruck_data"

    Returns
    -------
    path : str
        Path describing the location of the GitHub repo.
    data_folder : TYPE
        Name of the data folder which contains the .bin data you wish to process. An existing folder for this can be found in the folder RPi Copy and any new folders should be places here too.

    '''
    hostname = os.uname().nodename

    if hostname == "raspberrypi":
        path = "/home/pbrennan/Documents/CPOMDrone_container"
    else:
        path = "/home/kftg4/Git Repositories/MIDAS_cal_val"

    return path, data_folder, hostname

def radar_path_definitions(sensor, retrack, n):
    '''
    Path definitions for radar

    Parameters
    ----------
    sensor : str
        sensor name
    retrack : str
        retracker type
    n : int
        mean filter value

    Returns
    -------
    path : str
        path describing the location of the GitHub repo.
    data_folder : str
        location of where sensor data that needs to be processed can be found
    path_ssd : str
        path describing the location of the ssd (if using)
    path_save_loc : str
        path describing the location where data is saved

    '''
    path, data_folder, _ = path_fun(f"/data_for_repo/to_process/sensor_data/{sensor}_data")
    path_ssd = "/media/kftg4/T9/CPOMDrone_container/"
    path_save_loc = f"output_for_repo/sensor_output/{sensor}_output/{retrack}/{retrack} Average {n}"
    
    return path, data_folder, path_ssd, path_save_loc

def mini_laser_path_definitions():
    '''
    Path definitions for laser

    Returns
    -------
    path : str
        path describing the location of the GitHub repo.
    data_folder : str
        location of where sensor data that needs to be processed can be found
    path_ssd : str
        path describing the location of the ssd (if using)
    path_save_loc : str
        path describing the location where data is saved

    '''
    path, data_folder, _ = path_fun("/data_for_repo/to_process/sensor_data/mini_laser_data")
    path_ssd = "/media/kftg4/T9/CPOMDrone_container/"
    path_save_loc = "output_for_repo/sensor_output/mini_laser_output"
    
    return path, data_folder, path_ssd, path_save_loc 

def dji_gps_path_definitions():
    '''
    Path definitions for DJI GPS data

    Returns
    -------
    path : str
        path describing the location of the GitHub repo.
    data_folder : str
        location of where sensor data that needs to be processed can be found
    path_ssd : str
        path describing the location of the ssd (if using)
    path_save_loc : str
        path describing the location where data is saved

    '''
    path, data_folder, _ = path_fun("/data_for_repo/to_process/sensor_data/dji_gps_data")
    path_ssd = "/media/kftg4/T9/CPOMDrone_container/"
    path_save_loc = "output_for_repo/sensor_output/dji_gps_data"
    
    return path, data_folder, path_ssd, path_save_loc 

def combined_path_definitions():
    '''    
    Path definitions for combined drone and sensor data

    Returns
    -------
    path : str
        path describing the location of the GitHub repo.
    data_folder : str
        location of where sensor data that needs to be processed can be found
    path_ssd : str
        path describing the location of the ssd (if using)
    path_save_loc : str
        path describing the location where data is saved

    '''
    path, data_folder, _ = path_fun("/data_for_repo/to_process/combined_data/combined_sensor_data")
    path_ssd = "/media/kftg4/T9/CPOMDrone_container/"
    path_save_loc = "output_for_repo/combined_sensor_data"
    
    return path, data_folder, path_ssd, path_save_loc 

def plotting_choices(plot_deconv_wfs=False, plot_retracked_wfs=False, plot_z_scopes=True, generate_gps_maps=False, generate_gps_and_radar_maps=False):
    '''
    Toggle what to plot and what not to plot

    '''
    '''Uncomment to plot waveforms from the first iteration of deconvolution - will slow processing down significantly'''
    #plot_deconv_wfs=True
    '''Uncomment to plot waveforms from the second iteration of deconvolution - will slow process down significantly'''
    #plot_retracked_wfs=True
    '''Uncomment to plot z scopes/echograms'''
    plot_z_scopes=False

    return plot_deconv_wfs, plot_retracked_wfs, plot_z_scopes, generate_gps_maps, generate_gps_and_radar_maps

def py_choices(run_deconv=False, run_good_deconv=False, run_retrack=False, run_z_scope=True):
    '''
    Toggle what processing steps to run
    
    '''
    run_deconv=True
    run_good_deconv=True
    run_retrack=True
    run_z_scope=True

    return run_deconv, run_good_deconv, run_retrack, run_z_scope

def retrack_and_mf_choices_dreamrf(retrack="TCOG", mf=3):
    '''
    Define retrack and mean filter values for dream rf v band radar

    Parameters
    ----------
    retrack : TYPE, optional
        retracker type. The default is "TCOG".
    mf : TYPE, optional
        mean filter value. The default is 3.

    Returns
    -------
    str
        sensor name
    retrack : str
        retracker type
    mf : int
        mean filter value

    '''
    #retrack = "find_peak"
    #mf = 5

    return "DreamRF", retrack, mf

def retrack_and_mf_choices_urad(retrack="find_peak", mf=1):
    '''
    Define retrack and mean filter values for k band radar

    Parameters
    ----------
    retrack : str, optional
        retracker type. The default is "find_peak".
    mf : int, optional
        mean filter value. The default is 1.

    Returns
    -------
    str
        sensor name
    retrack : str
        retracker type
    mf : int
        mean filter value

    '''
    #retrack = "OCOG"
    #mf = 5

    return "uRAD", retrack, mf

def retrack_and_mf_choices_urad_indust(retrack="find_peak", mf=1):
    '''
    Define retrack and mean filter values for anteral v band radar

    Parameters
    ----------
    retrack : str, optional
        retracker type. The default is "find_peak".
    mf : int, optional
        mean filter value. The default is 1.

    Returns
    -------
    str
        sensor name
    retrack : str
        retracker type
    mf : int
        mean filter value

    '''
    #retrack = "OCOG"
    #mf = 5

    return "uRAD_Indust", retrack, mf

def rewrite(rewrite=False):
    '''
    toggle whether to overwrite files / rerun or not

    '''
    rewrite = True
    return rewrite

def reverse_val(reverse=False):
    '''
    toggle whether to run in ascending or descending order
    
    '''
    #reverse = True
    return reverse