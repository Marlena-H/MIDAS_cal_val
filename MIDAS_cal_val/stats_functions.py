#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:45:03 2026

@author: kftg4
"""
import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from laser_functions import read_data

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

def sensor_imports(sensor):
    '''
    Import sensor modules

    Parameters
    ----------
    sensor : str
        sensor name

    Returns
    -------
    None.

    '''
    
    #import relevant functions
    
    if sensor == "DreamRF":
        py_dir = add_path("../../../CPOMDrone/src/sensor_mh_processing/MMW-HAT-Release/")
        
        import z_scope
        import retrack as retracker
        import deconvolve
        
    if sensor == "uRAD":
        py_dir = add_path("../../../CPOMDrone/src/sensor_mh_processing/uRAD_RaspberryPi_SDK11/")

        import z_scope
        import retrack as retracker

def get_cal_data(path, path_save_loc, data_folder, files, trues, rerun=False):
    '''
    Collate all ground based calibration data. 

    Parameters
    ----------
    path : str
        path describing the location of the GitHub repo.
    path_save_loc : str
        path describing the location where data is saved
    data_folder : str
        location of where sensor data that needs to be processed can be found
    files : array
        array of files to process
    trues : array
        true ranges
    rerun : bool, optional
        ignore. The default is False.

    Returns
    -------
    trues : array
        true range
    means : array
        mean measured range
    ranges : nested array
        measured ranged per true range

    '''
    
    ranges = []
    
    for save_as in files:
        try:
            with open(f"{path}/{path_save_loc}/z_scope_array/retracked_z_scope{save_as[:-4]}.txt", 'r') as f:
                r = [list(map(float, line.split())) for line in f]                
            r_wf = []
            for i in r:
                r_wf += [i[0]]
            ranges += [r_wf]
        except:
            df = read_data(path, data_folder, save_as)
            ranges += [df["range"]]
                
    means = np.array([np.nanmean(i) for i in ranges])
    mask = means != 0
    trues = trues[mask]
    means = means[mask]
    
    ranges = [r for r, m in zip(ranges, mask) if m]

    drop = []
    for i, mean in enumerate(means):
        if mean <= 0.25: #remove where retracker has failed
            drop.append(i)
            
    trues = np.array([v for i, v in enumerate(trues) if i not in drop])
    means = [v for i, v in enumerate(means) if i not in drop]
    ranges = [v for i, v in enumerate(ranges) if i not in drop]
        
    return trues, means, ranges

def plot_calibration_figure(trues, means, ranges, limits):
    '''
    Plot true vs measured range for the ground based test

    Parameters
    ----------
    trues : array
        true range
    means : array
        mean measured range
    ranges : nested array
        measured ranged per true range
    limits : array
        plot axes limits

    Returns
    -------
    None.

    '''
    xlim, ylim = limits
    
    fig = plt.figure(figsize=(12, 5))
    
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.05, right=0.925,
                           bottom=0.1, top=0.9,
                           wspace=0.1)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    
    for ax in (ax1, ax2):
        ax.set_aspect('equal', adjustable='box')
        ax.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Perfect Correlation")
        ax.set_xticks(np.linspace(0, 15, 11))
        ax.set_yticks(np.linspace(0, 15, 11))
        ax.set_xlim(xlim, ylim)
        ax.set_ylim(xlim, ylim)
        ax.set_xlabel("True Range (m)")
    
    ax1.set_ylabel("Retracked Range (m)")
    ax2.tick_params(left=False, labelleft=False)
    
    means = np.array(means)
    
    indices = np.logical_not(np.logical_or(np.isnan(trues), np.isnan(means)))
    indices = np.array(indices)
    ranges = np.array(ranges, dtype=object)
    
    ranges = ranges[indices]
    trues = trues[indices]
    means = means[indices]
    
    m, b, *_ = stats.linregress(trues, means)
    diff = trues - means
    diff = np.nanmean(diff)
    
    ax1.axline(xy1=(0, b), slope=m, ls="--", c="green", label="Returned Correlation")
    ax1.scatter(trues, means, marker="o", c="cornflowerblue", label="Retracked Range")
    ax1.legend()
    
    ax2.axline(xy1=(0, b), slope=m, ls="--", c="green", label="Returned Correlation")
    ax2.scatter(trues, means, marker="o", c="cornflowerblue", label="Retracked Range")
    ax2.scatter(trues, means+diff, marker="o", c="red", label="Measurements with \nBias Correction")
    ax2.legend()
    
    plt.show()   
    
def calc_stats_ground(name, x, ranges):
    '''
    Calc stats for ground based test

    Parameters
    ----------
    name : str
        sensor name
    x : array
        true range
    ranges : nested array
        measured ranges per true range

    Returns
    -------
    summary : df
        dataframe containing overall summary stats
    per_measurement : df
        dataframe containing stats for each measurement set

    '''
    n = 5
    means = np.asarray([np.nanmean(row) for row in ranges])
    print(list(means))
    
    mask = ~np.isnan(means) & (means != 0)

    ranges = [r for r, m in zip(ranges, mask) if m]
    x = [xi for xi, m in zip(x, mask) if m]
    
    '''
    if name == "V Band":
        offset = 0.2
        print(offset)
        ranges = [np.array([float(x) for x in row]) + offset for row in ranges]'''
        
    means = np.asarray([np.nanmean(row) for row in ranges])
    within_std = np.asarray([np.nanstd(row) for row in ranges])
    n_samples = np.asarray([np.sum(~np.isnan(r)) for r in ranges])

    x = np.asarray(x)
    means = np.asarray(means)
    print(means)
    mask = ~np.isnan(x) & ~np.isnan(means)
    slope, intercept, r, p, se = stats.linregress(x[mask], means[mask])
    errors = means - x
    bias = np.nanmean(errors)
    rmse = np.sqrt(np.nanmean(errors**2))
    mae = np.nanmean(np.abs((errors)))
    mape = np.nanmean(np.abs((errors) / x)) * 100
    
    num_sum = 0
    for i in range(len(ranges)):
        num_sum += (n_samples[i] - 1)*(within_std[i]**2)
        
    mstd = np.sqrt(num_sum/(np.sum(n_samples)-len(ranges)))
    
    mcv = (mstd / np.nanmean(means)) * 100
    
    std = np.nanstd(means - x)
    cv = np.nanmean(std/np.nanmean(means))
    r2 = r**2
    
    means_wo_bias = means - bias
    errors_wo_bias = means_wo_bias - x
    rmse_wo_bias = np.sqrt(np.nanmean(errors_wo_bias**2))
    
    print(f"=== SENSOR: {name} ===")
    
    print("\n-- LINEARITY --\n")
    print(f"Slope: {slope:.{n}f} ")
    print(f"Intercept: {intercept:.{n}f} m")
    print(f"R: {r:.{n}f}")
    print(f"R2: {r2:.{n}f}")

    print("\n-- ACCURACY --\n")
    print(f"Mean Error (Bias): {bias:.{n}f} m")
    print(f"RMSE: {rmse:.{n}f}")
    print(f"RMSE (Bias Corrected) {rmse_wo_bias:.{n}f}")
    print(f"MAE: {mae:.{n}f}")
    print(f"MAPE: {mape:.{n}f} %")
    
    print("\n-- PRECISION --\n")
    print(f"Mean SD: {mstd:.{n}f}")
    print(f"Mean CV: {mcv:.{n}f} %")
    print(f"SD: {std:.{n}f}")
    print(f"CV: {cv:.{n}f} %")
    
    print("\n-- PER MEASURMENT RESULTS --\n")
    print(f"{'True (m)':>10} {'Mean (m)':>10} {'SD (m)':>10} {'CV':>10} {'No Points':>15}")
    for i in range(len(means)):
        print(f"{x[i]:>10.{n}f} {means[i]:>10.{n}f} {within_std[i]:>10.{n}f} {(within_std[i]/means[i])*100:>10.{n}f} % {n_samples[i]:>10}")

    summary = {
        "Sensor": name,
        "Slope": slope, "Intercept": intercept, "R": r, "R2": r2,
        "Bias": bias, "RMSE": rmse, "RMSE_BiasCorrected": rmse_wo_bias,
        "MAE": mae, "MAPE": mape,
        "Mean_SD": mstd, "Mean_CV": mcv, "SD": std, "CV": cv,
    }
    
    per_measurement = pd.DataFrame({
        "True (m)": x,
        "Mean (m)": means,
        "SD (m)": within_std,
        "CV (%)": (within_std/means)*100,
        "No Points": [len(r) for r in ranges],
    })
    
    return summary, per_measurement

def calc_stats_air(name, x, y):
    '''
    Calc stats for airborne test

    Parameters
    ----------
    name : str
        sensor name
    x : array
        true target heights
    y : array
        estimated target heights

    Returns
    -------
    summary : df
        dataframe containing overall summary stats
    per_measurement : df
        dataframe containing stats for each measurement set

    '''
    n = 5
    
    x = np.asarray(x)/100
    y = np.asarray(y)/100
    
    slope, intercept, r, p, se = stats.linregress(x, y)
    errors = y - x
    bias = np.nanmean(errors)
    rmse = np.sqrt(np.nanmean(errors**2))
    mae = np.nanmean(np.abs((errors)))
    mape = np.nanmean(np.abs((errors) / x)) * 100
    
    std = np.nanstd(y - x)
    cv = np.nanmean(std/np.nanmean(y))
    r2 = r**2
    
    y_wo_bias = y - bias
    errors_wo_bias = y_wo_bias - x
    rmse_wo_bias = np.sqrt(np.nanmean(errors_wo_bias**2))
    
    print(f"=== SENSOR: {name} ===")
    
    print("\n-- LINEARITY --\n")
    print(f"Slope: {slope:.{n}f} ")
    print(f"Intercept: {intercept:.{n}f} m")
    print(f"R: {r:.{n}f}")
    print(f"R2: {r2:.{n}f}")

    print("\n-- ACCURACY --\n")
    print(f"Mean Error (Bias): {bias:.{n}f} m")
    print(f"RMSE: {rmse:.{n}f}")
    print(f"RMSE (Bias Corrected) {rmse_wo_bias:.{n}f}")
    print(f"MAE: {mae:.{n}f}")
    print(f"MAPE: {mape:.{n}f} %")
    
    print("\n-- PRECISION --\n")
    print(f"SD: {std:.{n}f}")
    print(f"CV: {cv:.{n}f} %")
    
    print("\n-- PER MEASURMENT RESULTS --\n")
    print(f"{'True (m)':>10} {'Mean (m)':>10} {'Error (m)':>10}")
    for i in range(len(y)):
        print(f"{x[i]:>10.{n}f} {y[i]:>10.{n}f} {(y[i]-x[i]):>10.{n}f}")
    
    summary = {
        "Sensor": name,
        "Slope": slope, "Intercept": intercept, "R": r, "R2": r2,
        "Bias": bias, "RMSE": rmse, "RMSE_BiasCorrected": rmse_wo_bias,
        "MAE": mae, "MAPE": mape,
        "SD": std, "CV": cv,
    }
    
    per_measurement = pd.DataFrame({
        "True (m)": x,
        "Mean (m)": y,
        "Error (m)": y - x,
        "Relative Error (%)": (y - x) / x * 100,
    })
    
    return summary, per_measurement
    
def print_stats(results):
    #Claude generated
    print("=== Sensor Performance Statistics ===\n")

    print("-- Accuracy --")
    for k in ['mean_error (bias)', 'mean_absolute_error', 'RMSE', 'MAPE (%)']:
        print(f"  {k:<30} {results[k]:.4f}")

    print("\n-- Linearity --")
    for k in ['slope', 'intercept', 'R2']:
        print(f"  {k:<30} {results[k]:.4f}")
        
    try:
        print("\n-- Precision --")
        for k in ['mean_SD', 'mean_CV (%)']:
            print(f"  {k:<30} {results[k]:.4f}")

        print("\n-- Bland-Altman --")
        for k in ['BA_bias', 'BA_upper_LoA', 'BA_lower_LoA']:
            print(f"  {k:<30} {results[k]:.4f}")
    
        print("\n-- Per-condition breakdown --")
        print(f"  {'True':>8}  {'Mean':>8}  {'SD':>8}  {'CV (%)':>8}")
        for row in results['per_condition']:
            print(f"  {row['true_value']:>8.3f}  {row['mean']:>8.3f}  {row['SD']:>8.3f}  {row['CV (%)']:>8.3f}") 
    except:
        print("\n-- Per-condition breakdown --")
        print(f"  {'True':>8}  {'Mean':>8}")
        for row in results['per_condition']:
            print(f"  {row['true_value']:>8.3f}  {row['mean']:>8.3f}") 
        