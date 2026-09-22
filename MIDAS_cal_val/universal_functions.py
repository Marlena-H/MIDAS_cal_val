#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:08:43 2026

@author: kftg4
"""

import numpy as np
import sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

def running_in_ide():
    '''
    check if running in ide (controls plot displays)

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return (
        'spyder' in sys.modules or
        'IPython' in sys.modules or
        'ipykernel' in sys.modules
    )

def mean_filter(imgs, n):
    '''
    Applies a mean filter of n waveforms to the 2D imgs array

    Parameters
    ----------
    imgs : array
        2D echogram data (power with distance and time)
    n : int
        number of waveforms to average 

    Returns
    -------
    img_mean_filtered : array
        mean filtered version of imgs
        2D echogram data (power with distance and time)

    '''
    
    img_mean_filtered = np.zeros(np.shape(imgs))

    #can I improve?
    if n == 1:
        img_mean_filtered = imgs
    else:
        for i in range(int(n/2), len(imgs)-(int(n/2))):
            wf_mn = imgs[i-int(n/2)]
            wf = imgs[i]
            wf_pn = imgs[i+int(n/2)]
            img_mean_filtered[i] = np.mean([wf_mn, wf, wf_pn], axis=0)
    
    return img_mean_filtered

def find_pp(wf_og):
    '''
    Finds the pulse peakiness of a waveform and outputs it as a single value.

    Parameters
    ----------
    wf_og : numpy array
        Input waveform

    Returns
    -------
    pp : float
        pulse peakiness for specified waveform

    '''
    try:
        pp = np.max(wf_og)/np.mean(wf_og)
    except RuntimeWarning:
        pp = np.nan

    return pp

def apply_retracking(wf, retrack, dist):
    '''
    Retracks a waveform based on the chosen retracker e.g. tcog and the range window distance.

    Parameters
    ----------
    wf : numpy array
        Input waveform
    retrack : string
        Retracker name
    dist : float
        Range window maximum value e.g. 2 (meters)

    Returns
    -------
    surf_r : float
        Retracked range

    '''
    if retrack == "tcog":
        try:
            lep, p_2, amplitude = tcog(wf, 0.7)
        except:
            p_2 = None

        if p_2 == None or p_2 == 2:
            surf_r = np.nan
        else:
            surf_r = lep*dist/len(wf)

    elif retrack == "tfmra":
        a = TFMRA(wf, TL=0.7, thresh_noise=0.2)
        a = a[0]

        if np.isnan(a):
            surf_r = np.nan
        else:
            surf_r = a*dist/len(wf)

    elif retrack == "find_peak":
        a = np.argmax(wf)
        surf_r = a*dist/len(wf)
        
    elif retrack == "ocog":
        lep, cog, width, amplitude = ocog(wf)
        surf_r = lep*dist/len(wf)
        
    else:
        print("Retracker Does Not Exist")
        sys.exit()
        
    return surf_r

def ocog(waveform_data):
    '''
    Function for Offset Centre of Gravity Retracker. Originally written in IDL by Ines Otosaka.

    Parameters
    ----------
    waveform_data : list
        Power of the waveform with bin number

    Returns
    -------
    lep : float
        Leading Edge Position
    cog : float
        Centre of Gravity
    width : float
        Leading Edge Width
    amplitude : float
        Amplitude of Retracked Point

    '''

    a = np.array(waveform_data, dtype=np.float64)
    b = a.shape

    if len(b) == 1:
        c = a**2
        d = a**4
        n = np.arange(b[0], dtype=np.float64)
        width = np.zeros(b[0])
        lep = np.zeros(b[0])
        cog = np.zeros(b[0])
        amplitude = np.zeros(b[0])


        amplitude = np.sqrt(np.sum(d) / np.sum(c))
        width = (np.sum(c)**2) / np.sum(d)
        cog = np.sum(n * c) / np.sum(c)
        lep = cog - width / 2

    return lep, cog, width, amplitude

def tcog(waveform_data, threshold):
    '''
    Function for Threshold Centre of Gravity Retracker. Originally written in IDL by Ines Otosaka.

    Parameters
    ----------
    waveform_data : numpy array
        Power of the waveform with bin number
    threshold : float
        Threshold for retracking

    Returns
    -------
    lep : float
        Leading edge position
    p_2 : float
        Trailing edge position
    amplitude : float
        Amplitude of retracked point

    '''
    a = np.array(waveform_data)
    b = a.shape

    if len(b) == 1:
        c = a**2
        d = a**4
        n = np.arange(b[0], dtype=np.float64)
        lep = np.zeros(b[0])
        amplitude = np.zeros(b[0])
        amplitude = np.sqrt(np.sum(d)/np.sum(c))

        condition = a > threshold * amplitude
        test_lep = np.where(condition)[0]

        i0 = test_lep[0]
        lep = i0 - 1 + ((threshold*amplitude)-a[i0-1])/(a[i0]-a[i0-1])

        end = a[i0+10:]
        condition_2 = end > (threshold * amplitude)
        j0 = np.where(~condition_2)[0]

        if len(j0) == 0:
            p_2 = None
        else:
            p_2 = i0+j0[0]+10

    return lep, p_2, amplitude

def TFMRA(waveform_data, TL=0.3, thresh_noise=0.05):
    '''
    Threshold First-Maximum Retracking Algorithm

    Parameters
    ----------
    waveform_data : TYPE
        Power of waveform with bin number
    TL : TYPE, float
        Threshold value. The default is 0.7.
    thresh_noise : TYPE, optional
        Threshold for noise. The default is 0.05.

    Returns
    -------
    result
        Position of retracked point

    '''

    #For RPi only
    n = 6
    n_samples_interp = len(waveform_data)*10
    dt = 1.0 / 10.0

    if waveform_data.ndim == 1:
        waveform_data = np.stack([waveform_data, waveform_data], axis=1)
    elif waveform_data.shape[0] < waveform_data.shape[1]:
        waveform_data = waveform_data.T

    n_samples, n_waveforms = waveform_data.shape

    for i in range(n_waveforms):
        max_val = np.max(waveform_data[:, i])
        if max_val > 0:
            waveform_data[:, i] /= max_val
        else:
            waveform_data[:, i] = 0

    therm_noise = np.mean(waveform_data[:n-1, :], axis=0)

    good_wfm = np.where(therm_noise <= thresh_noise)[0]
    bad_wfm = np.setdiff1d(np.arange(n_waveforms), good_wfm)

    interp_range = np.linspace(0, n_samples - 1, n_samples_interp)
    interp_waves = np.zeros((n_samples_interp, len(good_wfm)))

    for i, idx in enumerate(good_wfm):
        interp_func = interp1d(np.arange(n_samples), waveform_data[:, idx], kind='linear', fill_value='extrapolate')
        interp_waves[:, i] = interp_func(interp_range)

    kernel = np.ones(15) / 15
    smooth_waves = np.zeros_like(interp_waves)
    for i in range(interp_waves.shape[1]):
        smooth_waves[:, i] = np.convolve(interp_waves[:, i], kernel, mode='same')

    dP = np.gradient(smooth_waves, axis=0)

    n_ret_tfmra = np.full(len(good_wfm), np.nan)

    for i in range(len(good_wfm)):
        wave = smooth_waves[:, i]
        deriv = dP[:, i]

        P_max = np.where(wave > therm_noise[i] + thresh_noise)[0]

        flag_P_max = np.zeros_like(wave)
        flag_P_max[P_max] = 1

        stationary_pts = np.where(deriv <= 0)[0]
        flag_dP_stationary = np.zeros_like(wave)
        flag_dP_stationary[stationary_pts] = 1

        maxima = np.where((flag_P_max == 1) & (flag_dP_stationary == 1))[0]

        if len(maxima) == 0:
            return [np.nan]
        else:
            P_max_bin = maxima[0]

            dP_vals = [deriv[P_max_bin - 1], deriv[P_max_bin], deriv[P_max_bin + 1]]
            dP_max1 = np.min(dP_vals)

            P_max1_bin = (
                np.where(deriv == dP_max1)[0][0]
                if dP_max1 != 0 else P_max_bin
            )

            thresh_power = wave[P_max1_bin] * TL + therm_noise[i]
            n_hat_P = np.where(wave > thresh_power)[0]

            n_hat = n_hat_P[0]

            y1 = wave[n_hat - 1]
            y2 = wave[n_hat]

            frac = (TL - y1) / (y2 - y1)
            n_ret = (n_hat - 1) + frac
            n_ret_tfmra[i] = n_ret * dt

    result = np.full(n_waveforms, np.nan)
    result[good_wfm] = n_ret_tfmra

    return result

def plot_waveforms(path, path_ssd, path_save_loc, save_as, x_axis, Magnitude, surf_r, seq, limits):
    '''
    Generic code to plot single waveform

    Parameters
    ----------
    path : str
        path describing the location of the GitHub repo.
    path_ssd : str
        path describing the location of the ssd (if using)
    path_save_loc : str
        path describing the location where data is saved
    save_as : str
        filename (for saving under)
    x_axis : array
        array of range with bin number for mapping as range
    Magnitude : array
        waveform magnitude with bin number
    surf_r : float
        retracked range
    seq : int
        measurement number
    limits : array
        plot axes limits
        xmin, xmax, ymin, ymax = limits

    Returns
    -------
    None.

    '''
    
    xmin, xmax, ymin, ymax = limits
    
    plt.plot(x_axis, Magnitude)
    plt.axvline(surf_r, c="r")
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xlabel('Range (m)')
    plt.ylabel('Power')
    plt.title(f"{save_as[:-4]}_{seq}")

    try:
        plt.savefig(f"{path_ssd}/{path_save_loc}/wf_retracked/{save_as[:-4]}_{seq}.png")
    except FileNotFoundError:
        plt.savefig(f"{path}/{path_save_loc}/wf_retracked/{save_as[:-4]}_{seq}.png")
    except PermissionError:
        plt.savefig(f"{path}/{path_save_loc}/wf_retracked/{save_as[:-4]}_{seq}.png")

    if running_in_ide():
        plt.show()

    plt.close()

def prep_time_arr(sensor, times):
    '''
    Create time (x) array for z scope plot. 

    Parameters
    ----------
    sensor : str
        sensor name
    times : array
        time array from txt file

    Returns
    -------
    time_arr : array
        time array suitable for plotting

    '''
    if sensor == "uRAD":
        time_arr = [
            None if (ts is None or math.isnan(float(ts)))
            else (dt := datetime.fromtimestamp(float(ts))).strftime('%H:%M:%S.') + dt.strftime('%f')[0]
            for ts in times
        ]
    
    if sensor == "DreamRF":
        time_arr = []
        for row in times:
            t = row[-1]
            parts = t.split(':')
            if len(parts) != 3:
                continue
            hh, mm, ss = parts
            time_arr.append(f"{hh}:{mm}:{float(ss):.1f}")
    
    else:
        time_arr = times
        
    return time_arr
                
def plotting_z_scopes(path, path_ssd, path_save_loc, save_as, times, imgs, sensor, n, dist, r_wf, PP, limits):
    '''
    Generic radargram plot

    Parameters
    ----------
    path : str
        path describing the location of the GitHub repo.
    path_ssd : str
        path describing the location of the ssd (if using)
    path_save_loc : str
        path describing the location where data is saved
    save_as : str
        filename (for saving under)
    times : array
        time array for plotting
    imgs : array
        1D array of range with time or 2D array of power with distance and time.
    sensor : str
        sensor name
    n : int
        mean filter value
    dist : float
        Maximum range that the sensor can range to (for consistent aspect ratio)
    r_wf : array
        1D array of range with time
    PP : array
        1D array of PP with time
    limits : array
        plot axes limits
        xmin, xmax, ymin, ymax = limits

    Returns
    -------
    None.

    '''
    ymin, ymax, vmin, vmax = limits
    time_arr = prep_time_arr(sensor, times)
    
    if n == 1:
        r = r_wf
    else:
        time_arr = time_arr[int(n/2):-(int(n/2))]
        r = r_wf[int(n/2):-(int(n/2))]
        
    fs = 7
    fig, ax = plt.subplots(figsize=(12,12))
    
    cax = ax.imshow(imgs, cmap='viridis', vmin=vmin, vmax=vmax, aspect=15, extent=[0, np.shape(imgs)[1], dist, 0])
    
    if sensor != "uRAD_Indust":
        s = int(len(time_arr)/6)+1
        plt.xticks(ticks=np.arange(np.shape(imgs)[1]+1)[::s], labels=time_arr[::s], fontsize=fs)
        plt.yticks(fontsize=fs)

    cbar = fig.colorbar(cax, ax=ax, label='Power', shrink=0.2, location="left")
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label('Power ', fontsize=fs)
    
    plt.scatter(np.arange(0, len(r)), r, c="red", label="Retracked Surface", s=0.1)
    ax.set_aspect((np.shape(imgs)[1]/(ymax-ymin))/3.2)
    ax.set_title(f"{save_as[:-4]}" , fontsize=fs)
    ax.set_ylabel('Range (m)', fontsize=fs)
    ax.set_xlabel('Time', fontsize=fs)

    plt.ylim(ymax, ymin)
    plt.legend(loc='upper left', fontsize="xx-small")
    
    try:
        plt.savefig(f"{path_ssd}/{path_save_loc}/z_scope/ZSCOPE_{str(save_as)[:-4]}.png", bbox_inches='tight', pad_inches=0.2)
    except FileNotFoundError:
        plt.savefig(f"{path}/{path_save_loc}/z_scope/ZSCOPE_{str(save_as)[:-4]}.png", bbox_inches='tight', pad_inches=0.2)
    except PermissionError:
        plt.savefig(f"{path}/{path_save_loc}/z_scope/ZSCOPE_{str(save_as)[:-4]}.png", bbox_inches='tight', pad_inches=0.2)

    if running_in_ide():
        plt.show()

    plt.close()
