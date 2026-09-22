#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:54:22 2026

@author: kftg4
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def crop_data_1d(imgs, r_wf, cutoff, lim, limits):
    '''
    Clean and crop 1D (e.g. laser) data to only include relevant sortie data and remove spurious signal
    Split 1D data into ground and target sections
    Calculate the estimated target height from the data

    Parameters
    ----------
    imgs : None
        None - as 1D array only
    r_wf : array
        1D array of range with time
    cutoff : array
        manually defined start and end cutoff values (indexes) for file i
        allows for removal of e.g. landing and takeoff sections
    lim : array
        manually defined indexes describing transition from ground to target and vice versa for file i
    limits : array
        decribe ranges to be filtered out e.g. low = 0.2, all ranges less than 0.2 will later be removed
        also referes to the ymax limit when plotting
        low, high, ymax = limits

    Returns
    -------
    estimated : float
        estimated target height
    range_g : array
        1D array of ranges associated with ground
    range_t : array
        1D array of ranges associated with the target
    range_g_dev : array
        1D array of ranges associated with the ground, with the ground aligned with 1 meter
    range_t_dev : array
        1D array of ranges associated with the target, shifted with respect to the ground which has been aligned with 1 meter
    xarr_g : TYPE
        1D array of indexes matching the ranges associated with ground
    xarr_t : TYPE
        1D array of indexes matching the ranges associated with target

    '''
    
    low, high, ymax = limits
    
    r_wf = np.array(r_wf)
    
    #remove any ranges that fall outside of the chosen low and high values
    r_wf[r_wf<low] = np.nan
    r_wf[r_wf>high] = np.nan
    
    #cuts out echogram data not relevant to study e.g. take-off / landing, leaving only clean sortie data over the ground / target 
    r_wf = r_wf[cutoff[0]:cutoff[1]]
    imgs = imgs[cutoff[0]:cutoff[1]]
    xarr = np.arange(0, len(r_wf))
    
    xarr = list(xarr)
    r_wf = list(r_wf)
    
    t = 2 #transition to cut out
    
    #if the target is surveyed twice
    if len(lim) == 2:
        
        #remove transition areas from ground to target and vice versa
        r_wf[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        r_wf[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        r_wf[lim[1][0]-t:lim[1][0]+t] = [np.nan]*2*t
        r_wf[lim[1][1]-t:lim[1][1]+t] = [np.nan]*2*t
        xarr[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        xarr[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        xarr[lim[1][0]-t:lim[1][0]+t] = [np.nan]*2*t
        xarr[lim[1][1]-t:lim[1][1]+t] = [np.nan]*2*t
        
        #splite echogram data into target and ground data
        range_g = r_wf[:lim[0][0]] + r_wf[lim[0][1]:lim[1][0]] + r_wf[lim[1][1]:]
        xarr_g = xarr[:lim[0][0]] + xarr[lim[0][1]:lim[1][0]] + xarr[lim[1][1]:]
        range_t = r_wf[lim[0][0]:lim[0][1]] + r_wf[lim[1][0]:lim[1][1]]
        xarr_t = xarr[lim[0][0]:lim[0][1]] + xarr[lim[1][0]:lim[1][1]]
    
    #if the target is only surveyed once
    if len(lim) == 1:
        
        #remove transition areas from ground to target and vice versa
        r_wf[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        r_wf[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        xarr[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        xarr[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        
        #splite echogram data into target and ground data
        range_g = r_wf[:lim[0][0]] + r_wf[lim[0][1]:]
        xarr_g = xarr[:lim[0][0]] + xarr[lim[0][1]:]
        range_t= r_wf[lim[0][0]:lim[0][1]] 
        xarr_t = xarr[lim[0][0]:lim[0][1]]
        
    range_g = np.array(range_g)
    xarr_g = np.array(xarr_g)
    
    mask = ~np.isnan(range_g)
    range_g = range_g[mask]
    xarr_g = xarr_g[mask]
    
    #align ground and target range so that the mean ground range is 1 (for easier comparison between sorties)
    range_g_dev = range_g - np.nanmean(range_g)
    range_t_dev = range_t - np.nanmean(range_t) + 1
    
    #calculate estimated target height
    estimated = (np.nanmean(range_g)-np.nanmean(range_t))*100
    
    return estimated, range_g, range_t, range_g_dev, range_t_dev, xarr_g, xarr_t

def crop_data_2d(imgs, r_wf, cutoff, lim, limits):
    '''
    Clean and crop 2D (e.g. radar) data to only include relevant sortie data and remove spurious signal
    Split 2D data into ground and target sections
    Calculate the estimated target height from the data

    Parameters
    ----------
    imgs : array
        2D echogram data (power with distance and time)
    r_wf : array
        1D array of range with time
    cutoff : array
        manually defined start and end cutoff values (indexes) for file i
        allows for removal of e.g. landing and takeoff sections
    lim : array
        manually defined indexes describing transition from ground to target and vice versa for file i
    limits : array
        decribe ranges to be filtered out e.g. low = 0.2, all ranges less than 0.2 will later be removed
        also referes to the ymax limit when plotting
        low, high, ymax = limits
        
    Returns
    -------
    estimated : float
        estimated target height
    range_g : array
        1D array of ranges associated with ground
    range_t : array
        1D array of ranges associated with the target
    range_g_dev : array
        1D array of ranges associated with the ground, with the ground aligned with 1 meter
    range_t_dev : array
        1D array of ranges associated with the target, shifted with respect to the ground which has been aligned with 1 meter
    xarr_g : TYPE
        1D array of indexes matching the ranges associated with ground
    xarr_t : TYPE
        1D array of indexes matching the ranges associated with target

    '''
    low, high, ymax = limits
    
    r_wf = np.array(r_wf)
    
    #remove any ranges that fall outside of the chosen low and high values
    r_wf[r_wf<low] = np.nan
    r_wf[r_wf>high] = np.nan
    
    #cuts out echogram data not relevant to study e.g. take-off / landing, leaving only clean sortie data over the ground / target 
    r_wf = r_wf[cutoff[0]:cutoff[1]]
    imgs = imgs[:,cutoff[0]:cutoff[1]]
    
    #3xception for if the survey ended whilst over the target (due to delay in saving data)
    start = cutoff[0]
    if cutoff[1] == -1:
        end = cutoff[0] + len(r_wf)
    else:
        end = cutoff[1]
    xarr = np.arange(start, end)
    
    xarr = list(xarr)
    r_wf = list(r_wf)
    
    t = 5 #transition to cut out
    if len(lim) == 2:
        
        #remove transition areas from ground to target and vice versa
        r_wf[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        r_wf[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        r_wf[lim[1][0]-t:lim[1][0]+t] = [np.nan]*2*t
        r_wf[lim[1][1]-t:lim[1][1]+t] = [np.nan]*2*t
        xarr[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        xarr[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        xarr[lim[1][0]-t:lim[1][0]+t] = [np.nan]*2*t
        xarr[lim[1][1]-t:lim[1][1]+t] = [np.nan]*2*t
        
        #splite echogram data into target and ground data
        range_g = r_wf[:lim[0][0]] + r_wf[lim[0][1]:lim[1][0]] + r_wf[lim[1][1]:]
        xarr_g = xarr[:lim[0][0]] + xarr[lim[0][1]:lim[1][0]] + xarr[lim[1][1]:]
        range_t = r_wf[lim[0][0]:lim[0][1]] + r_wf[lim[1][0]:lim[1][1]]
        xarr_t = xarr[lim[0][0]:lim[0][1]] + xarr[lim[1][0]:lim[1][1]]
    
    if len(lim) == 1:
        
        #remove transition areas from ground to target and vice versa
        r_wf[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        r_wf[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        xarr[lim[0][0]-t:lim[0][0]+t] = [np.nan]*2*t
        xarr[lim[0][1]-t:lim[0][1]+t] = [np.nan]*2*t
        
        #splite echogram data into target and ground data
        range_g = r_wf[:lim[0][0]] + r_wf[lim[0][1]:]
        xarr_g = xarr[:lim[0][0]] + xarr[lim[0][1]:]
        range_t = r_wf[lim[0][0]:lim[0][1]] 
        xarr_t = xarr[lim[0][0]:lim[0][1]]

    range_g = np.array(range_g)
    xarr_g = np.array(xarr_g)
    
    mask = ~np.isnan(range_g)
    range_g = range_g[mask]
    xarr_g = xarr_g[mask]
        
    #align ground and target range so that the mean ground range is 1 (for easier comparison between sorties)
    range_g_dev = range_g - np.nanmean(range_g)
    range_t_dev = range_t - np.nanmean(range_t) + 1
    
    #calculate estimated target height
    estimated = (np.nanmean(range_g)-np.nanmean(range_t))*100
    
    return estimated, range_g, range_t, range_g_dev, range_t_dev, xarr_g, xarr_t

def big_plot(i, axes, xarr_g, range_g, xarr_t, range_t, ymax, bins, height):
    '''
    Plots airborne validation echogram and histogram data

    Parameters
    ----------
    i : int
        subplot row identifier
    axes : matplotlib.axes
        figure axes
    xarr_g : TYPE
        1D array of indexes matching the ranges associated with ground
    range_g : array
        1D array of ranges associated with ground
    xarr_t : TYPE
        1D array of indexes matching the ranges associated with target
    range_t : array
        1D array of ranges associated with the target
    ymax : float
        ymax limit for plotting
    bins : array
        bins for histogram
    height : float
        true target height for file i

    Returns
    -------
    None.

    '''
    
    ax_left  = axes[i, 0]  #First plot
    ax_right = axes[i, 1]  #Second plot
    
    #plot retracked ranges with time
    ax_left.scatter(xarr_g, range_g, s=0.2, c="red")
    ax_left.scatter(xarr_t, range_t, s=1, c="darkorange")
    ax_left.set_ylim(ymax, 0)
    ax_left.set_xlabel("Measurement Number")
    ax_left.set_ylabel("Range (m)")
    ax_left.legend()
    
    #plot histogram of distributions
    ax_right.hist(range_g, bins=bins,alpha=0.7, color="red")
    ax_right.axvline(np.nanmean(range_g), linestyle="--", color="red", label=f"Ground: {round(np.nanmean(range_g), 4)} \u00B1 {round(np.nanstd(range_g), 4)} m")
    ax_right.hist(range_t, bins=bins,alpha=0.7, color="darkorange")
    ax_right.axvline(np.nanmean(range_t), linestyle="--", color="darkorange", label=f"Freeboard: {round(np.nanmean(range_t), 4)} \u00B1 {round(np.nanstd(range_t), 4)} m")
    ax_right.set_ylabel("Frequency")
    ax_right.set_xlabel("Range (m)")
    ax_right.legend()
    
    ax_left.set_title(f"{height} cm Freeboard")
    ax_right.set_title(f"Estimated {round((np.nanmean(range_g)-np.nanmean(range_t))*100, 4)} cm Freeboard")
        
def airborne_validation_plot(plot, i, axes, height, imgs, r_wf, cutoff, lim, limits, bins):
    '''
    Finds estimated target height from the airborne validation test
    Can also plot airborne validation echogram and histogram data

    Parameters
    ----------
    plot : bool
        if True, plot figure
    i : int
        subplot row identifier
    axes : matplotlib.axes
        figure axes
    height : float
        true target height for file i
    imgs : array
        2D echogram data (power with distance and time)
        can be None if data is 1D e.g. for laser
    r_wf : array
        1D array of range with time
    cutoff : array
        manually defined start and end cutoff values (indexes) for file i
        allows for removal of e.g. landing and takeoff sections
    lim : array
        manually defined indexes describing transition from ground to target and vice versa for file i
    limits : array
        decribe ranges to be filtered out e.g. low = 0.2, all ranges less than 0.2 will later be removed
        also referes to the ymax limit when plotting
        low, high, ymax = limits
    bins : array
        bins for histogram

    Returns
    -------
    estimated : float
        estimated target height for file i
    imgs : array
        filtered 2D echogram data (power with distance and time)
        can be None if data is 1D e.g. for laser
    range_g : array
        1D array of ranges associated with ground
    range_t : array
        1D array of ranges associated with the target
    range_g_dev : array
        1D array of ranges associated with the ground, with the ground aligned with 1 meter
    range_t_dev : array
        1D array of ranges associated with the target, shifted with respect to the ground which has been aligned with 1 meter
    xarr_g : TYPE
        1D array of indexes matching the ranges associated with ground
    xarr_t : TYPE
        1D array of indexes matching the ranges associated with target

    '''
    low, high, ymax = limits
    
    #if data is 1D e.g. laser
    if imgs is None:
        estimated, range_g, range_t, range_g_dev, range_t_dev, xarr_g, xarr_t = crop_data_1d(r_wf, r_wf, cutoff, lim, limits)
    
    #if data is 2D e.g. v-band / k-band
    else:
        estimated, range_g, range_t, range_g_dev, range_t_dev, xarr_g, xarr_t = crop_data_2d(imgs, r_wf, cutoff, lim, limits)
    
    #if plot is toggled, plot plot
    if plot:
        big_plot(i, axes, xarr_g, range_g, xarr_t, range_t, ymax, bins, height)

    return estimated, imgs, range_g, range_t, range_g_dev, range_t_dev, xarr_g, xarr_t

def est_v_true_height(heights, estimated):
    '''
    Plots estimated vs true target heights from airborne validation test

    Parameters
    ----------
    heights : array
        array of true target heights
    estimated : array
        array of estimated target heights

    Returns
    -------
    None.

    '''
    
    #plot estimated vs true target heights
    fig = plt.figure(figsize=(8, 8))
    plt.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Perfect Correlation", alpha=0.5)
    m, b, *_ = stats.linregress(heights, estimated)
    plt.axline(xy1=(0, b), slope=m, ls="--", c="green", label=f"Returned Fit: {round(m,4)}x + {round(b,4)} cm", alpha=0.5)
    plt.scatter(heights, estimated)
    plt.xlabel("Measured Freeboard (cm)")
    plt.ylabel("Estimated Freeboard (cm)")
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    plt.legend()
    plt.show()