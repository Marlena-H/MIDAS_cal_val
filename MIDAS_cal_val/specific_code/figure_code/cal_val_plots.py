#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:37:39 2026

@author: kftg4
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("../../../MIDAS_cal_val/")

from path_definitions import mini_laser_path_definitions

path, data_folder, path_ssd, path_save_loc = mini_laser_path_definitions()

def extract_data_from_output(output):
    x = [[] for i in range(8)]
    for i in output:
        estimated, imgs, range_g, range_t, range_g_dev, range_t_dev, xarr_g, xarr_t = i
        x[0].append(estimated)
        x[1].append(imgs)
        x[2].append(range_g)
        x[3].append(range_t)
        x[4].append(range_g_dev)
        x[5].append(range_t_dev)
        x[6].append(xarr_g)
        x[7].append(xarr_t)
    return x

def get_roll_adjusted(x, y, frac=0.15):
    x_new = []
    y_new = []
    diff = []

    for xi, yi in zip(x, y):
        xi = np.asarray(xi)
        yi = np.asarray(yi)
        
        #frac_len = len(yi)
        #frac = 60/frac_len

        mask = ~np.isnan(yi)  # LOWESS needs no NaNs
        trend = np.full_like(yi, np.nan, dtype=float)
        trend[mask] = lowess(yi[mask], xi[mask], frac=frac, return_sorted=False)

        diff_i = trend - np.nanmean(trend)
        yi_adj = yi - diff_i

        diff += [diff_i]
        x_new += [xi]
        y_new += [yi_adj]

    return x_new, y_new, diff


def plot_check(xarr_ground, ground, xarr_target, target, window=0.15):
    
    xarr_ground_new, ground_new, ground_diff = get_roll_adjusted(xarr_ground, ground, window)
    xarr_target_new, target_new, target_diff = get_roll_adjusted(xarr_target, target, window)
    
    '''
    for n, _ in enumerate(ground):
        print(n)
        plt.scatter(xarr_ground[n], ground[n], c="pink", s=1, label=f"OG_Ground: {round(np.nanstd(ground[n]), 3)}")
        plt.scatter(xarr_target[n], target[n], c="gold", s=1, label=f"OG_Target: {round(np.nanstd(target[n]), 3)}")
        #plt.plot(xarr_ground_new[n], ground_new[n]+ground_diff[n], linestyle="--", c="grey", linewidth=1)
        plt.scatter(xarr_ground_new[n], ground_new[n], c="red", s=1, label=f"NEW_Ground: {round(np.nanstd(ground_new[n]), 3)} {round(np.nanstd(ground_new[n])/np.nanstd(ground[n]), 2)}%")
        plt.scatter(xarr_target_new[n], target_new[n], c="darkorange", s=1, label=f"NEW_Target: {round(np.nanstd(target_new[n]), 3)} {round(np.nanstd(target_new[n])/np.nanstd(target[n]), 2)}%")
        #plt.plot(xarr_target_new[n], target_new[n]+target_diff[n], linestyle="--", c="grey", linewidth=1)
        plt.ylim(0, 2)
        plt.legend()
        plt.show()'''
        
    return xarr_ground_new, ground_new, ground_diff, xarr_target_new, target_new, target_diff

def big_plot_plot(xarr_g, range_g, xarr_t, range_t, ymax, bins, height, save_loc):

    fig, axes = plt.subplots(nrows=len(range_g), ncols=2, figsize=(10, 5 * len(range_g)), constrained_layout=True)
    for i, _ in enumerate(range_g):
        big_plot(i, axes, xarr_g[i], range_g[i], xarr_t[i], range_t[i], ymax, bins, height[i])
    plt.tight_layout()
    plt.savefig(save_loc, dpi=300)
    plt.show()
    
def img_diff_adjustment(x, x_diff, y_diff, y_diff_full):
    valid = ~np.isnan(y_diff)
    for xi, di in zip(x_diff[valid], y_diff[valid]):
        idx = np.argmin(np.abs(x - xi))
        y_diff_full[idx] = di
    return y_diff_full
    
def add_invisible_colorbar(ax, fig):
    """Add a transparent (invisible) colorbar to maintain layout symmetry."""
    # Create a dummy mappable with any colormap/norm
    dummy = cm.ScalarMappable(cmap='viridis')
    dummy.set_array([])
    
    cb = fig.colorbar(dummy, ax=ax)
    
    # Make every part of the colorbar invisible
    cb.ax.set_visible(False)
    
py_dir = add_path("../../../MIDAS_cal_val/")

from stats_functions import calc_stats_air, calc_stats_ground
from cal_val_functions import crop_data_1d, crop_data_2d, big_plot

py_dir = add_path("../../../MIDAS_cal_val/specific_code/calibration_code/")
from v_band_calibration import main as get_v_cal_data
from k_band_calibration import main as get_k_cal_data
from laser_calibration import main as get_laser_cal_data

py_dir = add_path("../../../MIDAS_cal_val/specific_code/validation_code/")

#%%
print("Getting Laser Data")
from laser_validation import main as get_laser_val_data
laser_x, laser_y, laser_r = get_laser_cal_data()
ml, bl, *_ = stats.linregress(laser_x, laser_y)
laser_y2 = np.array(laser_y) - np.nanmean(laser_y - laser_x)
l_heights, laser_ranges, laser_output = get_laser_val_data()

#%%
print("Getting V Band Data")
py_dir = add_path("../../../MIDAS_cal_val/src/sensor_mh_processing/MMW-HAT-Release/")
from v_band_validation import main as get_v_val_data
v_x, v_y, v_r = get_v_cal_data()
mv, bv, *_ = stats.linregress(v_x, v_y)
v_y2 = np.array(v_y) - np.nanmean(v_y - v_x)
v_heights, v_imgs, v_ranges, v_output = get_v_val_data()

#%%
print("Getting K Band Data")
py_dir = add_path("../../../MIDAS_cal_val/src/sensor_mh_processing/uRAD_RaspberryPi_SDK11/")
from k_band_validation import main as get_k_val_data
k_x, k_y, k_r = get_k_cal_data()
mk, bk, *_ = stats.linregress(k_x, k_y)
k_y2 = np.array(k_y) - np.nanmean(k_y - k_x)
k_heights, k_imgs, k_ranges, k_output = get_k_val_data()

#%%

l_estimates = [i[0] for i in laser_output]
v_estimates = [i[0] for i in v_output]
k_estimates = [i[0] for i in k_output]

#%%

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
#fig.suptitle(f"{sensor_name} — Performance Plots", fontsize=14, fontweight='bold')

col = 0

ax = axes[1, col]
ax.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
sign = "+" if bv >= 0 else "-"
ax.axline(xy1=(0, bv), slope=mv, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(mv, 3)}x {sign} {abs(round(bv, 3))} m", alpha=0.5)
ax.scatter(v_x, v_y, color="cornflowerblue", label="Measurements")
ax.scatter(v_x, v_y2, color="red", label="Bias Corrected Measurements")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xlabel("Measured Range (m)")
ax.set_ylabel("Observed (Sensor) Range (m)")
ax.set_title(f"(b) V-Band Radar\nBias: {round(np.nanmean(v_y - v_x)*100, 1)} cm")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(fontsize="small")

ax2 = axes[2, col]
ax2.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
sign = "+" if bk >= 0 else "-"
ax2.axline(xy1=(0, bk), slope=mk, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(mk, 3)}x {sign} {abs(round(bk, 3))} m", alpha=0.5)
ax2.scatter(k_x, k_y, color="cornflowerblue", label="Measurements")
ax2.scatter(k_x, k_y2, color="red", label="Bias Corrected Measurements")
ax2.set_xlim(1, 8)
ax2.set_ylim(1, 8)
ax2.set_xlabel("Measured Range (m)")
ax2.set_ylabel("Observed (Sensor) Range (m)")
ax2.set_title(f"(c) K-Band Radar\nBias: {round(np.nanmean(k_y - k_x)*100, 1)} cm")
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.legend(fontsize="small")

ax3=axes[0, col]
ax3.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
sign = "+" if bl >= 0 else "-"
ax3.axline(xy1=(0, bl), slope=ml, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(ml, 3)}x {sign} {abs(round(bl, 3))} m", alpha=0.5)
ax3.scatter(laser_x, laser_y, color="cornflowerblue", label="Measurements")
ax3.scatter(laser_x, laser_y2, color="red", label="Bias Corrected Measurements")
ax3.set_xlim(2, 3.5)
ax3.set_ylim(2, 3.5)
ax3.set_xlabel("Measured Range (m)")
ax3.set_ylabel("Observed (Sensor) Range (m)")
ax3.set_title(f"(a) Laser\nBias: {round(np.nanmean(laser_y - laser_x)*100, 1)} cm")
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.legend(fontsize="small")

col_titles = ['Laser', 'V-Band Radar', 'K-Band Radar']

datasets = [[l_heights, l_estimates], [v_heights, v_estimates], [k_heights, k_estimates]]
bias = [-0.81, 3.02, -2.22]
bias = [0, 0, 0]
box_data = [np.array(l_estimates)-np.array(l_heights)+bias[0], np.array(v_estimates)-np.array(v_heights)+bias[1], np.array(k_estimates)-np.array(k_heights)+bias[2]]

col = 1

lab = ["a", "b", "c"]
for row, title in enumerate(col_titles):
    axes[row, col].set_title(f"({lab[row]}) {title}")

for row, data in enumerate(datasets):
    heights = data[0]
    estimates = data[1]
    axes[row, col].set_ylabel("Sensor Estimated Height Change (cm)")
    axes[row, col].set_xlabel("Measured Height Change (cm)")
    axes[row, col].axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
    m, b, *_ = stats.linregress(np.array(heights), np.array(estimates))
    sign = "+" if b >= 0 else "-"
    axes[row, col].axline(xy1=(0, b), slope=m, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(m,3)}x {sign} {abs(round(b,3))} cm", alpha=0.5)
    axes[row, col].scatter(heights, estimates, color="cornflowerblue")
    axes[row, col].set_xlim(0, 65)
    axes[row, col].set_ylim(0, 65)
    axes[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[row, col].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[row, col].legend(fontsize="small")

plt.tight_layout()
plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/LinePlot.png", dpi=600)
plt.show()

#%%

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
#fig.suptitle(f"{sensor_name} — Performance Plots", fontsize=14, fontweight='bold')

col = 0

ax = axes[col, 1]
ax.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
sign = "+" if bv >= 0 else "-"
ax.axline(xy1=(0, bv), slope=mv, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(mv, 3)}x {sign} {abs(round(bv, 3))} m", alpha=0.5)
ax.scatter(v_x, v_y, color="cornflowerblue", label="Measurements")
ax.scatter(v_x, v_y2, color="red", label="Bias Corrected Measurements")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xlabel("Measured Range (m)")
ax.set_ylabel("Observed (Sensor) Range (m)")
ax.set_title(f"(b) V-Band Radar\nBias: {round(np.nanmean(v_y - v_x)*100, 2)} cm")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(frameon=False,fontsize="small")

ax2 = axes[col, 2]
ax2.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
sign = "+" if bk >= 0 else "-"
ax2.axline(xy1=(0, bk), slope=mk, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(mk, 3)}x {sign} {abs(round(bk, 3))} m", alpha=0.5)
ax2.scatter(k_x, k_y, color="cornflowerblue", label="Measurements")
ax2.scatter(k_x, k_y2, color="red", label="Bias Corrected Measurements")
ax2.set_xlim(1, 8)
ax2.set_ylim(1, 8)
ax2.set_xlabel("Measured Range (m)")
ax2.set_ylabel("Observed (Sensor) Range (m)")
ax2.set_title(f"(c) K-Band Radar\nBias: {round(np.nanmean(k_y - k_x)*100, 2)} cm")
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.legend(frameon=False,fontsize="small")

ax3=axes[col, 0]
ax3.axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
sign = "+" if bl >= 0 else "-"
ax3.axline(xy1=(0, bl), slope=ml, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(ml, 3)}x {sign} {abs(round(bl, 3))} m", alpha=0.5)
ax3.scatter(laser_x, laser_y, color="cornflowerblue", label="Measurements")
ax3.scatter(laser_x, laser_y2, color="red", label="Bias Corrected Measurements")
ax3.set_xlim(2, 3.5)
ax3.set_ylim(2, 3.5)
ax3.set_xlabel("Measured Range (m)")
ax3.set_ylabel("Observed (Sensor) Range (m)")
ax3.set_title(f"(a) Laser\nBias: {round(np.nanmean(laser_y - laser_x)*100, 2)} cm")
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.legend(frameon=False,fontsize="small")

col_titles = ['Laser', 'V-Band Radar', 'K-Band Radar']

datasets = [[l_heights, l_estimates], [v_heights, v_estimates], [k_heights, k_estimates]]
bias = [-0.81, 3.02, -2.22]
bias = [0, 0, 0]
box_data = [np.array(l_estimates)-np.array(l_heights)+bias[0], np.array(v_estimates)-np.array(v_heights)+bias[1], np.array(k_estimates)-np.array(k_heights)+bias[2]]

row = 1

lab = ["a", "b", "c"]
for col, title in enumerate(col_titles):
    axes[row, col].set_title(f"({lab[col]}) {title}")

for col, data in enumerate(datasets):
    heights = data[0]
    estimates = data[1]
    axes[row, col].set_ylabel("Sensor Estimated Height Change (cm)")
    axes[row, col].set_xlabel("Measured Height Change (cm)")
    axes[row, col].axline(xy1=(0, 0), slope=1, ls="--", color="black", label="Identity", alpha=0.5)
    m, b, *_ = stats.linregress(heights, estimates)
    sign = "+" if b >= 0 else "-"
    axes[row, col].axline(xy1=(0, b), slope=m, ls="--", c="cornflowerblue", label=f"Returned Fit: {round(m,3)}x {sign} {abs(round(b,3))} cm", alpha=0.5)
    axes[row, col].scatter(heights, estimates, color="cornflowerblue")
    axes[row, col].set_xlim(0, 65)
    axes[row, col].set_ylim(0, 65)
    axes[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[row, col].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[row, col].legend(frameon=False,fontsize="small")

plt.tight_layout()
plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/LinePlot_Inv.png", dpi=600)
plt.show()

#%%

l_estimates, l_imgs, l_ground, l_target, l_ground_dev, l_target_dev, l_xarr_ground, l_xarr_target = extract_data_from_output(laser_output)
v_estimates, v_imgs, v_ground, v_target, v_ground_dev, v_target_dev, v_xarr_ground, v_xarr_target = extract_data_from_output(v_output)
k_estimates, k_imgs, k_ground, k_target, k_ground_dev, k_target_dev, k_xarr_ground, k_xarr_target = extract_data_from_output(k_output)

#%%

all_summaries = []

for label, x, ranges in [
    ("Laser", laser_x, laser_r),
    ("V Band", v_x, v_r),
    ("K Band", k_x, k_r),
]:
    summary, per_measurement = calc_stats_ground(label, x, ranges)
    all_summaries.append(summary)
    per_measurement.to_csv(f"{path}/output_for_repo/specific_output/cal_val_output/{label.replace(' ', '_')}_ground_per_measurement.csv", index=False)
    print("\n")


for label, heights, estimates in [
    ("Laser", l_heights, l_estimates),
    ("V Band", v_heights, v_estimates),
    ("K Band", k_heights, k_estimates),
]:
    summary, per_measurement = calc_stats_air(label, heights, estimates)
    all_summaries.append(summary)
    per_measurement.to_csv(f"{path}/output_for_repo/specific_output/cal_val_output/{label.replace(' ', '_')}_air_per_measurement.csv", index=False)
    print("\n")

summary_table = pd.DataFrame(all_summaries)
summary_table.to_csv(f"{path}/output_for_repo/specific_output/cal_val_output/all_sensors_summary.csv", index=False)

#%%
'''
l_xarr_ground_new, l_ground_new, l_ground_diff = get_roll_adjusted(l_xarr_ground, l_ground, 5)
v_xarr_ground_new, v_ground_new, v_ground_diff = get_roll_adjusted(v_xarr_ground, v_ground, 30)
k_xarr_ground_new, k_ground_new, k_ground_diff = get_roll_adjusted(k_xarr_ground, k_ground, 80)

l_xarr_target_new, l_target_new, l_target_diff = get_roll_adjusted(l_xarr_target, l_target, 5)
v_xarr_target_new, v_target_new, v_target_diff = get_roll_adjusted(v_xarr_target, v_target, 30)
k_xarr_target_new, k_target_new, k_target_diff = get_roll_adjusted(k_xarr_target, k_target, 80)'''

#%%

l_xarr_ground, l_ground, l_ground_diff, l_xarr_target, l_target, l_target_diff = plot_check(l_xarr_ground, l_ground, l_xarr_target, l_target, 0.3)
v_xarr_ground, v_ground, v_ground_diff, v_xarr_target, v_target, v_target_diff = plot_check(v_xarr_ground, v_ground, v_xarr_target, v_target, 0.7)
k_xarr_ground, k_ground, k_ground_diff, k_xarr_target, k_target, k_target_diff = plot_check(k_xarr_ground, k_ground, k_xarr_target, k_target, 1)

big_plot_plot(l_xarr_ground, l_ground, l_xarr_target, l_target, 2, np.linspace(50, 150, 50)/100, l_heights, f"{path}/output_for_repo/specific_output/figure_output/LaserBigPlot.png")
big_plot_plot(v_xarr_ground, v_ground, v_xarr_target, v_target, 2, np.linspace(0, 150, 50)/100, v_heights, f"{path}/output_for_repo/specific_output/figure_output/VBandBigPlot.png")
big_plot_plot(k_xarr_ground, k_ground, k_xarr_target, k_target, 5, np.linspace(200, 400, 50)/100, k_heights, f"{path}/output_for_repo/specific_output/figure_output/KBandBigPlot.png")

#%%
fig, axes = plt.subplots(6, 3, figsize=(16, 16), gridspec_kw={'width_ratios': [1, 1, 1]})

plot_heights = [57, 25, 9]

col_titles = ['Laser', 'V-Band Radar', 'K-Band Radar']
row_titles = ['57 cm', '25 cm', '9 cm', '57 cm', '25 cm', '9 cm']

for row in range(len(row_titles)):
    for col in range(len(col_titles)):
        if row > 2:
            add_invisible_colorbar(axes[row, col], fig)
        elif row < 3 and col == 0:
            add_invisible_colorbar(axes[row, col], fig)   

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title)
    axes[3, col].set_title(title)
    
for row in range(len(row_titles)):
    for col in range(len(col_titles)):
        if row < 3:        
            axes[row, col].set_xlabel("Measurement Number")
            axes[row, col].set_ylabel("Range (m)", rotation=90)
        if row > 2:
            axes[row, col].set_xlabel("Range (m)")
            axes[row, col].set_ylabel("Frequency", rotation=90)

for row, title in enumerate(row_titles):
    if row != 1 or row != 4:
        if row < 3:
            axes[row, 0].set_ylabel(f"{title}\n\nRange (m)", rotation=90)
        if row > 2:
            axes[row, 0].set_ylabel(f"{title}\n\nFrequency", rotation=90)

axes[1, 0].set_ylabel(f"Target Height (m)\n\n{row_titles[1]}\n\nRange (m)", rotation=90)
axes[4, 0].set_ylabel(f"Target Height (m)\n\n{row_titles[1]}\n\nFrequency", rotation=90)

for index, i_height in enumerate(v_heights):
    if i_height not in plot_heights:
        continue
    
    col = 1
    row = plot_heights.index(i_height)
    
    i_ground = v_ground[index] 
    i_target = v_target[index]
    i_v_xarr_ground = v_xarr_ground[index]
    i_v_xarr_target = v_xarr_target[index]
    
    adjust = - np.nanmean(i_ground) + 1
    i_target = i_target + adjust 
    i_ground = i_ground + adjust
    img = v_imgs[index]
    
    if index == 0:
        n = 350
        axes[row, col].set_xlim(n, img.shape[1])

    x = np.arange(0, img.shape[1])
    y = np.linspace(0, 1.78, img.shape[0])
    y_diff_full = np.zeros(len(x))
    y_diff_full = img_diff_adjustment(x, v_xarr_ground[index], v_ground_diff[index], y_diff_full)
    y_diff_full = img_diff_adjustment(x, v_xarr_target[index], v_target_diff[index], y_diff_full)
    X, Y = np.meshgrid(x, y)
    Y = Y + adjust - np.array(y_diff_full)
    
    cax = axes[row, col].pcolormesh(X, Y, img, vmin=0, vmax=1, cmap="viridis", shading='auto')
    xlim = axes[row, col].get_xlim()
    ylim = axes[row, col].get_ylim()
    axes[row, col].fill_between(xlim, ylim[0], ylim[1], color='white', alpha=0.2)
    cbar = fig.colorbar(cax, ax=axes[row,col], label='Power', location="right")
    cbar_ax = cbar.ax
    cbar_ylim = cbar_ax.get_ylim()
    cbar_xlim = cbar_ax.get_xlim()
    
    cbar_ax.fill_betweenx(cbar_ylim, cbar_xlim[0], cbar_xlim[1],
                       color='white', alpha=0.2, zorder=10)
    cbar.set_label('Power')
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    axes[row, col].set_aspect('auto')
    axes[row, col].scatter(i_v_xarr_ground, i_ground, s=2, c="red")
    axes[row, col].scatter(i_v_xarr_target, i_target, s=2, c="orange")
    axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f')) 
    axes[row, col].set_ylim(1.78, 0.25)
    
    x_min, x_max = min(i_v_xarr_ground), max(i_v_xarr_ground)
    if min(i_v_xarr_target) < x_min:
        x_min = min(i_v_xarr_target)
    if max(i_v_xarr_target) > x_max:
        x_max = max(i_v_xarr_target)
        
    axes[row, col].set_xlim(x_min, x_max)
    
for index, i_height in enumerate(k_heights):
    if i_height not in plot_heights:
        continue
    
    n = 200
    col = 2
    row = plot_heights.index(i_height)
    
    i_ground = k_ground[index] 
    i_target = k_target[index]
    i_k_xarr_ground = k_xarr_ground[index]
    i_k_xarr_target = k_xarr_target[index]
    
    adjust = - np.nanmean(i_ground) + 1 
    i_target = i_target + adjust 
    i_ground = i_ground + adjust
    img = k_imgs[index]
    
    x = np.arange(0, img.shape[1]-n)
    y = np.linspace(0, 62.45676208333333, img.shape[0])
    i_diff_full = np.zeros(len(x))
    y_diff_full = np.zeros(len(x))
    y_diff_full = img_diff_adjustment(x, k_xarr_ground[index], k_ground_diff[index], y_diff_full)
    y_diff_full = img_diff_adjustment(x, k_xarr_target[index], k_target_diff[index], y_diff_full)
    X, Y = np.meshgrid(x, y)
    Y = Y + adjust - np.array(i_diff_full)
        
    cax = axes[row, col].pcolormesh(X, Y, img[:,:-n], vmin=0, vmax=0.15, cmap="viridis", shading='auto')
    xlim = axes[row, col].get_xlim()
    ylim = axes[row, col].get_ylim()
    axes[row, col].fill_between(xlim, ylim[0], ylim[1], color='white', alpha=0.2)
    cbar = fig.colorbar(cax, ax=axes[row,col], label='Power', location="right")
    cbar_ax = cbar.ax
    cbar_ylim = cbar_ax.get_ylim()
    cbar_xlim = cbar_ax.get_xlim()
    
    cbar_ax.fill_betweenx(cbar_ylim, cbar_xlim[0], cbar_xlim[1],
                       color='white', alpha=0.2, zorder=10)
    cbar.set_label('Power')
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    axes[row, col].set_aspect('auto')
    axes[row, col].scatter(i_k_xarr_ground[:-n], i_ground[:-n], s=0.2, c="red")
    axes[row, col].scatter(i_k_xarr_target[:-n], i_target[:-n], s=0.2, c="orange")
    axes[row, col].set_ylim(3, -0.5)
    axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f')) 
    
    x_min, x_max = min(i_k_xarr_ground), max(i_k_xarr_ground)
    if min(i_k_xarr_target) < x_min:
        x_min = min(i_k_xarr_target)
    if max(i_k_xarr_target) > x_max:
        x_max = max(i_k_xarr_target)

    axes[row, col].set_xlim(x_min, x_max)
    
for index, i_height in enumerate(l_heights):
    if i_height not in plot_heights:
        continue
    
    col = 0
    row = plot_heights.index(i_height)
    
    i_ground = l_ground[index] 
    i_target = l_target[index]
    i_l_xarr_ground = l_xarr_ground[index]
    i_l_xarr_target = l_xarr_target[index]
    
    adjust = - np.nanmean(i_ground) + 1 
    i_target = i_target + adjust
    i_ground = i_ground + adjust #- i_diff
    
    axes[row, col].scatter(i_l_xarr_ground, i_ground, s=2, c="red", label="Ground")
    axes[row, col].scatter(i_l_xarr_target, i_target, s=2, c="orange", label="Target")
    axes[row, col].set_ylim(1.75, 0.25)
    axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f')) 
    axes[row, col].set_aspect('auto')
    axes[row, col].legend()
    axes[row, col].legend(fontsize="x-small", loc='upper left')

for index, i_height in enumerate(v_heights):
    if i_height not in plot_heights:
        continue
    
    col = 1
    row = plot_heights.index(i_height) + 3
    
    i_ground = v_ground[index] 
    i_target = v_target[index]
    
    adjust = - np.nanmean(i_ground) + 1 
    i_target = i_target + adjust 
    i_ground = i_ground + adjust
    
    bins = (np.linspace(200, 400, 50)/100) - (np.nanmean(i_ground) + 1)
    axes[row, col].hist(i_ground, bins=bins, alpha=0.7, color="red")
    axes[row, col].axvline(np.nanmean(i_ground), linestyle="--", color="red", label=f"Ground: {round(np.nanmean(i_ground), 2)} \u00B1 {round(np.nanstd(i_ground), 2)} m")
    axes[row, col].hist(i_target, bins=bins, alpha=0.7, color="orange")
    axes[row, col].axvline(np.nanmean(i_target), linestyle="--", color="orange", label=f"Target: {round(np.nanmean(i_target), 2)} \u00B1 {round(np.nanstd(i_target), 2)} m")
    axes[row, col].set_xlim(0, 1.5)
    axes[row, col].text(.02, .77, f"Estimated  Height Change: {round((np.nanmean(i_ground)-np.nanmean(i_target))*100, 1)} cm", ha='left', va='top', transform=axes[row, col].transAxes, fontsize="x-small")
    axes[row, col].legend(loc='upper left', fontsize="x-small")
    
for index, i_height in enumerate(k_heights):
    if i_height not in plot_heights:
        continue
    
    col = 2
    row = plot_heights.index(i_height) + 3
    
    i_ground = k_ground[index] 
    i_target = k_target[index]
    
    adjust = - np.nanmean(i_ground) + 1 
    i_target = i_target + adjust 
    i_ground = i_ground + adjust
    
    bins = (np.linspace(200, 400, 50)/100) - (np.nanmean(i_ground) + 1)
    axes[row, col].hist(i_ground, bins=bins, alpha=0.7, color="red")
    axes[row, col].axvline(np.nanmean(i_ground), linestyle="--", color="red", label=f"Ground: {round(np.nanmean(i_ground), 2)} \u00B1 {round(np.nanstd(i_ground), 2)} m")
    axes[row, col].hist(i_target, bins=bins, alpha=0.7, color="orange")
    axes[row, col].axvline(np.nanmean(i_target), linestyle="--", color="orange", label=f"Target: {round(np.nanmean(i_target), 2)} \u00B1 {round(np.nanstd(i_target), 2)} m")
    axes[row, col].set_xlim(0, 1.5)
    axes[row, col].text(.02, .77, f"Estimated  Height Change: {round((np.nanmean(i_ground)-np.nanmean(i_target))*100, 1)} cm", ha='left', va='top', transform=axes[row, col].transAxes, fontsize="x-small")
    axes[row, col].legend(loc="upper left", fontsize="x-small")

for index, i_height in enumerate(l_heights):
    if i_height not in plot_heights:
        continue
    
    col = 0
    row = plot_heights.index(i_height) + 3
    
    i_ground = l_ground[index] 
    i_target = l_target[index]
    
    adjust = - np.nanmean(i_ground) + 1 
    i_target = i_target + adjust 
    i_ground = i_ground + adjust
    
    bins = np.linspace(0, 150, 50)/100
    axes[row, col].hist(i_ground, bins=bins, alpha=0.7, color="red")
    axes[row, col].axvline(np.nanmean(i_ground), linestyle="--", color="red", label=f"Ground: {round(np.nanmean(i_ground), 2)} \u00B1 {round(np.nanstd(i_ground), 2)} m")
    axes[row, col].hist(i_target, bins=bins, alpha=0.7, color="orange")
    axes[row, col].axvline(np.nanmean(i_target), linestyle="--", color="orange", label=f"Target: {round(np.nanmean(i_target), 2)} \u00B1 {round(np.nanstd(i_target), 2)} m")
    axes[row, col].set_xlim(0, 1.5)
    axes[row, col].text(.02, .77, f"Estimated  Height Change: {round((np.nanmean(i_ground)-np.nanmean(i_target))*100, 1)} cm", ha='left', va='top', transform=axes[row, col].transAxes, fontsize="x-small")
    axes[row, col].legend(loc="upper left", fontsize="x-small")


plt.tight_layout()
plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/DistPlot.png", dpi=600)
plt.show()

#%%

deconv_v = np.loadtxt(f"{path}/data_for_repo/to_process/specific_data/Deconv_V.txt", delimiter=",")
conv_v = np.loadtxt(f"{path}/data_for_repo/to_process/specific_data/Conv_V.txt", delimiter=",")
K = np.loadtxt(f"{path}/data_for_repo/to_process/specific_data/K_data.txt", delimiter=",")

#%%

fig, ax = plt.subplots(1, 3, figsize=(12, 8))
fig.subplots_adjust(wspace=0.45)

lab = ["a", "b", "c"]
col_titles = ['V-Band Radar (Uncorrected)', 'V-Band Radar (Corrected)', 'K-Band Radar']
data = [conv_v[:,100:300], deconv_v[:,100:300], K[30000:35000]]
asp = 4

col = 0
imgs = data[col]
imgs = np.flip(imgs, axis=1)
ax[col].set_title(f"({lab[col]}) {col_titles[col]}")
cax = ax[col].imshow(imgs, cmap="viridis", vmin=5.5, vmax=8.5, aspect=asp, extent=[0, np.shape(imgs)[1], 1.78, 0])
xlim = ax[col].get_xlim()
ylim = ax[col].get_ylim()
ax[col].fill_between(xlim, ylim[0], ylim[1], color='white', alpha=0.2)
cbar = fig.colorbar(cax, ax=ax[col], label='Power', location="right", shrink=0.62)
cbar_ax = cbar.ax
cbar_ylim = cbar_ax.get_ylim()
cbar_xlim = cbar_ax.get_xlim()

cbar_ax.fill_betweenx(cbar_ylim, cbar_xlim[0], cbar_xlim[1],
                   color='white', alpha=0.2, zorder=10)
cbar.set_label('Power')
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax[col].set_ylabel("Range (m)")
ax[col].set_aspect((np.shape(imgs)[1]/1.78)/0.5)

col = 1
imgs = data[col]
imgs = np.flip(imgs, axis=1)
ax[col].set_title(f"({lab[col]}) {col_titles[col]}")
cax = ax[col].imshow(imgs, cmap="viridis", vmin=0, vmax=1, aspect=asp, extent=[0, np.shape(imgs)[1], 1.78, 0])
xlim = ax[col].get_xlim()
ylim = ax[col].get_ylim()
ax[col].fill_between(xlim, ylim[0], ylim[1], color='white', alpha=0.2)
cbar = fig.colorbar(cax, ax=ax[col], label='Power', location="right", shrink=0.62)
cbar_ax = cbar.ax
cbar_ylim = cbar_ax.get_ylim()
cbar_xlim = cbar_ax.get_xlim()

cbar_ax.fill_betweenx(cbar_ylim, cbar_xlim[0], cbar_xlim[1],
                   color='white', alpha=0.2, zorder=10)
cbar.set_label('Power')
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax[col].set_ylabel("Range (m)")
ax[col].set_aspect((np.shape(imgs)[1]/1.78)/0.5)

col = 2
imgs = data[col]
imgs = np.rot90(imgs, k=3)
imgs = np.flip(imgs, axis=1)
ax[col].set_title(f"({lab[col]}) {col_titles[col]}")
cax = ax[col].imshow(imgs, cmap="viridis", vmin=0, vmax=0.15, aspect=asp, extent=[0, np.shape(imgs)[1], 62.45676208333333, 0])
xlim = ax[col].get_xlim()
ylim = ax[col].get_ylim()
ax[col].fill_between(xlim, ylim[0], ylim[1], color='white', alpha=0.2)
cbar = fig.colorbar(cax, ax=ax[col], label='Power', location="right", shrink=0.62)
cbar_ax = cbar.ax
cbar_ylim = cbar_ax.get_ylim()
cbar_xlim = cbar_ax.get_xlim()

cbar_ax.fill_betweenx(cbar_ylim, cbar_xlim[0], cbar_xlim[1],
                   color='white', alpha=0.2, zorder=10)
cbar.set_label('Power')
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax[col].set_ylabel("Range (m)")
ax[col].set_aspect((np.shape(imgs)[1]/10)/0.5)
ax[col].set_ylim(10, -0.05)

ax[0].get_xaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[2].get_xaxis().set_visible(False)
ax[2].set_xlabel('Time')

ax[1].set_title(f'Example Data from V- and K-Band Radars\n\n{col_titles[1]}')

plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/ZScope.png", dpi=600)
plt.show()

#%%
'''
v_wf = []
k_wf = []
k_x= []

with open('/home/kftg4/Documents/Code/mh_code/specific_plotting_code/v_wf.txt', 'r') as f:
    for line in f:
        line = line.strip()
        v_wf.append(np.fromstring(line, sep=','))
        
with open('/home/kftg4/Documents/Code/mh_code/specific_plotting_code/k_wf.txt', 'r') as f:
    for line in f:
        line = line.strip()
        k_wf.append(np.fromstring(line, sep=','))

with open('/home/kftg4/Git Repositories/MIDAS_cal_val/data_for_repo/to_process/specific_data/x_axis.txt', 'r') as f:
    for line in f:
        line = line.strip()
        k_x.append(np.fromstring(line, sep=','))

v_x = np.linspace(0, 1.78, 57)
    
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
fig.subplots_adjust(hspace=0.4)

v_r = v_wf[0]
k_r = k_x[np.argmax(k_wf)]

v_wf = v_wf[1:]
v_min_corr = v_wf[np.argmin(v_wf)]

v_wf = v_wf - v_min_corr

ax[0].set_title("(a) V Band Waveform")
ax[0].set_ylabel("Power")
ax[0].set_xlabel("Range (m)")
ax[0].plot(v_x, v_wf, label="Waveform")
ax[0].set_xlim(v_r-2, v_r+2)
ax[0].axvline(v_r, c="red", label="Retracked Point")

ax[1].set_title("(b) K Band Waveform")
ax[1].set_ylabel("Power")
ax[1].set_xlabel("Range (m)")
ax[1].plot(k_x, k_wf)
ax[1].set_xlim(k_r-2, k_r+2)
ax[1].axvline(k_r, c="red", label="Retracked Point")

ax[0].legend()

plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/Echomatch.png", dpi=600)
plt.show()

#%%
    
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
fig.subplots_adjust(hspace=0.4)

ax[0].set_title("(a) V Band Waveform")
ax[0].set_ylabel("Power")
ax[0].set_xlabel("Range (m)")
ax[0].plot(v_x, v_wf, label="Waveform")
ax[0].axvline(v_r, c="red", label="Retracked Point")

ax[1].set_title("(b) K Band Waveform")
ax[1].set_ylabel("Power")
ax[1].set_xlabel("Range (m)")
ax[1].plot(k_x, k_wf)
ax[1].set_xlim(0, 12)
ax[1].axvline(k_r, c="red", label="Retracked Point")

ax[0].legend()

plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/Echo.png", dpi=600)
plt.show()

#%% 

ref_wf = np.loadtxt(f"{path}/MIDAS_cal_val/src/sensor_mh_processing/uRAD_RaspberryPi_SDK11/ref_wf.txt")

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
fig.subplots_adjust(hspace=0.4)

ax[0].set_title("(a) K Band Waveform")
ax[0].set_ylabel("Power")
ax[0].set_xlabel("Range (m)")
ax[0].plot(k_x, k_wf)
ax[0].set_xlim(0, 12)
ax[0].set_ylim(-0.05, 0.15)
ax[0].axvline(k_r, c="red", label="Retracked Point")

ax[1].set_title("(b) Empty K Band Waveform")
ax[1].set_ylabel("Power")
ax[1].set_xlabel("Range (m)")
ax[1].plot(k_x, ref_wf)
ax[1].set_xlim(0, 12)
ax[1].set_ylim(-0.05, 0.15)

ax[0].legend()

plt.savefig(f"{path}/output_for_repo/specific_output/figure_output/K_Band.png", dpi=600)
plt.show()'''