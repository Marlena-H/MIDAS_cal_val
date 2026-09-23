#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:52:02 2025

@author: kftg4
"""

from natsort import natsorted
import os
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def add_path(where_to):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    py_dir = os.path.abspath(os.path.join(current_dir, where_to))
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    return py_dir

py_dir = add_path("../../..")

from path_definitions import path_fun, mini_laser_path_definitions
from universal_functions import running_in_ide
from laser_functions import read_data

path, data_folder, path_ssd, path_save_loc = mini_laser_path_definitions()

files = [f for f in os.listdir(path + data_folder) if f.endswith('.txt')]
files = natsorted(files)
n = 1

def main_for_big_laser(save_as, path):
    
    df = read_data(path, data_folder, save_as)

    if np.shape(df)[0] > 0:
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        time_arr = df["timestamp"].dt.round("100ms").dt.strftime("%H:%M:%S.%f").str[:-5].tolist()

        fs = 7
        fig, ax = plt.subplots()

        plt.scatter(time_arr, df["range"], label="Range", s=0.3)
        
        s = int(len(time_arr) / 6)
        
        plt.xticks(
            ticks=time_arr[::s],
            labels=[t[:10] for t in time_arr[::s]],  # "SS:mmm"
            fontsize=fs
        )
        
        plt.yticks(fontsize=fs)

        ax.set_aspect((np.shape(df["range"])[0]/7)/3.2)
        ax.set_title(f"{save_as[:-4]}" , fontsize=fs)
        ax.set_ylabel('Range (m)', fontsize=fs)
        ax.set_xlabel('Time', fontsize=fs)

        plt.ylim(7, -0.5)
        plt.legend(loc='upper left', fontsize="xx-small")

        try:
            plt.savefig(f"{path_ssd}/{path_save_loc}/ZSCOPE_{str(save_as)[:-4]}.png", bbox_inches='tight', pad_inches=0.2)
        except FileNotFoundError:
            plt.savefig(f"{path}/{path_save_loc}/ZSCOPE_{str(save_as)[:-4]}.png", bbox_inches='tight', pad_inches=0.2)
        except PermissionError:
            plt.savefig(f"{path}/{path_save_loc}/ZSCOPE_{str(save_as)[:-4]}.png", bbox_inches='tight', pad_inches=0.2)

        if running_in_ide():
            plt.show()

        plt.close()

def main():
    print("--- Started: plotting.py ---\n")
    start_time_1 = time.time()
    #Parallel(n_jobs=2)(delayed(main_for_parallel)(n) for n in files[:5])
    for save_as in files:
        main_for_big_laser(save_as, path)
        start_time = time.time()

        print("--- %s seconds ---\n" % (time.time() - start_time))

    print(f"--- Total Run Time: {round((time.time() - start_time_1)/60, 1)} minutes ---\n")

    print("--- Finished: plotting.py ---\n")

if __name__ == '__main__':
    main()