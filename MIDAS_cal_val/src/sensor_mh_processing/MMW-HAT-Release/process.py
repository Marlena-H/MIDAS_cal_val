#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:39:40 2025

@author: kftg4
"""

import os
import sys
import deconvolve
import retrack
import z_scope

current_dir = os.path.dirname(os.path.abspath(__file__))
a_py_dir = os.path.abspath(os.path.join(current_dir, "../../../.."))
if a_py_dir not in sys.path:
    sys.path.insert(0, a_py_dir)

from path_definitions import path_fun, plotting_choices, py_choices

path, data_folder, hostname = path_fun("/data_for_repo/to_process/sensor_data/DreamRF_data")

plot_deconv_wfs, plot_retracked_wfs, plot_z_scopes, generate_gps_maps, generate_gps_and_radar_maps = plotting_choices()
run_deconv, run_good_deconv, run_retrack, run_z_scope = py_choices()

if run_good_deconv or run_retrack:
    deconvolve.main()
if run_retrack:
    retrack.main()
if run_z_scope:
    z_scope.main()
