#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:12:55 2026

@author: kftg4
"""

import re
import pandas as pd

def read_data(path, data_folder, save_as):
    '''
    Read laser data from txt files

    Parameters
    ----------
    path : str
        path describing the location of the GitHub repo.
    data_folder : str
        location of where sensor data that needs to be processed can be found
    save_as : str
        filename

    Returns
    -------
    df : pandas dataframe
        pandas dataframe containing extracted laser data

    '''
    data = []
    with open(path + data_folder + "/" + save_as, 'r') as f:
        for line in f:
            #ChatGPT because of Regex :(
            match = re.match(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+),\s*System Time:\s*(?P<system_time>\d+),\s*Distance:\s*(?P<distance>[\d\.]+)\s*cm,\s*Status:\s*(?P<status>\d+),\s*Signal Strength:\s*(?P<signal_strength>\d+),\s*Range Precision:\s*(?P<range_precision>\d+)',
                line
                )
            if match:
                data.append({
                    'timestamp': pd.to_datetime(match.group('timestamp')),
                    'system_time': int(match.group('system_time')),
                    'range': float(match.group('distance'))/100,
                    'status': int(match.group('status')),
                    'signal_strength': int(match.group('signal_strength')),
                    'range_precision': int(match.group('range_precision'))
                })

        df = pd.DataFrame(data)
    return df