#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:25:26 2026

@author: kftg4
"""

import numpy as np
from scipy.fft import fft, fftshift

def get_IQ_data(filename, path, data_folder):
    '''
    Load IQ data from k band radar txt file

    Parameters
    ----------
    filename : str
        filename of file to load
    path : str
        path describing the location of the GitHub repo.
    data_folder : str
        location of where sensor data that needs to be processed can be found

    Returns
    -------
    params : dict
        dictionary of radar parameters used for these measurements e.g. bandwidth so conversions are all correct.
    imgs : array
        2D array of power with distance and time.
    x_axis : array
        array of range with bin number for mapping as range
    max_distance : float
        Maximum range that the sensor can range to
    all_times : array
        array containing time each measurement was taken
    PP : array
        1D array of PP with time

    '''
    params = {}
    imgs = []
    all_times = []
    PP = []
    
    #apply anteral method for getting waveforms
    with open(path + data_folder + "/" + filename, 'r') as f:
        for line in f:
            I = []
            Q = []

            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].strip().split(" = ")
                    if value == "True":
                        value = True
                    elif value == "False":
                        value = False
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value) if '.' in value else int(value)
                    params[key] = value
            elif line.strip():
                N = float(params["Ns"])
                max_voltage = 3.3
                ADC_bits = 12
                ADC_intervals = 2**ADC_bits
                Fs = 200000
                N_FFT = 4096
                BW_actual = float(params["BW"]) * (10**6)
                RampTimeReal = N/Fs
                c = 299792458
                
                first_comma = line.find(',')
                timestamp = line[:first_comma]
                try:
                    all_times += [float(timestamp)]
                except:
                    all_times += [np.nan]

                IQs_raw = line[first_comma+1:].strip()
                iq_strings = IQs_raw.split('],')

                for iq_str in iq_strings:
                    iq_str = iq_str.strip().strip('[]')
                    if iq_str:
                        try:
                            i_str, q_str = iq_str.split(',')
                        except:
                            pass

                    if i_str == '' or q_str == '':
                        I += [np.nan]
                        Q += [np.nan]
                    else:
                        try:
                            i_str = float(i_str)
                            q_str = float(q_str)
                        except:
                            i_str = np.nan
                            q_str = np.nan
                        I += [i_str]
                        Q += [q_str]

                I = [np.array(inner_list) for inner_list in I]
                Q = [np.array(inner_list) for inner_list in Q]

                I = np.array(I, dtype=float)
                Q = np.array(Q, dtype=float)

                #From URAD GUI Code
                I = I*max_voltage/ADC_intervals
                Q = Q*max_voltage/ADC_intervals
                I = I - np.mean(I)
                Q = Q - np.mean(Q)

                ComplexVector = I + 1j * Q

                Nm = len(I)
                ComplexVector = ComplexVector * np.hanning(Nm) * 2 / 3.3

                max_distance = c/(2*BW_actual) * Fs/2 * RampTimeReal

                FrequencyDomain = 2 * np.abs(fftshift(fft(ComplexVector / Nm, N_FFT))) #aka power

                f_axis = fftshift(np.fft.fftfreq(N_FFT, d=1/Fs))
                x_axis = c/(2 * BW_actual) * f_axis * RampTimeReal

                start = int(N_FFT/2)
                FrequencyDomain[start] = FrequencyDomain[start - 1]

                #FrequencyDomain = 20 * np.log10(FrequencyDomain)

                start = int(N_FFT/2)
                x_axis = x_axis[start:]
                Magnitude = FrequencyDomain[start:]
                imgs += [Magnitude]
                PP += [np.max(Magnitude)/np.nanmean(Magnitude)]
                
    return params, imgs, x_axis, max_distance, all_times, PP

