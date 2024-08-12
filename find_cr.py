'''

@author : Subham Raj

Application : Polarimetric SAR Calibration

Input : Corner Reflector Data in .csv format, LLH file, LKV file

Output : Corner Reflector Detected Image

Reference : https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl

'''

import os
import wget
from datetime import datetime
import requests
import struct
from tqdm import tqdm
import numpy as np
import pandas as pd
from uavsar_pytools.incidence_angle import calc_inc_angle
from bs4 import BeautifulSoup



def get_corner_reflector_info(kmz = False, csv = False):
    response = requests.get('https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl')
    soup = BeautifulSoup(response.text, 'html.parser')
    kmz_url = 'https://uavsar.jpl.nasa.gov' + soup.find('div', attrs = {'class' : 'main wrapper clearfix'}).findAll('a')[-5].get('href')
    csv_url = f'https://uavsar.jpl.nasa.gov/cgi-bin/corner-reflectors.pl?date=2024-08-11+00!00&project=rosamond_plate_location'
    if kmz:
        wget.download(kmz_url)
    elif csv:
        wget.download(csv_url)


def create_xyz_array(file_path, rows, cols, stacked = False):
    data = np.memmap(file_path, dtype=np.float32)
    data = data.reshape(-1, 3)
    if stacked:
        x = np.array(data)[:0].reshape((rows,cols))
        y = np.array(data)[:1].reshape((rows,cols))
        z = np.array(data)[:2].reshape((rows,cols))
        return np.dstack((x, y, z))

    else:
        return np.array(data)        


def create_inc_array_dem(llh_file, lkv_file, row_2x8 = 7669, col_2x8 = 4937):
    '''
    This function uses in built uavsar module to create incidence array for 2x8 SLC
    '''

    with open(llh_file, 'rb') as f, open(lkv_file, 'rb') as f2:
        total_pixels = row_2x8 * col_2x8
        
        dem = np.zeros(shape=(row_2x8, col_2x8), dtype='f')
        lkv_x = np.zeros(shape=(row_2x8, col_2x8), dtype='f')
        lkv_y = np.zeros(shape=(row_2x8, col_2x8), dtype='f')
        lkv_z = np.zeros(shape=(row_2x8, col_2x8), dtype='f')

        col = 0
        row = 0

        for _ in tqdm(range(total_pixels)):
            bytes_data = f.read(12)
            if not bytes_data:
                break

            lat = struct.unpack('f', bytes_data[:4])[0]
            lon = struct.unpack('f', bytes_data[4:8])[0]
            height = struct.unpack('f', bytes_data[8:12])[0]
            
            lkv_data = f2.read(12)
            px = struct.unpack('f', lkv_data[:4])[0]
            py = struct.unpack('f', lkv_data[4:8])[0]
            pz = struct.unpack('f', lkv_data[8:12])[0]
            
            dem[row][col] = height
            lkv_x[row][col] = px       
            lkv_y[row][col] = py
            lkv_z[row][col] = pz
            
            col += 1
            if col >= col_2x8:
                col = 0
                row += 1

    inc_arr = calc_inc_angle(dem, lkv_x, lkv_y, lkv_z)
    return inc_arr


def create_inc_array(min_look_angle = 21.32159622, max_look_angle = 66.17122143, row_1x1 = 61497, col_1x1 = 4937):
    '''
    This fucntion takes minimum and maximum look angle provided in .ann metadata file. It can be assumed that angle varies linearly from min to max across image width.
    '''
    array = np.linspace(min_look_angle, max_look_angle, col_1x1)
    stacked_inc_array = np.tile(array, (row_1x1, 1))
    return stacked_inc_array

def get(url = False):
    pass