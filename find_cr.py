'''

@author : Subham Raj

Application : Polarimetric SAR Calibration

Input : Corner Reflector Data in .csv format, LLH file, LKV file

Output : Corner Reflector Detected Image

Reference : https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl

'''

import wget
from datetime import datetime
import requests
import struct
from tqdm import tqdm
import numpy as np
import pandas as pd
from uavsar_pytools.incidence_angle import calc_inc_angle
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import torch



def get_corner_reflector_data(file_type = 'csv'):
    now = datetime.now()
    response = requests.get('https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl')
    soup = BeautifulSoup(response.text, 'html.parser')
    kmz_url = 'https://uavsar.jpl.nasa.gov' + soup.find('div', attrs = {'class' : 'main wrapper clearfix'}).findAll('a')[-5].get('href')
    csv_url = f'https://uavsar.jpl.nasa.gov/cgi-bin/corner-reflectors.pl?date={now.strftime("%Y")}-{now.strftime("%m")}-{now.strftime("%d")}+00!00&project=rosamond_plate_location'
    if file_type == 'kmz':
        wget.download(kmz_url)
    elif file_type == 'csv':
        wget.download(csv_url)
        csv_path = f'{now.strftime("%Y")}-{now.strftime("%m")}-{now.strftime("%d")}_0000_Rosamond-corner-reflectors_with_plate_motion.csv'
        df = pd.read_csv(csv_path)
        df = df.drop(df.columns[-1], axis = 1)
        df.columns = ['Corner ID', 'Latitude', 'Longitude', 'Height', 'Azimuth', 'Elevation Angle', 'Side Length']
        df.to_csv(csv_path)

    else:
        raise ValueError(f"Invalid datatype specified: {file_type}. Choose from 'kmz', or 'csv'.")    


def get_uavsar_data(flight_track_id, dt, date, multilooked = '1x1'):
    pols = ['HH', 'HV', 'VH', 'VV']
    
    for file in pols:
        print(f'Downloading SLC data in {file} polarization')
        slc_url = f'https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_{flight_track_id}_00{dt}_{date}_L090{file}_04_BC_s1_{multilooked}.slc'
        ann_url = f'https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_{flight_track_id}_00{dt}_{date}_L090{file}_04_BC.ann'
        wget.download(slc_url)
        wget.download(ann_url)

    print('Downloading Look Vector File')
    wget.download('https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_04_BC_s1_2x8.lkv')
    print('Downloading Lat Long File')
    wget.download('https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_04_BC_s1_2x8.llh')
    print('Download Complete.')
    

def create_xyz_array(file_path, rows, cols, datatype = 'array'):
    data = np.memmap(file_path, dtype=np.float32)
    data = data.reshape(-1, 3)
    if datatype == 'stacked':
        x = np.array(data)[:,0].reshape((rows,cols))
        y = np.array(data)[:,1].reshape((rows,cols))
        z = np.array(data)[:,2].reshape((rows,cols))
        return np.dstack((x, y, z))

    elif datatype == 'dataframe':
        return pd.DataFrame(data, columns = ['Latitude', 'Longitude', 'Height'])

    elif datatype == 'array':
        return np.array(data)

    else: 
        raise ValueError(f"Invalid datatype specified: {datatype}. Choose from 'stacked', 'dataframe', or 'array'.")
        


def create_inc_array_dem(llh_file, lkv_file, row_2x8 = 7669, col_2x8 = 4937, left_look = True):
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

    if left_look:
        return np.fliplr(inc_arr)
    return inc_arr


def create_inc_array_flat(min_look_angle = 21.32159622, max_look_angle = 66.17122143, row_1x1 = 61497, col_1x1 = 4937):
    '''
    This fucntion takes minimum and maximum look angle provided in .ann metadata file. Assuming flat terrain i.e., angle varies linearly from min to max across image width.
    '''
    array = np.linspace(min_look_angle, max_look_angle, col_1x1)
    stacked_inc_array = np.tile(array, (row_1x1, 1))
    return np.fliplr(stacked_inc_array)


def read_slc(file_path, rows = 61349, cols = 9874, gpu = False):
    image = np.memmap(file_path, shape = (rows, cols) ,dtype = np.complex64)
    
    if gpu:
        device = torch.device('cuda')
        array = torch.from_numpy(image).to(device)
        result = array.cpu()
        del array
        torch.cuda.empty_cache()
        return result

    else:
        return np.array(image)
