'''

@author : Subham Raj

Application : Polarimetric SAR Calibration

Input : Corner Reflector Data in .csv format, LLH file, LKV file

Output : Corner Reflector Detected Image

Reference : https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl

'''

import read_image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import create_inc
import config
import os
from dotenv import load_dotenv

load_dotenv()
    

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
        

import pandas as pd
import sys

def get_cr_location(csv_path, llh_file_path):
    cr_df = pd.read_csv(csv_path)
    llh_df = create_xyz_array(llh_file_path, rows=7669, cols=4937, datatype='dataframe')
    pixel_loc = []

    total_cr = len(cr_df) 
    found_cr_count = 0  

    for i in range(total_cr):
        llh_df['CR_Latitude'] = abs(llh_df['Latitude'] - cr_df['Latitude'][i])
        llh_df['CR_Longitude'] = abs(llh_df['Longitude'] - cr_df['Longitude'][i])


        matching_indices = llh_df[(llh_df['CR_Latitude'] < 0.00001) & (llh_df['CR_Longitude'] < 0.0001)].index.tolist()
        
        if matching_indices:  
            found_cr_count += 1  
            sys.stdout.write(f"\rFound {found_cr_count} CRs out of {total_cr} CRs") 
            sys.stdout.flush()

        pixel_loc.append(matching_indices)

    print()
    return pixel_loc


def get_dataframe(image, csv_path, llh_file_path):
    cr_df = pd.read_csv(csv_path)
    df = cr_df.copy()
        
    def calculate_indices(total_cols, count):
        row_index = (count - 1) // total_cols
        col_index = (count - 1) % total_cols
        return (row_index, col_index)

    def calculate_location_in_slc(loc):
        ml_row, ml_col = loc[0],loc[1]
        ml_factor_row, ml_factor_col = 8,2
        slc_row = ml_row * ml_factor_row
        slc_col = ml_col * ml_factor_col
        return slc_row, slc_col

    def exact_location(X, Y, window_size, image):
        top_left_x = X - window_size // 2
        top_left_y = Y - window_size // 2
        window = image[top_left_y:top_left_y + window_size, top_left_x:top_left_x + window_size]
        max_pixel_coordinates = np.unravel_index(np.argmax(window), window.shape)
        return top_left_y + max_pixel_coordinates[0], top_left_x + max_pixel_coordinates[1]

    total_cols = config.col_2x8
    df['Pixel Number'] = get_cr_location(csv_path, llh_file_path)
    df['Pixel Location in 2x8 SLC'] = df['Pixel Number'].apply(lambda x: [calculate_indices(total_cols, i) for i in x])
    df['Pixel Location in 1x1 SLC'] = df['Pixel Location in 2x8 SLC'].apply(lambda x: [calculate_location_in_slc(i) for i in x])
    df['Pixel Location in Local SLC'] = df['Pixel Location in 1x1 SLC'].apply(lambda x: [(i[0]-41000, i[1]-2000) for i in x])

    df['Tentative Location'] = df['Pixel Location in Local SLC'].apply(lambda x: x[x.index(max(x))])
    df['Tentative Y'] = df['Tentative Location'].apply(lambda x: x[0])
    df['Tentative X'] = df['Tentative Location'].apply(lambda x: x[1])

    df['Local Y'] = [exact_location(df['Tentative X'][i],df['Tentative Y'][i], window_size=16, image = image)[0] for i in range(len(df))]
    df['Local X'] = [exact_location(df['Tentative X'][i],df['Tentative Y'][i], window_size=16, image = image)[1] for i in range(len(df))]

    return df[['Corner ID', 'Latitude', 'Longitude', 'Height', 'Azimuth', 'Elevation Angle', 'Side Length', 'Local Y', 'Local X']]


if __name__ == '__main__':

    csv_path = os.getenv('csv_path')
    llh_file_path = os.getenv('llh_path')
    lkv_file_path = os.getenv('lkv_path')
    HH_path = os.getenv('HH_tif_path')
    HV_path = os.getenv('HV_tif_path')
    VH_path = os.getenv('VH_tif_path')
    VV_path = os.getenv('VV_tif_path')

    # create image subset
    start_coordinate = [config.x_start, config.y_start]
    image_size = [config.x_size, config.y_size]
    image = np.abs(read_image.image_array(HH_path))

    # calculate incidence angle
    inc_array_flat = create_inc.create_inc_array_flat(config.mininum_look_angle, config.maximum_look_angle, config.row_1x1, config.col_1x1)

    # Output csv
    df = get_dataframe(image, csv_path, llh_file_path)
    df['Global Y'] = start_coordinate[1] + df['Local Y']
    df['Global X'] = start_coordinate[0] + df['Local X']
    df['Incidence Angle'] = inc_array_flat[df['Global Y'],df['Global X']]
    df['HH'] = read_image.image_array(HH_path)[df['Local Y'],df['Local X']]
    df['HV'] = read_image.image_array(HV_path)[df['Local Y'],df['Local X']]
    df['VH'] = read_image.image_array(VH_path)[df['Local Y'],df['Local X']]
    df['VV'] = read_image.image_array(VV_path)[df['Local Y'],df['Local X']]
    df.to_csv(os.getenv('Input_csv_path'), index = False)

    # image plot
    plt.imshow(image, cmap='gray', vmin=1e-5, vmax=0.2)
    for i in range(len(df)):
        cv2.circle(image, (df['Local X'][i],df['Local Y'][i]), 50, 1, 1)
        text = df['Corner ID'][i]
        plt.text(df['Local X'][i] - 25, df['Local Y'][i] + 100, text, fontsize=8, color='red')


    plt.imshow(image, cmap='gray', vmin=1e-5, vmax=0.2)
    plt.title('Corner Reflectors Operating In-situ')
    plt.show()