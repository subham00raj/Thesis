'''

@author : Subham Raj

Application : Polarimetric SAR Calibration

Input : Corner Reflector Data in .csv format, LLH file, LKV file

Output : Corner Reflector Detected Image

Reference : https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl

'''
import read_image
import cv2
from datetime import datetime
import struct
from tqdm import tqdm
import numpy as np
import pandas as pd
from uavsar_pytools.incidence_angle import calc_inc_angle
import matplotlib.pyplot as plt
    

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

def get_cr_location(llh_file_path):
    now = datetime.now()
    csv_path = f'{now.strftime("%Y")}-{now.strftime("%m")}-{now.strftime("%d")}_0000_Rosamond-corner-reflectors_with_plate_motion.csv'
    cr_df = pd.read_csv(csv_path)
    llh_df = create_xyz_array(llh_file_path, rows = 7669, cols = 4937 , datatype = 'dataframe')
    pixel_loc = []

    for i in range(len(cr_df)):
        llh_df['CR_Latitude'] = abs(llh_df['Latitude'] - cr_df['Latitude'][i])
        llh_df['CR_Longitude'] = abs(llh_df['Longitude'] - cr_df['Longitude'][i])

        pixel_loc.append(llh_df[(llh_df['CR_Latitude'] < 0.00001) & (llh_df['CR_Longitude'] < 0.0001)].index.tolist())

    return pixel_loc

def get_dataframe(image, llh_file_path):
    now = datetime.now()
    csv_path = f'{now.strftime("%Y")}-{now.strftime("%m")}-{now.strftime("%d")}_0000_Rosamond-corner-reflectors_with_plate_motion.csv'
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

    total_cols = 4937
    df['Pixel Number'] = get_cr_location(llh_file_path)
    df['Pixel Location in 2x8 SLC'] = df['Pixel Number'].apply(lambda x: [calculate_indices(total_cols, i) for i in x])
    df['Pixel Location in 1x1 SLC'] = df['Pixel Location in 2x8 SLC'].apply(lambda x: [calculate_location_in_slc(i) for i in x])
    df['Pixel Location in Local SLC'] = df['Pixel Location in 1x1 SLC'].apply(lambda x: [(i[0]-41000, i[1]-2000) for i in x])

    df['Tentative Location'] = df['Pixel Location in Local SLC'].apply(lambda x: x[x.index(max(x))])
    df['Tentative Y'] = df['Tentative Location'].apply(lambda x: x[0])
    df['Tentative X'] = df['Tentative Location'].apply(lambda x: x[1])

    df['Y'] = [exact_location(df['Tentative X'][i],df['Tentative Y'][i], window_size=16, image = image)[0] for i in range(len(df))]
    df['X'] = [exact_location(df['Tentative X'][i],df['Tentative Y'][i], window_size=16, image = image)[1] for i in range(len(df))]

    return df[['Corner ID', 'Latitude', 'Longitude', 'Height', 'Azimuth', 'Elevation Angle', 'Side Length', 'Y', 'X']]


if __name__ == '__main__':
    image_path = r'HH.tiff'
    llh_file_path = r'Rosamd_35012_04_BC_s1_2x8.llh'
    image = read_image.image(image_path, start_coordinate = [4500, 17500], image_size = [3500, 3500])
    df = get_dataframe(image, llh_file_path)

    for i in range(len(df)):
        cv2.circle(image, (df['X'][i],df['Y'][i]), 50, 1, 1)

    plt.imshow(image, cmap='gray',vmin=1e-5,vmax=0.2)
    plt.show()



