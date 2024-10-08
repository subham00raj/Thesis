import numpy as np
import struct
import config
from uavsar_pytools.incidence_angle import calc_inc_angle


def create_inc_array_dem(llh_file, lkv_file, row_2x8, col_2x8):
    '''
    This function uses in built uavsar module to create incidence 
    array for 2x8 SLC using digital elevation model
    '''

    with open(llh_file, 'rb') as f, open(lkv_file, 'rb') as f2:
        total_pixels = row_2x8 * col_2x8
        
        dem = np.zeros(shape=(row_2x8, col_2x8), dtype='f')
        lkv_x = np.zeros(shape=(row_2x8, col_2x8), dtype='f')
        lkv_y = np.zeros(shape=(row_2x8, col_2x8), dtype='f')
        lkv_z = np.zeros(shape=(row_2x8, col_2x8), dtype='f')

        col = 0
        row = 0

        for _ in range(total_pixels):
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


def create_inc_array_flat(min_look_angle, max_look_angle, row_1x1, col_1x1, subset = False):
    '''
    This fucntion takes minimum and maximum look angle provided in .ann metadata file. 
    Assuming flat terrain i.e., angle varies linearly from min to max across image width.
    '''
    array = np.linspace(min_look_angle, max_look_angle, col_1x1)
    stacked_inc_array = np.tile(array, (row_1x1, 1))
    if subset:
        return stacked_inc_array[config.y_start:config.y_start + config.y_size, config.x_start:config.x_start + config.x_size]
    
    return stacked_inc_array
