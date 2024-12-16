import torch
import os
import numpy as np
from read_image import image_array
import ainsworth_ieee
from tqdm import tqdm
import tifffile
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv()

ainsworth_cal_output = os.getenv('ainsworth_cal_output')
if not os.path.exists(ainsworth_cal_output):
        os.mkdir(ainsworth_cal_output)

covariance_work_dir = os.getenv('covariance_file_path')
if not os.path.exists(covariance_work_dir):
     os.mkdir(covariance_work_dir)

output_folder_path = os.getenv('output_folder_path')
if not os.path.exists(output_folder_path):
     os.mkdir(output_folder_path)

def set_padding(input_array, x_window_size, y_window_size):
    pad_x = (x_window_size - 1) // 2
    pad_y = (y_window_size - 1) // 2
    padded_tensor = F.pad(torch.tensor(input_array), (pad_y, pad_y, pad_x, pad_x), mode='constant', value=0)
    return padded_tensor.numpy()


def sliding_window(HH, HV, VH, VV, window_size_x = 201, window_size_y = 201, stride_x = 1, stride_y = 1):
    result = []
    rows,cols = HH.shape
    total_iterations = ((rows - window_size_y) // stride_y + 1) * ((cols - window_size_x) // stride_x + 1)
    pbar = tqdm(total=total_iterations, desc="Sliding Window Progress")

    for i in range(0, rows - window_size_y + 1, stride_y):
        for j in range(0, cols - window_size_x + 1, stride_x):
            hh = HH[i:i + window_size_y, j:j + window_size_x]
            hv = HV[i:i + window_size_y, j:j + window_size_x]
            vh = VH[i:i + window_size_y, j:j + window_size_x]
            vv = VV[i:i + window_size_y, j:j + window_size_x]

            c11 = C11[i:i + window_size_y, j:j + window_size_x].mean()
            c12 = C12[i:i + window_size_y, j:j + window_size_x].mean()
            c13 = C13[i:i + window_size_y, j:j + window_size_x].mean()
            c14 = C14[i:i + window_size_y, j:j + window_size_x].mean()
            c21 = C21[i:i + window_size_y, j:j + window_size_x].mean()
            c22 = C22[i:i + window_size_y, j:j + window_size_x].mean()
            c23 = C23[i:i + window_size_y, j:j + window_size_x].mean()
            c24 = C24[i:i + window_size_y, j:j + window_size_x].mean()
            c31 = C31[i:i + window_size_y, j:j + window_size_x].mean()
            c32 = C32[i:i + window_size_y, j:j + window_size_x].mean()
            c33 = C33[i:i + window_size_y, j:j + window_size_x].mean()
            c34 = C34[i:i + window_size_y, j:j + window_size_x].mean()
            c41 = C41[i:i + window_size_y, j:j + window_size_x].mean()
            c42 = C42[i:i + window_size_y, j:j + window_size_x].mean()
            c43 = C43[i:i + window_size_y, j:j + window_size_x].mean()
            c44 = C44[i:i + window_size_y, j:j + window_size_x].mean()

            covariance = np.array([[c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44]]).reshape((4, 4))

            x = ainsworth_ieee.ainsworth_cal(covariance)
            out = x @ np.array([[hh[100, 100], hv[100, 100], vh[100, 100], vv[100, 100]]]).T
            result.append(out)
            pbar.update(1)

    pbar.close()
    return result

######################################################################### I M P L M E N T A T I O N ####################################################################################
        
if __name__ == '__main__':

    C11 = np.load(os.path.join(covariance_work_dir, 'C11.npy')) 
    C12 = np.load(os.path.join(covariance_work_dir, 'C12.npy')) 
    C13 = np.load(os.path.join(covariance_work_dir, 'C13.npy')) 
    C14 = np.load(os.path.join(covariance_work_dir, 'C14.npy')) 
    C21 = np.load(os.path.join(covariance_work_dir, 'C21.npy'))
    C22 = np.load(os.path.join(covariance_work_dir, 'C22.npy'))
    C23 = np.load(os.path.join(covariance_work_dir, 'C23.npy'))
    C24 = np.load(os.path.join(covariance_work_dir, 'C24.npy'))
    C31 = np.load(os.path.join(covariance_work_dir, 'C31.npy'))
    C32 = np.load(os.path.join(covariance_work_dir, 'C32.npy'))
    C33 = np.load(os.path.join(covariance_work_dir, 'C33.npy'))
    C34 = np.load(os.path.join(covariance_work_dir, 'C34.npy'))
    C41 = np.load(os.path.join(covariance_work_dir, 'C41.npy'))
    C42 = np.load(os.path.join(covariance_work_dir, 'C42.npy'))
    C43 = np.load(os.path.join(covariance_work_dir, 'C43.npy'))
    C44 = np.load(os.path.join(covariance_work_dir, 'C44.npy'))

    window_size_x, window_size_y = 201, 201

    HH = set_padding(image_array(os.path.join(output_folder_path),'HH_calibrated.tiff'), window_size_x, window_size_y)
    HV = set_padding(image_array(os.path.join(output_folder_path),'HV_calibrated.tiff'), window_size_x, window_size_y)
    VH = set_padding(image_array(os.path.join(output_folder_path),'VH_calibrated.tiff'), window_size_x, window_size_y)
    VV = set_padding(image_array(os.path.join(output_folder_path),'VV_calibrated.tiff'), window_size_x, window_size_y)

    result = sliding_window(HH, HV, VH, VV, window_size_x = 201, window_size_y = 201, stride_x = 1, stride_y = 1)

    image1 = []
    image2 = []
    image3 = []
    image4 = []

    for i in range(len(result)):
        image1.append(result[i][0,0])
        image2.append(result[i][1,0])
        image3.append(result[i][2,0])
        image4.append(result[i][3,0])
    
    HH_cal = np.array(image1).reshape(3000,3000)
    HV_cal = np.array(image2).reshape(3000,3000)
    VH_cal = np.array(image3).reshape(3000,3000)
    VV_cal = np.array(image4).reshape(3000,3000)

    tifffile.imwrite(file = os.path.join(ainsworth_cal_output,'HH_cal.tiff'), data = HH_cal)
    tifffile.imwrite(file = os.path.join(ainsworth_cal_output,'HV_cal.tiff'), data = HV_cal)
    tifffile.imwrite(file = os.path.join(ainsworth_cal_output,'VH_cal.tiff'), data = VH_cal)
    tifffile.imwrite(file = os.path.join(ainsworth_cal_output,'VV_cal.tiff'), data = VV_cal)