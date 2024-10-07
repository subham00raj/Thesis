import numpy as np
import utilities.create_inc as create_inc
import utilities.read_image as read_image
import config
import os
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def radar_cross_section(length, elevation_angle, incidence_angle, azimuth, wavelength):
  theta_cr = np.deg2rad(incidence_angle + elevation_angle)
  phi = np.deg2rad(azimuth)
  factor = (4 * np.pi * (length**4)) / wavelength**2
  a = np.cos(theta_cr) + np.sin(theta_cr) * (np.cos(phi) + np.sin(phi))
  b = 2/a
  rcs = factor * ((a - b)**2)
  return rcs



def measured_power(row, col, image):
  window_size_x = 51
  window_size_y = 101
  left = col - (window_size_x // 2)
  right = left + window_size_x
  top = row - (window_size_y // 2) + 1
  bottom = top + window_size_y
  img = image[top:bottom, left:right]

  horizontal = img[25:76,:]
  vertical = img[:,12:38]
  background_1 = img[0:25,0:12]
  background_2 = img[0:25,38:51]
  background_3 = img[76:101,0:12]
  background_4 = img[76:101,38:51]

  background_return_per_pixel = (np.sum(background_1) + np.sum(background_2) + np.sum(background_3) + np.sum(background_4)) / 1250
  power_from_cross = np.sum(horizontal) + np.sum(vertical) - np.sum(img[25:76,12:38])
  power_from_cr = power_from_cross - (background_return_per_pixel * 3563)

  return power_from_cr



def point_target_analysis(length, wavelength, elevation_angle, theta, azimuth, HH, VV):
  # correlator gain
  measured_rcs = np.abs(HH)**2
  theoritical_rcs = radar_cross_section(length, wavelength, elevation_angle, theta, azimuth)
  a = (theoritical_rcs / measured_rcs) ** 0.5
  a_dB = 10*np.log10(measured_rcs) - 10*np.log10(theoritical_rcs)

  # co channel imbalance amplitude
  f = ((np.abs(VV)**2) / (np.abs(HH))) ** 0.25

  # co channel imbalance phase
  co_phase = np.angle(np.array(VV) * np.conj(HH))

  return a, a_dB, f, co_phase, measured_rcs, theoritical_rcs




def distributed_target_analysis(HV_full_scene, VH_full_scene, gpu=False):
    # Cross-channel imbalance amplitude
    g = np.mean((np.abs(HV_full_scene) ** 2) / (np.abs(VH_full_scene))) ** 0.25

    # Cross-channel imbalance phase
    cross_phase = np.angle(np.mean(HV_full_scene * np.conj(VH_full_scene)))

    if gpu:
        import torch

        HV_tensor = torch.tensor(HV_full_scene, device='cuda')
        VH_tensor = torch.tensor(VH_full_scene, device='cuda')

        g_gpu = torch.mean((torch.abs(HV_tensor) ** 2) / torch.abs(VH_tensor)) ** 0.25
        cross_phase_gpu = torch.angle(torch.mean(HV_tensor * torch.conj(VH_tensor)))
        torch.cuda.empty_cache()
        return g_gpu.cpu().numpy(), cross_phase_gpu.cpu().numpy()

    return g, cross_phase


def apply_radio_cal(cr_length, image_path, input_csv_path):
  start = time.time()
  image = read_image.image_array(image_path)
  rows, cols = image.shape
  
  with tqdm(total=rows*cols, desc='Calibration Progress', unit = ' pixels') as pbar:
    df = pd.read_csv(input_csv_path).copy()
    df['HH'] = df['HH'].apply(lambda x: np.complex64(complex(x)))
    df['VV'] = df['VV'].apply(lambda x: np.complex64(complex(x)))
    df['a'], _ , df['f'], df['phase sum'], df['Measured RCS'], df['Theoritical RCS']  = point_target_analysis(length = df['Side Length'],
                                                                                                              wavelength = 0.234, 
                                                                                                              elevation_angle = df['Elevation Angle'], 
                                                                                                              theta = df['Incidence Angle'], 
                                                                                                              azimuth = df['Azimuth'], 
                                                                                                              HH = df['HH'], 
                                                                                                              VV = df['VV'])
    df = df[(df['Side Length'] == cr_length) & (df['Measured RCS'] > 10)]
    #g, phase_diff = distributed_target_analysis(HV, VH, gpu=False)

    inc = create_inc.create_inc_array_flat(min_look_angle = config.mininum_look_angle,
                                          max_look_angle = config.maximum_look_angle, 
                                          row_1x1 = config.row_1x1, 
                                          col_1x1 = config.col_1x1,
                                          subset = True)
    
    a_slope, a_intercept = np.polyfit(x = df['Incidence Angle'] - config.mount_antenna_angle, y = df['a'], deg = 1)
    p_slope, p_intercept = np.polyfit(x = df['Incidence Angle'] - config.mount_antenna_angle, y = df['phase sum'], deg = 1)
    

    calibrated_image = np.zeros_like(image)
    A = np.zeros_like(image)
    P = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            A[i, j] = a_slope * (inc[i, j] - config.mount_antenna_angle) + a_intercept
            calibrated_image[i, j] = image[i, j] / A[i, j]
            pbar.update(1)

  file_path = os.path.join(os.getenv('output_folder_path'),'Output.csv')
  df.to_csv(file_path, index = False)
  end = time.time()
  print(f'Executed in {(end - start):.2f} seconds.')
  return calibrated_image
        
if __name__ == '__main__':
   image_pol = os.getenv('HH_tif_path')
   dataframe = os.getenv('Input_csv_path')
   img = apply_radio_cal(cr_length = 4.8, image_path = image_pol, input_csv_path = dataframe)
   img = np.abs(img)
   plt.imshow(img, cmap='gray')
   plt.show()
  
   


