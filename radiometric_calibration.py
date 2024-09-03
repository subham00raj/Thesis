import numpy as np
import create_inc
import read_image
import config
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def radar_cross_section(length, elevation_angle, incidence_angle, azimuth, wavelength):
  theta_cr = np.deg2rad(incidence_angle + elevation_angle)
  phi = np.deg2rad(azimuth)
  factor = (4 * np.pi * (length**4)) / wavelength**2
  a = np.cos(theta_cr) + np.sin(theta_cr) * (np.cos(phi) + np.sin(phi))
  b = 2/a
  rcs = factor * ((a - b)**2)
  return rcs


def point_target_analysis(length, wavelength, elevation_angle, theta, azimuth, HH, VV):
  # correlator gain
  measured_rcs = np.abs(HH)**2
  theoritical_rcs = radar_cross_section(length, wavelength, elevation_angle, theta, azimuth)
  a = (theoritical_rcs / measured_rcs) ** 0.5
  a_dB = 10*np.log10(measured_rcs) - 10*np.log10(theoritical_rcs)

  # co channel imbalance amplitude
  f = ((np.abs(VV)**2) / (np.abs(HH))) ** 0.25

  # co channel imbalance phase
  phase_sum = np.angle(np.array(VV) * np.conj(HH))

  return a, a_dB, f, phase_sum


def distributed_target_analysis(HV, VH, gpu=False):
    # Cross-channel imbalance amplitude
    g = np.mean((np.abs(HV) ** 2) / (np.abs(VH))) ** 0.25

    # Cross-channel imbalance phase
    phase_diff = np.angle(np.mean(HV * np.conj(VH)))

    if gpu:
        import torch

        HV_tensor = torch.tensor(HV, device='cuda')
        VH_tensor = torch.tensor(VH, device='cuda')

        g_gpu = torch.mean((torch.abs(HV_tensor) ** 2) / torch.abs(VH_tensor)) ** 0.25
        phase_diff_gpu = torch.angle(torch.mean(HV_tensor * torch.conj(VH_tensor)))

        return g_gpu.cpu().numpy(), phase_diff_gpu.cpu().numpy()

    return g, phase_diff


def apply_radio_cal(image_path, input_csv_path, HV, VH):
  df = pd.read_csv(input_csv_path)

  df['HH'] = df['HH'].apply(lambda x: np.complex64(complex(x)))
  df['VV'] = df['VV'].apply(lambda x: np.complex64(complex(x)))
  df['a'], _ , df['f'], df['phase sum']  = point_target_analysis(length = df['Side Length'],
                                  wavelength = 0.234, 
                                  elevation_angle = df['Elevation Angle'], 
                                  theta = df['Incidence Angle'], 
                                  azimuth = df['Azimuth'], 
                                  HH = df['HH'], 
                                  VV = df['VV'])
  
  g, phase_diff = distributed_target_analysis(HV, VH, gpu=False)

  inc = create_inc.create_inc_array_flat(min_look_angle = config.mininum_look_angle,
                                        max_look_angle = config.maximum_look_angle, 
                                        row_1x1 = config.row_1x1, 
                                        col_1x1 = config.col_1x1,
                                        subset = True)
  
  a_slope, a_intercept = np.polyfit(x = df['Incidence Angle'], y = df['a'], deg = 1)
  p_slope, p_intercept = np.polyfit(x = df['Incidence Angle'], y = df['phase sum'], deg = 1)
  
  image = read_image.image_array(image_path)
  rows, cols = image.shape
  calibrated_image = np.zeros_like(image)
  A = np.zeros_like(image)
  P = np.zeros_like(image)

  with tqdm(total=rows*cols, desc='Progress', unit = 'pixels ') as pbar:
      for i in range(rows):
          for j in range(cols):
              A[i, j] = a_slope * (inc[i, j] - 45) + a_intercept
              calibrated_image[i, j] = image[i, j] / A[i, j]
              pbar.update(1)

  return calibrated_image
          

if __name__ == '__main__':
   image_pol = r'C:\Users\Vision IAS\Desktop\work\SAR\HH.tif'
   dataframe = r'C:\Users\Vision IAS\Desktop\work\SAR\Input.csv'
   img = apply_radio_cal(image_pol, dataframe)
   np.save('img.npy',img)


