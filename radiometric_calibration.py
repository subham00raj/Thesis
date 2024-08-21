import numpy as np

def radar_cross_section(length, wavelength, elevation_angle, theta, azimuth):
  theta_cr = np.deg2rad(theta + elevation_angle)
  phi = np.deg2rad(azimuth)
  factor = (4 * np.pi * (length**4)) / wavelength**2
  a = np.cos(theta_cr) + np.sin(theta_cr) * (np.cos(phi) + np.sin(phi))
  b = 2/a
  rcs = factor * ((a - b)**2)
  return rcs**0.5

def absolute_calibration_constant(measured_rcs, theoritical_rcs):
  return measured_rcs / theoritical_rcs
