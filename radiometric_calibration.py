import numpy as np

def radar_cross_section(length, wavelength, theta, phi):
  theta = np.deg2rad(theta)
  phi = np.deg2rad(phi)
  factor = (4 * np.pi * (length**4)) / wavelength**2
  a = np.cos(theta) + np.sin(theta) * (np.cos(phi) + np.sin(phi))
  b = 2/a
  rcs = factor * ((a - b)**2)
  return rcs**0.5