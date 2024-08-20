import tifffile
import math
from tqdm import tqdm
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

gdal.UseExceptions()

def read_slc(file_path, rows=61349, cols=9874, gpu=False):
    image = np.memmap(file_path, shape=(rows, cols), dtype=np.complex64)
    if gpu:
        import torch
        device = torch.device('cuda')
        array = torch.from_numpy(image).to(device)
        result = array.cpu()
        del array
        torch.cuda.empty_cache()
        return result
    return np.array(image)

def array_to_tiff(array, file_name):
    tifffile.imwrite(file_name, array)

def create_image(slc, type=None):
    type_dict = {
        'amplitude': np.real,
        'imaginary': np.imag,
        'magnitude': np.abs,
        'intensity': lambda x: np.abs(x)**2,
        'power': lambda x: 20 * np.log10(np.abs(x))
    }
    if type in type_dict:
        return type_dict[type](slc)
    raise ValueError(f"Invalid datatype specified: {type}. Choose from: 'amplitude', 'imaginary', 'magnitude', 'intensity', 'power'.")

def image(image_path, start_coordinate, image_size, return_array=False, min_threshold=None, max_threshold=None):
    dataset = gdal.Open(image_path)
    band = dataset.GetRasterBand(1)
    image_subset = band.ReadAsArray(start_coordinate[0], start_coordinate[1], image_size[0], image_size[1])

    if return_array:
        return image_subset
    plt.imshow(image_subset, cmap='gray', vmin=min_threshold, vmax=max_threshold)
    plt.show()

def multilooked(image, range_pixel=1.6, azimuth_pixel=0.6, return_array=False):
    factor = (int(range_pixel * azimuth_pixel * 100) // math.gcd(int(range_pixel * 10), int(azimuth_pixel * 10))) / 10
    range_looks = int(np.ceil(factor / range_pixel))
    azimuth_looks = int(np.ceil(factor / azimuth_pixel))

    rows, cols = image.shape[0] // azimuth_looks, image.shape[1] // range_looks
    multilook_image = np.zeros((rows, cols), dtype=image.dtype)

    with tqdm(total=rows * cols, desc="Progress", unit=" pixels") as progress_bar:
        for i in range(rows):
            for j in range(cols):
                start_row, end_row = i * azimuth_looks, (i + 1) * azimuth_looks
                start_col, end_col = j * range_looks, (j + 1) * range_looks
                multilook_image[i, j] = np.mean(image[start_row:end_row, start_col:end_col])
                progress_bar.update(1)

    if return_array:
        return multilook_image
    plt.imshow(multilook_image, cmap='gray', vmin=0.001, vmax=0.12)
    plt.show()

if __name__ == '__main__':
    coord = [4500, 17500]  # Starting coordinate
    size = [3500, 3500]    # Image size
    img = image('HH.tiff', start_coordinate=coord, image_size=size, return_array=True)
    multilooked(image=img, range_pixel=1.6, azimuth_pixel=0.6)