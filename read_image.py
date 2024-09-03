import math
import os
from tqdm import tqdm
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

gdal.UseExceptions()

def read_slc(file_path, rows=61349, cols=9874, subset = False, gpu=False):
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

def image_array(image_path):
    dataset = gdal.Open(image_path)
    band = dataset.GetRasterBand(1)
    image_subset = band.ReadAsArray()
    return image_subset


def multilooked(image, range_pixel=1.6, azimuth_pixel=0.6, gpu = False):

    factor = (int(range_pixel * azimuth_pixel * 100) // math.gcd(int(range_pixel * 10), int(azimuth_pixel * 10))) / 10
    range_looks = int(np.ceil(factor / range_pixel))
    azimuth_looks = int(np.ceil(factor / azimuth_pixel))

    if not gpu:
        rows, cols = image.shape[0] // azimuth_looks, image.shape[1] // range_looks
        multilook_image = np.zeros((rows, cols), dtype=image.dtype)

        with tqdm(total=rows * cols, desc="Progress", unit=" pixels") as progress_bar:
            for i in range(rows):
                for j in range(cols):
                    start_row, end_row = i * azimuth_looks, (i + 1) * azimuth_looks
                    start_col, end_col = j * range_looks, (j + 1) * range_looks
                    multilook_image[i, j] = np.mean(image[start_row:end_row, start_col:end_col])
                    progress_bar.update(1)

        return multilook_image
    
    else:

        import torch
        import torch.nn.functional as F

        device = torch.device("cuda")
        image = torch.tensor(image)
        image_tensor = image.unsqueeze(0).unsqueeze(0).to(device)

        output_tensor = F.avg_pool2d(image_tensor, kernel_size = (range_looks, azimuth_looks), stride = (range_looks, azimuth_looks), padding = 0)
        output_tensor = output_tensor.squeeze(0).squeeze(0).cpu()

        new = output_tensor.numpy().reshape(output_tensor.shape)
        return new


if __name__ == '__main__':
    
    image_path = os.path.join(os.getcwd(), 'HH.tif')
    img1 = np.abs(image_array(image_path))
    img2 = multilooked(image=img1, range_pixel=1.6, azimuth_pixel=0.6, gpu = False)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img1, cmap='gray', vmin=0.0001, vmax=0.0915)
    axs[0].set_title('Raw Distorted Image - Rectangular Pixels')
    
    axs[1].imshow(img2, cmap='gray', vmin=0.001, vmax=0.13)
    axs[1].set_title('Geometrically Corrected Image - Square Pixels')

    plt.show()

