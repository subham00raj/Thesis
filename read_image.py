from osgeo import gdal
import tifffile
import numpy as np
gdal.UseExceptions()
import matplotlib.pyplot as plt


def read_slc(file_path, rows = 61349, cols = 9874, gpu = False):
    image = np.memmap(file_path, shape = (rows, cols) ,dtype = np.complex64)
    
    if gpu:
        import torch
        device = torch.device('cuda')
        array = torch.from_numpy(image).to(device)
        result = array.cpu()
        del array
        torch.cuda.empty_cache()
        return result

    else:
        return np.array(image)
    

def array_to_tiff(array, file_name):
    tifffile.imwrite(file_name, array)

def create_image(slc, type = None):
    if type == 'amplitude':
        return np.real(slc)
    elif type == 'imaginary':
        return np.imag(slc)
    elif type == 'magnitude':
        return np.abs(slc)
    elif type == 'intensity':
        return np.abs(slc)**2
    elif type == 'power':
        return 20*np.log10(np.abs(slc))
    else:
        raise ValueError(f"Invalid datatype specified: {type}. Choose among : 'amplitude', 'imaginary', 'magnitude', 'intensity' or 'power'.")

def plot_image(image_path, start_coordinate, image_size, min_threshold = 0.001, max_threshold = 0.13, return_array = False):
    dataset = gdal.Open(image_path)
    band = dataset.GetRasterBand(1)  
    image_subset = band.ReadAsArray(start_coordinate[0], start_coordinate[1], image_size[0], image_size[1])
    if return_array:
        return image_subset
    else:
        plt.imshow(image_subset, cmap='gray', vmin = min_threshold, vmax = max_threshold)
        plt.show()


if __name__ == '__main__':
    coord = [4500,17500]  # Starting coordinate
    size = [3500,3500]    # Image size
    plot_image('HH.tiff', start_coordinate = coord, image_size = size, min_threshold = 0.1, max_threshold = 0.13)



