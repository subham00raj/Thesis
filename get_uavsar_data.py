import os
import sys
import wget
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime



def get_corner_reflector_data(date, file_type = 'csv'):
    date = datetime.strptime(date, '%d-%m-%Y')
    response = requests.get('https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl')
    soup = BeautifulSoup(response.text, 'html.parser')
    kmz_url = 'https://uavsar.jpl.nasa.gov' + soup.find('div', attrs = {'class' : 'main wrapper clearfix'}).findAll('a')[-5].get('href')
    csv_url = f'https://uavsar.jpl.nasa.gov/cgi-bin/corner-reflectors.pl?date={date.year}-{date.month:02d}-{date.day:02d}+00!00&project=rosamond_plate_location'
    
    if file_type == 'kmz':
        wget.download(kmz_url)

    elif file_type == 'csv':
        csv_path = f'{date.year}-{date.month:02d}-{date.day:02d}_0000_Rosamond-corner-reflectors_with_plate_motion.csv'

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.drop(df.columns[-1], axis = 1)
            df.columns = ['Corner ID', 'Latitude', 'Longitude', 'Height', 'Azimuth', 'Elevation Angle', 'Side Length']
            df.to_csv(csv_path)

            return df
        else:
            wget.download(csv_url)
            df = pd.read_csv(csv_path)
            df = df.drop(df.columns[-1], axis = 1)
            df.columns = ['Corner ID', 'Latitude', 'Longitude', 'Height', 'Azimuth', 'Elevation Angle', 'Side Length']
            df.to_csv(csv_path)

            return df

    else:
        raise ValueError(f"Invalid datatype specified: {file_type}. Choose from 'kmz', or 'csv'.")    
   


def get_uavsar_data(flight_track_id, dt, date, multilooked = '1x1'):
    pols = ['HH', 'HV', 'VH', 'VV']
    
    for file in pols:
        sys.stdout.write(f'Downloading SLC data in {file} polarization \r')
        sys.stdout.flush()
        slc_url = f'https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_{flight_track_id}_00{dt}_{date}_L090{file}_04_BC_s1_{multilooked}.slc'
        ann_url = f'https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_{flight_track_id}_00{dt}_{date}_L090{file}_04_BC.ann'
        wget.download(slc_url)
        wget.download(ann_url)
        sys.stdout.write(f'Downloaded SLC File in {file} polarization \n')
        sys.stdout.flush()

    sys.stdout.write('Downloading Look Vector File \r')
    sys.stdout.flush()
    wget.download('https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_04_BC_s1_2x8.lkv')
    sys.stdout.write('Downloading Lat Long File \r')
    sys.stdout.flush()
    wget.download('https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_04_BC_s1_2x8.llh')
    sys.stdout.write('Download Complete. \n')
    sys.stdout.flush()


if __name__ == '__main__':

    # Download CR data
    sys.stdout.write("Downloading Corner Reflector Data \r")
    sys.stdout.flush()
    df = get_corner_reflector_data('16-09-2020', file_type='csv')
    if not df.empty:
        sys.stdout.write("Downloaded Corner Reflector Data. \n")
        sys.stdout.flush()
    
    # Download Image data
    get_uavsar_data(flight_track_id=16074, dt=5, date=160920, multilooked = '1x1')
