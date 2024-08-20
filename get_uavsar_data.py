import os
import wget
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime



def get_corner_reflector_data(file_type = 'csv'):
    now = datetime.now()
    response = requests.get('https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl')
    soup = BeautifulSoup(response.text, 'html.parser')
    kmz_url = 'https://uavsar.jpl.nasa.gov' + soup.find('div', attrs = {'class' : 'main wrapper clearfix'}).findAll('a')[-5].get('href')
    csv_url = f'https://uavsar.jpl.nasa.gov/cgi-bin/corner-reflectors.pl?date={now.strftime("%Y")}-{now.strftime("%m")}-{now.strftime("%d")}+00!00&project=rosamond_plate_location'
    
    if file_type == 'kmz':
        wget.download(kmz_url)

    elif file_type == 'csv':
        csv_path = f'{now.strftime("%Y")}-{now.strftime("%m")}-{now.strftime("%d")}_0000_Rosamond-corner-reflectors_with_plate_motion.csv'

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
        print(f'Downloading SLC data in {file} polarization')
        slc_url = f'https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_{flight_track_id}_00{dt}_{date}_L090{file}_04_BC_s1_{multilooked}.slc'
        ann_url = f'https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_{flight_track_id}_00{dt}_{date}_L090{file}_04_BC.ann'
        wget.download(slc_url)
        wget.download(ann_url)

    print('Downloading Look Vector File')
    wget.download('https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_04_BC_s1_2x8.lkv')
    print('Downloading Lat Long File')
    wget.download('https://downloaduav2.jpl.nasa.gov/Release26/Rosamd_35012_04/Rosamd_35012_04_BC_s1_2x8.llh')
    print('Download Complete.')