import pandas as pd
import numpy as np

from Utility.GPS import GPSVis


def print_lat_lon_merge_drone(scenari):
    for scenario in scenari:
        print(' Stampo su mappa il volo dello scenario : ' + scenario)

        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scenario + '.csv',
                           delimiter=',')
        data_lat = data['latitude']
        data_lon = data['longitude']

        data = pd.concat([pd.DataFrame(data_lat), pd.DataFrame(data_lon)], axis=1)
        print_on_map(data, scenario)


def print_lat_lon_predette(scenari):
    for scenario in scenari:
        print(' Stampo su mappa il volo dello scenario : ' + scenario)

        data_lat = pd.read_csv('Steps/Step_3/Prediction/LAT_pred_scenario_' + scenario + '.csv',
                               delimiter=',')
        data_lat = data_lat.rename(columns={'0': 'latitude'})

        data_lon = pd.read_csv('Steps/Step_3/Prediction/LON_pred_scenario_' + scenario + '.csv',
                               delimiter=',')
        data_lon = data_lon.rename(columns={'0': 'longitude'})

        data = pd.concat([data_lat, data_lon], axis=1)
        print_on_map(data, scenario)


def print_lat_lon_predette_full(scenari):
    data_lat = pd.read_csv('Steps/Step_3/Prediction/LAT_pred_scenario_FULL.csv',
                           delimiter=',')
    data_lat = data_lat.rename(columns={'0': 'latitude'})

    data_lon = pd.read_csv('Steps/Step_3/Prediction/LON_pred_scenario_FULL.csv',
                           delimiter=',')
    data_lon = data_lon.rename(columns={'0': 'longitude'})

    data = pd.concat([data_lat, data_lon], axis=1)
    print_on_map(data, 'FULL')


def print_on_map(data, scenario):
    data = data[['latitude', 'longitude']]

    data.astype({"latitude": float})
    data.astype({"longitude": float})

    data['latitude'].map('{:,.6f}'.format)
    data['longitude'].map('{:,.6f}'.format)

    data.to_csv('Steps/Step_3/Prediction_Lat_Lon/lat_long_' + scenario + '.csv', header=False,
                index=False)

    vis = GPSVis(data_path='Steps/Step_3/Prediction_Lat_Lon/lat_long_' + scenario + '.csv',
                 map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
                 points=(
                     51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

    vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
    vis.plot_map(output='save', save_as='Steps/Step_3/Prediction/Prediction_Lat_Lon_scenario_' + scenario)
