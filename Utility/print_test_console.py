from Utility.GPS import GPSVis


def print_on_test_map(data,scenario):
    data = data[['TrackPositionLatitude', 'TrackPositionLongitude']]

    data.astype({"TrackPositionLatitude": float})
    data.astype({"TrackPositionLongitude": float})

    data['TrackPositionLatitude'].map('{:,.6f}'.format)
    data['TrackPositionLongitude'].map('{:,.6f}'.format)

    data.to_csv('Utility/app_csv_test/lat_long_' + scenario + '.csv', header=False,
                index=False)

    vis = GPSVis(data_path='Utility/app_csv_test/lat_long_' + scenario + '.csv',
                 map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
                 points=(
                     51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

    vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
    vis.plot_map(output='plot', save_as='Steps/Step_3/Prediction/Prediction_Lat_Lon_scenario_' + scenario)