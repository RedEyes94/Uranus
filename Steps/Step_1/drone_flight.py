import pandas as pd

from Utility.GPS import GPSVis


def print_drone_flight(scenari):
    for scenario in scenari:
        print(' Stampo su mappa il volo dello scenario : ' + scenario)

        if scenario[:1] == '1':
            data = pd.read_csv('Scenari/Scenario_1/Scenario_' + scenario + '/2020-09-29_' + scenario + '.csv',
                               delimiter=',')

            print_on_map(data, scenario)

        elif scenario[:1] == '2':
            data = pd.read_csv('Scenari/Scenario_2/Scenario_' + scenario + '/2020-09-30_' + scenario + 'a.csv',
                               delimiter=',')
            print_on_map(data, scenario)

            data = pd.read_csv('Scenari/Scenario_2/Scenario_' + scenario + '/2020-09-30_' + scenario + 'b.csv',
                               delimiter=',')
            print_on_map(data, scenario)

        elif scenario == '3':
            data = pd.read_csv('Scenari/Scenario_3/Scenario_3/2020_29-09-2020-13-58-31-Flight-Airdata_corrected.csv',
                               delimiter=',')

            print_on_map(data, scenario)


def print_on_map(data, scenario):
    data = data[['latitude', 'longitude']]

    data.astype({"latitude": float})
    data.astype({"longitude": float})

    data['latitude'].map('{:,.6f}'.format)
    data['longitude'].map('{:,.6f}'.format)

    data.to_csv('Steps/Step_1/AlviraMapsClustering/alvira_lat_long_' + scenario + '.csv', header=False,
                index=False)

    vis = GPSVis(data_path='Steps/Step_1/AlviraMapsClustering/alvira_lat_long_' + scenario + '.csv',
                 map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
                 points=(
                     51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

    vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
    vis.plot_map(output='save', save_as='Drone_flight_scenario_' + scenario)
