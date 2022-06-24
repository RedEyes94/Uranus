import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def merge_all_sensors(scenari):
    for scenario in scenari:
        print('-> Scenario : ' + scenario)
        alvira = pd.read_csv('AlviraClusteringResults/Training/Alvira_data_predict_' + scenario + '.csv',
                             delimiter=',')
        arcus = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        diana = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_DIANA_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        venus = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_VENUS_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        if scenario[:1] == '1':
            drone = pd.read_csv(
                'DataCleanResults/Scenario_' + scenario + '/[1]_DRONE_data_cleaned_' + scenario + '.csv',
                delimiter=',')
        if scenario[:1] == '2':
            drone_a = pd.read_csv(
                'DataCleanResults/Scenario_' + scenario + '/[1]_DRONE_data_cleaned_' + scenario + 'a.csv',
                delimiter=',')
            drone_b = pd.read_csv(
                'DataCleanResults/Scenario_' + scenario + '/[1]_DRONE_data_cleaned_' + scenario + 'b.csv',
                delimiter=',')
            drone = pd.concat([drone_a, drone_b], axis=0)
        if scenario[:1] == '3':
            drone = pd.read_csv(
                'DataCleanResults/Scenario_' + scenario + '/[1]_DRONE_data_cleaned_' + scenario + '.csv',
                delimiter=',')

        merge1 = pd.merge(alvira, arcus, how='outer', on=["datetime(utc)"]).drop_duplicates()
        merge2 = pd.merge(diana, venus, how='outer', on=["datetime(utc)"]).drop_duplicates()

        merge = pd.merge(merge1, merge2, how='inner', on=["datetime(utc)"]).drop_duplicates()
        drone = drone.groupby('datetime(utc)').last()
        merge_to_drone = pd.merge(merge, drone, how='inner', on=["datetime(utc)"]).drop_duplicates()
        merge_to_drone = merge_to_drone.fillna(value=0)
        merge_to_drone = shuffle(merge_to_drone)
        merge_to_drone.to_csv('MergeResults/Training/[3]_MERGE_scenario_' + scenario + '.csv',
                              float_format='%.6f')
        merge_to_drone.describe()
