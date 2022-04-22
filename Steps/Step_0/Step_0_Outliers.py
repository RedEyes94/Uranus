import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def calc_rem_outliers(scenari, print_outliers):
    print('--START OUTLIERS--')

    for scenario in scenari:
        print('-> Scenario : ' + scenario)
        alvira = pd.read_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_ALVIRA_data_cleaned_' + scenario + '.csv',
                             delimiter=',')
        if print_outliers:
            sns.boxplot(x=alvira['AlviraTracksTrackPosition_Altitude'])
            plt.show()
            sns.boxplot(x=alvira['AlviraTracksTrackVelocity_Azimuth'])
            plt.show()
            sns.boxplot(x=alvira['AlviraTracksTrackVelocity_Speed'])
            plt.show()
        alvira.describe().to_csv('Steps/Step_0/Alvira_Outliers_scenario_' + scenario + '.csv')

        arcus = pd.read_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        if print_outliers:
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Latitude'])
            plt.show()
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Longitude'])
            plt.show()
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Altitude'])
            plt.show()

        '''
        Con questa funzione vengono rimossi gli outliers su Arcus
        '''
        arcus = clean_arcus_outliers(arcus)

        if print_outliers:
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Latitude'])
            plt.show()
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Longitude'])
            plt.show()
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Altitude'])
            plt.show()

        arcus.to_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv')

        arcus.describe().to_csv('Steps/Step_0/Arcus_Outliers_scenario_' + scenario + '.csv')

        diana = pd.read_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_DIANA_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        if print_outliers:
            sns.boxplot(x=diana['DianaTarget_ID'])
            plt.show()
            sns.boxplot(x=diana['DianaTargetsTargetSignal_snr_dB'])
            plt.show()
            sns.boxplot(x=diana['DianaTargetsTargetSignal_bearing_deg'])
            plt.show()
            sns.boxplot(x=diana['DianaTargetsTargetSignal_range_m'])
            plt.show()
        diana.describe().to_csv('Steps/Step_0/Diana_Outliers_scenario_' + scenario + '.csv')

        venus = pd.read_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_VENUS_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        if print_outliers:
            sns.boxplot(x=venus['VenusTrigger_RadioId'])
            plt.show()
            sns.boxplot(x=venus['VenusTrigger_Azimuth'])
            plt.show()

        venus.describe().to_csv('Steps/Step_0/Venus_Outliers_scenario_' + scenario + '.csv')
        print('--END OUTLIERS--')


def clean_arcus_outliers(arcs1):
    min = arcs1['ArcusTracksTrackPosition_Latitude'].quantile(0.25)
    max = arcs1['ArcusTracksTrackPosition_Latitude'].quantile(0.75)

    arcs1["ArcusTracksTrackPosition_Latitude"] = np.where(arcs1["ArcusTracksTrackPosition_Latitude"] < min, min,
                                                          arcs1['ArcusTracksTrackPosition_Latitude'])
    arcs1["ArcusTracksTrackPosition_Latitude"] = np.where(arcs1["ArcusTracksTrackPosition_Latitude"] > max, max,
                                                          arcs1['ArcusTracksTrackPosition_Latitude'])

    min = arcs1['ArcusTracksTrackPosition_Longitude'].quantile(0.25)
    max = arcs1['ArcusTracksTrackPosition_Longitude'].quantile(0.75)

    arcs1["ArcusTracksTrackPosition_Longitude"] = np.where(arcs1["ArcusTracksTrackPosition_Longitude"] < min, min,
                                                           arcs1['ArcusTracksTrackPosition_Longitude'])
    arcs1["ArcusTracksTrackPosition_Longitude"] = np.where(arcs1["ArcusTracksTrackPosition_Longitude"] > max, max,
                                                           arcs1['ArcusTracksTrackPosition_Longitude'])

    arcs1 = arcs1.groupby('datetime(utc)').mean()
    arcs1 = arcs1.reset_index()

    return arcs1
