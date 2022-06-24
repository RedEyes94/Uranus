import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Sensors.Alvira import Alvira
from Sensors.Arcus import Arcus
from Sensors.Diana import Diana
from Sensors.Drone import Drone
from Sensors.Venus import Venus


def data_cleaning(scenari, sensori):
    for scenario in scenari:

        if 'Alvira' in sensori or 'All' in sensori:
            alvira = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/ALVIRA_scenario.csv',
                                 delimiter=',')
            alvira = Alvira(alvira, scenario)

            '''La funzione "clean_data" per Alvira si occupa di:
                - Eliminare timestamp diversi da 'datetime(utc)' che verrà invece mantenuto
                - Split categorico per la feature 'AlviraTracksTrack_Classification'
                - Cancellazione delle features che hanno il 100% dei valori nulli
                - Creazione del file ripulito nella DIR 'DataCleanResults/Scenario_SCENARIO/[1]_ALVIRA_data_cleaned_SCENARIO'.csv'''

            alvira.clean_data()

        if 'Arcus' in sensori or 'All' in sensori:

            arcs = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/ARCUS_scenario.csv',
                               delimiter=',')
            arcs = Arcus(arcs, scenario)

            '''
            La funzione "clean_data" per Arcus si occupa di:
                - Eliminare timestamp diversi da 'datetime(utc)' che verrà invece mantenuto
                - Split categorico per le features:
                         - 'ArcusSystemStatusSensor_OperationalState'
                         - 'ArcusTracksTrack_Classification' [UNKNOWN,VEHICLE,OTHER,DRONE,SUSPECTED_DRONE,HELICOPTER] 
                         - 'ArcusTracksTrack_Alarm' : TRUE o FALSE
                - Creazione del file ripulito nella DIR 'DataCleanResults/Scenario_SCENARIO/[1]_ARCUS_data_cleaned_SCENARIO'.csv
            '''
            arcs.clean_data()

        if 'Diana' in sensori or 'All' in sensori:

            diana = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/DIANA_scenario.csv',
                                delimiter=',')
            diana = Diana(diana, scenario)

            '''
                    La funzione "clean_data" per Diana si occupa di:
                        - Eliminare timestamp diversi da 'datetime(utc)' che verrà invece mantenuto
                        - Split categorico per le features:
                        - DianaTargetsTargetClassification_type (aircraft, controller)
                        - DianaTargetsTargetClassification_model
                            (DianaClasssification_Unknown
                                DJI-MAVIC-PRO-PLATINUM
                                Wifi-Bluetooth
                                DJI-MAVIC-2-PRO
                                DJI-Phantom-4F
                                Parrot-ANAFI
                                DJI-Phantom-4E
                                SPEKTRUM-DX5e
                                SYMA-X8HW
                                DJI-MAVIC-AIR
                                None
                                VISUO-Zen)
             
                        - Creazione del file ripulito nella DIR 'DataCleanResults/Scenario_SCENARIO/[1]_DIANA_data_cleaned_SCENARIO'.csv
                    '''
            diana.clean_data()

        if 'Venus' in sensori or 'All' in sensori:

            venus = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/VENUS_scenario.csv',
                                delimiter=',')
            venus = Venus(venus, scenario)

            '''
                            La funzione "clean_data" per Venus si occupa di:
                                - Eliminare timestamp diversi da 'datetime(utc)' che verrà invece mantenuto
                                - Split categorico per le features:
                                - VenusTrigger_VenusName 
                                    DJI OcuSync
                                    DJI Mavic Mini
                                    Cheerson Leopard 2
                                    DJI Mavic Pro long
                                    nan
                                    DJI Mavic Pro short
                                    Hubsan
                                    Futaba FASST-7CH Var. 1
                                    AscTec Falcon 8 Downlink, DJI Mavic Mini
                                    DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G
                                    DJI Mavic Mini, MJX R/C Technic
                                    Udi R/C 818A
                                    MJX R/C Technic
                                    TT Robotix Ghost
                                    Udi R/C
                                    'DJI Mini, DJI Phantom 4 Pro/Mavic Pro, DJI Phantom 4/Mavic Pro
                                    DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G
                                    Spektrum DSMX downlink
                                    Spektrum DSMX 12CH uplink
                                    MJX X901
                                    'DJI Mavic Pro long, DJI Phantom 4/Mavic Pro,DJI Phantom/Mavic Pro
                                    AscTec Falcon 8 Downlink
    
                                - Creazione del file ripulito nella DIR 'DataCleanResults/Scenario_SCENARIO/[1]_VENUS_data_cleaned_SCENARIO'.csv
                            '''
            venus.clean_data()

        '''
        Se sono nel secondo scenario leggo i file di log per entrambi i droni
        '''
        if scenario[:1] == '2':
            drone = pd.read_csv(
                'Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/2020-09-30_' + scenario + 'a.csv',
                delimiter=',')
            drone = Drone(drone, scenario, 'a')
            drone.clean_data()

            drone = pd.read_csv(
                'Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/2020-09-30_' + scenario + 'b.csv',
                delimiter=',')
            drone = Drone(drone, scenario, 'b')
            drone.clean_data()
        else:
            if scenario[:1] == '1':
                drone = pd.read_csv(
                    'Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/2020-09-29_' + scenario + '.csv',
                    delimiter=',')
                drone = Drone(drone, scenario, 'n')
                drone.clean_data()
            else:
                '''
                Se sono nello scenario Parrot leggo le rilevazioni del Parrot
                '''
                drone = pd.read_csv(
                    'Scenari/Scenario_' + scenario[
                                          :1] + '/Scenario_' + scenario + '/2020_29-09-2020-13-58-31-Flight-Airdata_corrected.csv',
                    delimiter=',')
                drone = Drone(drone, scenario, 'n')
                drone.clean_data()


def start_analisys(sensor,type='Train'):
    print('-------------------------------------------------------------------------------------------------------')
    print('                                      START - Analisi ' + sensor.name + ' scenario :  ' + sensor.scenario)
    print('-------------------------------------------------------------------------------------------------------')

    print(' --------------------------------------------------')
    print(' 1 - Analisi dei valori nulli HEATMAP:  ' + sensor.scenario)
    print(' --------------------------------------------------')
    missing_data_heatmap(sensor.data, sensor.name, sensor.scenario, type)

    print(' --------------------------------------------------')
    print(' 2 - Analisi dei valori nulli ISTOGRAMMA:  ' + sensor.scenario)
    print(' --------------------------------------------------')
    missing_data_histogram(sensor.data, sensor.name, sensor.scenario,type)

    print(' --------------------------------------------------')
    print(' 3 - Calcolo percentuale valori nulli:  ' + sensor.scenario)
    print(' --------------------------------------------------')
    feature_missing_rate(sensor.data, sensor.name, sensor.scenario,type)

    print('-------------------------------------------------------------------------------------------------------')
    print('                                      END - Analisi ' + sensor.name + ' scenario :  ' + sensor.scenario)
    print('-------------------------------------------------------------------------------------------------------')


def data_analisys(scenari, sensori):
    for scenario in scenari:

        if 'Alvira' in sensori or 'All' in sensori:
            alvira = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/ALVIRA_scenario.csv',
                                 delimiter=',')
            alvira = Alvira(alvira, scenario)

            start_analisys(alvira)

        if 'Arcus' in sensori or 'All' in sensori:
            arcs = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/ARCUS_scenario.csv',
                               delimiter=',')
            arcs = Arcus(arcs, scenario)
            arcs.data = arcs.data[:4000]
            start_analisys(arcs)

        if 'Diana' in sensori or 'All' in sensori:
            diana = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/DIANA_scenario.csv',
                                delimiter=',')
            diana = Diana(diana, scenario)

            start_analisys(diana)

        if 'Venus' in sensori or 'All' in sensori:
            venus = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/VENUS_scenario.csv',
                                delimiter=',')
            venus = Venus(venus, scenario)

            start_analisys(venus)


'''
La funzione stampa a video la heatmap per tutte le features rispetto ai valori nulli
'''


def missing_data_heatmap(df, sensore, scenario, type):
    plt.subplots_adjust(bottom=0.55)
    cols = df.columns[:]
    colours = ['#000099', '#ffff00']
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

    if type == 'Test':
        plt.savefig('DataAnalysisResults/Heatmap/Test/heatmap_' + sensore + '_' + scenario + '.pdf')
    else:
        plt.savefig('DataAnalysisResults/Heatmap/Training/heatmap_'+sensore+'_'+scenario+'.pdf')

    plt.show()


'''
La funzione stampa a video l'istogramma per tutte le features rispetto ai valori nulli
'''


def missing_data_histogram(df, sensore, scenario, type):
    for col in df.columns:
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            print('created missing indicator for: {}'.format(col))
            df['{}_ismissing'.format(col)] = missing

    ismissing_cols = [col for col in df.columns if 'ismissing' in col]
    df['num_missing'] = df[ismissing_cols].sum(axis=1)

    df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')
    if type == 'Test':
        plt.savefig('DataAnalysisResults/Histogram/Test/missing_values_histogram_' + sensore + '_' + scenario + '.pdf')
    else:
        plt.savefig('DataAnalysisResults/Histogram/Training/missing_values_histogram_' + sensore + '_' + scenario + '.pdf')

    #plt.show()


'''
La funzione calcola in termini % la numerosità dei valori nulli per ogni feature
'''


def feature_missing_rate(df, sensor, scenario, type):
    result = np.array([])
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, (pct_missing * 100)))
        result = np.append(result, [col, (pct_missing * 100)])
    if type == 'Test':
        pd.DataFrame(result).to_csv('DataAnalysisResults/FeaturesMissingRate/Test/missing_rate_'+sensor+'_'+scenario+'.csv', index=None)
    else:
        pd.DataFrame(result).to_csv('DataAnalysisResults/FeaturesMissingRate/Training/missing_rate_'+sensor+'_'+scenario+'.csv', index=None)
