from Steps.Step_0.Step_0_DataAnalisysAndClean import data_analisys, data_cleaning
from Steps.Step_0.Step_0_Outliers import calc_rem_outliers
from Steps.Step_1.Step_1_clustering import clustering_alvira
from Steps.Step_1.drone_flight import print_drone_flight
from Steps.Step_2.merge_all_sensors import merge_all_sensors
from Steps.Step_3.Step_3_1_predict_lat_lon import predict_lat_full, predict_lon_full, predict_lat_per_scenario, \
    predict_lon_per_scenario
from Steps.Step_3.Step_3_2_predict_speed import predict_speed_per_scenario, predict_speed
from Steps.Step_3.Step_3_3_predict_drone_type import predict_drone_type
from Steps.Step_3.anova import anova_feature_selection
from Steps.Step_3.predict_altitude import predict_altitude_per_scenario
from Steps.Step_3.print_pred_lat_lon import print_lat_lon_predette, print_lat_lon_predette_full, \
    print_lat_lon_merge_drone
from Steps.Step_4.Step_4_test_scenario import start_prediction
from Utility.arcus_meno_alvira import arcus_meno_alvira
from Utility.check_parameters import check_parameters
from Utility.plot_speed_alt import print_altitude, print_velocity

import sys

print('')
print('*************************************************')
print('               ***  URANUS **')
print('*************************************************')
print('')

n = len(sys.argv)
print("Total arguments passed:", n)

print("\nName of Python script:", sys.argv[0])

print("\nArguments passed:", end=" ")
for i in range(1, n):
    print(sys.argv[i], end=" ")

#check_parameters(sys.argv)

"""
    La funzione "data_analisys" si occupa di:
         - Stampare heatmap valori nulli per feature
         - Stampare istogramma valori nulli per feature
         - Stampare % valori nulli per feature

    :param scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]

    :return: 0

"""
scenari = ['1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '3']
sensori = ['Alvira', 'Arcus', 'Diana', 'Venus']

if 'Analisys' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: data_analisys <<<<')
    data_analisys(scenari, sensori)
    print('')
    print('')
    print('>>>> End function: data_analisys <<<<')

if 'DataCleaning' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: data_cleaning <<<<')
    data_cleaning(scenari, sensori)
    print('')
    print('')
    print('>>>> End function: data_cleaning <<<<')

'''
 La funzione "arcus_meno_alvira" si occupa di:
 
    Per ogni scenario indicato in input genera un file al path: 
    1-DataCleaned/Scenario_<SCENARIO>/[1]_ARCUS_data_cleaned_<SCENARIO>.csv'
    contenente tutti i dati presenti in arcus che non sono presenti in Alvira
    
    :param scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]

    :return: 0
'''
print('')
print('')
print('>>>> Start function: data_cleaning <<<<')
arcus_meno_alvira(scenari)
print('')
print('')
print('>>>> End function: data_cleaning <<<<')
'''
 La funzione "calc_rem_outliers" si occupa di:

    Per ogni scenario indicato in input genera un file al path: 
    Steps/Step_0/Venus_Outliers_scenario_<SCENARIO>.csv
    contenente tutti i dati di arcus ripuliti dagli outliers in relazione alle 
    features latitude longitude. Inoltre stampa i box plot, se il parametro print_outliers
    è valorizzato a True per tutti i sensori

    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
        print_outliers : True per stampare i box plot relativi agli outliers, False altrimenti
    :return: 0
'''
if 'Outliers' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: calc_rem_outliers <<<<')
    calc_rem_outliers(scenari, print_outliers=False)
    print('')
    print('')
    print('>>>> End function: calc_rem_outliers <<<<')
'''
 La funzione "print_drone_flight" si occupa di:

    Per ogni scenario indicato in input legge dal csv presente al path: 
    Scenari/Scenario_<n_scenario>/Scenario_<SCENARIO>/2020-09-29_<SCENARIO>.csv'
    le latitudini e longitudini del volo reale del drone e le stampa su mappa

    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
    :return: 0
'''
if 'PrintDroneFlight' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: print_drone_flight <<<<')
    print_drone_flight(scenari)
    print('')
    print('')
    print('>>>> End function: print_drone_flight <<<<')
'''
 La funzione "clustering_alvira" si occupa di:

    Per ogni scenario indicato in input genera un csv nel path: 
    '2-<SENSORE>Clustering/[2]_<SENSORE>_data_predict_<SCENARIO>.csv'
    contenente i dati clusterizzati con Kmeans.
    L'alogritmo di default è il k-means che è risultato il migliore 
    per perfomance ottenute

    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
        calc_k : se True mostra il diagramma a "gomito" per la scelta del valore k, False altrimenti
        k      : numero totale di cluster sul quale calcolare il calcolo del migliore valore di k
        clustering_alg : nome dell'algoritmo di clustering (Kmeans, GMM, Agglomerative)
    :return: 0
'''
if 'ClusteringAlvira' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: clustering_alvira <<<<')
    clustering_alvira(scenari, calc_k=False, k=5, clustering_alg='Kmeans')
    print('')
    print('')
    print('>>>> End function: clustering_alvira <<<<')

'''
 La funzione "merge_all_sensors" si occupa di:

    Per ogni scenario indicato in input genera un csv nel path: 
    '3-DataMerged/[3]_MERGE_scenario_<SCENARIO>.csv''
    contenente i dati mergiati di tutti i sensori rispetto allo scenario specifico
    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
        
    :return: 0
'''
print('')
print('')
print('>>>> Start function: merge_all_sensors <<<<')
merge_all_sensors(scenari)
print('')
print('')
print('>>>> End function: merge_all_sensors <<<<')
'''
 La funzione "anova_feature_selection" si occupa di:

    Per ogni scenario e feature indicata in input stampare il diagramma Anova  
    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
        feature : Indicare la feature da analizzare con anova (drone, latitude, longitude, speed, altitude)
    :return: 0
'''
if 'Anova' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: merge_all_sensors <<<<')
    anova_feature_selection(scenari, feature='latitude')
    print('')
    print('')
    print('>>>> End function: merge_all_sensors <<<<')
'''
 Le funzioni "predict_lat_full" e "predict_lon_full" si occupano di:

    Per ogni scenario indicato legge i dati dal path 3-DataMerged/[3]_MERGE_scenario_<SCENARIO>.csv
     creano il predittore rispettivamente per LATITUDINE e LONGITUDINE
    salvano il modello generato ai path:
        - ModelliEstratti/model_latitude_FULL_1.pickle', 'wb'
        - ModelliEstratti/model_longitude_FULL_2.pickle', 'wb'
        - ModelliEstratti/model_longitude_FULL_3.pickle', 'wb'
    e generano il csv con le relative latitudini e longitudini predette al path:
        - Steps/Step_3/Prediction/LAT_pred_scenario_<SCENARIO>.csv'
        - Steps/Step_3/Prediction/LON_pred_scenario_<SCENARIO>.csv'
    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
    :return: 0
'''
if 'TrainModel_Lat_Lon' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: predict_lat, predict_lon <<<<')
    predict_lat_per_scenario(scenari)
    predict_lon_per_scenario(scenari)
    print('')
    print('')
    print('>>>> End function: predict_lat, predict_lon <<<<')
'''
 La funzione "print_lat_lon_predette" si occupa di:
    
    Stampare su mappa le latitudini e longitudini predette recuperandole dai csv
        - Steps/Step_3/Prediction/LAT_pred_scenario_<SCENARIO>.csv'
        - Steps/Step_3/Prediction/LON_pred_scenario_<SCENARIO>.csv'
    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
    :return: 0
'''
if 'Print_Lat_Lon_Training' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: print_lat_lon_predette <<<<')
    print_lat_lon_predette(scenari)
    print('')
    print('')
    print('>>>> End function: print_lat_lon_predette <<<<')
# obsoleta
# print_lat_lon_merge_drone(scenari)

# obsoleta
# predict_speed(scenari)

'''
 La funzione "predict_speed_per_scenario" si occupa di:

    Creare un predittore per la feature speed, legge i dati da 
    3-DataMerged/[3]_MERGE_scenario_<SCENARIO>.csv e crea un predittore nella folder
    - ModelliEstratti/model_speed_FULL<SCENARIO>.pickle'
    
    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
    :return: 0
'''
if 'TrainModel_Speed' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: predict_speed_per_scenario <<<<')
    predict_speed_per_scenario(scenari)
    print('')
    print('')
    print('>>>> End function: predict_speed_per_scenario <<<<')
#  OBS
# predict_altitude(scenari)
'''
 La funzione "predict_altitude_per_scenario" si occupa di:

    Creare un predittore per la feature speed, legge i dati da 
    3-DataMerged/[3]_MERGE_scenario_<SCENARIO>.csv e crea un predittore nella folder
    - ModelliEstratti/model_altitude_FULL<SCENARIO>.pickle'

    :param 
        scenari: Indicare gli scenari da analizzare nel formato [1_1, 2_1, 3]
    :return: 0
'''
if 'TrainModel_Altitude' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: predict_altitude_per_scenario <<<<')
    predict_altitude_per_scenario(scenari)
    print('')
    print('')
    print('>>>> End function: predict_altitude_per_scenario <<<<')
'''
 La funzione "predict_drone_type" si occupa di:

    Creare un predittore per la feature speed, legge i dati da 
    3-DataMerged/[3]_MERGE_scenario_<SCENARIO>.csv e crea un predittore nella folder
    - ModelliEstratti/model_<NOME_DRONE>.pickle'

    :param :
    :return: 0
'''
if 'TrainModel_DroneType' in sys.argv or 'All' in sys.argv:
    print('')
    print('')
    print('>>>> Start function: predict_drone_type <<<<')
    predict_drone_type()
    print('')
    print('')
    print('>>>> End function: predict_drone_type <<<<')
'''
 La funzione "start_prediction" si occupa di:

    Avviare la predizione sugli scenari di test, esegue gli step di pulizia e merge.
    Subito dopo applica i modelli generati negli step precedenti e stampa su mappa i risultati
    ottenuti in un file "SubmissionFileScenario_<SCENARIO>.csv
    
    :param 
            -numero : identificativo numerico dello scenario
            -lettera: identificativo lettera dello scenario
            -model  : indica il modello che si vuole utilizzare per le previsioni (di default FULL3, gli altri
                      modelli sono obsoleti)
    :return: 0
'''
if 'StartPrediction' in sys.argv or 'All' in sys.argv:
    print('')
    print('***  Maps with results of prediction will be located at Steps/Step_4/Latitude_Longitude_predette - '
          'Full_prediction_Lat_lon_scenario_<scenario>.png ***')
    print('')
    if '1a' in sys.argv or 'All' in sys.argv:
        start_prediction(numero='1', lettera='a', model='FULL3')
    elif '1b' in sys.argv or 'All' in sys.argv:
        start_prediction(numero='1', lettera='b', model='FULL3')
    elif '2a' in sys.argv or 'All' in sys.argv:
        start_prediction(numero='2', lettera='a', model='FULL3')
    elif '2b' in sys.argv or 'All' in sys.argv:
        start_prediction(numero='2', lettera='b', model='FULL3')
    elif '2c' in sys.argv or 'All' in sys.argv:
        start_prediction(numero='2', lettera='c', model='FULL3')
    elif '2d' in sys.argv or 'All' in sys.argv:
        start_prediction(numero='2', lettera='d', model='FULL3')
    elif '3a' in sys.argv or 'All' in sys.argv:
        start_prediction(numero='3', lettera='a', model='FULL3')
    else:
        print('Scenario not found')
'''
 Le funzioni "print_altitude" e "print_velocity" si occupano di:

    Mostrare un grafico con la velocità e l'altitudine predetta per lo scenario indicato in input

    :param 
            -scenario : identificativo lo scenario di test 1a,1b,2a,2b,2c,2d,3a
    :return: 0
'''

if 'PrintAltitudePrediction' in sys.argv or 'All' in sys.argv:
    if '1a' in sys.argv:
        print_altitude('1a')
    elif '1b' in sys.argv:
        print_altitude('1b')
    elif '2a' in sys.argv:
        print_altitude('2a')
    elif '2b' in sys.argv:
        print_altitude('2b')
    elif '2c' in sys.argv:
        print_altitude('2c')
    elif '2d' in sys.argv:
        print_altitude('2d')
    elif '3a' in sys.argv:
        print_altitude('3a')

if 'PrintVelocityPrediction' in sys.argv or 'All' in sys.argv:
    if '1a' in sys.argv:
        print_velocity('1a')
    elif '1b' in sys.argv:
        print_velocity('1b')
    elif '2a' in sys.argv:
        print_velocity('2a')
    elif '2b' in sys.argv:
        print_velocity('2b')
    elif '2c' in sys.argv:
        print_velocity('2c')
    elif '2d' in sys.argv:
        print_velocity('2d')
    elif '3a' in sys.argv:
        print_velocity('3a')

