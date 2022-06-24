from Steps.Step_0.DataAnalisysAndClean import data_analisys, data_cleaning
from Steps.Step_0.Outliers import calc_rem_outliers
from Steps.Step_1.clustering import clustering_alvira
from Steps.Step_1.drone_flight import print_drone_flight
from Steps.Step_2.merge_all_sensors import merge_all_sensors
from Steps.Step_3.ANOVA import anova_feature_selection
from Steps.Step_3.predict_altitude import predict_altitude
from Steps.Step_3.predict_drone_type import predict_drone_type
from Steps.Step_3.predict_latitude_longitude import predict_latitude, predict_longitude
from Steps.Step_3.predict_speed import predict_speed
from Steps.Step_3.print_pred_lat_lon import print_lat_lon_predette
from Steps.Step_4.test_scenario import start_prediction
from Utility.arcus_meno_alvira import arcus_minus_alvira

print('')
print('******************************************')
print('              ***  URANUS  ***            ')
print('******************************************')
print('')

scenari = ['1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '3']
sensori = ['Alvira', 'Arcus', 'Diana', 'Venus']

print('')
print('')
print('>>>> Start function: data_analisys <<<<')
"""
    The "data_analys" function deals with:
         - Printing the null values of the heat map for each feature
         - Printing the null values of the histogram by feature
         - Print null values % per element

    Param 
         - scenarios: Indicates the scenarios to be analyzed in the format [1_1, 2_1, 3].
         - sensors  : Indicates sensors (Arcus, Alvira etc.)
    :return: 0

"""
data_analisys(scenari, sensori)
print('')
print('')
print('>>>> End function: data_analisys <<<<')

data_cleaning(scenari, sensori)

'''
The " arcus_minus_alvira" function deals with:
 
    For each scenario given as input it generates a file at path: 
    'DataCleanResults/Scenario_<SCENARIO>/[1]_ARCUS_data_cleaned_<SCENARIO>.csv'
    containing all data present in arcus that is not present in Alvira
    
    :param scenarios: Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3].

    :return: 0
'''

print('')
print('')
print('>>>> Start function: arcus_meno_alvira <<<<')
arcus_minus_alvira(scenari)
print('>>>> End function: arcus_meno_alvira <<<<')
print('')
print('')
'''
 The "calc_rem_outliers" function deals with:

    For each scenario given as input it generates a file at path: 
    OutliersResults/ResultValues/<Sensor>_Outliers_scenario_<SCENARIO>.csv
    containing all arcus data cleaned of outliers in relation to the 
    latitude longitude features. It also prints box plots and save it in OutliersResults/BoxPlot/, 

    :param 
        scenarios : Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
        print_outliers : True to print box plots related to outliers, False otherwise.
    :return: 0

'''
print('')
print('')
print('>>>> Start function: calc_rem_outliers <<<<')
calc_rem_outliers(scenari, print_outliers=True)
print('>>>> End function: calc_rem_outliers <<<<')
print('')
print('')
'''
 The "print_drone_flight" function deals with:

    For each scenario given as input reads from the csv present at path: 
    'Scenarios/Scenario_<n_scenario>/Scenario_<SCENARIO>/2020-09-29_<SCENARIO>.csv'
    the latitudes and longitudes of the actual flight of the drone and prints them to a map

    :param 
        scenarios: Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
    :return: 0
'''
print('')
print('')
print('>>>> Start function: print_drone_flight <<<<')
print_drone_flight(scenari)
print('')
print('')
print('>>>> End function: print_drone_flight <<<<')
'''
 The "clustering_alvira" function deals with:

    For each scenario given in input it generates a csv in the path: 
    'AlviraClusteringResults/Training/Alvira_data_predict_<SCENARIO>.csv'
    containing the data clustered with Kmeans. 
    Generates in the path AlviraClusteringResults/Training/LatitudeLongitudeValues
    the latitude and longitude used to print on map the flight path.
    The default clustering alogrithm is k-means which was found to be the best 
    for perfomance obtained

    :param 
        scenarios : Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
        calc_k : if True shows the "elbow" diagram for choosing the k value, False otherwise
        k : total number of clusters on which to calculate the calculation of the best value of k
        clustering_alg : name of clustering algorithm (Kmeans, GMM, Agglomerative)
    :return: 0

'''
print('')
print('')
print('>>>> Start function: clustering_alvira <<<<')

clustering_alvira(scenari, calc_k=False, k=5, clustering_alg='Agglomerative')

print('')
print('')
print('>>>> End function: clustering_alvira <<<<')
'''
 The "merge_all_sensors" function deals with:

    For each scenario given as input it generates a csv in the path: 
    'MergeResults/Training/[3]_MERGE_scenario_<SCENARIO>.csv''
    containing the merged data of all sensors with respect to the specific scenario
    :param 
        scenarios: Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
        
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
 The "anova_feature_selection" function deals with:

    For each scenario and feature given as input print the Anova diagram  
    :param 
        scenarios : Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
        feature : Indicate the feature to be analyzed with anova (drone, latitude, longitude, speed, altitude)
    :return: 0
'''
print('')
print('')
print('>>>> Start function: anova_feature_selection <<<<')
features = ['latitude','longitude','speed(mps)','altitude(m)','Dronetype']
k_sensors = ['Alvira','Arcus']
r_sensors = ['Diana','Venus']

for sens in k_sensors:
    for feat in features:
        anova_feature_selection(scenari, sensore=sens, feature=feat)

anova_feature_selection(scenari,sensore='Diana', feature='Dronetype')
anova_feature_selection(scenari,sensore='Venus', feature='Dronetype')

print('')
print('')
print('>>>> End function: anova_feature_selection <<<<')
'''
 The "predict_latitude" and "predict_longitude" functions deal with:

    For each given scenario it reads data from the MergeResults/Training/[3]_MERGE_scenario_<SCENARIO>.csv path
     create the predictor for LATITUDE and LONGITUDE, respectively
    save the generated model to the paths:
        - Models/model_latitude.pickle', 'wb'
        - Models/model_longitude.pickle', 'wb'
        - Models/model_longitude.pickle', 'wb'
    and generate the csv with their predicted latitudes and longitudes at the path:
        - PredictionResults/Training/latitude_scenario_<SCENARIO>.csv
    :param 
        scenarios: Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
    :return: 0
'''
print('')
print('')
print('>>>> Start function: predict_latitude, predict_longitude <<<<')
predict_latitude(scenari)
predict_longitude(scenari)
print('')
print('')
print('>>>> End function: predict_lat_per_scenario, predict_lon_per_scenario <<<<')
'''
 The "print_lat_lon_predict" function deals with:
    
    Print the predicted latitudes and longitudes on the map by retrieving them from the csvs
        - PredictionResults/Training/Latitude_Longitude/latitude_pred_scenario_<SCENARIO>.csv
        - PredictionResults/Training/Latitude_Longitude/longitude_pred_scenario_<SCENARIO>.csv
    :param 
        scenarios: Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
    :return: 0
'''
print('')
print('')
print('>>>> Start function: print_lat_lon_predette <<<<')
print_lat_lon_predette(scenari)
print('')
print('')
print('>>>> End function: print_lat_lon_predette <<<<')

'''
 The "predict_speed" function deals with:

    Create a predictor for feature speed, reads data from 
    MergeResults/[3]_MERGE_scenario_<SCENARIO>.csv and creates a predictor in the folder
    - Models/model_speed.pickle' 
    :param 
        scenarios: Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
    :return: 0
'''
print('')
print('')
print('>>>> Start function: predict_speed_per_scenario <<<<')
predict_speed(scenari)
print('')
print('')
print('>>>> End function: predict_speed_per_scenario <<<<')

'''
 The "predict_altitude" function deals with:

    Create a predictor for feature speed, reads data from 
    MergeResults/[3]_MERGE_scenario_<SCENARIO>.csv and creates a predictor in the folder
    - Models/model_altitude.pickle'

    :param 
        scenarios: Indicate the scenarios to be analyzed in the format [1_1, 2_1, 3]
    :return: 0
'''
print('')
print('')
print('>>>> Start function: predict_altitude_per_scenario <<<<')
predict_altitude(scenari)
print('')
print('')
print('>>>> End function: predict_altitude_per_scenario <<<<')
'''
 The "predict_drone_type" function deals with:

    Create a predictor for feature speed, reads data from 
    MergeResults/[3]_MERGE_scenario_<SCENARIO>.csv and creates a predictor in the folder
    - Models/model_<DRONE_NAME>.pickle'

    :param :
    :return: 0
'''
print('')
print('')
print('>>>> Start function: predict_drone_type <<<<')
predict_drone_type()

print('')
print('')
print('>>>> End function: predict_drone_type <<<<')
'''
 The "start_prediction" function is responsible for:

    Start prediction on the test scenarios, performs the cleanup and merge steps.
    Immediately afterwards it applies the models generated in the previous steps and prints the results on a map
    obtained in a file "SubmissionFileScenario_<SCENARIO>.csv.
    
    :param 
            -number : numerical identifier of the scenario (1,2,3)
            -letter : letter identifier of the scenario ('a','b','c','d')
                - All scenarios are : 1a, 1b, 2a, 2b, 2c, 2d, 3a
            -model : indicates the model you want to use for forecasting (by default FULL3, other
                      models are obsolete)
    :return: 0
'''
start_prediction(numero='1', lettera='a')
start_prediction(numero='1', lettera='b')
start_prediction(numero='2', lettera='a')
start_prediction(numero='2', lettera='b')
start_prediction(numero='2', lettera='c')
start_prediction(numero='2', lettera='d')
start_prediction(numero='3', lettera='a')
