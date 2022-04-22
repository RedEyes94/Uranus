import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale, MinMaxScaler
import numpy as np
from Sensori.Alvira import Alvira
from Sensori.Arcus import Arcus
from Sensori.Diana import Diana
from Sensori.Venus import Venus
from Steps.Step_0.Step_0_Outliers import clean_arcus_outliers
from Steps.Step_1.Step_1_clustering import k_means
from Utility.GPS import GPSVis
import pickle
import matplotlib.pyplot as plt


def merge_all_sensor_test(alvira, arcus, diana, venus, scenario):
    merge1 = pd.merge(alvira, arcus, how='outer', on=["datetime(utc)"]).drop_duplicates()
    merge2 = pd.merge(diana, venus, how='outer', on=["datetime(utc)"]).drop_duplicates()

    merge = pd.merge(merge1, merge2, how='inner', on=["datetime(utc)"]).drop_duplicates()

    merge = merge.fillna(value=0)
    merge.to_csv('3-DataMerged/[3-TEST]_MERGE_scenario_' + scenario + '.csv',
                 float_format='%.6f')


def arcus_meno_alvira_test(alvira, arcus, scenario):
    arcus_meno_alvira = arcus[~arcus['datetime(utc)'].isin(alvira['datetime(utc)'])].drop_duplicates()

    arcus_meno_alvira.to_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv')

    return arcus_meno_alvira


def calc_rem_outliers_test(arcus):
    arcus = clean_arcus_outliers(arcus)

    return arcus


def print_lat_long_prediction_for_drone(scenario, numero, lettera, track_id):
    vis = GPSVis(
        data_path='Steps/Step_4/Latitude_Longitude_predette/[TrackID-' + track_id + ']-LatitudeLongitude_test_prediction_scenario_' + numero + lettera + '_test' + '.csv',
        map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
        points=(51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

    vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
    vis.plot_map(output='save', save_as='[TrackID' + track_id + ']-Prediction_Lat_Lon_scenario_' + scenario)


def predict_lat_test(scn, model):
    scenario = pd.read_csv('3-DataMerged/[3-TEST]_MERGE_scenario_' + scn + '_test.csv')
    scenario_bk = scenario
    scenario = scenario[['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
                             'AlviraTracksTrackPosition_Altitude',
                             'AlviraTracksTrack_Reflection', 'AlviraTracksTrackVelocity_Azimuth',
                             'AlviraTracksTrack_Score',
                             'AlviraTracksTrackVelocity_Speed', 'ArcusTracksTrackPosition_Latitude',
                             'ArcusTracksTrackPosition_Longitude', 'ArcusPotentialDronPlot_id',
                             'ArcusPotentialDronPlot_rcs',
                             'ArcusPotentiaDronPlotPlotPosition_altitude',
                             'ArcusPotentialDronPlotsPlotPosition_latitude',
                             'ArcusPotentialDronPlotsPlotPosition_longitude',
                             'ArcusSystemStatusSensorPosition_Latitude',
                             'ArcusSystemStatusSensorPosition_Longitude',
                             'ArcusSystemStatusSensorPosition_Altitude',
                             'ArcusSystemStatusSensorStatusOrientetation_Azimuth']]

    scenario.astype(np.float64)

    scenario = minmax_scale(scenario, feature_range=(0, 1), axis=0, copy=True)

    scenario = pd.DataFrame(scenario)
    scenario = scenario.fillna(value=0)

    X = np.c_[np.ones(scenario.shape[0]), scenario]

    with open('ModelliEstratti/model_latitude_' + model + '.pickle', 'rb') as file:
        latitude = pickle.load(file)

    lat_pred = latitude.predict(X)
    lat_pred = pd.DataFrame(lat_pred)

    lat_pred['datetime(utc)'] = scenario_bk['datetime(utc)']

    lat_pred.rename(columns={0: 'TrackPositionLatitude'}, inplace=True)
    lat_pred.rename(columns={'datetime(utc)': 'TrackDateTimeUTC'}, inplace=True)
    return lat_pred


def predict_lon_test(scn, model):
    scenario = pd.read_csv('3-DataMerged/[3-TEST]_MERGE_scenario_' + scn + '_test.csv')
    scenario_bk = scenario
    scenario = scenario[['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
                             'AlviraTracksTrackPosition_Altitude',
                             'AlviraTracksTrack_Reflection', 'AlviraTracksTrackVelocity_Azimuth',
                             'AlviraTracksTrack_Score',
                             'AlviraTracksTrackVelocity_Speed', 'ArcusTracksTrackPosition_Latitude',
                             'ArcusTracksTrackPosition_Longitude', 'ArcusPotentialDronPlot_id',
                             'ArcusPotentialDronPlot_rcs',
                             'ArcusPotentiaDronPlotPlotPosition_altitude',
                             'ArcusPotentialDronPlotsPlotPosition_latitude',
                             'ArcusPotentialDronPlotsPlotPosition_longitude',
                             'ArcusSystemStatusSensorPosition_Latitude',
                             'ArcusSystemStatusSensorPosition_Longitude',
                             'ArcusSystemStatusSensorPosition_Altitude',
                             'ArcusSystemStatusSensorStatusOrientetation_Azimuth']]

    scenario.astype(np.float64)

    scenario = minmax_scale(scenario, feature_range=(0, 1), axis=0, copy=True)

    scenario = pd.DataFrame(scenario)
    scenario = scenario.fillna(value=0)

    X = np.c_[np.ones(scenario.shape[0]), scenario]

    with open('ModelliEstratti/model_longitude_' + model + '.pickle', 'rb') as file:
        longitude = pickle.load(file)

    lon_pred = longitude.predict(X)
    lon_pred = pd.DataFrame(lon_pred)

    lon_pred['datetime(utc)'] = scenario_bk['datetime(utc)']

    lon_pred.rename(columns={0: 'TrackPositionLongitude'}, inplace=True)

    return lon_pred


def print_lat_long_prediction(scenario, numero, lettera):
    vis = GPSVis(
        data_path='Steps/Step_4/Latitude_Longitude_predette/LatitudeLongitude_test_prediction_scenario_' + numero + lettera + '_test' + '.csv',
        map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
        points=(51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

    vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
    vis.plot_map(output='save', save_as='Steps/Step_4/Latitude_Longitude_predette/Full_Prediction_Lat_Lon_scenario_' + scenario)


def print_lat_lon_test_predette(lat_pred, lon_pred, numero, lettera):
    test_lat = pd.concat([lat_pred, lon_pred], axis=1).drop_duplicates()
    test_lat = test_lat[['TrackPositionLatitude', 'TrackPositionLongitude']]
    test_lat.to_csv(
        'Steps/Step_4/Latitude_Longitude_predette/LatitudeLongitude_test_prediction_scenario_' + numero + lettera + '_test' + '.csv',
        header=False, index=False, float_format='%.6f')
    print_lat_long_prediction(scenario=numero + lettera + '_test', numero=numero, lettera=lettera)


def predict_speed_test(scn, model):
    scenario = pd.read_csv('3-DataMerged/[3-TEST]_MERGE_scenario_' + scn + '_test.csv')

    scenario_bk = scenario

    scenario = scenario[['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
                             'AlviraTracksTrackPosition_Altitude',
                             'AlviraTracksTrack_Reflection', 'AlviraTracksTrackVelocity_Azimuth',
                             'AlviraTracksTrack_Score',
                             'AlviraTracksTrackVelocity_Speed', 'ArcusTracksTrackPosition_Latitude',
                             'ArcusTracksTrackPosition_Longitude', 'ArcusPotentialDronPlot_id',
                             'ArcusPotentialDronPlot_rcs',
                             'ArcusPotentiaDronPlotPlotPosition_altitude',
                             'ArcusPotentialDronPlotsPlotPosition_latitude',
                             'ArcusPotentialDronPlotsPlotPosition_longitude',
                             'ArcusSystemStatusSensorPosition_Latitude',
                             'ArcusSystemStatusSensorPosition_Longitude',
                             'ArcusSystemStatusSensorPosition_Altitude',
                             'ArcusSystemStatusSensorStatusOrientetation_Azimuth'
                         ]]
    scenario = scenario.fillna(value=0)

    X = scenario

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    X = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

    with open('ModelliEstratti/model_speed_' + model + '.pickle', 'rb') as file:
        speed = pickle.load(file)

    speed_pred = speed.predict(X)
    speed_pred = pd.DataFrame(speed_pred)

    speed_pred['datetime(utc)'] = scenario_bk['datetime(utc)']

    speed_pred.rename(columns={0: 'TrackPositionSpeed'}, inplace=True)

    return speed_pred


def predict_altitude_test(scn, model):
    scenario = pd.read_csv('3-DataMerged/[3-TEST]_MERGE_scenario_' + scn + '_test.csv')

    scenario_bk = scenario
    scenario = scenario[['AlviraTracksTrackVelocity_Azimuth', 'AlviraTracksTrackVelocity_Speed',
                             'AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
                             'AlviraTracksTrack_Score',
                             'AlviraTracksTrack_Reflection',
                             'ArcusTracksTrackPosition_Latitude',
                             'ArcusTracksTrackPosition_Longitude',
                             'ArcusTracksTrackPosition_Altitude',
                             'ArcusTracksTrackVelocity_Azimuth',
                             'ArcusTracksTrackVelocity_Elevation',
                             'ArcusTracksTrackVelocity_Speed',
                             'ArcusTracksTrack_Reflection',
                             'ArcusTracksTrack_Score']]

    scenario = scenario.fillna(value=0)

    X = scenario

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    X = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

    with open('ModelliEstratti/model_altitude_' + model + '.pickle', 'rb') as file:
        altitude = pickle.load(file)

    alt_pred = altitude.predict(X)
    alt_pred = pd.DataFrame(alt_pred)
    alt_pred['datetime(utc)'] = scenario_bk['datetime(utc)']

    alt_pred.rename(columns={0: 'TrackPositionAltitude'}, inplace=True)

    return alt_pred


def predict_drone_type_test(scn):
    scenario = pd.read_csv('3-DataMerged/[3-TEST]_MERGE_scenario_' + scn + '_test.csv')
    scenario_bk = scenario
    scenario = scenario[[
        'DRONE', 'NO_DRONE', 'SUSPECTED_DRONE', 'Arcus_OperationalState_NaN',
        'Arcus_OperationalState_Idle', 'Arcus_OperationalState_Operational',
        'Arcus_Classification_NaN', 'Arcus_Classification_UNKNOWN',
        'Arcus_Classification_VEHICLE', 'Arcus_Classification_OTHER',
        'Arcus_Classification_DRONE', 'Arcus_Classification_SUSPECTED_DRONE',
        'Arcus_Classification_HELICOPTER', 'Arcus_Alarm_NaN',
        'Arcus_Alarm_FALSE', 'Arcus_Alarm_TRUE',
        'ArcusTracksTrackPosition_Altitude', 'ArcusTracksTrackVelocity_Azimuth',
        'ArcusTracksTrackVelocity_Elevation', 'ArcusTracksTrackVelocity_Speed',
        'ArcusTracksTrack_Reflection',
        'DianaSensorPosition_latitude_deg',
        'DianaSensorPosition_longitude_deg', 'DianaSensorPosition_altitude_m',
        'DianaTargets_band', 'DianaTarget_ID',
        'DianaTargetsTargetSignal_snr_dB',
        'DianaTargetsTargetSignal_bearing_deg',
        'DianaTargetsTargetSignal_range_m',
        'DianaTargetsTargetClassification_score', 'channels_x',
        'DianaTarget_Aircraft', 'DianaTarget_Controller', 'DianaTarget_None',
        'DianaClasssification_Unknown',
        'DianaClasssification_DJI-MAVIC-PRO-PLATINUM',
        'DianaClasssification_Wifi-Bluetooth',
        'DianaClasssification_DJI-MAVIC-2-PRO',
        'DianaClasssification_DJI-Phantom-4F',
        'DianaClasssification_Parrot-ANAFI',
        'DianaClasssification_DJI-Phantom-4E',
        'DianaClasssification_SPEKTRUM-DX5e', 'DianaClasssification_SYMA-X8HW',
        'DianaClasssification_DJI-MAVIC-AIR', 'DianaClasssification_None',
        'DianaClasssification_VISUO-Zen',
        'channels_y', 'VenusTriggerVenusName_isThreat',
        'VenusTrigger_RadioId',
        'VenusTrigger_Frequency', 'VenusTrigger_OnAirStartTime',
        'VenusTrigger_StopTime', 'VenusTrigger_Azimuth',
        'VenusTrigger_Deviation', 'DJI OcuSync', 'DJI Mavic Mini',
        'Cheerson Leopard 2', 'DJI Mavic Pro long',
        'DJI Mavic Pro short', 'Hubsan', 'Futaba FASST-7CH Var. 1',
        'AscTec Falcon 8 Downlink, DJI Mavic Mini',
        'DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G',
        'DJI Mavic Mini, MJX R/C Technic', 'Udi R/C 818A', 'MJX R/C Technic',
        'TT Robotix Ghost', 'Udi R/C',
        'DJI Mini, DJI Phantom 4 Pro/Mavic Pro, DJI Phantom 4/Mavic Pro',
        'DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G',
        'Spektrum DSMX downlink', 'Spektrum DSMX 12CH uplink', 'MJX X901',
        'DJI Mavic Pro long, DJI Phantom 4/Mavic Pro,DJI Phantom/Mavic Pro',
        'AscTec Falcon 8 Downlink', 'VenusName NaN', 'ISM 2.4 GHz',
        'ISM 5.8 GHz', 'FreqBand_Null'
    ]]
    scenario.astype(np.float64)
    data = scenario.fillna(value=0)
    data = data.reset_index()

    scenario = data.drop('index', axis=1)

    from scipy import stats
    scenario = stats.zscore(scenario)
    scenario = scenario.fillna(value=0)

    scenario = np.c_[np.ones(scenario.shape[0]), scenario]

    with open('ModelliEstratti/model_Mavic2.pickle', 'rb') as file:
        mavic_2 = pickle.load(file)

    mavic_2_pred = mavic_2.predict(scenario)
    mavic_2_pred = pd.DataFrame(mavic_2_pred)

    with open('ModelliEstratti/model_MavicPro.pickle', 'rb') as file:
        mavic_pro = pickle.load(file)

    mavic_pro_pred = mavic_pro.predict(scenario)
    mavic_pro_pred = pd.DataFrame(mavic_pro_pred)

    with open('ModelliEstratti/model_Parrot.pickle', 'rb') as file:
        parrot = pickle.load(file)

    parrot_pred = parrot.predict(scenario)
    parrot_pred = pd.DataFrame(parrot_pred)

    with open('ModelliEstratti/model_ProfessionalV2.pickle', 'rb') as file:
        profV2 = pickle.load(file)

    profV2_pred = profV2.predict(scenario)
    profV2_pred = pd.DataFrame(profV2_pred)

    mavic_2_pred['datetime(utc)'] = scenario_bk['datetime(utc)']
    mavic_2_pred['TrackClassification'] = 'UNKNOWN'

    for i in range(len(mavic_2_pred)):
        if mavic_2_pred.loc[i, 0] == 1:
            mavic_2_pred.loc[i, 'TrackClassification'] = 'DRONE'

    mavic_2_pred.rename(columns={0: 'Mavic2'}, inplace=True)

    mavic_pro_pred['datetime(utc)'] = scenario_bk['datetime(utc)']
    mavic_pro_pred['TrackClassification'] = 'UNKNOWN'

    for i in range(len(mavic_pro_pred)):
        if mavic_pro_pred.loc[i, 0] == 1:
            mavic_pro_pred.loc[i, 'TrackClassification'] = 'DRONE'

    mavic_pro_pred.rename(columns={0: 'MavicPro'}, inplace=True)

    parrot_pred['datetime(utc)'] = scenario_bk['datetime(utc)']
    parrot_pred['TrackClassification'] = 'UNKNOWN'
    for i in range(len(parrot_pred)):
        if parrot_pred.loc[i, 0] == 1:
            parrot_pred.loc[i, 'TrackClassification'] = 'FIXED_WING'

    parrot_pred.rename(columns={0: 'Parrot'}, inplace=True)

    profV2_pred['datetime(utc)'] = scenario_bk['datetime(utc)']
    profV2_pred['TrackClassification'] = 'UNKNOWN'
    for i in range(len(profV2_pred)):
        if profV2_pred.loc[i, 0] == 1:
            profV2_pred.loc[i, 'TrackClassification'] = 'DRONE'

    profV2_pred.rename(columns={0: 'Professional V2'}, inplace=True)

    return mavic_2_pred, mavic_pro_pred, profV2_pred, parrot_pred


def create_drone_detection_label(numero, mavic_2_pred, mavic_pro_pred, profV2_pred, parrot_pred):
    drone_detection = pd.DataFrame(columns=['TrackClassification', 'TrackIdentification'])

    if numero == '3':
        for i in range(len(parrot_pred)):
            if parrot_pred.loc[i, 'Parrot'] == 1:
                drone_detection.loc[i, 'TrackIdentification'] = 'FIXED_WING'
                drone_detection.loc[i, 'TrackClassification'] = 'Parrot'
                drone_detection.loc[i, 'TrackID'] = '3'
            else:
                drone_detection.loc[i, 'TrackIdentification'] = 'UNKNOWN'
                drone_detection.loc[i, 'TrackClassification'] = 'UNKNOWN'
                drone_detection.loc[i, 'TrackID'] = '0'
    else:
        for i in range(len(mavic_2_pred)):
            if mavic_2_pred.loc[i, 'Mavic2'] > mavic_pro_pred.loc[i, 'MavicPro'] and \
                    mavic_2_pred.loc[i, 'Mavic2'] > profV2_pred.loc[i, 'Professional V2']:
                # mavic_2_pred.loc[i, 'Mavic2'] > parrot_pred.loc[i, 'Parrot'] and \
                drone_detection.loc[i, 'TrackIdentification'] = 'DRONE'
                drone_detection.loc[i, 'TrackClassification'] = 'Mavic 2'
                drone_detection.loc[i, 'TrackID'] = '1'

            elif mavic_pro_pred.loc[i, 'MavicPro'] > mavic_2_pred.loc[i, 'Mavic2'] and \
                    mavic_pro_pred.loc[i, 'MavicPro'] > profV2_pred.loc[i, 'Professional V2']:
                # mavic_pro_pred.loc[i, 'MavicPro'] > parrot_pred.loc[i, 'Parrot'] and \
                drone_detection.loc[i, 'TrackIdentification'] = 'DRONE'
                drone_detection.loc[i, 'TrackClassification'] = 'Mavic Pro'
                drone_detection.loc[i, 'TrackID'] = '2'

            elif profV2_pred.loc[i, 'Professional V2'] > mavic_2_pred.loc[i, 'Mavic2'] and \
                    profV2_pred.loc[i, 'Professional V2'] > mavic_pro_pred.loc[i, 'MavicPro']:
                # profV2_pred.loc[i, 'Professional V2'] > parrot_pred.loc[i, 'Parrot']
                drone_detection.loc[i, 'TrackIdentification'] = 'DRONE'
                drone_detection.loc[i, 'TrackClassification'] = 'Professional V2'
                drone_detection.loc[i, 'TrackID'] = '4'
            else:
                drone_detection.loc[i, 'TrackIdentification'] = 'UNKNOWN'
                drone_detection.loc[i, 'TrackClassification'] = 'UNKNOWN'
                drone_detection.loc[i, 'TrackID'] = '0'

    return drone_detection


def print_lat_lon_test_drone(numero, lettera):
    test_lat = pd.read_csv('SubmissionFileTestResults/SubmissionFileScenario_' + numero + lettera + '.csv')
    #test_lat = test_lat[test_lat['TrackID'] == track_id]

    test_lat = test_lat[['TrackPositionLatitude', 'TrackPositionLongitude']]
    test_lat.to_csv(
        'Steps/Step_4/Latitude_Longitude_predette/[TrackID-FULL]-LatitudeLongitude_test_prediction_scenario_' + numero + lettera + '_test' + '.csv',
        header=False, index=False, float_format='%.6f')
    print_lat_long_prediction(scenario=numero + lettera + '_test', numero=numero, lettera=lettera)
    #print_lat_long_prediction_for_drone(numero + lettera, numero, lettera, str(track_id))


def start_prediction(numero, lettera, model):
    alvira, arcus, diana, venus = data_clean(numero, lettera)

    # ricalcolo arucs
    if lettera != 'd':
        arcus = arcus_meno_alvira_test(alvira, arcus, numero + lettera + '_test')
    arcus = calc_rem_outliers_test(arcus)
    arcus_time = arcus['datetime(utc)'].unique()

    # merge
    merge_all_sensor_test(alvira, arcus, diana, venus, scenario=numero + lettera + '_test')

    # predici lat lon

    lat_pred = predict_lat_test(numero + lettera, model)
    lon_pred = predict_lon_test(numero + lettera, model)

    print_lat_lon_test_predette(lat_pred, lon_pred, numero, lettera)

    # predici speed

    speed_pred = predict_speed_test(numero + lettera, model)

    # predici altitude

    altitude_pred = predict_altitude_test(numero + lettera, model)

    # predici tipo di drone

    mavic_2_pred, mavic_pro_pred, profV2_pred, parrot_pred = predict_drone_type_test(numero + lettera)

    drone_detection = create_drone_detection_label(numero, mavic_2_pred, mavic_pro_pred, profV2_pred, parrot_pred)

    submission = pd.concat([lat_pred, lon_pred, speed_pred, altitude_pred, drone_detection], axis=1).drop(
        'datetime(utc)', axis=1)

    submission['TrackSource'] = 'None'
    for i in range(0, len(submission)):
        if submission.loc[i, 'TrackDateTimeUTC'] in arcus_time:
            submission.loc[i, 'TrackSource'] = 'Arcus'
        else:
            submission.loc[i, 'TrackSource'] = 'Alvira'
    submission = submission.reindex(
        columns=['TrackDateTimeUTC', 'TrackID', 'TrackPositionLatitude', 'TrackPositionLongitude',
                 'TrackPositionAltitude','TrackPositionSpeed', 'TrackClassification', 'TrackIdentification', 'TrackSource'])
    submission = submission.sort_values('TrackDateTimeUTC')

    submission.to_csv('SubmissionFileTestResults/SubmissionFileScenario_' + numero + lettera + '.csv', index=False,
                      float_format='%.6f')

    print_lat_lon_test_drone(numero, lettera)

    print('End')
    # genera file submission


def data_clean(numero, lettera):
    ###  ALVIRA  ###

    alvira = pd.read_csv(
        'Scenari/Scenario_' + numero + lettera + '_test/ALVIRA_scenario.csv',
        delimiter=',')

    alvira = Alvira(alvira, numero + lettera + '_test')
    alvira.clean_data()

    if lettera != 'd':
        clustering_alvira_test(alvira,numero)
        alvira = pd.read_csv('2-AlviraClustering/[2-TEST]_ALVIRA_data_predict_' + numero + lettera + '_test.csv')
    else:
        alvira.data['AlviraTracksTrackPosition_Latitude'] = 0
        alvira.data['AlviraTracksTrackPosition_Longitude'] = 0
        alvira.data['AlviraTracksTrackPosition_Altitude'] = 0
        alvira.data['AlviraTracksTrack_Reflection'] = 0
        alvira.data['AlviraTracksTrackVelocity_Azimuth'] = 0
        alvira.data['AlviraTracksTrack_Score'] = 0
        alvira.data['AlviraTracksTrackVelocity_Speed'] = 0
        alvira.data['NO_DRONE'] = 0
        alvira.data['SUSPECTED_DRONE'] = 0

        alvira = alvira.data
    ###  ARCUS  ###

    arcus = pd.read_csv(
        'Scenari/Scenario_' + numero + lettera + '_test/ARCUS_scenario.csv',
        delimiter=',')
    arcus = Arcus(arcus, numero + lettera + '_test')
    arcus.clean_data()

    arcus = pd.read_csv(
        '1-DataCleaned/Scenario_' + numero + lettera + '_test/[1]_ARCUS_data_cleaned_' + numero + lettera + '_test.csv')

    ###  DIANA  ###

    diana = pd.read_csv(
        'Scenari/Scenario_' + numero + lettera + '_test/DIANA_scenario.csv',
        delimiter=',')
    diana = Diana(diana, numero + lettera + '_test')
    diana.clean_data()

    diana = pd.read_csv(
        '1-DataCleaned/Scenario_' + numero + lettera + '_test/[1]_DIANA_data_cleaned_' + numero + lettera + '_test.csv')

    ###  ARCUS  ###

    venus = pd.read_csv(
        'Scenari/Scenario_' + numero + lettera + '_test/VENUS_scenario.csv',
        delimiter=',')
    venus = Venus(venus, numero + lettera + '_test')
    venus.clean_data()

    venus = pd.read_csv(
        '1-DataCleaned/Scenario_' + numero + lettera + '_test/[1]_VENUS_data_cleaned_' + numero + lettera + '_test.csv')

    return alvira, arcus, diana, venus


def calc_k_value(points):
    if 'datetime(utc)' in points:
        points = points.drop('datetime(utc)', axis=1)
    cost = []
    for i in range(1, 5):
        KM = KMeans(n_clusters=i, max_iter=1500)
        KM.fit(points)

        # calculates squared error
        # for the clustered points
        cost.append(KM.inertia_)

        # plot the cost against K values
    plt.plot(range(1, 5), cost, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Squared Error (Cost)")
    plt.show()  # clear the plot
    return 0


def clustering_alvira_test(alvira,scenario):
    alvira.data = alvira.data[
        ['datetime(utc)', 'AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
         'AlviraTracksTrackPosition_Altitude', 'AlviraTracksTrackVelocity_Azimuth',
         'AlviraTracksTrackVelocity_Elevation', 'AlviraTracksTrackVelocity_Speed', 'DRONE', 'NO_DRONE',
         'SUSPECTED_DRONE',
         'AlviraTracksTrack_Score', 'AlviraTracksTrack_Reflection']]
    if scenario == '1' or scenario == '3':
        k = 2
    else:
        k = 3
    calc_k_value(alvira.data)
    print('K-means')
    k_means(alvira, k)

    data = pd.read_csv('2-AlviraClustering/[2-TEST]_ALVIRA_data_predict_' + alvira.scenario + '.csv')
    print_on_map(data, scenario=alvira.scenario)


def print_on_map(data, scenario):
    data = data[['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude']]

    data.astype({"AlviraTracksTrackPosition_Latitude": float})
    data.astype({"AlviraTracksTrackPosition_Longitude": float})

    data['AlviraTracksTrackPosition_Latitude'].map('{:,.6f}'.format)
    data['AlviraTracksTrackPosition_Longitude'].map('{:,.6f}'.format)

    data.to_csv('Steps/Step_3/Prediction_Lat_Lon/lat_long_' + scenario + '.csv', header=False,
                index=False)

    vis = GPSVis(data_path='Steps/Step_3/Prediction_Lat_Lon/lat_long_' + scenario + '.csv',
                 map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
                 points=(
                     51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

    vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
    vis.plot_map(output='save', save_as='Prediction_Lat_Lon_scenario_' + scenario)
