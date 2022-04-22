import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler


def predict_drone_type(imblearn=None):
    data1 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_1.csv')
    data2 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_2.csv')
    data3 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_3.csv')
    data4 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_4.csv')
    data5 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_2_1.csv')
    data6 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_2_1.csv')
    data7 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_3.csv')

    data = pd.concat([data1, data2, data3, data4, data5, data6, data7], axis=0)
    data = data.reset_index()
    data = data.drop('index', axis=1)
    data = data.drop('VenusTriggerLinkType_Uplink', axis=1)

    '''
    'DianaSensorPosition_latitude_deg',
                        'DianaSensorPosition_longitude_deg',
                        'DianaSensorPosition_altitude_m',
                        'DianaTargets_band',
                        'DianaTarget_ID',
                        'DianaTargetsTargetSignal_snr_dB',
                        'DianaTargetsTargetSignal_bearing_deg',
                        'DianaTargetsTargetSignal_range_m',
                        'DianaTargetsTargetClassification_score',
                        'DianaTarget_Aircraft',
                        'DianaTarget_Controller',
                        'DianaTarget_None',
                        'DianaClasssification_Unknown',
                        'DianaClasssification_DJI-MAVIC-PRO-PLATINUM',
                        'DianaClasssification_Wifi-Bluetooth',
                        'DianaClasssification_DJI-MAVIC-2-PRO',
                        'DianaClasssification_DJI-Phantom-4F',
                        'DianaClasssification_Parrot-ANAFI',
                        'DianaClasssification_DJI-Phantom-4E',
                        'DianaClasssification_SPEKTRUM-DX5e',
                        'DianaClasssification_SYMA-X8HW',
                        'DianaClasssification_DJI-MAVIC-AIR',
                        'DianaClasssification_None',
                        'DianaClasssification_VISUO-Zen'
    
    '''

    data = data[[

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
        'ISM 5.8 GHz', 'FreqBand_Null', 'Dronetype'

    ]]
    data = data.fillna(value=0)
    data['Dronetype'] = data['Dronetype'].apply("int64")
    y = data['Dronetype']

    rus = RandomUnderSampler(random_state=42, replacement=True)  # fit predictor and target variable
    x_rus, y_rus = rus.fit_resample(data, y)

    print('original dataset shape:', len(y))
    print('Resample dataset shape', len(y_rus))

    # x_rus = x_rus.drop('index', axis=1)
    x_rus.insert(x_rus.shape[1], 'Dronetype1', y_rus)

    data = x_rus

    data['MAVIC 2'] = 0
    data['MAVIC Pro'] = 0
    data['Phantom 4 Pro'] = 0
    data['Parrot'] = 0

    for i in range(0, data.shape[0]):
        if data.loc[i, 'Dronetype1'] == 13:
            data.loc[i, 'MAVIC Pro'] = 1

        elif data.loc[i, 'Dronetype1'] == 23:
            data.loc[i, 'Phantom 4 Pro'] = 1

        elif data.loc[i, 'Dronetype1'] == 27:
            data.loc[i, 'MAVIC 2'] = 1
        else:
            data.loc[i, 'Parrot'] = 1

    from sklearn.utils import shuffle

    data = shuffle(data)
    data = data.reset_index()
    data = data.drop('index', axis=1)

    y_mavic2 = data['MAVIC 2']
    y_mavicPro = data['MAVIC Pro']
    y_profV2 = data['Phantom 4 Pro']
    y_parrot = data['Parrot']

    data = data[[

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

    data.astype(np.float64)
    data = data.fillna(value=0)
    data = data.reset_index()

    scenario = data.drop('index', axis=1)

    from scipy import stats
    scenario = stats.zscore(scenario)
    scenario = scenario.fillna(value=0)

    scenario = np.c_[np.ones(scenario.shape[0]), scenario]
    print('-----------------------MAVIC 2-----------------------------')
    regression_drone(scenario, y_mavic2, 'Mavic2')
    print('-----------------------MAVIC PRO-----------------------------')
    regression_drone(scenario, y_mavicPro, 'MavicPro')
    print('-----------------------PROF V2-----------------------------')
    regression_drone(scenario, y_profV2, 'ProfessionalV2')
    print('-----------------------PARROT-----------------------------')
    print('-------------------------------------------------')
    regression_drone(scenario, y_parrot, 'Parrot')


def regression_drone(scenario, y, drone):
    kf = KFold(n_splits=5, random_state=None)
    acc_score = []
    prediction_values = []
    precision_values = []
    recall_values = []
    roc_values = []
    f1_values = []
    X_train_acc = []
    X_test_acc = []
    y_train_acc = []
    y_test_acc = []

    X = pd.DataFrame(scenario)
    m2 = np.array(y)
    model = RandomForestClassifier(max_depth=10, random_state=0)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = m2[train_index], m2[test_index]

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)

        prediction_values.append(pred_values)
        # acc = accuracy_score(pred_values, y_test)
        acc = model.score(X_test, y_test)

        roc = roc_auc_score(y_test, pred_values)
        acc = accuracy_score(y_test, pred_values)
        recall = recall_score(y_test, pred_values, average='weighted')
        f1 = f1_score(y_test, pred_values, average='weighted')
        precision = precision_score(y_test, pred_values, average='weighted')

        if len(acc_score) == 0:
            X_train_acc = X_train
            X_test_acc = X_test
            y_train_acc = y_train
        elif acc > max(acc_score):
            X_train_acc = X_train
            y_train_acc = y_train
            y_test_acc = y_test

        roc_values.append(roc)
        acc_score.append(acc)
        recall_values.append(recall)
        f1_values.append(f1)
        precision_values.append(precision)

    avg_acc_score = sum(acc_score) / 5
    avg_precision_score = sum(precision_values) / 5
    avg_recall_score = sum(recall_values) / 5
    avg_f1_score = sum(f1_values) / 5
    avg_roc_score = sum(roc_values) / 5
    print(drone)

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    print('Precision of each fold - {}'.format(precision_values))
    print('Avg Precision : {}'.format(avg_precision_score))

    print('Recall of each fold - {}'.format(recall_values))
    print('Avg Recall : {}'.format(avg_recall_score))

    print('F1 of each fold - {}'.format(f1_values))
    print('Avg F1 : {}'.format(avg_f1_score))

    print('ROC AUC of each fold - {}'.format(roc_values))
    print('Avg ROC : {}'.format(avg_roc_score))


    with open('ModelliEstratti/model_' + drone + '.pickle', 'wb') as f:
        import pickle
        pickle.dump(model, f)
