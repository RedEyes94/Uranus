import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import f_classif
from matplotlib import pyplot


def anova_feature_selection(scenari, feature, sensore):
    alvira_features = ['AlviraTracksTrackPosition_Latitude',
                       'AlviraTracksTrackPosition_Longitude',
                       'AlviraTracksTrackPosition_Altitude',
                       'AlviraTracksTrackVelocity_Azimuth',
                       'AlviraTracksTrackVelocity_Speed',
                       'DRONE',
                       'NO_DRONE',
                       'SUSPECTED_DRONE',
                       'AlviraTracksTrack_Score',
                       'AlviraTracksTrack_Reflection'

                       ]
    arcus_features = [

        'ArcusPotentialDronPlot_id',
        'ArcusPotentialDronPlot_rcs',
        'ArcusPotentiaDronPlotPlotPosition_altitude',
        'ArcusPotentialDronPlotsPlotPosition_latitude',
        'ArcusPotentialDronPlotsPlotPosition_longitude',
        'ArcusTracksTrack_id',
        'ArcusTracksTrackPosition_Latitude',
        'ArcusTracksTrackPosition_Longitude',
        'ArcusTracksTrackPosition_Altitude',
        'ArcusTracksTrackVelocity_Azimuth',
        'ArcusTracksTrackVelocity_Elevation',
        'ArcusTracksTrackVelocity_Speed',
        'ArcusTracksTrack_Reflection',
        'ArcusTracksTrack_Score',
        'ArcusSystemStatusSensorPosition_Latitude',
        'ArcusSystemStatusSensorPosition_Longitude',
        'ArcusSystemStatusSensorPosition_Altitude',
        'ArcusSystemStatusSensorStatusOrientetation_Azimuth',
        'ArcusSystemStatusSensorStatusOrientetation_Elevation',
        'ArcusSystemStatusSensorStatusBlankSector_Angle',
        'ArcusSystemStatusSensorStatusBlankSector_Span',
        'ArcusSystemStatusSensorStatusProcessing_Sensitivity'

    ]

    diana_features = [
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
    ]

    venus_features = [
        'VenusTriggerVenusName_isThreat',
        'VenusTrigger_RadioId',
        'VenusTrigger_Frequency',
        'VenusTrigger_OnAirStartTime',
        'VenusTrigger_StopTime',
        'VenusTrigger_Azimuth',
        'VenusTrigger_Deviation',
        'DJI OcuSync', 'DJI Mavic Mini',
        'Cheerson Leopard 2', 'DJI Mavic Pro long', 'nan',
        'DJI Mavic Pro short', 'Hubsan', 'Futaba FASST-7CH Var. 1',
        'AscTec Falcon 8 Downlink, DJI Mavic Mini',
        'DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G',
        'DJI Mavic Mini, MJX R/C Technic', 'Udi R/C 818A', 'MJX R/C Technic',
        'TT Robotix Ghost', 'Udi R/C',
    ]
    merge1 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_1.csv')
    merge2 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_2.csv')
    merge3 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_3.csv')
    merge4 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_4.csv')
    merge5 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_2_1.csv')
    merge6 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_2_2.csv')
    merge7 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_3.csv')

    total = pd.concat([merge1,merge2,merge3,merge4,merge5,merge6,merge7], axis=0)
    total = total.fillna(value=0)
    feature_ = total[feature]
    if sensore == 'Alvira':
        merge = total[alvira_features]
    if sensore == 'Arcus':
        merge = total[arcus_features]
    if sensore == 'Diana':
        merge = total[diana_features]
    if sensore == 'Venus':
        merge = total[venus_features]
    anova_ftest_feature_selection(merge, feature_, feature, sensore)


        # anova_test(merge,latitude)


def load_dataset(dataset, feature):
    # split into input (X) and output (y) variables
    X = dataset
    y = feature
    return X, y


# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')  # qui indica quante migliori features vuoi mantenere
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def anova_ftest_feature_selection(data, feature_, feature, sensore):
    # load the dataset
    X, y = load_dataset(data, feature_)
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print(' %s;%f' % (str(fs.feature_names_in_[i]), fs.scores_[i]))
    # plot the scores
    pyplot.subplots_adjust(bottom=0.55)

    pyplot.bar([str(fs.feature_names_in_[i]) for i in range(len(fs.scores_))], fs.scores_)

    pyplot.xticks(rotation=90)
    pyplot.savefig('AnovaResults/'+sensore+'_anova_results_' + feature + '.pdf')
    pyplot.show()


def anova_test(X, y):
    # feature selection
    def select_features(X_train, y_train, X_test):
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k='all')
        # learn relationship from training data
        fs.fit(X_train, y_train)
        # transform train input data
        X_train_fs = fs.transform(X_train)
        # transform test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    app = X
    # app = app.drop('datetime(utc)', axis=1)
    '''app = app.drop('flightmode', axis=1)
    app = app.iloc[:, :49]'''
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(app, y, test_size=0.33, random_state=1)
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([fs.feature_names_in_[i] for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
