import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, learning_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import minmax_scale
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

def predict_lat_full(scenari):
    data1 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_1.csv')
    data2 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_2.csv')
    data3 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_3.csv')
    data4 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_4.csv')
    data5 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_2_1.csv')
    data6 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_2_1.csv')
    data7 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_1.csv')

    data = pd.concat([data1, data2, data3, data4, data5, data6, data7], axis=1)
    from sklearn.utils import shuffle
    # scenario = shuffle(data)
    # scenario = scenario.iloc[np.random.permutation(len(scenario))]

    scenario = data.reset_index()
    scenario = scenario.drop('index', axis=1)

    y_lat = scenario['latitude']
    y_lat = np.array(y_lat)
    y_lon = scenario['longitude']
    y_lon = np.array(y_lon)

    scenario = scenario[['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
                         'AlviraTracksTrackPosition_Altitude',
                         'AlviraTracksTrack_Reflection', 'AlviraTracksTrackVelocity_Azimuth',
                         'AlviraTracksTrack_Score',
                         'AlviraTracksTrackVelocity_Speed', 'ArcusTracksTrackPosition_Latitude',
                         'ArcusTracksTrackPosition_Longitude']]

    scenario.astype(np.float64)

    scenario = minmax_scale(scenario, feature_range=(0, 1), axis=0, copy=True)

    scenario = pd.DataFrame(scenario)
    scenario = scenario.fillna(value=0)

    k = 5
    X = np.c_[np.ones(scenario.shape[0]), scenario]

    kf = KFold(n_splits=k, random_state=42)
    scores = []
    X = pd.DataFrame(X)

    X = X.astype(np.float64)
    y_lat = y_lat.astype(np.float64)
    y_lon = y_lon.astype(np.float64)

    X = np.nan_to_num(X)
    y_lat = np.nan_to_num(y_lat)
    y_lon = np.nan_to_num(y_lon)

    X = pd.DataFrame(X)
    y_lat = np.array(y_lat)
    y_lon = np.array(y_lon)

    mae_error_rate = []
    mse_error_rate = []
    # neigh = KNeighborsRegressor(n_neighbors=10, p=2)
    neigh = RandomForestRegressor(max_depth=80, random_state=42)
    # neigh = KNeighborsRegressor(n_neighbors=6)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y_lat[train_index], y_lat[test_index]

        # neigh = KNeighborsRegressor(n_neighbors=10)
        neigh.fit(X_train, y_train)

        pred_i = neigh.predict(X_test)
        # score = neigh.score(X_test, y_test)
        from sklearn.metrics import r2_score
        score = r2_score(y_test, pred_i)
        mae = mean_absolute_error(y_test, pred_i)
        mse = sqrt(mean_squared_error(y_test, pred_i))

        # error_rate.append((pred_i-y_test)*40000*math.cos((pred_i+y_test)*math.pi/360)/360)
        # (pred_i[0] - y_test[0]) * 40000 * math.cos((pred_i[0] + y_test[0]) * math.pi / 360) / 360   0.03851
        if len(scores) == 0:
            X_train_acc = X_train
            y_train_acc = y_train
            X_test_acc = X_test
        elif score < min(scores):
            X_train_acc = X_train
            y_train_acc = y_train
            y_test_acc = y_test
        scores.append(score)
        mae_error_rate.append(mae)
        mse_error_rate.append(mse)
    avg = sum(scores) / k
    print('accuracy of each fold - {}'.format(scores))
    print('MAE error of each fold - {}'.format(mae_error_rate))
    print('MSE error of each fold - {}'.format(mse_error_rate))
    print('Avg accuracy : {}'.format(avg))

    scenario = np.c_[np.ones(scenario.shape[0]), scenario]
    pred = neigh.predict(scenario)
    pd.DataFrame(pred).to_csv('Steps/Step_3/Prediction/LAT_pred_scenario_FULL.csv', float_format='%.6f',
                              index=False)


def learning_curves(estimator, data, features, target, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, data[features], data[target], train_sizes=train_sizes,
        cv=cv, scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 40)

import matplotlib.pyplot as plt

def predict_latitude(scenari):
    print(' ---- Start predict LATITUDE ----')
    neigh = RandomForestRegressor(max_depth=10, random_state=42)



    for scn in scenari:

        data = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_' + scn + '.csv')


        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        y_lat = scenario['latitude']
        y_lat = np.array(y_lat)
        y_lon = scenario['longitude']
        y_lon = np.array(y_lon)
        scenario = scenario[['latitude', 'AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
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
        features = ['latitude', 'AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
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
                    'ArcusSystemStatusSensorStatusOrientetation_Azimuth']


        scenario.astype(np.float64)

        #scenario = minmax_scale(scenario, feature_range=(0, 1), axis=0, copy=True)

        scenario = pd.DataFrame(scenario)
        scenario = scenario.fillna(value=0)

        k = 5
        X = np.c_[np.ones(scenario.shape[0]), scenario]

        kf = KFold(n_splits=k, random_state=42)
        scores = []
        X = pd.DataFrame(X)

        X = X.astype(np.float64)
        y_lat = y_lat.astype(np.float64)
        y_lon = y_lon.astype(np.float64)

        X = np.nan_to_num(X)
        y_lat = np.nan_to_num(y_lat)
        y_lon = np.nan_to_num(y_lon)

        X = pd.DataFrame(X)
        y_lat = np.array(y_lat)
        y_lon = np.array(y_lon)

        mae_error_rate = []
        mse_error_rate = []
        # neigh = KNeighborsRegressor(n_neighbors=10, p=2)
        # neigh = KNeighborsRegressor(n_neighbors=6)


        for train_index, test_index in kf.split(X):




            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_lat[train_index], y_lat[test_index]

            #################

            train_results = []
            test_results = []
            list_nb_trees = [5, 10, 15, 30, 45, 60, 80, 100]

            for nb_trees in list_nb_trees:
                rf = RandomForestRegressor(n_estimators=nb_trees)
                rf.fit(X_train, y_train)

                train_results.append(mean_squared_error(y_train, rf.predict(X_train)))
                test_results.append(mean_squared_error(y_test, rf.predict(X_test)))

            pd.DataFrame(train_results).to_csv('Train_Score_curve.csv')
            pd.DataFrame(train_results).to_csv('Test_Score_curve.csv')

            line1, = plt.plot(list_nb_trees, train_results, color="r", label="Training Score")
            line2, = plt.plot(list_nb_trees, test_results, color="g", label="Testing Score")

            plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
            plt.ylabel('MSE')
            plt.xlabel('n_estimators')
            plt.show()

            #################


            neigh.fit(X_train, y_train)

            pred_i = neigh.predict(X_test)
            from sklearn.metrics import r2_score
            score = r2_score(y_test, pred_i)
            mae = mean_absolute_error(y_test, pred_i)
            mse = sqrt(mean_squared_error(y_test, pred_i))

            if len(scores) == 0:
                X_train_acc = X_train
                y_train_acc = y_train
                X_test_acc = X_test
            elif score < min(scores):
                X_train_acc = X_train
                y_train_acc = y_train
                y_test_acc = y_test
            scores.append(score)
            mae_error_rate.append(mae)
            mse_error_rate.append(mse)
        avg = sum(scores) / k
        print('accuracy of each fold - {}'.format(scores))
        print('MAE error of each fold - {}'.format(mae_error_rate))
        print('MSE error of each fold - {}'.format(mse_error_rate))
        print('Avg accuracy : {}'.format(avg))

        scenario = np.c_[np.ones(scenario.shape[0]), scenario]
        pred = neigh.predict(scenario)
        pd.DataFrame(pred).to_csv(
            'PredictionResults/Training/Latitude_Longitude/latitude_pred_scenario_' + scn + '.csv', float_format='%.6f',
            index=False)

    import pickle
    with open('Models/model_latitude.pickle', 'wb') as f:
        pickle.dump(neigh, f)


import matplotlib.pyplot as plt


def learning_curves(estimator, data, features, target, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, data[features], data[target], train_sizes=train_sizes,
        cv=cv, scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 40)


def predict_lat(scenari):
    neigh = RandomForestRegressor(max_depth=80, random_state=42)

    for scn in scenari:

        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')
        from sklearn.utils import shuffle
        # scenario = shuffle(data)
        # scenario = scenario.iloc[np.random.permutation(len(scenario))]

        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        y_lat = scenario['latitude']
        y_lat = np.array(y_lat)
        y_lon = scenario['longitude']
        y_lon = np.array(y_lon)

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

        scenario.astype(np.float64)

        scenario = minmax_scale(scenario, feature_range=(0, 1), axis=0, copy=True)

        scenario = pd.DataFrame(scenario)
        scenario = scenario.fillna(value=0)

        k = 5
        X = np.c_[np.ones(scenario.shape[0]), scenario]

        kf = KFold(n_splits=k, random_state=42)
        scores = []
        X = pd.DataFrame(X)

        X = X.astype(np.float64)
        y_lat = y_lat.astype(np.float64)
        y_lon = y_lon.astype(np.float64)

        X = np.nan_to_num(X)
        y_lat = np.nan_to_num(y_lat)
        y_lon = np.nan_to_num(y_lon)

        X = pd.DataFrame(X)
        y_lat = np.array(y_lat)
        y_lon = np.array(y_lon)

        mae_error_rate = []
        mse_error_rate = []
        # neigh = KNeighborsRegressor(n_neighbors=10, p=2)
        # neigh = KNeighborsRegressor(n_neighbors=6)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_lat[train_index], y_lat[test_index]

            # neigh = KNeighborsRegressor(n_neighbors=10)
            neigh.fit(X_train, y_train)

            pred_i = neigh.predict(X_test)
            # score = neigh.score(X_test, y_test)
            from sklearn.metrics import r2_score
            score = r2_score(y_test, pred_i)
            mae = mean_absolute_error(y_test, pred_i)
            mse = sqrt(mean_squared_error(y_test, pred_i))

            # error_rate.append((pred_i-y_test)*40000*math.cos((pred_i+y_test)*math.pi/360)/360)
            # (pred_i[0] - y_test[0]) * 40000 * math.cos((pred_i[0] + y_test[0]) * math.pi / 360) / 360   0.03851
            if len(scores) == 0:
                X_train_acc = X_train
                y_train_acc = y_train
                X_test_acc = X_test
            elif score < min(scores):
                X_train_acc = X_train
                y_train_acc = y_train
                y_test_acc = y_test
            scores.append(score)
            mae_error_rate.append(mae)
            mse_error_rate.append(mse)
        avg = sum(scores) / k
        print('accuracy of each fold - {}'.format(scores))
        print('MAE error of each fold - {}'.format(mae_error_rate))
        print('MSE error of each fold - {}'.format(mse_error_rate))
        print('Avg accuracy : {}'.format(avg))

        scenario = np.c_[np.ones(scenario.shape[0]), scenario]
        pred = neigh.predict(scenario)
        pd.DataFrame(pred).to_csv('Steps/Step_3/Prediction/LAT_pred_scenario_' + scn + '.csv', float_format='%.6f',
                                  index=False)
        import pickle
        with open('Models/model_latitude_' + scn + '.pickle', 'wb') as f:
            pickle.dump(neigh, f)


def predict_longitude(scenari):
    print(' ---- Start predict LONGITUDE ----')

    neigh = RandomForestRegressor(max_depth=10, random_state=42)

    for scn in scenari:

        data = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_' + scn + '.csv')

        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        y_lon = scenario['longitude']
        y_lon = np.array(y_lon)

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

        k = 5
        X = np.c_[np.ones(scenario.shape[0]), scenario]

        kf = KFold(n_splits=k, random_state=42)
        scores = []
        X = pd.DataFrame(X)

        X = X.astype(np.float64)
        y_lon = y_lon.astype(np.float64)

        X = np.nan_to_num(X)
        y_lon = np.nan_to_num(y_lon)

        X = pd.DataFrame(X)
        y_lon = np.array(y_lon)

        mae_error_rate = []
        mse_error_rate = []
        # neigh = KNeighborsRegressor(n_neighbors=10, p=2)
        # neigh = KNeighborsRegressor(n_neighbors=6)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_lon[train_index], y_lon[test_index]

            # neigh = KNeighborsRegressor(n_neighbors=10)
            neigh.fit(X_train, y_train)

            pred_i = neigh.predict(X_test)
            # score = neigh.score(X_test, y_test)
            from sklearn.metrics import r2_score
            score = r2_score(y_test, pred_i)
            mae = mean_absolute_error(y_test, pred_i)
            mse = sqrt(mean_squared_error(y_test, pred_i))

            if len(scores) == 0:
                X_train_acc = X_train
                y_train_acc = y_train
                X_test_acc = X_test
            elif score < min(scores):
                X_train_acc = X_train
                y_train_acc = y_train
                y_test_acc = y_test
            scores.append(score)
            mae_error_rate.append(mae)
            mse_error_rate.append(mse)
        avg = sum(scores) / k
        print('accuracy of each fold - {}'.format(scores))
        print('MAE error of each fold - {}'.format(mae_error_rate))
        print('MSE error of each fold - {}'.format(mse_error_rate))
        print('Avg accuracy : {}'.format(avg))

        scenario = np.c_[np.ones(scenario.shape[0]), scenario]
        pred = neigh.predict(scenario)
        pd.DataFrame(pred).to_csv(
            'PredictionResults/Training/Latitude_Longitude/longitude_pred_scenario_' + scn + '.csv',
            float_format='%.6f',
            index=False)
    print(' ---- Stop predict LONGITUDE ----')

    import pickle
    with open('Models/model_longitude.pickle', 'wb') as f:
        pickle.dump(neigh, f)


def predict_lon(scenari):
    neigh = RandomForestRegressor(max_depth=80, random_state=42)

    for scn in scenari:

        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')

        from sklearn.utils import shuffle
        # scenario = shuffle(data)
        # scenario = scenario.iloc[np.random.permutation(len(scenario))]

        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        y_lon = scenario['longitude']
        y_lon = np.array(y_lon)

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

        k = 5
        X = np.c_[np.ones(scenario.shape[0]), scenario]

        kf = KFold(n_splits=k, random_state=42)
        scores = []
        X = pd.DataFrame(X)

        X = X.astype(np.float64)
        y_lon = y_lon.astype(np.float64)

        X = np.nan_to_num(X)
        y_lon = np.nan_to_num(y_lon)

        X = pd.DataFrame(X)
        y_lon = np.array(y_lon)

        mae_error_rate = []
        mse_error_rate = []
        # neigh = KNeighborsRegressor(n_neighbors=10, p=2)
        # neigh = KNeighborsRegressor(n_neighbors=6)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_lon[train_index], y_lon[test_index]

            # neigh = KNeighborsRegressor(n_neighbors=10)
            neigh.fit(X_train, y_train)

            pred_i = neigh.predict(X_test)
            # score = neigh.score(X_test, y_test)
            from sklearn.metrics import r2_score
            score = r2_score(y_test, pred_i)
            mae = mean_absolute_error(y_test, pred_i)
            mse = sqrt(mean_squared_error(y_test, pred_i))

            # error_rate.append((pred_i-y_test)*40000*math.cos((pred_i+y_test)*math.pi/360)/360)
            # (pred_i[0] - y_test[0]) * 40000 * math.cos((pred_i[0] + y_test[0]) * math.pi / 360) / 360   0.03851
            if len(scores) == 0:
                X_train_acc = X_train
                y_train_acc = y_train
                X_test_acc = X_test
            elif score < min(scores):
                X_train_acc = X_train
                y_train_acc = y_train
                y_test_acc = y_test
            scores.append(score)
            mae_error_rate.append(mae)
            mse_error_rate.append(mse)
        avg = sum(scores) / k
        print('accuracy of each fold - {}'.format(scores))
        print('MAE error of each fold - {}'.format(mae_error_rate))
        print('MSE error of each fold - {}'.format(mse_error_rate))
        print('Avg accuracy : {}'.format(avg))

        scenario = np.c_[np.ones(scenario.shape[0]), scenario]
        pred = neigh.predict(scenario)
        pd.DataFrame(pred).to_csv('Steps/Step_3/Prediction/LON_pred_scenario_' + scn + '.csv', float_format='%.6f',
                                  index=False)
        import pickle
        with open('Models/model_longitude_' + scn + '.pickle', 'wb') as f:
            pickle.dump(neigh, f)


def predict_lon_full(scenari):
    data1 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_1.csv')
    data2 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_2.csv')
    data3 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_3.csv')
    data4 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_4.csv')
    data5 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_2_1.csv')
    data6 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_2_1.csv')
    data7 = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_1_1.csv')

    data = pd.concat([data1, data2, data3, data4, data5, data6, data7], axis=1)
    from sklearn.utils import shuffle
    # scenario = shuffle(data)
    # scenario = scenario.iloc[np.random.permutation(len(scenario))]

    scenario = data.reset_index()
    scenario = scenario.drop('index', axis=1)

    y_lon = scenario['longitude']
    y_lon = np.array(y_lon)

    scenario = scenario[['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
                         'AlviraTracksTrackPosition_Altitude',
                         'AlviraTracksTrack_Reflection', 'AlviraTracksTrackVelocity_Azimuth',
                         'AlviraTracksTrack_Score',
                         'AlviraTracksTrackVelocity_Speed', 'ArcusTracksTrackPosition_Latitude',
                         'ArcusTracksTrackPosition_Longitude']]

    scenario.astype(np.float64)

    scenario = minmax_scale(scenario, feature_range=(0, 1), axis=0, copy=True)

    scenario = pd.DataFrame(scenario)
    scenario = scenario.fillna(value=0)

    k = 5
    X = np.c_[np.ones(scenario.shape[0]), scenario]

    kf = KFold(n_splits=k, random_state=42)
    scores = []
    X = pd.DataFrame(X)

    X = X.astype(np.float64)
    y_lon = y_lon.astype(np.float64)

    X = np.nan_to_num(X)
    y_lon = np.nan_to_num(y_lon)

    X = pd.DataFrame(X)
    y_lon = np.array(y_lon)

    mae_error_rate = []
    mse_error_rate = []
    # neigh = KNeighborsRegressor(n_neighbors=10, p=2)
    neigh = RandomForestRegressor(max_depth=80, random_state=42)
    # neigh = KNeighborsRegressor(n_neighbors=6)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y_lon[train_index], y_lon[test_index]

        # neigh = KNeighborsRegressor(n_neighbors=10)
        neigh.fit(X_train, y_train)

        pred_i = neigh.predict(X_test)
        # score = neigh.score(X_test, y_test)
        from sklearn.metrics import r2_score
        score = r2_score(y_test, pred_i)
        mae = mean_absolute_error(y_test, pred_i)
        mse = sqrt(mean_squared_error(y_test, pred_i))

        # error_rate.append((pred_i-y_test)*40000*math.cos((pred_i+y_test)*math.pi/360)/360)
        # (pred_i[0] - y_test[0]) * 40000 * math.cos((pred_i[0] + y_test[0]) * math.pi / 360) / 360   0.03851
        if len(scores) == 0:
            X_train_acc = X_train
            y_train_acc = y_train
            X_test_acc = X_test
        elif score < min(scores):
            X_train_acc = X_train
            y_train_acc = y_train
            y_test_acc = y_test
        scores.append(score)
        mae_error_rate.append(mae)
        mse_error_rate.append(mse)
    avg = sum(scores) / k
    print('accuracy of each fold - {}'.format(scores))
    print('MAE error of each fold - {}'.format(mae_error_rate))
    print('MSE error of each fold - {}'.format(mse_error_rate))
    print('Avg accuracy : {}'.format(avg))

    scenario = np.c_[np.ones(scenario.shape[0]), scenario]
    pred = neigh.predict(scenario)
    pd.DataFrame(pred).to_csv('Steps/Step_3/Prediction/LON_pred_scenario_FULL.csv', float_format='%.6f',
                              index=False)
