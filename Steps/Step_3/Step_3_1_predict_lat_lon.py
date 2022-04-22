import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale
from math import sqrt


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

    kf = KFold(n_splits=k, random_state=None)
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
    neigh = RandomForestRegressor(max_depth=80, random_state=0)
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


def predict_lat_per_scenario(scenari):
    print(' ---- Start predict LATITUDE ----')
    neigh = RandomForestRegressor(max_depth=10, random_state=0)
    for scn in scenari:

        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')

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
                             'ArcusSystemStatusSensorStatusOrientetation_Azimuth']]

        scenario.astype(np.float64)

        scenario = minmax_scale(scenario, feature_range=(0, 1), axis=0, copy=True)

        scenario = pd.DataFrame(scenario)
        scenario = scenario.fillna(value=0)

        k = 5
        X = np.c_[np.ones(scenario.shape[0]), scenario]

        kf = KFold(n_splits=k, random_state=None)
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
        pd.DataFrame(pred).to_csv('Steps/Step_3/Prediction/LAT_pred_scenario_' + scn + '.csv', float_format='%.6f',
                                  index=False)
    print(' ---- Stop predict LATITUDE ----')

    import pickle
    with open('ModelliEstratti/model_latitude_FULL' + scn[:1] + '.pickle', 'wb') as f:
        pickle.dump(neigh, f)


def predict_lat(scenari):
    neigh = RandomForestRegressor(max_depth=80, random_state=0)

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

        kf = KFold(n_splits=k, random_state=None)
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
        with open('ModelliEstratti/model_latitude_' + scn + '.pickle', 'wb') as f:
            pickle.dump(neigh, f)


def predict_lon_per_scenario(scenari):
    print(' ---- Start predict LONGITUDE ----')

    neigh = RandomForestRegressor(max_depth=10, random_state=0)

    for scn in scenari:

        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')

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

        kf = KFold(n_splits=k, random_state=None)
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
        pd.DataFrame(pred).to_csv('Steps/Step_3/Prediction/LON_pred_scenario_' + scn + '.csv', float_format='%.6f',
                                  index=False)
    print(' ---- Stop predict LONGITUDE ----')

    import pickle
    with open('ModelliEstratti/model_longitude_FULL' + scn[:1] + '.pickle', 'wb') as f:
        pickle.dump(neigh, f)


def predict_lon(scenari):
    neigh = RandomForestRegressor(max_depth=80, random_state=0)

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

        kf = KFold(n_splits=k, random_state=None)
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
        with open('ModelliEstratti/model_longitude_' + scn + '.pickle', 'wb') as f:
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

    kf = KFold(n_splits=k, random_state=None)
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
    neigh = RandomForestRegressor(max_depth=80, random_state=0)
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
