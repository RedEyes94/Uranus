import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pickle


def predict_speed_per_scenario(scenari):
    neigh = RandomForestRegressor(max_depth=10, random_state=0)

    for scn in scenari:
        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')

        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        if scn == '3':
            scenario['speed(mph)'] = scenario['speed(mph)'] * 0.44704
            scenario.rename(columns={'speed(mph)': 'speed(mps)'},
                            inplace=True)

        y_speed = scenario['speed(mps)']
        y_speed = np.array(y_speed)

        scenario = scenario[
            ['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
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

        kf = KFold(n_splits=5, random_state=None)
        scores = []
        X = pd.DataFrame(X)
        mae_rate = []
        mse_rate = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_speed[train_index], y_speed[test_index]

            # neigh = RandomForestRegressor(max_depth=10, random_state=0)
            # neigh = SGDRegressor(max_iter=1000, tol=1e-3)
            # neigh = GradientBoostingRegressor(random_state=0)
            # neigh = KNeighborsRegressor(n_neighbors=4)
            neigh.fit(X_train, y_train)

            pred_i = neigh.predict(X_test)
            # score = neigh.score(X_test, y_test)
            score = r2_score(y_test, pred_i)
            mae = mean_absolute_error(y_test, pred_i)
            mse = sqrt(mean_squared_error(y_test, pred_i))

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
            mae_rate.append(mae)
            mse_rate.append(mse)

        avg = sum(scores) / 5
        print('accuracy of each fold - {}'.format(scores))
        print('Avg accuracy : {}'.format(avg))
        print('MAE of each fold - {}'.format(mae_rate))
        print('MSE of each fold - {}'.format(mse_rate))
        print('----------------------'.format(mse_rate))

    with open('ModelliEstratti/model_speed_FULL' + scn + '.pickle', 'wb') as f:
        pickle.dump(neigh, f)


def predict_speed(scenari):
    neigh = RandomForestRegressor(max_depth=10, random_state=0)

    for scn in scenari:
        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')

        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        if scn == '3':
            scenario['speed(mph)'] = scenario['speed(mph)'] * 0.44704
            scenario.rename(columns={'speed(mph)': 'speed(mps)'},
                            inplace=True)

        y_speed = scenario['speed(mps)']
        y_speed = np.array(y_speed)

        scenario = scenario[
            ['AlviraTracksTrackVelocity_Azimuth', 'AlviraTracksTrackVelocity_Speed',
             'AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
             'AlviraTracksTrack_Score', 'AlviraTracksTrack_Reflection',
             'ArcusTracksTrackVelocity_Azimuth', 'ArcusTracksTrackVelocity_Elevation', 'ArcusTracksTrackVelocity_Speed'
             ]]
        scenario = scenario.fillna(value=0)

        X = scenario

        scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        X = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

        kf = KFold(n_splits=5, random_state=None)
        scores = []
        X = pd.DataFrame(X)
        mae_rate = []
        mse_rate = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_speed[train_index], y_speed[test_index]

            # neigh = RandomForestRegressor(max_depth=10, random_state=0)
            # neigh = SGDRegressor(max_iter=1000, tol=1e-3)
            # neigh = GradientBoostingRegressor(random_state=0)
            # neigh = KNeighborsRegressor(n_neighbors=4)
            neigh.fit(X_train, y_train)

            pred_i = neigh.predict(X_test)
            # score = neigh.score(X_test, y_test)
            score = r2_score(y_test, pred_i)
            mae = mean_absolute_error(y_test, pred_i)
            mse = sqrt(mean_squared_error(y_test, pred_i))

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
            mae_rate.append(mae)
            mse_rate.append(mse)

        avg = sum(scores) / 5
        print('accuracy of each fold - {}'.format(scores))
        print('Avg accuracy : {}'.format(avg))
        print('MAE of each fold - {}'.format(mae_rate))
        print('MSE of each fold - {}'.format(mse_rate))
        print('----------------------'.format(mse_rate))

        with open('ModelliEstratti/model_speed_' + scn + '.pickle', 'wb') as f:
            pickle.dump(neigh, f)
