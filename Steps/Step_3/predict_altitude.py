import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from math import sqrt
import pickle


def predict_altitude_per_scenario(scenari):
    neigh = RandomForestRegressor(max_depth=10, random_state=0)

    for scn in scenari:
        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')

        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        y_altitude = scenario['altitude(m)']
        y_altitude = np.array(y_altitude)

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

        kf = KFold(n_splits=5, random_state=None)
        scores = []
        X = pd.DataFrame(X)
        mae_rate = []
        mse_rate = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_altitude[train_index], y_altitude[test_index]

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

    with open('ModelliEstratti/model_altitude_FULL' + scn + '.pickle', 'wb') as f:
        pickle.dump(neigh, f)


def predict_altitude(scenari):
    neigh = RandomForestRegressor(max_depth=10, random_state=0)

    for scn in scenari:
        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')

        scenario = data.reset_index()
        scenario = scenario.drop('index', axis=1)

        y_altitude = scenario['altitude(m)']
        y_altitude = np.array(y_altitude)

        scenario = scenario[['AlviraTracksTrackVelocity_Azimuth', 'AlviraTracksTrackVelocity_Speed',
                             'AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
                             'AlviraTracksTrack_Score',
                             'AlviraTracksTrack_Reflection',
                             'ArcusSystemStatusSensorStatusOrientetation_Elevation',
                             'ArcusSystemStatusSensorStatusOrientetation_Azimuth',
                             'ArcusPotentiaDronPlotPlotPosition_altitude',
                             'ArcusTracksTrackPosition_Altitude',
                             'ArcusTracksTrackVelocity_Elevation',
                             'ArcusSystemStatusSensorPosition_Altitude']]

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
            y_train, y_test = y_altitude[train_index], y_altitude[test_index]

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

        with open('ModelliEstratti/model_altitude' + scn + '.pickle', 'wb') as f:
            pickle.dump(neigh, f)
