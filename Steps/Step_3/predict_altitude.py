import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from math import sqrt
import pickle


def predict_altitude(scenari):
    neigh = RandomForestRegressor(max_depth=10, random_state=42)

    for scn in scenari:
        data = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_' + scn + '.csv')
        data_bk = data
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

        kf = KFold(n_splits=5, random_state=42)
        scores = []
        X = pd.DataFrame(X)
        mae_rate = []
        mse_rate = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_altitude[train_index], y_altitude[test_index]

            import matplotlib.pyplot as plt
            #################

            train_results = []
            test_results = []
            list_nb_trees = [5, 10, 15, 30, 45, 60, 80, 100]

            for nb_trees in list_nb_trees:
                rf = RandomForestRegressor(n_estimators=nb_trees)
                rf.fit(X_train, y_train)

                train_results.append(mean_squared_error(y_train, rf.predict(X_train)))
                test_results.append(mean_squared_error(y_test, rf.predict(X_test)))

            line1, = plt.plot(list_nb_trees, train_results, color="r", label="Training Score")
            line2, = plt.plot(list_nb_trees, test_results, color="g", label="Testing Score")

            plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
            plt.ylabel('MSE')
            plt.xlabel('n_estimators')
            plt.show()

            #################
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
        data['datetime(utc)'] = data_bk['datetime(utc)']
        data = data.sort_values('datetime(utc)')
        import matplotlib.pyplot as plt
        x = data['datetime(utc)'].str.slice(10, 19)
        data['altitude(m)'] = neigh.predict(X)
        data.to_csv('PredictionResults/Training/Altitude/altitude_data_result_'+scn+'.csv')

        y = data['altitude(m)']
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ticks = ax.get_xticks()
        ax.set_xticks([ticks[i] for i in [0, round(len(ticks) / 2), len(ticks) - 1]])
        plt.savefig('PredictionResults/Training/Altitude/altitude_result_'+scn)

    with open('Models/model_altitude.pickle', 'wb') as f:
        pickle.dump(neigh, f)


'''def predict_altitude(scenari):
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

        with open('Models/model_altitude' + scn + '.pickle', 'wb') as f:
            pickle.dump(neigh, f)
'''