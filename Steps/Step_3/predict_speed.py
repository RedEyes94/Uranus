import pandas as pd
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pickle


def predict_speed(scenari):
    neigh = RandomForestRegressor(max_depth=30, random_state=42)

    for scn in scenari:
        data = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_' + scn + '.csv')
        data_bk = data
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

        kf = KFold(n_splits=5, random_state=42)
        scores = []
        X = pd.DataFrame(X)
        mae_rate = []
        mse_rate = []
        import matplotlib.pyplot as plt

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y_speed[train_index], y_speed[test_index]
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
        import matplotlib.pyplot as plt
        data['speed(mps)'] = neigh.predict(X)
        data['datetime(utc)'] = data_bk['datetime(utc)']
        data = data.sort_values('datetime(utc)')
        data.to_csv('PredictionResults/Training/Speed/speed_data_result_'+scn+'.csv')
        x = data['datetime(utc)'].str.slice(10, 19)
        y = data['speed(mps)']

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ticks = ax.get_xticks()
        ax.set_xticks([ticks[i] for i in [0, round(len(ticks) / 2), len(ticks) - 1]])
        # plt.gcf().autofmt_xdate()
        plt.savefig('PredictionResults/Training/Speed/speed_result_'+scn)
        plt.show()
        '''
        plt.plot(x, y)
        plt.gcf().autofmt_xdate()'''

    with open('Models/model_speed.pickle', 'wb') as f:
        pickle.dump(neigh, f)


'''
def predict_speed(scenari):
    neigh = RandomForestRegressor(max_depth=10, random_state=0)

    for scn in scenari:
        data = pd.read_csv('3-DataMerged/[3]_MERGE_scenario_' + scn + '.csv')
        data_bk = data
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
        import matplotlib.pyplot as plt
        data['speed(mps)'] = neigh.predict(X)
        data['datetime(utc)'] = data_bk['datetime(utc)']
        x = data['datetime(utc)']
        y = data['speed(mps)']
        plt.plot(x, y)
        plt.gcf().autofmt_xdate()

        plt.show()
        with open('Models/model_speed_' + scn + '.pickle', 'wb') as f:
            pickle.dump(neigh, f)
'''