import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


class Arcus:

    def __init__(self, data, scenario):
        self.name = 'ARCUS'
        self.data = data
        self.features = self.data.columns
        self.shape = self.data.shape
        self.scenario = scenario
        self.latitude = 51.52147
        self.longitude = 5.87056833

    '''def data_analysis(self):
        analysis = DataAnalysis(self.data)
        analysis.start_analysis()'''

    def normalize_lat_long(self):

        self.data.astype({"ArcusTracksTrackPosition_Latitude": float})
        self.data.astype({"ArcusTracksTrackPosition_Longitude": float})

        self.data['ArcusTracksTrackPosition_Latitude'].map('{:,.6f}'.format)
        self.data['ArcusTracksTrackPosition_Longitude'].map('{:,.6f}'.format)

    def clean_data(self):

        print('***************************************************')
        print(' Start pulizia ARCUS scenario :  ' + self.scenario)
        print('***************************************************')

        print(' 1 - Cancello i timestamp e altre features, scenario:  ' + self.scenario)
        print(' --------------------------------------------------')
        self.data = self.data.drop('SystemStatusSensorStatus_Messages', axis=1)
        self.data = self.data.drop('ArcusSystemStatus_timestamp', axis=1)
        self.data = self.data.drop('ArcusPotentialDronePlots_timestamp', axis=1)
        self.data = self.data.drop('ArcusTracks_timestamp', axis=1)
        self.data = self.data.drop('ArcusTracksTrack_Timestamp', axis=1)

        print(' --------------------------------------------------')
        print(' 2 - Splitto features categoriche,  scenario:  ' + self.scenario)
        print(' --------------------------------------------------')
        self.split_categorical()
        print(' --------------------------------------------------')

        print(' --------------------------------------------------')
        print(' 3 - Cancello features con il 100% dei valori nulli ,  scenario:  ' + self.scenario)
        print(' --------------------------------------------------')

        self.drop_feature_100_missing_rate()

        print(' --------------------------------------------------')        #self.data['datetime(utc)'] = pd.to_datetime(self.data['datetime(utc)'])
        self.data.to_csv('1-DataCleaned/Scenario_'+self.scenario+'/[1]_ARCUS_data_cleaned_' + self.scenario + '.csv')

        print('***************************************************')
        print(' Stop pulizia ARCUS scenario :  ' + self.scenario)
        print('***************************************************')
        print('')

    def drop_feature_100_missing_rate(self):
        to_drop = []
        for col in self.data.columns:
            pct_missing = np.mean(self.data[col].isnull())
            if (pct_missing * 100) == 100:  # ritorno le features con solo NULL
                to_drop.append(col)
            # print('{} - {}%'.format(col, (pct_missing * 100)))
        for i in range(0, len(to_drop)):
            if to_drop[i] =='ArcusPotentialDronePlots_timestamp' or to_drop[i] =='ArcusTracksTrack_id':
                self.data = self.data.drop(to_drop[i], axis=1)
        return 0

    def split_arcus_timestamp(self):
        for i in range(0, self.data.shape[0]):
            if self.data['datetime(utc)'][i] != 0:
                val = self.data['datetime(utc)'][i][11:]
                self.data['datetime(utc)'][i] = val
            if self.data['ArcusTracksTrack_Timestamp'][i] != 0:
                val = self.data['ArcusTracksTrack_Timestamp'][i][11:23]
                self.data['ArcusTracksTrack_Timestamp'][i] = val
            if self.data['ArcusTracks_timestamp'][i] != 0:
                val = self.data['ArcusTracks_timestamp'][i][11:23]
                self.data['ArcusTracks_timestamp'][i] = val
            if self.data['ArcusPotentialDronePlots_timestamp'][i] != 0:
                val = self.data['ArcusPotentialDronePlots_timestamp'][i][11:23]
                self.data['ArcusPotentialDronePlots_timestamp'][i] = val
            if self.data['ArcusPotentiaDronPlotsPlot_timestamp'][i] != 0:
                val = self.data['ArcusPotentiaDronPlotsPlot_timestamp'][i][11:23]
                self.data['ArcusPotentiaDronPlotsPlot_timestamp'][i] = val
            if self.data['ArcusSystemStatus_timestamp'][i] != 0:
                val = self.data['ArcusSystemStatus_timestamp'][i][11:23]
                self.data['ArcusSystemStatus_timestamp'][i] = val

        self.data[['datetime_hh', 'datetime_mi', 'datetime_ss']] = self.data['datetime(utc)'].str.split(':',
                                                                                                        expand=True)
        del self.data['datetime(utc)']

        self.data[['arcus_hh', 'arcus_mi', 'arcus_ss_1']] = self.data['ArcusTracks_timestamp'].str.split(':',
                                                                                                         expand=True)
        self.data[['arcus_ss', 'arcus_ms']] = self.data['arcus_ss_1'].str.split('.', expand=True)
        del self.data['arcus_ss_1']
        del self.data['ArcusTracks_timestamp']

        self.data[['arcus_tracks_hh', 'arcus_tracks_mi', 'arcus_tracks_ss_1']] = self.data[
            'ArcusTracksTrack_Timestamp'].str.split(':', expand=True)
        self.data[['arcus_tracks_ss', 'arcus_tracks_ms']] = self.data['arcus_tracks_ss_1'].str.split('.',
                                                                                                     expand=True)
        del self.data['arcus_tracks_ss_1']
        del self.data['ArcusTracksTrack_Timestamp']

        self.data[['arcus_plot_hh', 'arcus_plot_mi', 'arcus_plot_ss_1']] = self.data[
            'ArcusPotentialDronePlots_timestamp'].str.split(':', expand=True)
        self.data[['arcus_plot_ss', 'arcus_plot_ms']] = self.data['arcus_plot_ss_1'].str.split('.', expand=True)
        del self.data['arcus_plot_ss_1']
        del self.data['ArcusPotentialDronePlots_timestamp']

        self.data[['arcus_plots_plot_hh', 'arcus_plots_plot_mi', 'arcus_plots_plot_ss_1']] = self.data[
            'ArcusPotentiaDronPlotsPlot_timestamp'].str.split(':', expand=True)
        self.data[['arcus_plots_plot_ss', 'arcus_plots_plot_ms']] = self.data['arcus_plots_plot_ss_1'].str.split(
            '.', expand=True)
        del self.data['arcus_plots_plot_ss_1']
        del self.data['ArcusPotentiaDronPlotsPlot_timestamp']

        self.data[['arcus_status_hh', 'arcus_status_mi', 'arcus_status_ss_1']] = self.data[
            'ArcusSystemStatus_timestamp'].str.split(':', expand=True)
        self.data[['arcus_status_ss', 'arcus_status_ms']] = self.data['arcus_status_ss_1'].str.split('.',
                                                                                                     expand=True)
        del self.data['arcus_status_ss_1']
        del self.data['ArcusSystemStatus_timestamp']

        self.data = self.data.fillna(value=0)

        return self.data

    def split_categorical(self):


        self.data = self.data.fillna(value=0)

        self.data['Arcus_OperationalState_NaN'] = 0
        self.data['Arcus_OperationalState_Idle'] = 0
        self.data['Arcus_OperationalState_Operational'] = 0

        for i in range(0, self.data.shape[0]):
            if self.data.loc[i, 'ArcusSystemStatusSensor_OperationalState'] == 'idle':
                self.data.loc[i, 'Arcus_OperationalState_Idle'] = 1
            elif self.data.loc[i, 'ArcusSystemStatusSensor_OperationalState'] == 'operational':
                self.data.loc[i, 'Arcus_OperationalState_Operational'] = 1
            elif self.data.loc[i, 'ArcusSystemStatusSensor_OperationalState'] == 0:
                self.data.loc[i, 'Arcus_OperationalState_NaN'] = 1

        self.data['Arcus_Classification_NaN'] = 0
        self.data['Arcus_Classification_UNKNOWN'] = 0
        self.data['Arcus_Classification_VEHICLE'] = 0
        self.data['Arcus_Classification_OTHER'] = 0
        self.data['Arcus_Classification_DRONE'] = 0
        self.data['Arcus_Classification_SUSPECTED_DRONE'] = 0
        self.data['Arcus_Classification_HELICOPTER'] = 0

        for i in range(0, self.data.shape[0]):
            if self.data.loc[i, 'ArcusTracksTrack_Classification'] == 0:
                self.data.loc[i, 'Arcus_Classification_NaN'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Classification'] == 'UNKNOWN':
                self.data.loc[i, 'Arcus_Classification_UNKNOWN'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Classification'] == 'VEHICLE':
                self.data.loc[i, 'Arcus_Classification_VEHICLE'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Classification'] == 'OTHER':
                self.data.loc[i, 'Arcus_Classification_OTHER'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Classification'] == 'DRONE':
                self.data.loc[i, 'Arcus_Classification_DRONE'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Classification'] == 'SUSPECTED_DRONE':
                self.data.loc[i, 'Arcus_Classification_SUSPECTED_DRONE'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Classification'] == 'HELICOPTER':
                self.data.loc[i, 'Arcus_Classification_HELICOPTER'] = 1

        self.data['Arcus_Alarm_NaN'] = 0
        self.data['Arcus_Alarm_FALSE'] = 0
        self.data['Arcus_Alarm_TRUE'] = 0

        for i in range(0, self.data.shape[0]):
            if self.data.loc[i, 'ArcusTracksTrack_Alarm'] == 0:
                self.data.loc[i, 'Arcus_Alarm_NaN'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Alarm'] == 'False':
                self.data.loc[i, 'Arcus_Alarm_FALSE'] = 1

            elif self.data.loc[i, 'ArcusTracksTrack_Alarm'] == 'True':
                self.data.loc[i, 'Arcus_Alarm_TRUE'] = 1

        self.data.drop('ArcusTracksTrack_Classification',axis=1)
        self.data.drop('ArcusSystemStatusSensor_OperationalState', axis=1)
        self.data.drop('ArcusTracksTrack_Alarm', axis=1)


def set_value_one_to_not_null(data):
    to_update = []
    for (columnName, columnData) in data.iteritems():
        # Per ARCUS saltiamo ArcusSystemStatusSensorStatusBlankSector_Span e ArcusSystemStatusSensorStatusProcessing_Sensitivity in quanto presentano diversi valori
        # nei diversi scenari
        if columnName != 'ArcusSystemStatusSensorStatusProcessing_Sensitivity' or columnName != 'ArcusSystemStatusSensorStatusBlankSector_Span':
            all_columns_values = np.array(
                data[columnName].dropna())  # elimino i null e vedo se i dati che restano sono tutti uguali
            if len(all_columns_values) != 0:
                if (all_columns_values[0] == all_columns_values).all():
                    to_update.append(columnName)

    data = data.fillna(value=0)

    for column in to_update:
        if column != 'ArcusTracksTrack_Alarm':
            data.loc[data[column] != 0, column] = 1
            data.loc[data[column] == 0, column] = 0
    return data
