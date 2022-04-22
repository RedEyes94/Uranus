from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class Alvira:
    def __init__(self, data, scenario):

        self.name = 'ALVIRA'
        self.data = data
        self.features = self.data.columns
        self.shape = self.data.shape
        self.scenario = scenario
        self.latitude = 51.519271
        self.longitude = 5.8579155

    def clean_data(self):

        print('-------------------------------------------------------------------------------------------------------')
        print('                                      START - Pulizia ALVIRA scenario :  ' + self.scenario)
        print('-------------------------------------------------------------------------------------------------------')

        print(' 1 - Cancello i timestamp, scenario:  ' + self.scenario)
        print(' --------------------------------------------------')
        del self.data['AlviraTracks_timestamp']
        del self.data['AlviraTracksTrack_Timestamp']
        del self.data['AlviraSystemStatus_timestamp']

        del self.data['AlviraSystemStatus_Name']
        del self.data['AlviraSystemStatus_Version']
        del self.data['AlviraSystemStatus_OperationalState']
        del self.data['AlviraSystemStatusSensor_Name']
        del self.data['AlviraSystemStatusSensor_Version']
        del self.data['AlviraSystemStatusSensor_OperationalState']
        del self.data['AlviraSystemStatusSensorStatus_SensorType']

        print(' --------------------------------------------------')
        print(' 2 - Splitto features categoriche,  scenario:  ' + self.scenario)
        print(' --------------------------------------------------')
        self.split_categorical()
        print(' --------------------------------------------------')
        print(' 3 - Elimino le features con 100% null,  scenario:  ' + self.scenario)
        print(' --------------------------------------------------')
        self.drop_feature_100_missing_rate()

        self.data = self.data.fillna(value=0)
        self.data.to_csv(
            '1-DataCleaned/Scenario_' + self.scenario + '/[1]_ALVIRA_data_cleaned_' + self.scenario + '.csv',
            float_format='%.6f', index=False)
        print('-------------------------------------------------------------------------------------------------------')
        print('                                      END - Pulizia ALVIRA scenario : ' + self.scenario)
        print('-------------------------------------------------------------------------------------------------------')

    def drop_feature_100_missing_rate(self):
        to_drop = []
        for col in self.data.columns:
            pct_missing = np.mean(self.data[col].isnull())
            if (pct_missing * 100) == 100:  # ritorno le features con solo NULL
                to_drop.append(col)
            # print('{} - {}%'.format(col, (pct_missing * 100)))
        for i in range(0, len(to_drop)):
            self.data = self.data.drop(to_drop[i], axis=1)
        return 0

    def normalize_lat_long(self):
        self.data.astype({"AlviraTracksTrackPosition_Latitude": float})
        self.data.astype({"AlviraTracksTrackPosition_Longitude": float})

        self.data['AlviraTracksTrackPosition_Latitude'].map('{:,.6f}'.format)
        self.data['AlviraTracksTrackPosition_Longitude'].map('{:,.6f}'.format)

    def split_alvira_time_stamp(self):

        for i in range(0, self.data.shape[0]):
            if self.data['datetime(utc)'][i] != 0:
                val = self.data['datetime(utc)'][i][11:]
                self.data['datetime(utc)'][i] = val

            # 1) Rimozione della data, mantengo solo data ora secondi
        for i in range(0, self.data.shape[0]):
            if self.data['AlviraTracks_timestamp'][i] != 0:
                val = self.data['AlviraTracks_timestamp'][i][11:23]
                self.data['AlviraTracks_timestamp'][i] = val

            if self.data['AlviraTracksTrack_Timestamp'][i] != 0:
                val = self.data['AlviraTracksTrack_Timestamp'][i][11:23]
                self.data['AlviraTracksTrack_Timestamp'][i] = val

            # 1) Rimozione della data, mantengo solo data ora secondi
            if self.data['AlviraSystemStatus_timestamp'][i] != 0:
                val = self.data['AlviraSystemStatus_timestamp'][i][11:23]
                self.data['AlviraSystemStatus_timestamp'][i] = val

            # 2) splitto la data in HH:MI:SS
        self.data[['datetime_hh', 'datetime_mi', 'datetime_ss']] = self.data['datetime(utc)'].str.split(':',
                                                                                                        expand=True)

        self.data[['alvira_hh', 'alvira_mi', 'alvira_ss_1']] = self.data['AlviraTracks_timestamp'].str.split(':',
                                                                                                             expand=True)
        self.data[['alvira_ss', 'alvira_ms']] = self.data['alvira_ss_1'].str.split('.', expand=True)

        self.data[['alvira_tracks_hh', 'alvira_tracks_mi', 'alvira_tracks_ss_1']] = self.data[
            'AlviraTracksTrack_Timestamp'].str.split(':',
                                                     expand=True)
        self.data[['alvira_tracks_ss', 'alvira_tracks_ms']] = self.data['alvira_tracks_ss_1'].str.split('.',
                                                                                                        expand=True)

        self.data[['alvira_status_hh', 'alvira_status_mi', 'alvira_status_ss_1']] = self.data[
            'AlviraSystemStatus_timestamp'].str.split(':', expand=True)
        self.data[['alvira_status_ss', 'alvira_status_ms']] = self.data['alvira_status_ss_1'].str.split('.',
                                                                                                        expand=True)

        # 3) Riempio i valori nulli e cancello le colonne di appoggio o quelle sostituite
        self.data.alvira_hh.fillna(self.data.alvira_status_hh, inplace=True)

        del self.data['alvira_status_hh']
        self.data.alvira_mi.fillna(self.data.alvira_status_mi, inplace=True)
        del self.data['alvira_status_mi']
        self.data.alvira_ss.fillna(self.data.alvira_status_ss, inplace=True)
        del self.data['alvira_status_ss']
        self.data.alvira_ms.fillna(self.data.alvira_status_ms, inplace=True)
        del self.data['alvira_status_ms']

        del self.data['alvira_ss_1']
        del self.data['alvira_tracks_ss_1']
        del self.data['alvira_status_ss_1']

    def split_categorical(self):

        self.data.AlviraTracksTrack_Classification = self.data.AlviraTracksTrack_Classification.replace(0, 'NO_DRONE')

        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(self.data[['AlviraTracksTrack_Classification']]).toarray())
        self.data = self.data.join(enc_df)
        del self.data['AlviraTracksTrack_Classification']
        self.data.rename(columns={0: 'DRONE', 1: 'NO_DRONE', 2: 'SUSPECTED_DRONE'}, inplace=True)

        for i in range(0, self.data.shape[0]):
            if self.data.loc[i, 'AlviraTracksTrack_Alarm'] == True:
                self.data.loc[i, 'AlviraTracksTrack_Alarm'] = 1
            else:
                self.data.loc[i, 'AlviraTracksTrack_Alarm'] = 0

    def drop_feature(self):

        self.data = self.data.drop('AlviraSystemStatus_Name', axis=1)
        self.data = self.data.drop('AlviraSystemStatus_Version', axis=1)
        self.data = self.data.drop('AlviraSystemStatus_OperationalState', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensor_Name', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensor_Version', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensor_OperationalState', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensorPosition_Latitude', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensorPosition_Longitude', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensorPosition_Altitude', axis=1)
        self.data = self.data.drop('total in seconds', axis=1)
        self.data = self.data.drop('AlviraTracksTrackVelocity_Elevation', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensorStatus_SensorType', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensorStatusBlankSector_Angle', axis=1)
        self.data = self.data.drop('AlviraSystemStatusSensorStatusBlankSector_Span', axis=1)

        return self.data

    def map_label_to_string(self, scenario):

        self.data['LABEL'] = self.data['LABEL'].map(str)
        if scenario == '1_1':
            self.data['LABEL'] = self.data['LABEL'].str.replace('1', 'DRONE')
            self.data['LABEL'] = self.data['LABEL'].str.replace('2', 'DRONE')
            self.data['LABEL'] = self.data['LABEL'].str.replace('0', 'UNKNOWN')
        if scenario == '1_2':
            self.data['LABEL'] = self.data['LABEL'].str.replace('1', 'DRONE')
            self.data['LABEL'] = self.data['LABEL'].str.replace('2', 'UNKNOWN')
            self.data['LABEL'] = self.data['LABEL'].str.replace('0', 'UNKNOWN')
        if scenario == '1_3':
            self.data['LABEL'] = self.data['LABEL'].str.replace('1', 'UNKNOWN')
            self.data['LABEL'] = self.data['LABEL'].str.replace('2', 'DRONE')
            self.data['LABEL'] = self.data['LABEL'].str.replace('0', 'DRONE')
        if scenario == '1_4':
            self.data['LABEL'] = self.data['LABEL'].str.replace('1', 'UNKNOWN')
            self.data['LABEL'] = self.data['LABEL'].str.replace('2', 'DRONE')
            self.data['LABEL'] = self.data['LABEL'].str.replace('0', 'UNKNOWN')
        if scenario == 'a':
            self.data['LABEL'] = self.data['LABEL'].str.replace('1', 'UNKNOWN')
            self.data['LABEL'] = self.data['LABEL'].str.replace('2', 'DRONE')
            self.data['LABEL'] = self.data['LABEL'].str.replace('0', 'UNKNOWN')

        for i in range(0, self.data.shape[0]):
            if self.data['LABEL'][i] == 'DRONE':
                self.data['LABEL'][i] = 1
            else:
                self.data['LABEL'][i] = 0
