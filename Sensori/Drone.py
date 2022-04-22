from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class Drone:

    def __init__(self, data, scenario, parte):
        self.features = data.columns
        self.data = data
        self.name = 'DRONE'
        self.scenario = scenario
        self.parte = parte

    def clean_data(self):
        print('***************************************************')
        print(' Start pulizia DRONE scenario :  ' + self.scenario)
        print('***************************************************')
        print(' 1 - Normalizzo latitudine e longitudine, scenario:  ' + self.scenario)
        print('---------------------------------------------------')
        self.normalize_lat_lon()
        print('---------------------------------------------------')
        print(' 2 - Normalizzo datetime e longitudine, scenario:  ' + self.scenario)
        print('---------------------------------------------------')
        if self.scenario != '3':
            if len(self.data['datetime(utc)'][0]) == 23:
                self.data['datetime(utc)'] = self.data['datetime(utc)'].str.slice(0, 19)
                self.data['datetime(utc)'] = pd.to_datetime(self.data['datetime(utc)'])
        print('---------------------------------------------------')

        if self.parte == 'a' or self.parte == 'b':
            self.data.to_csv('1-DataCleaned/Scenario_'+self.scenario+'/[1]_DRONE_data_cleaned_' + self.scenario + self.parte +'.csv')
        else:
            self.data.to_csv(
                '1-DataCleaned/Scenario_' + self.scenario + '/[1]_DRONE_data_cleaned_' + self.scenario + '.csv')

    def normalize_lat_lon(self):
        if type(self.data['latitude'][0]) is str:
            self.data['latitude'] = self.data['latitude'].str.replace('.', '')
            self.data['latitude'] = self.data['latitude'].str[:2] + '.' + self.data['latitude'].str[2:]

            self.data['longitude'] = self.data['longitude'].str.replace('.', '')
            self.data['longitude'] = self.data['longitude'].str[:1] + '.' + self.data['longitude'].str[1:]

            self.data['latitude'] = self.data['latitude'].astype(float)
            self.data['longitude'] = self.data['longitude'].astype(float)

        self.data['latitude'].map('{:,.6f}'.format)
        self.data['longitude'].map('{:,.6f}'.format)

    def split_categorical(self):
        enc = OneHotEncoder(handle_unknown='ignore')

        if 'flightmode' in self.data:
            enc_df = pd.DataFrame(enc.fit_transform(self.data[['flightmode']]).toarray())
            self.data.join(enc_df)
            # del data['flightmode']
            if self.scenario == '1_1':
                self.data.rename(
                    columns={0: 'flight_mode_gps', 1: 'flight_mode_landing', 2: 'flight_mode_sport',
                             3: 'flight_mode_wp'},
                    inplace=True)
            if self.scenario == '1_2':
                self.data.rename(
                    columns={0: 'flight_mode_gps', 1: 'flight_mode_go_home', 2: 'flight_mode_landing',
                             3: 'flight_mode_wp'},
                    inplace=True)
            if self.scenario == '1_3':
                self.data.rename(
                    columns={0: 'flight_mode_gps', 1: 'flight_mode_landing', 2: 'flight_mode_sport',
                             3: 'flight_mode_wp'},
                    inplace=True)
            if self.scenario == '1_4':
                self.data.rename(columns={0: 'flight_mode_gps', 1: 'flight_mode_go_home', 2: 'flight_mode_landing',
                                          3: 'flight_mode_sport', 4: 'flight_mode_take_off', 5: 'flight_mode_wp'},
                                 inplace=True)
        return self.data