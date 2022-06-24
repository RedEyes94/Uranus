from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class Diana:
    def __init__(self, data, scenario):
        self.name = 'DIANA'
        self.data = data
        self.features = self.data.columns
        self.shape = self.data.shape
        self.scenario = scenario
        self.latitude = 51.519137
        self.longitude = 5.857951


    ''' def data_analysis(self):
        analysis = DataAnalysis(self.data)
        analysis.start_analysis()'''

    def clean_data(self):
        print('***************************************************')
        print(' Start pulizia DIANA scenario :  ' + self.scenario)
        print('***************************************************')

        print(' --------------------------------------------------')
        print(' 1 - Splitto features categoriche,  scenario:  ' + self.scenario)
        print(' --------------------------------------------------')
        self.split_categorical()
        self.data = self.data.fillna(value=0)

        self.data.to_csv('DataCleanResults/Scenario_'+self.scenario+'/[1]_DIANA_data_cleaned_' + self.scenario + '.csv')

        print('***************************************************')
        print(' Stop pulizia DIANA scenario :  ' + self.scenario)
        print('***************************************************')

        return self.data

    def split_timestamp(self):

        for i in range(0, self.data.shape[0]):
            if self.data['datetime(utc)'][i] != 0:
                val = self.data['datetime(utc)'][i][11:]
                self.data['datetime(utc)'][i] = val

        self.data[['datetime_hh', 'datetime_mi', 'datetime_ss']] = self.data['datetime(utc)'].str.split(':',
                                                                                                        expand=True)
        del self.data['datetime(utc)']

    def drop_diana_features(self, features):
        for i in range(0, features.shape[0]):
            X = X.drop(features[i], axis=1)

    def split_categorical(self):
        enc = OneHotEncoder(handle_unknown='ignore')

        if 'DianaTargetsTargetClassification_type' in self.data:
            self.data.DianaTargetsTargetClassification_type = self.data.DianaTargetsTargetClassification_type.replace(0, 'none')
            self.data['DianaTarget_Aircraft'] = 0
            self.data['DianaTarget_Controller'] = 0
            self.data['DianaTarget_None'] = 0
            for i in range(0, self.data.shape[0]):
                if self.data.loc[i,'DianaTargetsTargetClassification_type'] == 'controller':
                    self.data.loc[i,'DianaTarget_Controller'] = 1
                elif self.data.loc[i,'DianaTargetsTargetClassification_type'] == 'aircraft':
                    self.data.loc[i,'DianaTarget_Aircraft'] = 1
                elif self.data.loc[i,'DianaTargetsTargetClassification_type'] == 0:
                    self.data.loc[i,'DianaTarget_None'] = 1

        del self.data['DianaTargetsTargetClassification_type']

        if 'DianaTargetsTargetClassification_model' in self.data:
            self.data.DianaTargetsTargetClassification_model = self.data.DianaTargetsTargetClassification_model.replace(0,'none')
            if 'DianaTargetsTargetClassification_model' in self.data:

                self.data['DianaClasssification_Unknown'] = 0
                self.data['DianaClasssification_DJI-MAVIC-PRO-PLATINUM'] = 0
                self.data['DianaClasssification_Wifi-Bluetooth'] = 0
                self.data['DianaClasssification_DJI-MAVIC-2-PRO'] = 0
                self.data['DianaClasssification_DJI-Phantom-4F'] = 0
                self.data['DianaClasssification_Parrot-ANAFI'] = 0
                self.data['DianaClasssification_DJI-Phantom-4E'] = 0
                self.data['DianaClasssification_SPEKTRUM-DX5e'] = 0
                self.data['DianaClasssification_SYMA-X8HW'] = 0
                self.data['DianaClasssification_DJI-MAVIC-AIR'] = 0
                self.data['DianaClasssification_None'] = 0
                self.data['DianaClasssification_VISUO-Zen'] = 0

                for i in range(0, self.data.shape[0]):
                    if self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'Unknown':
                        self.data.loc[i, 'DianaClasssification_Unknown'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'DJI-MAVIC-PRO-PLATINUM':
                        self.data.loc[i, 'DianaClasssification_DJI-MAVIC-PRO-PLATINUM'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 0:
                        self.data.loc[i, 'DianaTarget_None'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'Wifi-Bluetooth':
                        self.data.loc[i, 'DianaClasssification_Wifi-Bluetooth'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'DJI-MAVIC-2-PRO':
                        self.data.loc[i, 'DianaClasssification_DJI-MAVIC-2-PRO'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'DJI-Phantom-4F':
                        self.data.loc[i, 'DianaClasssification_DJI-Phantom-4F'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'Parrot-ANAFI':
                        self.data.loc[i, 'DianaClasssification_Parrot-ANAFI'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'DJI-Phantom-4E':
                        self.data.loc[i, 'DianaClasssification_DJI-Phantom-4E'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'SPEKTRUM-DX5e':
                        self.data.loc[i, 'DianaClasssification_SPEKTRUM-DX5e'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'SYMA-X8HW':
                        self.data.loc[i, 'DianaClasssification_SYMA-X8HW'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'DJI-MAVIC-AIR':
                        self.data.loc[i, 'DianaClasssification_DJI-MAVIC-AIR'] = 1

                    elif self.data.loc[i, 'DianaTargetsTargetClassification_model'] == 'VISUO-Zen':
                        self.data.loc[i, 'DianaClasssification_VISUO-Zen'] = 1

                del self.data['DianaTargetsTargetClassification_model']

                '''enc_df = pd.DataFrame(enc.fit_transform(self.data[['DianaTargetsTargetClassification_model']]).toarray())
                self.data = self.data.join(enc_df)
                del self.data['DianaTargetsTargetClassification_model']
                if self.scenario == '2_1':
                    self.data.rename(
                        columns={0: 'DianaClasssification_DJI_MAVIC_2_PRO',
                                 1: 'DianaClasssification_DJI_MAVIC_PRO_PLATINUM',
                                 2: 'DianaClasssification_DJI-Phantom-4E',
                                 3: 'DianaClasssification_DJI-Phantom-4F', 4: 'DianaClasssification_Parrot-ANAFI', 5: 'DianaClasssification_SPECTRUM-DX5e',
                                 6: 'DianaClasssification_unknown', 7: 'DianaClasssification_none'},
                        inplace=True)
                if self.scenario == '2_2':
                    self.data.rename(
                        columns={0: 'DianaClasssification_DJI_MAVIC_2_PRO',
                                 1: 'DianaClasssification_DJI_MAVIC_AIR',
                                 2: 'DianaClasssification_DJI_MAVIC_2_PRO_PLATINUM',
                                 3: 'DianaClasssification_DJI-Phantom-4E', 4: 'DianaClasssification_DJI-Phantom-4F', 5: 'DianaClasssification_Parrot-ANAFI',
                                 6: 'DianaClasssification_spektrum-DX5E', 7: 'DianaClasssification_unknown'},
                        inplace=True)
                elif self.scenario == '1_4':
                    self.data.rename(
                        columns={0: 'DianaClasssification_DJI_MAVIC_2_PRO', 1: 'DianaClasssification_DJI_MAVIC_PRO_PLATINUM',
                                 2: 'DianaClasssification_DJI-Phantom-4F',
                                 3: 'DianaClasssification_Parrot-ANAFI', 4: '', 5: 'DianaClasssification_unknown',
                                 6: 'DianaClasssification_WIFI_BLUETOOTH', 7: 'DianaClasssification_NONE', 8: 'SYMA - X8HW'},
                        inplace=True)
                else:
                    self.data.rename(
                        columns={0: 'DianaClasssification_DJI_MAVIC_2_PRO', 1: 'DianaClasssification_DJI_MAVIC_PRO_PLATINUM',
                                 2: 'DianaClasssification_DJI-Phantom-4F',
                                 3: 'DianaClasssification_Parrot-ANAFI', 4: '', 5: 'DianaClasssification_unknown',
                                 6: 'DianaClasssification_WIFI_BLUETOOTH', 7: 'DianaClasssification_NONE'}, inplace=True)'''
