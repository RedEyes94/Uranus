from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np


class Venus:
    def __init__(self, data, scenario):
        self.name = 'VENUS'
        self.data = data
        self.features = self.data.columns
        self.scenario = scenario
        self.shape = self.data.shape
        self.scenario = scenario
        self.latitude = 51.52126391
        self.longitude = 5.85862734
    

    '''def data_analysis(self):
        analysis = DataAnalysis(self.data)
        analysis.start_analysis()'''
        
    def clean_data(self):

        print('***************************************************')
        print(' Start pulizia VENUS scenario :  ' + self.scenario)
        print('***************************************************')

        self.split_categorical()

        print('***************************************************')
        print(' Stop pulizia VENUS scenario :  ' + self.scenario)
        print('***************************************************')
        self.data = self.data.fillna(value=0)

        self.data.to_csv('DataCleanResults/Scenario_'+self.scenario+'/[1]_VENUS_data_cleaned_' + self.scenario + '.csv')
    
    def split_timestamp(self):

        for i in range(0, self.data.shape[0]):
            if self.data['datetime(utc)'][i] != 0:
                val = self.data['datetime(utc)'][i][11:]
                self.data['datetime(utc)'][i] = val

        self.data[['datetime_hh', 'datetime_mi', 'datetime_ss']] = self.data['datetime(utc)'].str.split(':', expand=True)
        del self.data['datetime(utc)']
    
    def split_categorical(self):

        self.data = self.data.fillna(value=0)

        self.data['DJI OcuSync'] = 0
        self.data['DJI Mavic Mini'] = 0
        self.data['Cheerson Leopard 2'] = 0
        self.data['DJI Mavic Pro long'] = 0
        self.data['nan'] = 0
        self.data['DJI Mavic Pro short'] = 0
        self.data['Hubsan'] = 0
        self.data['Futaba FASST-7CH Var. 1'] = 0
        self.data['AscTec Falcon 8 Downlink, DJI Mavic Mini'] = 0
        self.data['DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G'] = 0
        self.data['DJI Mavic Mini, MJX R/C Technic'] = 0
        self.data['Udi R/C 818A'] = 0
        self.data['MJX R/C Technic'] = 0
        self.data['TT Robotix Ghost'] = 0
        self.data['Udi R/C'] = 0
        self.data[
            'DJI Mini, DJI Phantom 4 Pro/Mavic Pro, DJI Phantom 4/Mavic Pro'] = 0
        self.data['DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G'] = 0
        self.data['Spektrum DSMX downlink'] = 0
        self.data['Spektrum DSMX 12CH uplink'] = 0
        self.data['MJX X901'] = 0
        self.data[
            'DJI Mavic Pro long, DJI Phantom 4/Mavic Pro,DJI Phantom/Mavic Pro'] = 0
        self.data['AscTec Falcon 8 Downlink'] = 0

        for i in range(0, self.data.shape[0]):
            if self.data.loc[i, 'VenusTrigger_VenusName'] == 'DJI OcuSync':
                self.data.loc[i, 'DJI OcuSync'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'DJI Mavic Mini':
                self.data.loc[i, 'DJI Mavic Mini'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'Cheerson Leopard 2':
                self.data.loc[i, 'Cheerson Leopard 2'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'DJI Mavic Pro long':
                self.data.loc[i, 'DJI Mavic Pro long'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 0 :
                self.data.loc[i,'VenusName NaN'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'DJI Mavic Pro short':
                self.data.loc[i,'DJI Mavic Pro short'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'Hubsan':
                self.data.loc[i,'Hubsan'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'Futaba FASST-7CH Var. 1':
                self.data.loc[i, 'Futaba FASST-7CH Var. 1'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'AscTec Falcon 8 Downlink, DJI Mavic Mini':
                self.data.loc[i, 'AscTec Falcon 8 Downlink, DJI Mavic Mini'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G':
                self.data.loc[i, 'DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'DJI Mavic Mini, MJX R/C Technic':
                self.data.loc[i, 'DJI Mavic Mini, MJX R/C Technic'] = 1


            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'Udi R/C 818A':
                self.data.loc[i, 'Udi R/C 818A'] = 1


            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'MJX R/C Technic':
                self.data.loc[i, 'MJX R/C Technic'] = 1


            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'TT Robotix Ghost':
                self.data.loc[i,'TT Robotix Ghost'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'Udi R/C':
                self.data.loc[i,'Udi R/C'] = 1

            elif self.data.loc[
                i, 'VenusTrigger_VenusName'] == 'DJI Mavic Mini, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G':
                self.data.loc[i,'DJI Mini, DJI Phantom 4 Pro/Mavic Pro, DJI Phantom 4/Mavic Pro'] = 1

            elif self.data.loc[
                i, 'VenusTrigger_VenusName'] == 'DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G':
                self.data.loc[i,'DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'Spektrum DSMX downlink':
                self.data.loc[i,'Spektrum DSMX downlink'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'Spektrum DSMX 12CH uplink':
                self.data.loc[i,'Spektrum DSMX 12CH uplink'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'MJX X901':
                self.data.loc[i,'MJX X901'] = 1

            elif self.data.loc[
                i, 'VenusTrigger_VenusName'] == 'DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G':
                self.data.loc[i,'DJI Mavic Pro long, DJI Phantom 4/Mavic Pro,DJI Phantom/Mavic Pro'] = 1

            elif self.data.loc[i, 'VenusTrigger_VenusName'] == 'AscTec Falcon 8 Downlink':
                self.data.loc[i,'AscTec Falcon 8 Downlink'] = 1

        del self.data['VenusTrigger_VenusName']

        self.data['ISM 2.4 GHz'] = 0
        self.data['ISM 5.8 GHz'] = 0
        self.data['FreqBand_Null'] = 0

        for i in range(0, self.data.shape[0]):
            if self.data.loc[i, 'VenusTrigge_FrequencyBand'] == 'ISM 2.4 GHz':
                self.data.loc[i, 'ISM 2.4 GHz'] = 1
            elif self.data.loc[i, 'VenusTrigge_FrequencyBand'] == 'ISM 5.8 GHz':
                self.data.loc[i, 'ISM 5.8 GHz'] = 1
            elif self.data.loc[i, 'VenusTrigge_FrequencyBand'] == 0:
                self.data['FreqBand_Null'] = 1

        del self.data['VenusTrigge_FrequencyBand']
        del self.data['VenusTrigge_LifeStatus']

        return self.data
        '''self.data.VenusTrigge_FrequencyBand = self.data.VenusTrigge_FrequencyBand.replace(0, 'operation')
        self.data.VenusTrigge_FrequencyBand = self.data.VenusTrigge_FrequencyBand.replace(1, 'operation')
        self.data.VenusTrigge_FrequencyBand = self.data.VenusTrigge_FrequencyBand.replace(2, 'operation')

        if 'VenusTrigge_FrequencyBand' in self.data:
            if self.scenario == '3':
                self.data.VenusTrigge_FrequencyBand = self.data.VenusTrigge_FrequencyBand.replace('ISM 2.4 GHz', 1)
                self.data['ISM 2.4 GHz'] = self.data['VenusTrigge_FrequencyBand']

            else:
                enc = OneHotEncoder(handle_unknown='ignore')
                enc_df = pd.DataFrame(enc.fit_transform(self.data[['VenusTrigge_FrequencyBand']]).toarray())
                self.data = self.data.join(enc_df)

                if self.scenario == '1_1':
                    self.data.rename(
                        columns={0: 'ISM 2.4 GHz'}, inplace=True)

                self.data.rename(
                    columns={0: 'ISM 2.4 GHz', 1: 'ISM 5.8 GHz'}, inplace=True)'''


