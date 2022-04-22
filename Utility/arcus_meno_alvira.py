import pandas as pd


def arcus_meno_alvira(scenari):
    for scenario in scenari:
        alvira = pd.read_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_ALVIRA_data_cleaned_' + scenario + '.csv',
                             delimiter=',')
        arcus = pd.read_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv',
                            delimiter=',')

        arcus_meno_alvira1 = arcus[~arcus['datetime(utc)'].isin(alvira['datetime(utc)'])].drop_duplicates()

        arcus_meno_alvira1.to_csv('1-DataCleaned/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv')
