import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from numpy import mean
from numpy import std
from scipy.stats import norm


def calc_rem_outliers(scenari, print_outliers):
    for scenario in scenari:
        print('>> Outliers Scenario : ' + scenario)
        alvira = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_ALVIRA_data_cleaned_' + scenario + '.csv',
                             delimiter=',')
        if print_outliers:
            sns.boxplot(x=alvira['AlviraTracksTrackPosition_Altitude'])
            plt.savefig('OutliersResults/BoxPlot/BoxPlot_Outliers_Alvira_Altitude_' + '_' + scenario + '.pdf')
            plt.show()

            sns.boxplot(x=alvira['AlviraTracksTrackVelocity_Azimuth'])
            plt.savefig('OutliersResults/BoxPlot/BoxPlot_Outliers_Alvira_Azimuth_' + '_' + scenario + '.pdf')
            plt.show()

            sns.boxplot(x=alvira['AlviraTracksTrackVelocity_Speed'])
            plt.savefig('OutliersResults/BoxPlot/BoxPlot_Outliers_Alvira_Speed_' + '_' + scenario + '.pdf')
            plt.show()

        alvira.describe().to_csv('OutliersResults/ResultValues/Alvira_Outliers_scenario_' + scenario + '.csv')

        arcus = pd.read_csv('Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/ARCUS_scenario.csv',
                            delimiter=',')
        if print_outliers:
            sample = arcus['ArcusPotentialDronPlot_rcs']

            # generate a sample
            # sample = normal(loc=50, scale=5, size=1000)
            # calculate parameters
            sample_mean = mean(sample)
            sample_std = std(sample)
            print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
            # define the distribution
            dist = norm(sample_mean, sample_std)
            # sample probabilities for a range of outcomes
            values = [value for value in range(int(np.min(arcus['ArcusPotentialDronPlot_rcs'])),
                                               int(np.max(arcus['ArcusPotentialDronPlot_rcs'])))]
            probabilities = [dist.pdf(value) for value in values]
            # plot the histogram and pdf
            pyplot.hist(sample, bins=10, density=True)
            pyplot.plot(values, probabilities)
            pyplot.show()

            '''sns.boxplot(x=arcus['ArcusTracksTrackPosition_Longitude'])
            plt.savefig('OutliersResults/BoxPlot/BoxPlot_Outliers_Arcus_Longitude_'+'_' + scenario + '.pdf')
            plt.show()

            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Altitude'])
            plt.savefig('OutliersResults/BoxPlot/BoxPlot_Outliers_Arcus_Altitude_'+'_' + scenario + '.pdf')
            plt.show()'''

        '''
        Con questa funzione vengono rimossi gli outliers su Arcus
        '''
        arcus = clean_arcus_outliers(arcus)

        '''if print_outliers:
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Latitude'])
            plt.show()
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Longitude'])
            plt.show()
            sns.boxplot(x=arcus['ArcusTracksTrackPosition_Altitude'])
            plt.show()'''

        arcus.to_csv('DataCleanResults/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv')

        arcus.describe().to_csv('OutliersResults/ResultValues/Arcus_Outliers_scenario_' + scenario + '.csv')

        diana = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_DIANA_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        if print_outliers:
            sns.boxplot(x=diana['DianaTarget_ID'])
            plt.savefig('OutliersResults/BoxPlot/BoxPlot_Outliers_Diana_DianaTarget_ID_' + '_' + scenario + '.pdf')
            plt.show()

            sns.boxplot(x=diana['DianaTargetsTargetSignal_snr_dB'])
            plt.savefig(
                'OutliersResults/BoxPlot/BoxPlot_Outliers_Diana_DianaTargetsTargetSignal_snr_dB_' + '_' + scenario + '.pdf')
            plt.show()

            sns.boxplot(x=diana['DianaTargetsTargetSignal_bearing_deg'])
            plt.savefig(
                'OutliersResults/BoxPlot/BoxPlot_Outliers_Diana_DianaTargetsTargetSignal_bearing_deg_' + '_' + scenario + '.pdf')
            plt.show()

            sns.boxplot(x=diana['DianaTargetsTargetSignal_range_m'])
            plt.savefig(
                'OutliersResults/BoxPlot/BoxPlot_Outliers_Diana_DianaTargetsTargetSignal_range_m_' + '_' + scenario + '.pdf')
            plt.show()

        diana.describe().to_csv('OutliersResults/ResultValues/Diana_Outliers_scenario_' + scenario + '.csv')

        venus = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_VENUS_data_cleaned_' + scenario + '.csv',
                            delimiter=',')
        if print_outliers:
            sns.boxplot(x=venus['VenusTrigger_RadioId'])
            plt.savefig(
                'OutliersResults/BoxPlot/BoxPlot_Outliers_Venus_VenusTrigger_RadioId_' + '_' + scenario + '.pdf')
            plt.show()

            sns.boxplot(x=venus['VenusTrigger_Azimuth'])
            plt.savefig(
                'OutliersResults/BoxPlot/BoxPlot_Outliers_Venus_VenusTrigger_Azimuth_' + '_' + scenario + '.pdf')
            plt.show()

        venus.describe().to_csv('OutliersResults/ResultValues/Venus_Outliers_scenario_' + scenario + '.csv')
        print('--END OUTLIERS--')


def clean_arcus_outliers(arcs1):
    min = arcs1['ArcusTracksTrackPosition_Latitude'].quantile(0.25)
    max = arcs1['ArcusTracksTrackPosition_Latitude'].quantile(0.75)

    arcs1["ArcusTracksTrackPosition_Latitude"] = np.where(arcs1["ArcusTracksTrackPosition_Latitude"] < min, min,
                                                          arcs1['ArcusTracksTrackPosition_Latitude'])
    arcs1["ArcusTracksTrackPosition_Latitude"] = np.where(arcs1["ArcusTracksTrackPosition_Latitude"] > max, max,
                                                          arcs1['ArcusTracksTrackPosition_Latitude'])

    min = arcs1['ArcusTracksTrackPosition_Longitude'].quantile(0.25)
    max = arcs1['ArcusTracksTrackPosition_Longitude'].quantile(0.75)

    arcs1["ArcusTracksTrackPosition_Longitude"] = np.where(arcs1["ArcusTracksTrackPosition_Longitude"] < min, min,
                                                           arcs1['ArcusTracksTrackPosition_Longitude'])
    arcs1["ArcusTracksTrackPosition_Longitude"] = np.where(arcs1["ArcusTracksTrackPosition_Longitude"] > max, max,
                                                           arcs1['ArcusTracksTrackPosition_Longitude'])

    arcs1 = arcs1.groupby('datetime(utc)').mean()
    arcs1 = arcs1.reset_index()

    return arcs1
