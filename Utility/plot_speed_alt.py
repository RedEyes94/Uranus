import pandas as pd
import matplotlib.pyplot as plt


def print_altitude(scenari, tipo):
    if tipo == 'Test':
        for scenario in scenari:
            data = pd.read_csv('SubmissionFileTestResults/SubmissionFileScenario_' + scenario + '.csv')
            x = data['TrackDateTimeUTC'].str.slice(10, 23)
            y = data['TrackPositionAltitude']
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ticks = ax.get_xticks()
            ax.set_xticks([ticks[i] for i in [0, round(len(ticks) / 2), len(ticks) - 1]])
            # plt.gcf().autofmt_xdate()
            plt.savefig('PredictionResults/Test/Altitude/altitude_result_'+scenario)

            plt.show()
    elif tipo == 'Real':
        for scenario in scenari:
            data = pd.read_csv(
                'Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/2020-09-29_' + scenario + '.csv')
            x = data['datetime(utc)'].str.slice(10, 19)
            y = data['altitude(m)']
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ticks = ax.get_xticks()
            ax.set_xticks([ticks[i] for i in [0, round(len(ticks) / 2), len(ticks) - 1]])
            # plt.gcf().autofmt_xdate()
            plt.savefig('PredictionResults/Training/Altitude/real_altitude_result_'+scenario)

            plt.show()
    elif tipo == 'Train':
        for scenario in scenari:
            data = pd.read_csv(
                'Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/2020-09-29_' + scenario + '.csv')
            x = data['datetime(utc)'].str.slice(10, 19)
            y = data['altitude(m)']
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ticks = ax.get_xticks()
            ax.set_xticks([ticks[i] for i in [0, round(len(ticks) / 2), len(ticks) - 1]])
            # plt.gcf().autofmt_xdate()
            plt.savefig('PredictionResults/Training/Altitude/real_altitude_result_'+scenario)

            plt.show()


def print_speed(scenari, tipo):
    if tipo == 'Test':
        for scenario in scenari:
            data = pd.read_csv('SubmissionFileTestResults/SubmissionFileScenario_' + scenario + '.csv')
            x = data['TrackDateTimeUTC'].str.slice(10, 23)
            y = data['TrackPositionSpeed']

            fig, ax = plt.subplots()
            ax.plot(x, y)
            ticks = ax.get_xticks()
            ax.set_xticks([ticks[i] for i in [0, round(len(ticks) / 2), len(ticks) - 1]])
            # plt.gcf().autofmt_xdate()
            plt.savefig('PredictionResults/Test/Speed/speed_result_'+scenario)
            plt.show()
    elif tipo == 'Training':
        for scenario in scenari:
            data = pd.read_csv(
                'Scenari/Scenario_' + scenario[:1] + '/Scenario_' + scenario + '/2020-09-29_' + scenario + '.csv')
            x = data['datetime(utc)'].str.slice(10, 19)
            y = data['speed(mps)']

            fig, ax = plt.subplots()
            ax.plot(x, y)
            ticks = ax.get_xticks()
            ax.set_xticks([ticks[i] for i in [0, round(len(ticks) / 2), len(ticks) - 1]])
            # plt.gcf().autofmt_xdate()
            plt.savefig('PredictionResults/Training/Speed/speed_result_'+scenario)

            plt.show()
