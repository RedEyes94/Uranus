import pandas as pd
import matplotlib.pyplot as plt


def print_altitude(scenario):
    data = pd.read_csv('SubmissionFileTestResults/SubmissionFileScenario_'+scenario+'.csv')
    x = data['TrackDateTimeUTC'].str.slice(10, 23)
    y = data['TrackPositionAltitude']
    plt.plot(x, y)
    plt.gcf().autofmt_xdate()

    plt.show()


def print_velocity(scenario):
    data = pd.read_csv('SubmissionFileTestResults/SubmissionFileScenario_'+scenario+'.csv')
    x = data['TrackDateTimeUTC'].str.slice(10, 23)
    y = data['TrackPositionSpeed']
    plt.plot(x, y)
    plt.gcf().autofmt_xdate()

    plt.show()
