from itertools import cycle

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, \
    roc_curve, auc
from sklearn.model_selection import KFold, train_test_split, learning_curve
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from matplotlib import pyplot


def plot_roc_curve(y_test, y_pred):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i], drop_intermediate=False)
    roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # plt.figure(figsize=(10,5))
    plt.figure(dpi=600)
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
             color="deeppink", linestyle=":", linewidth=4, )

    plt.plot(fpr["macro"], tpr["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
             color="navy", linestyle=":", linewidth=4, )

    colors = cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]), )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend()


def regression_full(scenario, y):
    kf = KFold(n_splits=5, random_state=42)
    acc_score = []
    prediction_values = []
    precision_values = []
    recall_values = []
    roc_values = []
    f1_values = []
    X_train_acc = []
    X_test_acc = []
    y_train_acc = []
    y_test_acc = []

    X = pd.DataFrame(scenario)
    m2 = np.array(y)
    model = RandomForestClassifier(max_depth=4, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = m2[train_index], m2[test_index]

        model.fit(X_train, y_train)
        pred_values = model.predict_proba(X_test)
        # plot_roc_curve(y_test, pred_values)
        prediction_values.append(pred_values)
        # acc = accuracy_score(pred_values, y_test)
        acc = model.score(X_test, y_test)

        # roc = roc_auc_score(y_test, pred_values)
        '''acc = accuracy_score(y_test, pred_values)
        recall = recall_score(y_test, pred_values, average='weighted')
        f1 = f1_score(y_test, pred_values, average='weighted')
        precision = precision_score(y_test, pred_values, average='weighted')'''

        if len(acc_score) == 0:
            X_train_acc = X_train
            X_test_acc = X_test
            y_train_acc = y_train
        elif acc > max(acc_score):
            X_train_acc = X_train
            y_train_acc = y_train
            y_test_acc = y_test

        # roc_values.append(roc)
        acc_score.append(acc)
        '''recall_values.append(recall)
        f1_values.append(f1)
        precision_values.append(precision)'''

    avg_acc_score = sum(acc_score) / 5
    avg_precision_score = sum(precision_values) / 5
    avg_recall_score = sum(recall_values) / 5
    avg_f1_score = sum(f1_values) / 5
    avg_roc_score = sum(roc_values) / 5

    pred_values = model.predict(X)
    pd.DataFrame(m2).to_csv('True_Label.csv')
    pd.DataFrame(pred_values).to_csv('Pred_Label.csv')
    # STAMPO HEATMAP CONFUSION MATRIX
    matrix = confusion_matrix(m2, pred_values)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    pd.DataFrame(matrix).to_csv('PredictionResults/Training/DroneType/ConfusionMatrix.csv')
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=2.3)

    sns.heatmap(matrix, annot=True, annot_kws={'size': 20},
                cmap=plt.cm.Blues, linewidths=0.2)

    class_names = ['Mavic Pro', 'DJI Phantom 4 Pro', 'Mavic 2',
                   'Parrot']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.subplots_adjust(bottom=0.35, left=0.25)
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('')
    plt.savefig('PredictionResults/Training/DroneType/ConfusionMatrix.pdf')
    plt.show()

    # print(drone)

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    print('Precision of each fold - {}'.format(precision_values))
    print('Avg Precision : {}'.format(avg_precision_score))

    print('Recall of each fold - {}'.format(recall_values))
    print('Avg Recall : {}'.format(avg_recall_score))

    print('F1 of each fold - {}'.format(f1_values))
    print('Avg F1 : {}'.format(avg_f1_score))

    print('ROC AUC of each fold - {}'.format(roc_values))
    print('Avg ROC : {}'.format(avg_roc_score))


def predict_drone_type():
    data1 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_1.csv')
    data2 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_2.csv')
    data3 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_3.csv')
    data4 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_1_4.csv')
    data5 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_2_1.csv')
    data6 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_2_2.csv')
    data7 = pd.read_csv('MergeResults/Training/[3]_MERGE_scenario_3.csv')

    data = pd.concat([data1, data2, data3, data4, data5, data6, data7], axis=0)
    data = data.reset_index()
    data = data.drop('index', axis=1)
    data = data.drop('VenusTriggerLinkType_Uplink', axis=1)

    data = data[[
        'ArcusPotentialDronPlot_rcs',
        'DRONE', 'NO_DRONE', 'SUSPECTED_DRONE',
        'ArcusTracksTrackPosition_Altitude', 'ArcusTracksTrackVelocity_Azimuth',
        'ArcusTracksTrackVelocity_Elevation', 'ArcusTracksTrackVelocity_Speed',
        'ArcusTracksTrack_Reflection',
        'DianaSensorPosition_latitude_deg',
        'DianaSensorPosition_longitude_deg', 'DianaSensorPosition_altitude_m',
        'DianaTargets_band', 'DianaTarget_ID',
        'DianaTargetsTargetSignal_snr_dB',
        'DianaTargetsTargetSignal_bearing_deg',
        'DianaTargetsTargetSignal_range_m',
        'DianaTargetsTargetClassification_score', 'channels_x',
        'DianaTarget_Aircraft', 'DianaTarget_Controller', 'DianaTarget_None',
        'DianaClasssification_Unknown',
        'DianaClasssification_DJI-MAVIC-PRO-PLATINUM',
        'DianaClasssification_Wifi-Bluetooth',
        'DianaClasssification_DJI-MAVIC-2-PRO',
        'DianaClasssification_DJI-Phantom-4F',
        'DianaClasssification_Parrot-ANAFI',
        'DianaClasssification_DJI-Phantom-4E',
        'DianaClasssification_SPEKTRUM-DX5e', 'DianaClasssification_SYMA-X8HW',
        'DianaClasssification_DJI-MAVIC-AIR', 'DianaClasssification_None',
        'DianaClasssification_VISUO-Zen',
        'channels_y', 'VenusTriggerVenusName_isThreat',
        'VenusTrigger_RadioId',
        'VenusTrigger_Frequency', 'VenusTrigger_OnAirStartTime',
        'VenusTrigger_StopTime', 'VenusTrigger_Azimuth',
        'VenusTrigger_Deviation', 'DJI OcuSync', 'DJI Mavic Mini',
        'Cheerson Leopard 2', 'DJI Mavic Pro long',
        'DJI Mavic Pro short', 'Hubsan', 'Futaba FASST-7CH Var. 1',
        'AscTec Falcon 8 Downlink, DJI Mavic Mini',
        'DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G',
        'DJI Mavic Mini, MJX R/C Technic', 'Udi R/C 818A', 'MJX R/C Technic',
        'TT Robotix Ghost', 'Udi R/C',
        'DJI Mini, DJI Phantom 4 Pro/Mavic Pro, DJI Phantom 4/Mavic Pro',
        'DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G',
        'Spektrum DSMX downlink', 'Spektrum DSMX 12CH uplink', 'MJX X901',
        'DJI Mavic Pro long, DJI Phantom 4/Mavic Pro,DJI Phantom/Mavic Pro',
        'AscTec Falcon 8 Downlink', 'VenusName NaN', 'ISM 2.4 GHz',
        'ISM 5.8 GHz', 'FreqBand_Null', 'Dronetype'

    ]]
    data = data.fillna(value=0)
    data['Dronetype'] = data['Dronetype'].apply("int64")
    y = data['Dronetype']

    rus = RandomUnderSampler(random_state=42, replacement=True)  # fit predictor and target variable
    #rus = RandomOverSampler(sampling_strategy='minority')  # fit predictor and target variable

    x_rus, y_rus = rus.fit_resample(data, y)

    print('original dataset shape:', len(y))
    print('Resample dataset shape', len(y_rus))

    # x_rus = x_rus.drop('index', axis=1)
    x_rus.insert(x_rus.shape[1], 'Dronetype1', y_rus)

    data = x_rus

    data['MAVIC 2'] = 0
    data['MAVIC Pro'] = 0
    data['Phantom 4 Pro'] = 0
    data['Parrot'] = 0

    ''' Per stampare la confusion matix sostituisci gli 1 con 1,2,3,4 per permettere la stampa di tutte
        le classi nella heatmap di confusione   '''
    for i in range(0, data.shape[0]):
        if data.loc[i, 'Dronetype1'] == 13:
            data.loc[i, 'MAVIC Pro'] = 1

        elif data.loc[i, 'Dronetype1'] == 23:
            data.loc[i, 'Phantom 4 Pro'] = 1

        elif data.loc[i, 'Dronetype1'] == 27:
            data.loc[i, 'MAVIC 2'] = 1
        else:
            data.loc[i, 'Parrot'] = 1

    from sklearn.utils import shuffle

    data = shuffle(data)
    data = data.reset_index()
    data = data.drop('index', axis=1)

    y_mavic2 = data['MAVIC 2']
    y_mavicPro = data['MAVIC Pro']
    y_profV2 = data['Phantom 4 Pro']
    y_parrot = data['Parrot']

    freq = data['ArcusPotentialDronPlot_rcs']
    data = data[[

        'DRONE', 'NO_DRONE', 'SUSPECTED_DRONE',

        'ArcusTracksTrackPosition_Altitude', 'ArcusTracksTrackVelocity_Azimuth',
        'ArcusTracksTrackVelocity_Elevation', 'ArcusTracksTrackVelocity_Speed',
        'ArcusTracksTrack_Reflection',
        'DianaSensorPosition_latitude_deg',
        'DianaSensorPosition_longitude_deg', 'DianaSensorPosition_altitude_m',
        'DianaTargets_band', 'DianaTarget_ID',
        'DianaTargetsTargetSignal_snr_dB',
        'DianaTargetsTargetSignal_bearing_deg',
        'DianaTargetsTargetSignal_range_m',
        'DianaTargetsTargetClassification_score', 'channels_x',
        'DianaTarget_Aircraft', 'DianaTarget_Controller', 'DianaTarget_None',
        'DianaClasssification_Unknown',
        'DianaClasssification_DJI-MAVIC-PRO-PLATINUM',
        'DianaClasssification_Wifi-Bluetooth',
        'DianaClasssification_DJI-MAVIC-2-PRO',
        'DianaClasssification_DJI-Phantom-4F',
        'DianaClasssification_Parrot-ANAFI',
        'DianaClasssification_DJI-Phantom-4E',
        'DianaClasssification_SPEKTRUM-DX5e', 'DianaClasssification_SYMA-X8HW',
        'DianaClasssification_DJI-MAVIC-AIR', 'DianaClasssification_None',
        'DianaClasssification_VISUO-Zen',
        'channels_y', 'VenusTriggerVenusName_isThreat',
        'VenusTrigger_RadioId',
        'VenusTrigger_Frequency', 'VenusTrigger_OnAirStartTime',
        'VenusTrigger_StopTime', 'VenusTrigger_Azimuth',
        'VenusTrigger_Deviation', 'DJI OcuSync', 'DJI Mavic Mini',
        'Cheerson Leopard 2', 'DJI Mavic Pro long',
        'DJI Mavic Pro short', 'Hubsan', 'Futaba FASST-7CH Var. 1',
        'AscTec Falcon 8 Downlink, DJI Mavic Mini',
        'DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G',
        'DJI Mavic Mini, MJX R/C Technic', 'Udi R/C 818A', 'MJX R/C Technic',
        'TT Robotix Ghost', 'Udi R/C',
        'DJI Mini, DJI Phantom 4 Pro/Mavic Pro, DJI Phantom 4/Mavic Pro',
        'DJI Mavic Pro long, DJI Phantom 4 Pro+ V2.0 / Mavic Pro V2.0 2.4G',
        'Spektrum DSMX downlink', 'Spektrum DSMX 12CH uplink', 'MJX X901',
        'DJI Mavic Pro long, DJI Phantom 4/Mavic Pro,DJI Phantom/Mavic Pro',
        'AscTec Falcon 8 Downlink', 'VenusName NaN', 'ISM 2.4 GHz',
        'ISM 5.8 GHz', 'FreqBand_Null'

    ]]

    data.astype(np.float64)
    data = data.fillna(value=0)
    data = data.reset_index()

    scenario = data.drop('index', axis=1)

    from scipy import stats
    scenario = stats.zscore(scenario)
    scenario = scenario.fillna(value=0)

    '''sizes, training_scores, testing_scores = learning_curve(RandomForestClassifier(), scenario, y_mavic2, cv=10, scoring='accuracy',
                                                            train_sizes=np.linspace(0.01, 1.0, 50))

    # Mean and Standard Deviation of training scores
    mean_training = np.mean(training_scores, axis=1)
    Standard_Deviation_training = np.std(training_scores, axis=1)

    # Mean and Standard Deviation of testing scores
    mean_testing = np.mean(testing_scores, axis=1)
    Standard_Deviation_testing = np.std(testing_scores, axis=1)
    import matplotlib.pyplot as plt
    # dotted blue line is for training scores and green line is for cross-validation score
    plt.plot(sizes, mean_training, '--', color="b", label="Training score")
    plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

    # Drawing plot
    plt.title("LEARNING CURVE FOR KNN Classifier")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()'''

    scenario = np.c_[np.ones(scenario.shape[0]), scenario]

    ''' La funziona seguente è da usare per la Confusion matrix'''
    '''totale = pd.concat([y_mavic2, y_mavicPro, y_profV2, y_parrot], axis=1).sum(axis=1)
    regression_full(scenario, totale)'''

    print('-----------------------MAVIC 2-----------------------------')
    rsc1 = regression_drone(scenario, y_mavic2, 'Mavic2', freq)
    calc_roc_curve(scenario, y_mavic2, 'Mavic2')

    print('-----------------------MAVIC PRO-----------------------------')
    rsc2 = regression_drone(scenario, y_mavicPro, 'MavicPro', freq)
    #calc_roc_curve(scenario, y_mavicPro, 'MavicPro')

    print('-----------------------PROF V2-----------------------------')
    rsc3 = regression_drone(scenario, y_profV2, 'ProfessionalV2', freq)
    #calc_roc_curve(scenario, y_profV2, 'ProfessionalV2')

    print('-----------------------PARROT-----------------------------')
    print('-------------------------------------------------')
    rsc4 = regression_drone(scenario, y_parrot, 'Parrot', freq)
    #calc_roc_curve(scenario, y_parrot, 'Parrot')

    # PLOT BAR CHART#

    import scipy.stats as stats
    import matplotlib.pyplot as plt

    cdf = stats.binom.cdf

    '''fig, ax = plt.subplots()
    ########### INIZIO GRAFICO 1 ###########

    markerline, stemlines, baseline = ax.stem(rsc1, cdf(rsc1, 50, 0.2),markerfmt='o', label='Mavic2')
    markerline1, stemlines1, baseline1 = ax.stem(rsc2, cdf(rsc2, 50, 0.2), markerfmt='o',label='MavicPro')
    markerline2, stemlines2, baseline2 = ax.stem(rsc3, cdf(rsc3, 50, 0.2),markerfmt='o', label='ProfessionalV2')
    markerline3, stemlines3, baseline3 = ax.stem(rsc4, cdf(rsc4, 50, 0.2), markerfmt='o',label='Parrot')

    pd.DataFrame(rsc1).to_csv('CDF_RDS_Stem/x_Mavic2.csv')
    pd.DataFrame(cdf(rsc1, 50, 0.2)).to_csv('CDF_RDS_Stem/y_Mavic2.csv')

    pd.DataFrame(rsc2).to_csv('CDF_RDS_Stem/x_MavicPro.csv')
    pd.DataFrame(cdf(rsc2, 50, 0.2)).to_csv('CDF_RDS_Stem/y_MavicPro.csv')

    pd.DataFrame(rsc3).to_csv('CDF_RDS_Stem/x_ProfessionalV2.csv')
    pd.DataFrame(cdf(rsc3, 50, 0.2)).to_csv('CDF_RDS_Stem/y_ProfessionalV2.csv')

    pd.DataFrame(rsc4).to_csv('CDF_RDS_Stem/x_Parrot.csv')
    pd.DataFrame(cdf(rsc4, 50, 0.2)).to_csv('CDF_RDS_Stem/y_Parrot.csv')


    plt.setp(stemlines, 'color', plt.getp(markerline, 'color'),  color='b')
    plt.setp(stemlines, 'linestyle', 'solid')

    plt.setp(stemlines1, 'color', plt.getp(markerline1, 'color'),  color='r')
    plt.setp(stemlines1, 'linestyle', 'solid')

    plt.setp(stemlines2, 'color', plt.getp(markerline2, 'color'), color='g')
    plt.setp(stemlines2, 'linestyle', 'solid')

    plt.setp(stemlines3, 'color', plt.getp(markerline3, 'color'), color='y')
    plt.setp(stemlines3, 'linestyle', 'solid')
    plt.xlabel('RCS (dBsm)')
    plt.ylabel('CDF')
    plt.title('')
    plt.legend()
    plt.show()'''
    ########### FINE GRAFICO 1 ###########


    plt.scatter(rsc1, cdf(rsc1, 50, 0.2), color="b",label="Mavic2")
    plt.scatter(rsc2, cdf(rsc2, 50, 0.2), color="g",label="MavicPro")
    plt.scatter(rsc3, cdf(rsc3, 50, 0.2), color="r",label="ProfessionalV2")
    plt.scatter(rsc4, cdf(rsc4, 50, 0.2), color="y",label="Parrot")
    plt.xlabel('RCS (dBsm)')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

    rcs_full = pd.concat([rsc1,rsc2,rsc3,rsc4],axis=0)
    # getting data of the histogram
    count, bins_count = np.histogram(rcs_full, bins=10)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    pd.DataFrame(pdf).to_csv('PDF_result.csv')
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    pd.DataFrame(cdf).to_csv('CDF_result.csv')

    # plotting PDF and CDF
    plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.xlabel('RCS(dBsm)')
    plt.ylabel('Cumulative Density Function')
    plt.title('CDF of RCS (dBsm)')

    pd.DataFrame(bins_count[1:]).to_csv('CDF_RCS/x_CDF.csv')
    pd.DataFrame(cdf).to_csv('CDF_RCS/y_CDF.csv')

    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()


    ### FINE PLOT SOLO CDF ###
    '''
    rcs_full = pd.concat([rsc1,rsc2,rsc3,rsc4],axis=0)

    ### plot normal dist
    import scipy
    # test values for the bw_method option ('None' is the default value)
    bw_values = [None, 0.1, 0.01]

    # generate a list of kde estimators for each bw
    kde = [scipy.stats.gaussian_kde(rcs_full, bw_method=bw) for bw in bw_values]

    # plot (normalized) histogram of the data
    import matplotlib.pyplot as plt
    plt.hist(rcs_full, 50, density=1, stacked = True)#facecolor='green', alpha=0.5);
    plt.show()

    # plot density estimates
    t_range = np.linspace(-2, 2, 200)
    for i, bw in enumerate(bw_values):
        if i == 2:
            plt.plot(t_range, kde[i](t_range), lw=2, label='bw = ' + str(bw))
            plt.fill_between(t_range, kde[i](t_range), where=((t_range >= -0.25) & (t_range <= 0.25)), color='red')
            pd.DataFrame(t_range).to_csv('Curve_PDF_x.csv')
            pd.DataFrame(kde[i](t_range)).to_csv('Curve_PDF_y.csv')
    plt.xlabel('RCS(dBsm)')
    plt.ylabel('Probability Density Function')
    plt.title('Parrot')
    plt.xlim(-1, 1)
    plt.legend(loc='best')'''

def regression_drone(scenario, y, drone, freq):
    print('-->' + drone)
    kf = KFold(n_splits=5, random_state=42)
    acc_score = []
    prediction_values = []
    precision_values = []
    recall_values = []
    roc_values = []
    f1_values = []
    X_train_acc = []
    X_test_acc = []
    y_train_acc = []
    y_test_acc = []

    X = pd.DataFrame(scenario)
    m2 = np.array(y)
    model = RandomForestClassifier(max_depth=10, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = m2[train_index], m2[test_index]

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        prediction_values.append(pred_values)
        # acc = accuracy_score(pred_values, y_test)
        acc = model.score(X_test, y_test)

        roc = roc_auc_score(y_test, pred_values)
        acc = accuracy_score(y_test, pred_values)
        recall = recall_score(y_test, pred_values, average='weighted')
        f1 = f1_score(y_test, pred_values, average='weighted')
        precision = precision_score(y_test, pred_values, average='weighted')

        roc_values.append(roc)
        acc_score.append(acc)
        recall_values.append(recall)
        f1_values.append(f1)
        precision_values.append(precision)

    avg_acc_score = sum(acc_score) / 5
    avg_precision_score = sum(precision_values) / 5
    avg_recall_score = sum(recall_values) / 5
    avg_f1_score = sum(f1_values) / 5
    avg_roc_score = sum(roc_values) / 5

    preds = model.predict(X)
    total = pd.concat([pd.DataFrame(preds), pd.DataFrame(freq)], axis=1)
    rsc = total[total[0] != 0]['ArcusPotentialDronPlot_rcs']
    #total = total[total[0] != 0]
    #total = total[total['VenusTrigger_Frequency'] != 2440000000.0]
    #total = total[total['VenusTrigger_Frequency'] != 0]
    #total = total[total['VenusTrigger_Frequency'] != 2406500000.0]

    #total.to_csv('GRAFICI_WORD/Parrot.csv')
    total = total[total[0] != 0]
    print(drone)

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    print('Precision of each fold - {}'.format(precision_values))
    print('Avg Precision : {}'.format(avg_precision_score))

    print('Recall of each fold - {}'.format(recall_values))
    print('Avg Recall : {}'.format(avg_recall_score))

    print('F1 of each fold - {}'.format(f1_values))
    print('Avg F1 : {}'.format(avg_f1_score))

    print('ROC AUC of each fold - {}'.format(roc_values))
    print('Avg ROC : {}'.format(avg_roc_score))

    with open('Models/model_' + drone + '.pickle', 'wb') as f:
        import pickle
        pickle.dump(model, f)

    return rsc

def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

def calc_roc_curve(scenario, y, drone):
    print('-->' + drone)
    kf = KFold(n_splits=5, random_state=42)
    acc_score = []
    prediction_values = []
    precision_values = []
    recall_values = []
    roc_values = []
    f1_values = []
    X_train_acc = []
    X_test_acc = []
    y_train_acc = []
    y_test_acc = []

    X = pd.DataFrame(scenario)
    m2 = np.array(y)
    model = RandomForestClassifier(max_depth=10, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = m2[train_index], m2[test_index]

        model.fit(X_train, y_train)

        lr_probs = model.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calcolo cut-off
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(y_test, lr_probs)
        cut_off = cutoff_youdens_j(fpr, tpr, thresholds)
        best_tpr = tpr[1:][::-1]
        best_fpr = fpr[1:][::-1]
        df_test = best_tpr-best_fpr
        cut_off_1 = pd.DataFrame(df_test).sort_values(
            by=0, ascending=False, ignore_index=True).iloc[0]
        print(cut_off_1)
        print('------------------------')
        # calculate scores
        ns_probs = [0 for _ in range(len(y_test))]
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        pd.DataFrame(lr_tpr).to_csv('PredictionResults/Training/DroneType/ROC_curve_Data_TPR_' + drone + '.csv')
        pd.DataFrame(lr_fpr).to_csv('PredictionResults/Training/DroneType/ROC_curve_Data_FPR_' + drone + '.csv')
        # plot the roc curve for the model
        pyplot.rcParams.update({'font.size': 22})
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.savefig('PredictionResults/Training/DroneType/ROC_curve_' + drone + '.pdf')
        pyplot.show()

        '''prediction_values.append(pred_values)
        # acc = accuracy_score(pred_values, y_test)
        acc = model.score(X_test, y_test)

        roc = roc_auc_score(y_test, pred_values)
        acc = accuracy_score(y_test, pred_values)
        recall = recall_score(y_test, pred_values, average='weighted')
        f1 = f1_score(y_test, pred_values, average='weighted')
        precision = precision_score(y_test, pred_values, average='weighted')'''

        if len(acc_score) == 0:
            X_train_acc = X_train
            X_test_acc = X_test
            y_train_acc = y_train
        '''elif acc > max(acc_score):
            X_train_acc = X_train
            y_train_acc = y_train
            y_test_acc = y_test'''

        '''roc_values.append(roc)
        acc_score.append(acc)
        recall_values.append(recall)
        f1_values.append(f1)
        precision_values.append(precision)'''

    avg_acc_score = sum(acc_score) / 5
    avg_precision_score = sum(precision_values) / 5
    avg_recall_score = sum(recall_values) / 5
    avg_f1_score = sum(f1_values) / 5
    avg_roc_score = sum(roc_values) / 5

    print(drone)

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    print('Precision of each fold - {}'.format(precision_values))
    print('Avg Precision : {}'.format(avg_precision_score))

    print('Recall of each fold - {}'.format(recall_values))
    print('Avg Recall : {}'.format(avg_recall_score))

    print('F1 of each fold - {}'.format(f1_values))
    print('Avg F1 : {}'.format(avg_f1_score))

    print('ROC AUC of each fold - {}'.format(roc_values))
    print('Avg ROC : {}'.format(avg_roc_score))
