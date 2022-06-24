import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from Sensors.Arcus import Arcus
from Utility.GPS import GPSVis
from Utility.drone_label_clustering import definisci_label_drone
from Sensors.Alvira import Alvira
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc


def k_means(X, k):
    scenari_test = ['1a_test', '1b_test', '2a_test', '2b_test', '2c_test', '2d_test', '3a_test']
    sensore = X.name
    scenario = X.scenario
    restore = X.data
    X_ = X.data
    if 'datetime(utc)' in X.data:
        X_ = X.data.drop('datetime(utc)', axis=1)

    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=15, random_state=0)
    labels = kmeans.fit_predict(X_)
    app = X_
    app.insert(app.shape[1], 'LABEL', labels)
    sil = metrics.silhouette_score(X_, X_['LABEL'])
    app['datetime(utc)'] = restore['datetime(utc)']

    print('Silhouette coef. per scenario ' + scenario + ' : ' + str(sil))

    app = definisci_label_drone(scenario, app, sensore)

    if scenario not in scenari_test:
        '''app.to_csv('2-' + sensore + 'Clustering/[2]_' + sensore + '_data_predict_' + scenario + '.csv',
                   float_format='%.6f')'''

        app.to_csv('AlviraClusteringResults/Training/Alvira_data_predict_' + scenario + '.csv',
                   float_format='%.6f')
    else:
        '''app.to_csv('2-AlviraClustering/[2-TEST]_' + sensore + '_data_predict_' + scenario + '.csv',
                   float_format='%.6f')'''
        app.to_csv('AlviraClusteringResults/Test/Alvira_data_predict_' + scenario + '.csv',
                   float_format='%.6f')


def agglomerative_clustering(X, scenario):
    if scenario == '1_1': n = 2
    if scenario == '1_2': n = 2
    if scenario == '1_3': n = 3
    if scenario == '1_4': n = 3
    if scenario == '2_1': n = 2
    if scenario == '2_2': n = 2
    if scenario == '3': n = 2
    if 'datetime(utc)' in X.data:
        X.data = X.data.drop('datetime(utc)', axis=1)
    data_scaled = normalize(X.data)
    data_scaled = pd.DataFrame(data_scaled, columns=X.data.columns)

    # cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='average')  # per arcus da 0.547
    labels = cluster.fit_predict(data_scaled)
    app = X.data
    app.insert(X.data.shape[1], 'LABEL', labels)
    sil = metrics.silhouette_score(X.data, X.data['LABEL'])
    print('Silhouette coef. per scenario ' + X.scenario + ' : ' + str(sil))

    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    #plt.axhline(y=5, color='r', linestyle='--')
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.show()
    X.data = X.data.drop('LABEL', axis=1)


def GMM(X, features_to_plot):
    app = X
    if 'datetime(utc)' in X:
        X = X.drop('datetime(utc)', axis=1)
    GMM = GaussianMixture(n_components=3, random_state=0).fit(X)
    labels = GMM.predict(X[:])
    X["LABEL"] = labels
    features_to_plot.append('LABEL')
    '''plt.figure(figsize=(9, 7))
    sns.pairplot(X[features_to_plot], hue="LABEL")
    plt.show()'''
    app = metrics.silhouette_score(X, X['LABEL'])
    print(app)
    X = X.drop('LABEL', axis=1)


def calc_k_value(points, kmax, scenario):
    if 'datetime(utc)' in points:
        points = points.drop('datetime(utc)', axis=1)
    cost = []
    for i in range(1, 5):
        KM = KMeans(n_clusters=i, max_iter=1500)
        KM.fit(points)

        cost.append(KM.inertia_)

    plt.plot(range(1, 5), cost, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Squared Error (Cost)")
    plt.savefig('AlviraClusteringResults/ElbowMethodResults/Elbow_results_'+ scenario + '.pdf')
    plt.show()  # clear the plot
    return 0


def clustering_alvira(scenari, calc_k, k, clustering_alg='Kmeans'):
    for scenario in scenari:
        data = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_ALVIRA_data_cleaned_' + scenario + '.csv',
                           delimiter=',')
        alvira = Alvira(data, scenario)
        alvira.data = alvira.data[
            ['datetime(utc)', 'AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude',
             'AlviraTracksTrackPosition_Altitude', 'AlviraTracksTrackVelocity_Azimuth',
             'AlviraTracksTrackVelocity_Elevation', 'AlviraTracksTrackVelocity_Speed', 'DRONE', 'NO_DRONE',
             'SUSPECTED_DRONE',
             'AlviraTracksTrack_Score', 'AlviraTracksTrack_Reflection']]

        if calc_k:
            calc_k_value(alvira.data, k, alvira.scenario)

        if scenario[:1] == '1' or scenario[:1] == '3':
            k = 2
        else:
            k = 3

        if clustering_alg == 'kmeans':
            print('K-means')
            k_means(alvira, k)
            data = pd.read_csv('AlviraClusteringResults/Training/Alvira_data_predict_' + scenario + '.csv',
                               delimiter=',')

            '''Questa funzione stampa su mappa i risultati ottenuti con il K-Means'''

            print_on_map(data, scenario, 'Alvira')
        if clustering_alg == 'GMM':
            print('GMM scenario : ' + scenario)
            GMM(alvira.data, ['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude'])

        if clustering_alg == 'Agglomerative':
            print('Aglomerative scenario : ' + scenario)
            agglomerative_clustering(alvira, scenario)


def clustering_arcus(scenari):
    for scenario in scenari:
        data = pd.read_csv('DataCleanResults/Scenario_' + scenario + '/[1]_ARCUS_data_cleaned_' + scenario + '.csv',
                           delimiter=',')
        arcus = Arcus(data, scenario)
        arcus.data = arcus.data[
            ['datetime(utc)', 'ArcusTracksTrackPosition_Latitude', 'ArcusTracksTrackPosition_Longitude',
             'ArcusTracksTrackPosition_Latitude',
             'ArcusSystemStatusSensorPosition_Latitude',
             'ArcusSystemStatusSensorPosition_Longitude',
             'ArcusSystemStatusSensorPosition_Altitude',
             'ArcusSystemStatusSensorStatusOrientetation_Azimuth',
             'ArcusSystemStatusSensorStatusOrientetation_Elevation',
             'ArcusSystemStatusSensorStatusBlankSector_Angle',
             'ArcusSystemStatusSensorStatusBlankSector_Span',
             'ArcusSystemStatusSensorStatusProcessing_Sensitivity',
             'ArcusPotentialDronPlot_id',
             'ArcusPotentialDronPlot_rcs',
             'ArcusPotentiaDronPlotPlotPosition_altitude',
             'ArcusPotentialDronPlotsPlotPosition_latitude',
             'ArcusPotentialDronPlotsPlotPosition_longitude'

             ]]

        print('K-means')
        k_means(arcus, 3)
        data = pd.read_csv('2-ArcusClustering/[2]_ARCUS_data_predict_' + scenario + '.csv',
                           delimiter=',')
        '''print('GMM scenario : ' + scenario)
        GMM(alvira.data, ['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude'])

        print('Aglomerative scenario : ' + scenario)
        agglomerative_clustering(alvira, scenario)'''

        '''
        Questa stampa su mappa
        '''
        print_on_map(data, scenario, 'Arcus')


def print_on_map(data, scenario, sensore):
    scenari_test = ['1a_test', '1b_test', '2a_test', '2b_test', '2c_test', '2d_test', '3a_test']

    if sensore == 'Alvira':
        data = data[['AlviraTracksTrackPosition_Latitude', 'AlviraTracksTrackPosition_Longitude']]

        data.astype({"AlviraTracksTrackPosition_Latitude": float})
        data.astype({"AlviraTracksTrackPosition_Longitude": float})

        data['AlviraTracksTrackPosition_Latitude'].map('{:,.6f}'.format)
        data['AlviraTracksTrackPosition_Longitude'].map('{:,.6f}'.format)
    else:
        data = data[['ArcusTracksTrackPosition_Latitude', 'ArcusTracksTrackPosition_Longitude']]

        '''data.astype({"ArcusTracksTrackPosition_Latitude": float})
        data.astype({"ArcusTracksTrackPosition_Longitude": float})'''

        data['ArcusTracksTrackPosition_Latitude'].map('{:,.6f}'.format)
        data['ArcusTracksTrackPosition_Longitude'].map('{:,.6f}'.format)

    if scenario not in scenari_test:
        data.to_csv('AlviraClusteringResults/Training/LatitudeLongitudeValues/AlviraClustering_LatLong_'+scenario+'.csv',
                    header=False, index=False)

        vis = GPSVis(data_path='AlviraClusteringResults/Training/LatitudeLongitudeValues/AlviraClustering_LatLong_'+scenario+'.csv',
                     map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
                     points=(51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

        vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
        vis.plot_map(output='save', save_as='AlviraClusteringResults/Training/MapResults/AlviraClustering_Map_'+scenario)
    else:
        data.to_csv('AlviraClusteringResults/Test/LatitudeLongitudeValues/AlviraClustering_LatLong_' + scenario + '.csv',
                    header=False, index=False)

        vis = GPSVis(
            data_path='AlviraClusteringResults/Test/LatitudeLongitudeValues/AlviraClustering_LatLong_' + scenario + '.csv',
            map_path='Utility/map1.png',  # Path to map downloaded from the OSM.
            points=(51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

        vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
        vis.plot_map(output='save',
                     save_as='AlviraClusteringResults/Test/MapResults/AlviraClustering_Map_' + scenario)
