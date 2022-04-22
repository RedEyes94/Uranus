from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import pandas as pd

from Utility.drone_label_clustering import definisci_label_drone

'''def DBSCAN(self, scaled_dataframe, X, eps, n_sample, features_to_plot):
    results_noise, results_clusters, X = start_dbscan(scaled_dataframe, X, eps, n_sample, metrics=True)
    show_dbscan_results(results_noise, results_clusters, X, features_to_plot)
    app = metrics.silhouette_score(X, X['LABEL'])
    print(app)'''


def GMM(X, features_to_plot):
    GMM = GaussianMixture(n_components=4, random_state=0).fit(X)
    labels = GMM.predict(X[:])
    X["LABEL"] = labels
    features_to_plot.append('LABEL')
    plt.figure(figsize=(9, 7))
    sns.pairplot(X[features_to_plot], hue="LABEL")
    plt.show()
    app = metrics.silhouette_score(X, X['LABEL'])
    print(app)
    X.data = X.data.drop('LABEL', axis=1)


def agglomerative_clustering(X, scenario):
    if scenario == '1_1': n = 2
    if scenario == '1_2': n = 2
    if scenario == '1_3': n = 3
    if scenario == '1_4': n = 3
    if scenario == '2_1': n = 4
    if scenario == '2_2': n = 4
    if scenario == '3': n = 2
    if 'datetime(utc)' in X.data:
        X = X.data.drop('datetime(utc)', axis=1)
    data_scaled = normalize(X.data)
    data_scaled = pd.DataFrame(data_scaled, columns=X.data.columns)

    # cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='average')  # per arcus da 0.547
    labels = cluster.fit_predict(data_scaled)
    app = X.data
    app.insert(X.data.shape[1], 'LABEL', labels)
    sil = metrics.silhouette_score(X.data, X.data['LABEL'])
    print('Silhouette coef. per scenario ' + X.scenario + ' : ' + str(sil))

    # plt.figure(figsize=(10, 7))
    # plt.title("Dendrograms")
    # plt.axhline(y=5, color='r', linestyle='--')
    # dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    # plt.show()
    X.data = X.data.drop('LABEL', axis=1)


def k_means(X, k):
    sensore = X.name
    scenario = X.scenario
    restore = X.data

    if 'datetime(utc)' in X.data:
        X = X.data.drop('datetime(utc)', axis=1)

    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=15, random_state=0)
    labels = kmeans.fit_predict(X)
    app = X
    app.insert(app.shape[1], 'LABEL', labels)
    sil = metrics.silhouette_score(X, X['LABEL'])
    app['datetime(utc)'] = restore['datetime(utc)']

    print('Silhouette coef. per scenario ' + scenario + ' : ' + str(sil))

    app = definisci_label_drone(scenario, app)

    app.to_csv('2-AlviraClustering/[2]_' + sensore + '_data_predict_' + scenario + '.csv', float_format='%.6f')


def GMM(X, scenario, features_to_plot=False):
    GMM = GaussianMixture(n_components=3, random_state=0).fit(X.data)
    labels = GMM.predict(X.data[:])
    app = X.data
    app.insert(X.data.shape[1], 'LABEL', labels)
    # features_to_plot.append('LABEL')
    # plt.figure(figsize=(9, 7))
    # sns.pairplot(X[features_to_plot], hue="LABEL")
    # plt.show()
    sil = metrics.silhouette_score(X.data, X.data['LABEL'])
    print('Silhouette coef. per scenario ' + scenario + ' : ' + str(sil))
    X.data = X.data.drop('LABEL', axis=1)
