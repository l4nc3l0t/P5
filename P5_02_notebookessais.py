# %%
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, \
    DBSCAN, OPTICS, Birch
from sklearn.metrics import silhouette_score

from P5_00_fonctions import visuPCA, searchClusters, graphScores, graphClusters
# %%
write_data = True

# True : création d'un dossier Figures et Tableau
# dans lesquels seront créés les éléments qui serviront à la présentation
# et écriture des figures et tableaux dans ces dossier
#
# False : pas de création de dossier ni de figures ni de tableaux

if write_data is True:
    try:
        os.mkdir("./Figures/")
    except OSError as error:
        print(error)
    try:
        os.mkdir("./Tableaux/")
    except OSError as error:
        print(error)
else:
    print("""Visualisation uniquement dans le notebook
    pas de création de figures ni de tableaux""")

# %%
DataRFM = pd.read_csv('OlistDataRFM.csv')
# %%
Data_fit = StandardScaler().fit(DataRFM)
ScaledData = Data_fit.transform(DataRFM)
# %%
# ACP
pca = PCA().fit(ScaledData)
components = pca.transform(ScaledData)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# %%
# visualisation de la variance expliquée de chaque composante (cumulée)
exp_var_cum = np.cumsum(pca.explained_variance_ratio_)
fig = px.area(x=range(1, exp_var_cum.shape[0] + 1),
              y=exp_var_cum,
              labels={
                  'x': 'Composantes',
                  'y': 'Variance expliquée cumulée'
              })
fig.update_layout(title='Scree plot')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScreePlotRFM.pdf', height=300)

# %%
# graph PCA 3D
fig = px.scatter_3d(components, x=0, y=1, z=2)
fig.show(renderer='notebook')
# %%
# création des graphiques
for a1, a2 in [[0, 1], [0, 2], [1, 2]]:
    fig = visuPCA(DataRFM, pca, components, loadings, [(a1, a2)], color=None)
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/PCARFMF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=500,
                        height=500)
"""
# %%
# MDS
mds3 = MDS(n_components=3, n_jobs=-1)
ScaledData_transformed3 = mds3.fit(ScaledData).embedding_
# %%
fig = px.scatter_3d(ScaledData_transformed3, x=0, y=1, z=2)
fig.show(renderer='notebook')
"""
# %%
# TSNE
tsne = TSNE(n_components=2,
            perplexity=50,
            init='pca',
            learning_rate='auto',
            n_jobs=-1)
ScaledData_TSNEfit = tsne.fit_transform(ScaledData)

fig = px.scatter(ScaledData_TSNEfit, x=0, y=1)
fig.show(renderer='notebook')

# %%
# KMeans
KMeansClusters = searchClusters(KMeans, ScaledData, {}, 'n_clusters',
                                [*range(2, 13)])
fig = graphScores(KMeansClusters)
fig.show(renderer='notebook')

# %%
best_KMeans = KMeansClusters.sort_values(by='silhouette_score',
                                         ascending=False).iloc[0]
fig = graphClusters(
    'KMeans', DataRFM, best_KMeans.labels,
    pd.DataFrame(Data_fit.inverse_transform(best_KMeans.clusters_centers),
                 columns=DataRFM.columns))
fig.show(renderer='notebook')
# %%
# SpectralClustering
SpectralClusters = searchClusters(SpectralClustering, ScaledData, {
    'n_jobs': -1,
    'affinity': 'nearest_neighbors'
}, 'n_clusters', [*range(2, 13)])
fig = graphScores(SpectralClusters)
fig.show(renderer='notebook')

# %%
best_Spectral = SpectralClusters.sort_values(by='silhouette_score',
                                             ascending=False).iloc[0]
fig = graphClusters('SpectralClustering', DataRFM, best_Spectral.labels)
fig.show(renderer='notebook')

# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_Spectral.labels.astype(str))
fig.show(renderer='notebook')

# %%
# AgglomerativeClustering
AggloClusters = searchClusters(AgglomerativeClustering, ScaledData, {},
                               'n_clusters', range(2, 13))
fig = graphScores(AggloClusters)
fig.show(renderer='notebook')

# %%
best_Agglo = AggloClusters.sort_values(by='silhouette_score',
                                       ascending=False).iloc[0]
fig = graphClusters('AgglomerativeClusters', DataRFM, best_Agglo.labels)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_Agglo.labels.astype(str))
fig.show(renderer='notebook')
# %%
# DBSCAN
DBSCANClusters = searchClusters(
    DBSCAN, ScaledData, {'n_jobs': -1}, 'eps',
    [round(e, 2) for e in np.linspace(.2, 1.9, 10)])
fig = graphScores(DBSCANClusters)
fig.show(renderer='notebook')

# %%
best_DBSCAN = DBSCANClusters.sort_values(by='silhouette_score',
                                         ascending=False).iloc[0]
fig = graphClusters('DBSCAN', DataRFM, best_DBSCAN.labels)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_DBSCAN.labels.astype(str))
fig.show(renderer='notebook')
# %%
# Birch
BirchClusters = searchClusters(Birch, ScaledData, {}, 'n_clusters',
                               [*range(2, 13)])
fig = graphScores(BirchClusters)
fig.show(renderer='notebook')

# %%
best_Birch = BirchClusters.sort_values(by='silhouette_score',
                                       ascending=False).iloc[0]
fig = graphClusters('Birch', DataRFM, best_Birch.labels)
fig.show(renderer='notebook')

# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_Birch.labels.astype(str))
fig.show(renderer='notebook')

# %%
