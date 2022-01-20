# %%
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, \
    DBSCAN, Birch
from sklearn.metrics import silhouette_score

from P5_00_fonctions import visuPCA, searchClusters, graphScores, graphClusters, \
    pieNbCustClust, graphClustersRFMS
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
# %% [markdown]
### Comparaison des algorithmes sur les données RFM
# %%
# échantillon de 10000 client ce qui fait un peu plus de 10% de la population totale
DataRFM = pd.read_csv('OlistDataRFM.csv').sample(n=10000, random_state=0)
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

# %% [markdown]
# Le scree plot nous montre que les trois composantes expliquent toutes la même
# part (1/3)
# %%
# création des graphiques
for a1, a2 in [[0, 1], [0, 2], [1, 2]]:
    fig = visuPCA(DataRFM, pca, components, loadings, [(a1, a2)], color=None)
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/PCARFMF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=500,
                        height=500)

# %% [markdown]
# Les trois variables sont nécessaire à expliquer ce jeux de données
# %%
# graph PCA 3D
fig = px.scatter_3d(components,
                    x=0,
                    y=1,
                    z=2,
                    labels={
                        '0': 'F1',
                        '1': 'F2',
                        '2': 'F3'
                    })
fig.update_layout(title='Visualisation 3D de la PCA')
fig.show(renderer='notebook')

# %% [markdown]
# On retrouve des séparation en fonction du nombre de commandes surtout
# %% [markdown]
#### KMeans
# %%
KMeansClusters = searchClusters(KMeans, ScaledData, {'random_state': 50},
                                'n_clusters', [*range(3, 10)])
fig = graphScores(KMeansClusters)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresKMeans.pdf', width=500)
# %% [markdown]
# Le meilleurs résultat est pour 4 clusters
# %%
best_KMeans = KMeansClusters.sort_values(by='metascore',
                                         ascending=False).iloc[0]
fig = graphClusters(
    'KMeans', DataRFM, best_KMeans.labels,
    pd.DataFrame(Data_fit.inverse_transform(best_KMeans.clusters_centers),
                 columns=DataRFM.columns))
fig.update_layout(scene_camera=dict(eye=dict(x=-1.9, y=1, z=1)))
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/VisuKMeansClusters.pdf', width=760, height=700)
# %% [markdown]
# Les 4 clusters sont clairement interprétables
# - 0 : clients qui ont fait au moins 2 commandes
# - 1 : clients qui ont fait 1 commande d'une valeur moyenne < 500 et plutôt recemment
# (environ moins de 300j)
# - 2 : clients qui ont fait 1 commande il y a plus longtemps (> 300j) et d'une valeur
# plus faible (< 5000)
# - 3 : clients qui ont fait 1 commande de plutôt grande valeur (> 500)
# %%
# diagramme circulaire
fig = pieNbCustClust('KMeans', best_KMeans.labels)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/pieKMeans.pdf', width=500)
# %% [markdown]
#### SpectralClustering
# %%
SpectralClusters = searchClusters(SpectralClustering, ScaledData, {
    'n_jobs': -1,
    'random_state': 50,
    'affinity': 'nearest_neighbors'
}, 'n_clusters', [*range(3, 10)])
fig = graphScores(SpectralClusters)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresSpectral.pdf', width=500)
# %% [markdown]
# Le meilleur résultat est pour 4 clusters
# %%
best_Spectral = SpectralClusters.sort_values(by='metascore',
                                             ascending=False).iloc[0]
fig = graphClusters('SpectralClustering', DataRFM, best_Spectral.labels)
fig.update_layout(scene_camera=dict(eye=dict(x=-1.9, y=1, z=1)))
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/VisuSpectralClusters.pdf',
                    width=760,
                    height=700)
# %% [markdown]
# Les 4 clusters sont clairement interprétables
# - 0 : clients qui ont fait 1 commande il y a plus longtemps ou plutôt recemment d'une
# d'une valeur > 500
# - 1 : clients qui ont fait 3 commandes
# - 2 : clients qui ont fait 1 commande plus récemment (< 250j) et d'une valeur moins
# importante (< 500)
# - 3 : clients qui ont fait 2 commandes
# %%
# diagramme circulaire
fig = pieNbCustClust('SpectralClustering', best_Spectral.labels)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/pieSpectral.pdf', width=500)

# %% [markdown]
#### AgglomerativeClustering
# %%
AggloClusters = searchClusters(AgglomerativeClustering, ScaledData, {},
                               'n_clusters', range(3, 10))
fig = graphScores(AggloClusters)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresAgglo.pdf', width=500, height=530)
# %% [markdown]
# Le meilleurs résultat est pour 5 clusters
# %%
best_Agglo = AggloClusters.sort_values(by='metascore', ascending=False).iloc[0]
fig = graphClusters('AgglomerativeClusters', DataRFM, best_Agglo.labels)
fig.update_layout(scene_camera=dict(eye=dict(x=-1.9, y=1, z=1)))
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/VisuAggloClusters.pdf', width=760, height=700)
# %% [markdown]
# Les 5 clusters sont plutôt bien définis
# - 0 : clients qui ont fait 1 commande plutôt recemment (< 300j)
# d'une valeur moyenne < 300
# - 1 : clients qui ont fait 1 commande d'une valeur moyenne > 200 mais < 1400
# - 2 : clients qui ont fait 1 commande il y a plus longtemps (> 300j)
# - 3 : clients qui ont fait 2 ou 3 commandes
# - 4 : clients qui ont fait 1 commande d'une valeur importante (> 1400)
# %%
# diagramme circulaire
fig = pieNbCustClust('AgglomerativeClustering', best_Agglo.labels)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/pieAgglo.pdf', width=500)
# %% [markdown]
#### DBSCAN
# %%
DBSCANClusters = searchClusters(
    DBSCAN, ScaledData, {'n_jobs': -1}, 'eps',
    [round(e, 2) for e in np.linspace(.2, 1.9, 10)])
fig = graphScores(DBSCANClusters)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresDBSCAN.pdf', width=500)
# %%
best_DBSCAN = DBSCANClusters.sort_values(by='metascore',
                                         ascending=False).iloc[0]
fig = graphClusters('DBSCAN', DataRFM, best_DBSCAN.labels)
fig.update_layout(scene_camera=dict(eye=dict(x=-1.9, y=1, z=1)))
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/VisuDBSCANClusters.pdf', width=760, height=700)
# %% [markdown]
# Cet algorithme cherche lui même le nombre de cluster.
# Il en a créé 3 et éliminé quelques points. Les trois clusters correspondent
# au nombre de commandes effectuées
# - 0 : clients qui ont fait 1 commande
# - 1 : clients qui ont fait 2 commandes
# - 2 : clients qui ont fait 3 commandes
# %%
# diagramme circulaire
fig = pieNbCustClust('DBSCAN', best_DBSCAN.labels)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/pieDBSCAN.pdf', width=500)
# %% [markdown]
# Les séparation sont très simples en fonction du nombre de commandes effectuées
# %% [markdown]
#### Birch
# %%
BirchClusters = searchClusters(Birch, ScaledData, {}, 'n_clusters',
                               [*range(3, 10)])
fig = graphScores(BirchClusters)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresBirch.pdf', width=500)
# %%
best_Birch = BirchClusters.sort_values(by='metascore', ascending=False).iloc[0]
fig = graphClusters('Birch', DataRFM, best_Birch.labels)
fig.update_layout(scene_camera=dict(eye=dict(x=-1.9, y=1, z=1)))
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/VisuBirchClusters.pdf', width=760, height=700)
# %% [markdown]
# Nous avons 3 clusters:
# - 0 : clients ayant fait au moins 2 commandes
# - 1 : clients ayant fait 1 commande d'une valeur > 1000
# - 2 : clients ayant fait 1 commande d'une valeur < 1000
# %%
# diagramme circulaire
fig = pieNbCustClust('Birch', best_Birch.labels)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/pieBirch.pdf', width=500)

# %%
# aggrégation des scores pour les différents algorithmes
CompareScores = pd.DataFrame(
    columns=['model', 'silhouette', 'calinski_harabasz', 'davies_bouldin'])
for best in [best_KMeans, best_Spectral, best_Agglo, best_DBSCAN, best_Birch]:
    CompareScores = CompareScores.merge(
        pd.DataFrame([{
            'model': best.model,
            'silhouette': best.silhouette_score,
            'calinski_harabasz': best.calinski_harabasz_score,
            'davies_bouldin': best.davies_bouldin_score
        }]),
        on=['model', 'silhouette', 'calinski_harabasz', 'davies_bouldin'],
        how='outer')
CompareScores.set_index('model', inplace=True)
# %%
# visualisation comparaison des scores 
fig = make_subplots(len(CompareScores.columns),
                    1,
                    row_titles=(CompareScores.columns.to_list()),
                    shared_xaxes=True)
for r, c in enumerate(CompareScores):
    fig.add_trace(go.Bar(x=CompareScores.index, y=CompareScores[c]),
                  row=r + 1,
                  col=1)
fig.update_layout(
    title_text="Comparaison des scores des modèles de classification",
    showlegend=False,
    height=700)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/CompareScores.pdf', height=700)
# %% [markdown]
### Classification avec données de satisfaction
#
# Nous allons à présent essayer d'ajouter les données de notation pour voir
# quel est l'influence de cette donnée sur notre classification
# %%
DataRFMS = pd.read_csv('OlistDataRFMS.csv').sample(n=5000, random_state=0)
# %%
Data_fit = StandardScaler().fit(DataRFMS)
ScaledData = Data_fit.transform(DataRFMS)
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
    fig.write_image('./Figures/ScreePlotRFMS.pdf', height=300)

# %% [markdown]
# Le scree plot nous montre que les quatres composantes expliquent toutes la même
# part (1/4). Les quatres variables sont nécessaires à expliquer ce jeux de données
# %%
# création des graphiques
for a1, a2 in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
    fig = visuPCA(DataRFMS, pca, components, loadings, [(a1, a2)], color=None)
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/PCARFMSF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=500,
                        height=500)

# %%
# graph PCA 3D
fig = px.scatter_3d(components,
                    x=0,
                    y=1,
                    z=2,
                    labels={
                        '0': 'F1',
                        '1': 'F2',
                        '2': 'F3'
                    })
fig.update_layout(title='Visualisation 3D de la PCA')
fig.show(renderer='notebook')

# %% [markdown]
# On retrouve des séparation en fonction du nombre de commande surtout
# %%
# KMeans
KMeansClusters = searchClusters(KMeans, ScaledData, {'random_state': 50},
                                'n_clusters', [*range(3, 10)])
fig = graphScores(KMeansClusters)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/ScoresKMeansRFMS.pdf', width=500)
# %% [markdown]
# Le meilleurs résultat est pour 6 clusters
# %%
best_KMeans = KMeansClusters.sort_values(by='metascore',
                                         ascending=False).iloc[0]
# %% [markdown]
# Les sont bien visualisable sur la PCA mais il est difficile d'expliquer ce qu'ils
# représentent
# %%
# graph PCA 3D KMeans labels colors
fig = px.scatter_3d(components,
                    x=0,
                    y=1,
                    z=2,
                    color=best_KMeans.labels.astype(str),
                    opacity=1,
                    labels={
                        '0': 'F1',
                        '1': 'F2',
                        '2': 'F3'
                    })
fig.update_traces(marker_size=3)
fig.update_layout(title='Visualisation 3D de la PCA',
                  legend={'itemsizing': 'constant'})
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/pcaKMeansRFMScolor.pdf')
# %%
fig = graphClustersRFMS(
    'KMeans', DataRFMS, best_KMeans.labels,
    pd.DataFrame(Data_fit.inverse_transform(best_KMeans.clusters_centers),
                 columns=DataRFMS.columns))
fig.update_layout(scene_camera=dict(eye=dict(x=-1.9, y=1, z=1)))
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/VisuKMeansClustersRFMS.pdf',
                    width=760,
                    height=700)

# %% [markdown]
# La visualisation sur les données brutes est plus difficile néanmoins on peut
# s'y essayer.
#
# Les 6 clusters sont
# - 0 : clients qui ont fait 1 commande récente (< 300j) d'une valeur moyenne < 400 
# et plutôt satisfaits (note entre 3 et 5)
# - 1 : clients qui ont fait 2 ou 3 commandes
# - 2 : clients qui ont fait 1 commande d'une valeur < 600 et peu satisfaits
# (note entre 1 et 3)
# - 3 : clients qui ont fait 1 commande il y a plus longtemps (> 300j) et d'une valeur
# plus faible (< 250)
# - 4 : clients qui ont fait 1 commande de grande valeur (> 1200)
# - 5 : clients qui ont fait 1 commande il y a plus longtemps (> 300j) d'une valeur 
# moyenne < 400 et plutôt satisfaits (note entre 3 et 5)

# %%
# diagramme circulaire
fig = pieNbCustClust('KMeans', best_KMeans.labels)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/pieKMeansRFMS.pdf', width=500)
# %%