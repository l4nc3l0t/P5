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

from P5_00_fonctions import visuPCA, searchClusters, graphScores, graphClusters, \
    pieNbCustClust
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
# On retrouve des séparation en fonction du nombre d'achat surtout

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

# %% [markdown]
# Ici les séparations sont plus difficiles à expliquer nous visualiseront
# les clusters créés par chaque fonction sur ce graph pour comprendre comment
# il fonctionne
# %%
# KMeans
KMeansClusters = searchClusters(KMeans, ScaledData, {'random_state': 50},
                                'n_clusters', [*range(2, 13)])
fig = graphScores(KMeansClusters)
fig.show(renderer='notebook')
# %% [markdown]
# Le meilleurs résultat est pour 5 clusters
# %%
best_KMeans = KMeansClusters.sort_values(by='silhouette_score',
                                         ascending=False).iloc[0]
fig = graphClusters(
    'KMeans', DataRFM, best_KMeans.labels,
    pd.DataFrame(Data_fit.inverse_transform(best_KMeans.clusters_centers),
                 columns=DataRFM.columns))
fig.show(renderer='notebook')

# %% [markdown]
# Les 5 clusters sont clairement interprétables
# - 0 : clients qui ont fait 3 ou 4 achats d'une valeur moyenne < 450
# - 1 : clients qui ont fait 2 achats il y a plus longtemps (environ plus de 250j)
# d'une valeur moyenne < 600
# - 2 : clients qui ont fait 2 achats plutôt recemment (environ moins de 250j)
# d'une valeur moyenne < 350
# - 3 : clients qui ont fait plus de 5 achats d'une valeur moyenne < 400
# - 4 : client qui ont fait 2, 3 ou 4 achats de valeurs plus importante
# (environ > 350)

# %%
# diagramme circulaire
fig = pieNbCustClust('KMeans', best_KMeans.labels)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_KMeans.labels.astype(str))
fig.show(renderer='notebook')
# %% [markdown]
# On retrouve sur cette visualisation par t-SNE certains des clusters définis
# par KMeans mais certains sont rassemblés (2 et 4 par exemple)
# %%
# SpectralClustering
SpectralClusters = searchClusters(SpectralClustering, ScaledData, {
    'n_jobs': -1,
    'random_state': 50,
    'affinity': 'nearest_neighbors'
}, 'n_clusters', [*range(2, 13)])
fig = graphScores(SpectralClusters)
fig.show(renderer='notebook')
# %% [markdown]
# Pour cet algorithme nous allons prendre le nombre de clusters pour lequel
# le score de Calinski-Harabasz est le plus élevé car sinon il n'y a que
# 2 clusters ce qui est peu intéressant
# %%
best_Spectral = SpectralClusters.sort_values(by='calinski_harabasz_score',
                                             ascending=False).iloc[0]
fig = graphClusters('SpectralClustering', DataRFM, best_Spectral.labels)
fig.show(renderer='notebook')
# %% [markdown]
# Les 3 clusters sont clairement interprétables
# - 0 : clients qui ont fait plus de 3 achats d'une valeur moyenne < 700
# - 1 : clients qui ont fait 2 achats plutôt recemment (environ moins de 250j)
# et d'une valeur moyenne < 400
# - 2 : clients qui ont fait 2 achats il y a plus longtemps (environ plus de 250j)
# et/ou d'une valeur moyenne > 400
# %%
# diagramme circulaire
fig = pieNbCustClust('SpectralClustering', best_Spectral.labels)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_Spectral.labels.astype(str))
fig.show(renderer='notebook')
# %% [markdown]
# Les séparation correspondent presque on a un peu de mélange entre les clusters
# 1 et 2
# %%
# AgglomerativeClustering
AggloClusters = searchClusters(AgglomerativeClustering, ScaledData, {},
                               'n_clusters', range(2, 13))
fig = graphScores(AggloClusters)
fig.show(renderer='notebook')
# %% [markdown]
# Le meilleurs résultat est pour 6 clusters
# %%
best_Agglo = AggloClusters.sort_values(by='silhouette_score',
                                       ascending=False).iloc[0]
fig = graphClusters('AgglomerativeClusters', DataRFM, best_Agglo.labels)
fig.show(renderer='notebook')
# %% [markdown]
# Les 6 clusters sont plutôt bien définis mais se chevauches légèrement plus que
# ceux de l'algorithme KMeans
# - 0 : clients qui ont fait 2 achats plutôt recemment (environ moins de 250j)
# d'une valeur moyenne < 500
# - 1 : clients qui ont fait plus de 4 achats d'une valeur moyenne < 450
# - 2 : clients qui ont fait entre 2 et 4 achats d'une valeur moyenne > 400
# - 3 : clients qui ont fait 2 il y a plus longtemps (> 300j pour les montants
# les moins importants et > 150j pour les montants plus importants)
# - 4 : clients qui ont fait 3 achats ou 4 achats de valeurs plus importante
# (environ > 400) ou il y a plus longtemps (> 250j)
# - 5 : clients qui ont fait 2 achats de valeurs très importante (> 3500)
# %%
# diagramme circulaire
fig = pieNbCustClust('AgglomerativeClustering', best_Agglo.labels)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_Agglo.labels.astype(str))
fig.show(renderer='notebook')
# %% [markdown]
# On retrouve les chevauchements plus importants sur cette visualisation aussi
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
# %% [markdown]
# Cet algorithme cherche lui même le nombre de cluster.
# Il en a créé 4 et éliminé quelques points. Les quatres clusters correspondent
# au nombre d'achats effectués
# - 0 : clients qui ont fait 2 achats
# - 1 : clients qui ont fait 3 achats
# - 2 : clients qui ont fait 4 achats
# - 3 : clients qui ont fait 5 achats
# %%
# diagramme circulaire
fig = pieNbCustClust('DBSCAN', best_DBSCAN.labels)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_DBSCAN.labels.astype(str))
fig.show(renderer='notebook')
# %% [markdown]
# Les séparation concordent seulement le cluster principale n'est pas séparé
# en deux comme avec d'autres algorithme en fonction du temps écoulé depuis
# la dernière commande
# %%
# Birch
BirchClusters = searchClusters(Birch, ScaledData, {}, 'n_clusters',
                               [*range(2, 13)])
fig = graphScores(BirchClusters)
fig.show(renderer='notebook')
# %% [markdown]
# Pour cet algorithme nous allons prendre le nombre de clusters pour lequel
# le score de Calinski-Harabasz est le plus élevé car sinon il n'y a que
# 3 clusters ce qui est peu intéressant
# %%
best_Birch = BirchClusters.sort_values(by='calinski_harabasz_score',
                                       ascending=False).iloc[0]
fig = graphClusters('Birch', DataRFM, best_Birch.labels)
fig.show(renderer='notebook')
# %% [markdown]
# Nous avons alors 9 clusters:
# - 0 : clients ayant fait entre 6 et 9 commandes
# - 1 : clients ayant fait des commandes d'une valeur moyenne plutôt élevé (>800)
# - 2 : clients ayant fait 2 ou 3 commandes d'une valeur moyenne plutôt moyenne
# (entre 400 et 800)
# - 3 : clients ayant fait 3 ou 4 commandes d'une valeur moyenne plutôt basse
# (<400) et il y a plus longtemps (>300j)
# - 4 : clients ayant fait 2 ou 3 commandes d'une valeur moyenne plutôt basse
# (<500) et plus récemment (<300j)
# - 5 : clients ayant fait 2 commandes d'une valeur moyenne très élevée (>3500)
# - 6 : clients ayant fait 4 ou 5 commandes plutôt récemment (<250j)
# - 7 : clients ayant fait un très grand nombre de commandes (15)
# - 8 : clients ayant fait 2 commandes d'une valeur moyenne plutôt basse (<450)
# et il y a plus longtemps (>350j)
# %%
# diagramme circulaire
fig = pieNbCustClust('Birch', best_Birch.labels)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit,
                 x=0,
                 y=1,
                 color=best_Birch.labels.astype(str))
fig.show(renderer='notebook')
# %% [markdown]
# Il y a plus de clusters que ce que nous permet de visualiser la t-SNE on a donc
# des ensembles scindés en plusieurs clusters
# %%
