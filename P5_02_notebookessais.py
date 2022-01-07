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
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, DBSCAN, \
    OPTICS, Birch
from sklearn.metrics import silhouette_score

from P5_00_fonctions import visuPCA
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
ScaledData = StandardScaler().fit_transform(DataRFM)
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
# graph PCA 3D
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

# %%
# MDS
mds3 = MDS(n_components=3, n_jobs=-1)
ScaledData_transformed3 = mds3.fit(ScaledData).embedding_
# %%
fig = px.scatter_3d(ScaledData_transformed3, x=0, y=1, z=2)
fig.show(renderer='notebook')

# %%
# TSNE
tsne = TSNE(n_components=2,
            perplexity=50,
            init='pca',
            learning_rate='auto',
            n_jobs=-1)
ScaledData_TSNEfit = tsne.fit_transform(ScaledData)
# %%
fig = px.scatter(ScaledData_TSNEfit, x=0, y=1)
fig.show(renderer='notebook')

# %%
distortions = []
silhouettesKM = []
for i in range(2, 13):
    km = KMeans(n_clusters=i)
    km.fit(ScaledData)
    distortions.append(km.inertia_)
    silhouettesKM.append(silhouette_score(ScaledData, km.labels_))

# %%
fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_trace(go.Scatter(x=[*range(2, 13)], y=distortions, name='distortion'),
              secondary_y=False)
fig.add_trace(go.Scatter(x=[*range(2, 13)], y=silhouettesKM,
                         name='silhouette'),
              secondary_y=True)
fig.update_xaxes(title_text='n_clusters')
fig.update_yaxes(title_text="distortion", secondary_y=False)
fig.update_yaxes(title_text="silhouette", secondary_y=True)
fig.show(renderer='notebook')

# %%
# clustering KMeans
KMeansClustering = KMeans(n_clusters=5).fit(ScaledData)
# %%
fig = px.scatter_3d(DataRFM,
                    x='last_purchase_days',
                    y='orders_number',
                    z='mean_payement',
                    color=KMeansClustering.labels_)
fig.show(renderer='notebook')

# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit, x=0, y=1, color=KMeansClustering.labels_)
fig.show(renderer='notebook')
# %%
# MeanShift
MSClustering = MeanShift(n_jobs=-1).fit(ScaledData)
# %%
fig = px.scatter_3d(DataRFM,
                    x='last_purchase_days',
                    y='orders_number',
                    z='mean_payement',
                    color=MSClustering.labels_)
fig.show(renderer='notebook')

# %%
fig = px.scatter(ScaledData_TSNEfit, x=0, y=1, color=MSClustering.labels_)
fig.show(renderer='notebook')

# %%
# SpectralClustering
# %%
silhouettesSC = []
for i in range(2, 13):
    sc = SpectralClustering(n_clusters=i,
                            n_jobs=-1,
                            affinity='nearest_neighbors')
    sc.fit(ScaledData)
    silhouettesSC.append(silhouette_score(ScaledData, sc.labels_))

# %%
fig = px.line(x=[*range(2, 13)], y=silhouettesSC)
fig.update_xaxes(title_text='n_clusters')
fig.update_yaxes(title_text="silhouette")
fig.show(renderer='notebook')
# %%
SClustering = SpectralClustering(n_clusters=5,
                                 affinity='nearest_neighbors',
                                 n_jobs=-1).fit(ScaledData)
# %%
fig = px.scatter_3d(DataRFM,
                    x='last_purchase_days',
                    y='orders_number',
                    z='mean_payement',
                    color=SClustering.labels_)
fig.show(renderer='notebook')

# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit, x=0, y=1, color=SClustering.labels_)
fig.show(renderer='notebook')
# %%
# DBSCAN
# %%
silhouettesDB = []
for i in range(1, 10):
    db = DBSCAN(
        eps=i,
        n_jobs=-1,
    )
    db.fit(ScaledData)
    silhouettesDB.append(silhouette_score(ScaledData, db.labels_))

# %%
fig = px.line(x=[*range(1, 10)], y=silhouettesDB)
fig.update_xaxes(title_text='eps')
fig.update_yaxes(title_text="silhouette")
fig.show(renderer='notebook')
# %%
DBClustering = DBSCAN(eps=1, n_jobs=-1).fit(ScaledData)
# %%
fig = px.scatter_3d(DataRFM,
                    x='last_purchase_days',
                    y='orders_number',
                    z='mean_payement',
                    color=DBClustering.labels_)
fig.show(renderer='notebook')
# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit, x=0, y=1, color=DBClustering.labels_)
fig.show(renderer='notebook')
# %%
# Birch
# %%
silhouettesB = []
for i in range(2, 10):
    b = Birch(n_clusters=i)
    b.fit(ScaledData)
    silhouettesB.append(silhouette_score(ScaledData, b.labels_))

# %%
fig = px.line(x=[*range(2, 10)], y=silhouettesB)
fig.update_xaxes(title_text='n_clusters')
fig.update_yaxes(title_text="silhouette")
fig.show(renderer='notebook')
# %%
BClustering = Birch(n_clusters=5).fit(ScaledData)
# %%
fig = px.scatter_3d(DataRFM,
                    x='last_purchase_days',
                    y='orders_number',
                    z='mean_payement',
                    color=BClustering.labels_)
fig.show(renderer='notebook')

# %%
# TSNE
fig = px.scatter(ScaledData_TSNEfit, x=0, y=1, color=BClustering.labels_)
fig.show(renderer='notebook')
# %%
