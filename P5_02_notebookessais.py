# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
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
OlistData = pd.read_csv('OlistData.csv')
# %%
DataRFM = OlistData[['last_purchase_days', 'orders_number',
                     'mean_payement']].dropna()
ScaledData = StandardScaler().fit_transform(DataRFM)

# %%
# MDS sur les 500 premières lignes
stress = []
for n_comp in range(2, 13):
    mds = MDS(n_components=n_comp, n_jobs=-1)
    ScaledData_fit = mds.fit(ScaledData[:500])
    stress.append(mds.stress_)
# %%
fig = px.line(x=range(2, 13),
              y=stress,
              labels={
                  'x': 'n_components',
                  'y': 'stress'
              })
fig.show(renderer='notebook')
# %%
mds3 = MDS(n_components=4, n_jobs=-1)
ScaledData_fit3 = mds3.fit(ScaledData[:500]).embedding_
# %%
fig = px.scatter_3d(ScaledData_fit3, x=0, y=1, z=2, color=3)
fig.show(renderer='notebook')


# %%
distortions = []
silhouettes = []
for i in range(2, 23):
    km = KMeans(n_clusters=i)
    km.fit(ScaledData)
    Clusters = km.predict(ScaledData)
    distortions.append(km.inertia_)
    silhouettes.append(silhouette_score(ScaledData, Clusters))

# %%
fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_trace(go.Scatter(x=[*range(2, 23)], y=distortions, name='distortion'),
              secondary_y=False)
fig.add_trace(go.Scatter(x=[*range(2, 23)], y=silhouettes, name='silhouette'),
              secondary_y=True)
fig.update_xaxes(title_text='n_clusters')
fig.update_yaxes(title_text="distortion", secondary_y=False)
fig.update_yaxes(title_text="silhouette", secondary_y=True)
fig.show(renderer='notebook')
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
    fig.write_image('./Figures/ScreePlotKMeans.pdf', height=300)

# %%
# création des graphiques
for a1, a2 in [[0, 1], [0, 2], [1, 2]]:
    fig = visuPCA(DataRFM,
                  pca,
                  components,
                  loadings, [(a1, a2)],
                  color=None)
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/PCAKMeansF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=500,
                        height=500)
# %%
# %%
# clustering KMeans
DataClustering = KMeans(n_clusters=3).fit(ScaledData)
ScaledData_KMeansTransformed = DataClustering.transform(ScaledData)
Clusters = DataClustering.predict(ScaledData)
# %%
fig = px.scatter_3d(ScaledData_KMeansTransformed,
                 x=0,
                 y=1,
                 z=2,
                 color=DataClustering.labels_)
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
fig = px.scatter(ScaledData_TSNEfit, x=0, y=1, color=DataClustering.labels_)
fig.show(renderer='notebook')
# %%