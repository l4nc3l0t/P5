# %%
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
Data = OlistData.drop(columns='customer_unique_id').dropna()
ScaledData = StandardScaler().fit_transform(Data)

# %%
DataClustering = KMeans(n_clusters=4).fit(ScaledData)
Clusters = DataClustering.predict(ScaledData)
# %%
fig = px.scatter_3d(components, x=0, y=1, z=2, color=DataClustering.labels_)
fig.show(renderer='notebook')
# %%
distortions = []
for i in range(1, 51):
    km = KMeans(n_clusters=i)
    km.fit(ScaledData)
    distortions.append(km.inertia_)

fig = px.line(x=range(1, 51), y=distortions)
fig.show(renderer='notebook')
# %%
# ACP des features retenues avec l'energystar score
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
for a1, a2 in [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
               [14, 15]]:
    fig = visuPCA(Data,
                  pca,
                  components,
                  loadings, [(a1, a2)],
                  color=DataClustering.labels_)
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/PCAKMeansF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=500,
                        height=500)
# %%
