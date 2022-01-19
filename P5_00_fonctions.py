import pandas as pd
import numpy as np
import plotly.express as px


# graphique visualisation vecteurs et données
def visuPCA(df, pca, components, loadings, axis, color=None):
    for f1, f2 in axis:
        if color is None:
            fig = px.scatter(components, x=f1, y=f2)
        else:
            fig = px.scatter(components, x=f1, y=f2, color=color)  #,
            #labels={'color': 'Score<br>Nutriscore'})
    for f, feature in enumerate(df.columns):
        fig.add_shape(type='line',
                      x0=0,
                      y0=0,
                      x1=loadings[f, f1] * 10,
                      y1=loadings[f, f2] * 10,
                      line=dict(color='yellow'))
        fig.add_annotation(x=loadings[f, f1] * 10,
                           y=loadings[f, f2] * 10,
                           ax=0,
                           ay=0,
                           xanchor="center",
                           yanchor="bottom",
                           text=feature,
                           bgcolor='white')
    fig.update_layout(
        title='PCA F{} et F{}'.format(f1 + 1, f2 + 1),
        xaxis_title='F{} '.format(f1 + 1) + '(' + str(
            (pca.explained_variance_ratio_[f1] * 100).round(2)) + '%' + ')',
        yaxis_title='F{} '.format(f2 + 1) + '(' + str(
            (pca.explained_variance_ratio_[f2] * 100).round(2)) + '%' + ')')
    return (fig)


from time import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# variation d'un paramètre de modèle de classification
def searchClusters(model, data, paramfix: dict, paramtuned: str,
                   paramrange: list):
    Results = pd.DataFrame()
    for paramval in paramrange:
        paramfix[paramtuned] = paramval
        paramodel = model(**paramfix)
        start_time = time()
        pred_labels = paramodel.fit_predict(data)
        fit_pred_time = time() - start_time
        result = {
            'model':
            str(paramodel).replace(' ', '').replace(',', '').replace(
                '\n', '').replace('n_jobs=-1',
                                  '').replace('random_state=50', '').replace(
                                      "affinity='nearest_neighbors'", ''),
            'n_clusters':
            pred_labels.max() + 1,
            'labels':
            paramodel.labels_,
            'clusters_centers':
            paramodel.cluster_centers_
            if hasattr(paramodel, 'cluster_centers_') else None,
            'inertia':
            paramodel.inertia_ if hasattr(paramodel, 'inertia_') else None,
            'time':
            fit_pred_time,
            'silhouette_score':
            silhouette_score(data, pred_labels),
            'calinski_harabasz_score':
            calinski_harabasz_score(data, pred_labels),
            'davies_bouldin_score':
            davies_bouldin_score(data, pred_labels)
        }
        Results = Results.append(result, ignore_index=True)
        scaledScores = pd.DataFrame(
            data=StandardScaler().fit_transform(Results[[
                'silhouette_score', 'calinski_harabasz_score',
                'davies_bouldin_score'
            ]]),
            columns=[
                'scaled_silhouette_score', 'scaled_calinski_harabasz_score',
                'scaled_davies_bouldin_score'
            ])
        Results['metascore'] = (scaledScores['scaled_silhouette_score'] +
                                scaledScores['scaled_calinski_harabasz_score'] -
                                scaledScores['scaled_davies_bouldin_score'])

    return Results


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# visualisation des scores des modèles de classification
def graphScores(Results):
    if (Results.inertia.unique().all() == None) is True:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.silhouette_score],
                                 name='Silhouette'),
                      row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.calinski_harabasz_score],
                                 name='Calinski-Harabasz'),
                      row=2,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.davies_bouldin_score],
                                 name='Davies-Bouldin'),
                      row=3,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.metascore],
                                 name='meta-score'),
                      row=4,
                      col=1)
    else:
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.silhouette_score],
                                 name='Silhouette'),
                      row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.calinski_harabasz_score],
                                 name='Calinski Harabasz'),
                      row=2,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.davies_bouldin_score],
                                 name='Davies-Bouldin'),
                      row=3,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.metascore],
                                 name='meta-score'),
                      row=4,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.inertia],
                                 name='Distortion'),
                      row=5,
                      col=1)
    fig.update_layout(
        title_text=
        'Visualisation des scores de classification<br>selon le paramètre du modèle'
    )
    return fig


# graph data avec couleurs par clusters et centroïdes
def graphClusters(modelname, data, labels: list, clusters_centers=None):
    fig = px.scatter_3d(data,
                        x='last_purchase_days',
                        y='orders_number',
                        z='mean_payment',
                        title='Clusters créés par {}'.format(modelname),
                        color=labels.astype(str),
                        labels={'color': 'Clusters'})
    fig.update_traces(marker_size=3)
    if clusters_centers is not None:
        fig.add_trace(
            go.Scatter3d(
                x=clusters_centers['last_purchase_days'],
                y=clusters_centers['orders_number'],
                z=clusters_centers['mean_payment'],
                mode='markers',
                marker_symbol='x',
                marker_size=5,
                hovertemplate=
                "recency: %{x}<br>frequency: %{y}<br>monetary: %{z}",
                name="Cluster center",
            ))
    fig.update_layout(legend={'itemsizing': 'constant'})
    return fig

# graph data avec couleurs par clusters et centroïdes
def graphClustersRFMS(modelname, data, labels: list, clusters_centers=None):
    fig = px.scatter_3d(data,
                        x='last_purchase_days',
                        y='mean_payment',
                        z='review_score',
                        size='orders_number',
                        title='Clusters créés par {}'.format(modelname),
                        color=labels.astype(str),
                        opacity=1,
                        labels={'color': 'Clusters'})
    #fig.update_traces(marker_size=3)
    if clusters_centers is not None:
        fig.add_trace(
            go.Scatter3d(
                x=clusters_centers['last_purchase_days'],
                y=clusters_centers['mean_payment'],
                z=clusters_centers['review_score'],
                mode='markers',
                marker_symbol='x',
                marker_size=5,
                hovertemplate=
                "recency: %{x}<br>frequency: %{y}<br>monetary: %{z}",
                name="Cluster center",
            ))
    fig.update_layout(legend={'itemsizing': 'constant'})
    return fig

# visualisation du nombre de clients par clusters
def pieNbCustClust(modelname: str, labels):
    NbCustClust = pd.Series(labels).value_counts().sort_index()
    fig = px.pie(NbCustClust,
                 values=NbCustClust.values,
                 names=NbCustClust.index,
                 labels={
                     'index': 'Cluster',
                     'values': 'Nombre de clients'
                 })
    fig.update_layout(
        title="Nombre de clients par clusters de l'algorithme<br>{}".format(
            modelname))
    return fig
