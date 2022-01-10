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
from sklearn.metrics import silhouette_score, calinski_harabasz_score


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
            str(paramodel).replace(', n_jobs=-1',
                                   '').replace('n_jobs=-1', '').replace(
                                       "affinity='nearest_neighbors', ", ''),
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
            calinski_harabasz_score(data, pred_labels)
        }
        Results = Results.append(result, ignore_index=True)
    return Results


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# visualisation des scores des modèles de classification
def graphScores(Results):
    if (Results.inertia.unique().all() == None) is True:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
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
                                 y=[*Results.time],
                                 name='Temps'),
                      row=3,
                      col=1)
        fig.update_yaxes(title_text='secondes', row=3, col=1)
    else:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
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
                                 y=[*Results.inertia],
                                 name='Distortion'),
                      row=3,
                      col=1)
        fig.add_trace(go.Scatter(x=[*Results.model],
                                 y=[*Results.time],
                                 name='Temps'),
                      row=4,
                      col=1)

        fig.update_yaxes(title_text='secondes', row=4, col=1)
    fig.update_layout(
        title_text=
        'Visualisation des scores et du temps de classification<br>selon le paramètre du modèle'
    )
    return fig


# graph data avec couleurs par clusters et centroïdes
def graphClusters(modelname, data, labels: list, clusters_centers=None):
    fig = px.scatter_3d(data,
                        x='last_purchase_days',
                        y='orders_number',
                        z='mean_payement',
                        title='Clusters créés par {}'.format(modelname),
                        color=labels.astype(str))
    if clusters_centers is not None:
        fig.add_trace(
            go.Scatter3d(
                x=clusters_centers['last_purchase_days'],
                y=clusters_centers['orders_number'],
                z=clusters_centers['mean_payement'],
                mode='markers',
                marker_symbol='x',
                hovertemplate=
                "recency: %{x}<br>frequency: %{y}<br>monetary: %{z}",
                name="Cluster center",
            ))
    fig.update_layout(coloraxis_colorbar=dict(yanchor='top', y=.9))
    return fig