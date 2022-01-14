# %%
import os
import pandas as pd
import numpy as np

import datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import plotly.express as px
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
Customers = pd.read_csv('olist_customers_dataset.csv')
Payments = pd.read_csv('olist_order_payments_dataset.csv')
Orders = pd.read_csv('olist_orders_dataset.csv')


# %%
# fonction de sélection de données au modèle RFM
def selectDataRFM(Customers, Payments, Orders, base_data_date=None):

    # conservation des commandes livrées
    DelivOrders = Orders[Orders.order_status == 'delivered'].dropna(
        axis=1).drop(columns='order_status')
    DelivOrders[['order_purchase_timestamp',
                 'order_estimated_delivery_date']] = DelivOrders[[
                     'order_purchase_timestamp',
                     'order_estimated_delivery_date'
                 ]].astype('datetime64[ns]')

    if base_data_date is None:
        DelivOrders = DelivOrders
    else:
        # sélection des commandes ayant été effectuées avant une certaine date
        DelivOrders = DelivOrders.set_index(
            ['order_purchase_timestamp']).loc[:base_data_date].reset_index()

    # identifiant des clients ayant réalisés plus de 2 commandes livrées
    CustomersMultID = Customers.merge(
        DelivOrders, on='customer_id',
        how='right').groupby('customer_unique_id').count()['customer_id'][
            Customers.merge(DelivOrders, on='customer_id',
                            how='right').groupby('customer_unique_id').count()
            ['customer_id'] >= 2].reset_index().rename(
                columns={'customer_id': 'orders_number'})
    # dataframe clients ayant réalisés plus de 2 commandes
    CustomersMulti = CustomersMultID.merge(
        Customers[['customer_unique_id', 'customer_id']],
        on='customer_unique_id',
        how='left')

    # dataframe du temps depuis le dernier achat
    LastPurchase = CustomersMultID.merge(
        Customers, on='customer_unique_id',
        how='left').merge(DelivOrders, on='customer_id', how='right').groupby(
            'customer_unique_id').order_purchase_timestamp.max().reset_index(
            ).rename(
                columns={'order_purchase_timestamp': 'last_purchase_date'})
    LastPurchase['last_purchase_days'] = -(
        LastPurchase.last_purchase_date -
        LastPurchase.last_purchase_date.max()) / (np.timedelta64(1, 'D'))

    # selection des paiements de commandes livrées
    DelivPayments = DelivOrders[['order_id', 'customer_id']].merge(
        Payments, on='order_id', how='left').drop(
            columns={'payment_sequential', 'payment_installments'})

    # somme des paiements par commandes
    PaymentsGroup = DelivPayments.groupby('order_id').sum('payment_value')

    # groupement par client et calcul de la somme et de la moyenne de ses achats
    PaymentsCust = CustomersMultID.merge(
        Customers[['customer_unique_id', 'customer_id']],
        on='customer_unique_id',
        how='left').merge(DelivOrders[['customer_id', 'order_id']],
                          on='customer_id').merge(
                              PaymentsGroup, on='order_id',
                              how='left').groupby('customer_unique_id').agg({
                                  'payment_value': {'sum', 'mean'},
                              }).reset_index()
    PaymentsCust['total_payment'] = PaymentsCust.payment_value['sum']
    PaymentsCust['mean_payment'] = PaymentsCust.payment_value['mean']
    PaymentsCust.columns = PaymentsCust.columns.droplevel(1)
    PaymentsCust.drop(columns='payment_value', inplace=True)

    # agrégation des jeux de données entre eux
    Data = LastPurchase[['customer_unique_id', 'last_purchase_days']].merge(
        CustomersMulti.customer_unique_id.drop_duplicates(),
        on='customer_unique_id',
        how='right').merge(PaymentsCust, on='customer_unique_id')

    return Data


# %%
# calcul et visualisation du score ARI entre les données les plus récentes et 
# celles d'une période donnée
def simulationMAJData(customers, payments, orders, base_data_date, period):
    # utilisation des commandes livrées pour avoir les dates de commandes
    DelivOrders = Orders[Orders.order_status == 'delivered'].dropna(
        axis=1).drop(columns='order_status')
    DelivOrders[['order_purchase_timestamp',
                 'order_estimated_delivery_date']] = DelivOrders[[
                     'order_purchase_timestamp',
                     'order_estimated_delivery_date'
                 ]].astype('datetime64[ns]')

    Data = {}
    # simulation mensuelle
    if period == 'mensuelle':
        # calcul du nombre de mois entre la date de base et la date la dernière
        # commande dans nos données
        months = range(
            0,
            int(
                len(
                    list(
                        rrule(MONTHLY,
                              dtstart=datetime.datetime.strptime(
                                  '2017-09', '%Y-%m'),
                              until=DelivOrders.order_purchase_timestamp.max().
                              to_pydatetime()))) + 1))
        for m, rm in zip(months, reversed(months)):
            # pour chaque nombre de mois ajout des commandes effectuées pendant
            # ces mois
            Data['M{}'.format(rm)] = selectDataRFM(
                Customers, Payments, Orders,
                (datetime.datetime.strptime(base_data_date, '%Y-%m') +
                 relativedelta(months=m)).strftime('%Y-%m'))
    # simulation trimestrielle
    elif period == 'trimestrielle':
        # calcul du nombre de trimestres entre la date de base et la date de la
        # dernière commande dans nos données
        trimestre = range(
            0,
            int(
                len(
                    list(
                        rrule(MONTHLY,
                              dtstart=datetime.datetime.strptime(
                                  '2017-09', '%Y-%m'),
                              until=DelivOrders.order_purchase_timestamp.max().
                              to_pydatetime()))) / 3 + 1))
        for t, rt in zip(trimestre, reversed(trimestre)):
            # pour chaque nombre de trimestre ajout des commandes effectuées
            # pendant ces mois
            Data['T{}'.format(rt)] = selectDataRFM(
                Customers, Payments, Orders,
                (datetime.datetime.strptime(base_data_date, '%Y-%m') +
                 relativedelta(months=t * 3)).strftime('%Y-%m'))
    # simulation semestrielle
    elif period == 'semestrielle':
        # calcul du nombre de semestres entre la date de base et la date de la
        # dernière commande dans nos données
        semestre = range(
            0,
            int(
                len(
                    list(
                        rrule(MONTHLY,
                              dtstart=datetime.datetime.strptime(
                                  '2017-09', '%Y-%m'),
                              until=DelivOrders.order_purchase_timestamp.max().
                              to_pydatetime()))) / 6 + 1))
        for s, rs in zip(semestre, reversed(semestre)):
            # pour chaque nombre de semestre ajout des commandes effectuées
            # pendant ces mois
            Data['S{}'.format(rs)] = selectDataRFM(
                Customers, Payments, Orders,
                (datetime.datetime.strptime(base_data_date, '%Y-%m') +
                 relativedelta(months=s * 6)).strftime('%Y-%m'))

    else:
        print('Période mal renseignée (mensuelle, trimestrielle, semestrielle')

    true_labels = {}
    pred_labels = {}
    ARI = {}
    for k in Data.keys():
        DataFull = selectDataRFM(Customers, Payments, Orders)
        # sélection des données clients finales qui correspondent aux clients
        # présents dans les données de la période choisie
        DataFull = DataFull[DataFull.customer_unique_id.isin(
            Data[k].customer_unique_id)].drop(columns='customer_unique_id')
        DataFull_fit = StandardScaler().fit(DataFull)
        ScaledDataFull = DataFull_fit.transform(DataFull)
        true_labels[k] = KMeans(n_clusters=5,
                                random_state=50).fit_predict(ScaledDataFull)
        # KMeans sur les données de la période choisie
        DataP = Data[k].drop(columns='customer_unique_id')
        DataP_fit = StandardScaler().fit(DataP)
        ScaledDataP = DataP_fit.transform(DataP)
        pred_labels[k] = KMeans(n_clusters=5,
                                random_state=50).fit_predict(ScaledDataP)
        # calcul du score ARI
        ARI[k] = (adjusted_rand_score(true_labels[k], pred_labels[k]))

    fig = px.line(
        x=ARI.keys(),
        y=ARI.values(),
        labels={
            'x': 'Anciennetée du modèle',
            'y': 'ARI'
        },
        title="ARI du modèle en fonction de l'anciennetée {} du modèle".format(
            period))
    fig.update_xaxes(autorange='reversed')
    return fig

# %% [markdown]
# L'utilisation du score ARI (adjusted rand score) permet de regarder si les 
# points sont toujours dans le mêmes clusters entre deux classifications 
# différentes. Nous comparons pour chaque période le clusters dans lequel sont
# les clients dans les données actuelles et le cluster dans lequel sont les données
# de la période correspondante.
# %%
# évolution du score ARI avec ajouts de données semestrielles
figS = simulationMAJData(Customers, Payments, Orders, '2017-09',
                         'semestrielle')
figS.show(renderer='notebook')
if write_data is True:
    figS.write_image('./Figures/simMAJS.pdf')
    figS.write_image('./Figures/simMAJS.pdf')
# %% [markdown]
# Le premier semestre semble être encore assez bien corrélé (ARI>0.9) le second
# chute assez brutalement (ARI<0.6)
# %%
# évolution du score ARI avec ajouts de données trimestrielles
figT = simulationMAJData(Customers, Payments, Orders, '2017-09',
                         'trimestrielle')
figT.show(renderer='notebook')
if write_data is True:
    figT.write_image('./Figures/simMAJT.pdf')
# %% [markdown]
# Le premier trimestre reste plutôt bien corrélé (ARI>0.98). La pente augmente 
# ensuite jusqu'au 3è trimestre. On retrouve la chute importante au 4è trimestre
# %%
# évolution du score ARI avec ajouts de données mensuelles
figM = simulationMAJData(Customers, Payments, Orders, '2017-09', 'mensuelle')
figM.show(renderer='notebook')
if write_data is True:
    figM.write_image('./Figures/simMAJM.pdf')
#%% [markdown]
# Les trois premiers mois les données restent bien corrélées (ARI>0.98) mais après
# on observe une pente plus importante entre le 3è et le 11è mois avec une chute
# au 12è mois

#### Conclusion

# Afin de conserver des données toujours au plus près de la réalité (ARI>0.98) 
# il me semble idéal de renouveler le modèle tout les 3 mois (mise à jours
# trimestrielle). Si le modèle n'est renouvelé que tout les 6 mois on sera moins
# proche de la réalité mais celà me semble rester pertinent car l'ARI reste autour 
# de 0.9. Par contre au delà la chute du score s'accélère et au bout d'un an l'ARI
# passe en dessous de 0.6.