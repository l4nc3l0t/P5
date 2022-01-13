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
                              to_pydatetime())))))
        for m, rm in zip(months, reversed(months)):
            # pour chaque nombre de mois ajout des commandes effectuées pendant
            # ces mois
            Data['M{}'.format(rm)] = selectDataRFM(
                Customers, Payments, Orders,
                (datetime.datetime.strptime(base_data_date, '%Y-%m') +
                 relativedelta(months=m)).strftime('%Y-%m'))

    elif period == 'trimestrielle':
        trimestre = range(
            0,
            int(
                len(
                    list(
                        rrule(MONTHLY,
                              dtstart=datetime.datetime.strptime(
                                  '2017-09', '%Y-%m'),
                              until=DelivOrders.order_purchase_timestamp.max().
                              to_pydatetime()))) / 3))
        for t, rt in zip(trimestre, reversed(trimestre)):
            Data['T{}'.format(rt)] = selectDataRFM(
                Customers, Payments, Orders,
                (datetime.datetime.strptime(base_data_date, '%Y-%m') +
                 relativedelta(months=t * 3)).strftime('%Y-%m'))

    elif period == 'semestrielle':
        semestre = range(
            0,
            int(
                len(
                    list(
                        rrule(MONTHLY,
                              dtstart=datetime.datetime.strptime(
                                  '2017-09', '%Y-%m'),
                              until=DelivOrders.order_purchase_timestamp.max().
                              to_pydatetime()))) / 6))
        for s, rs in zip(semestre, reversed(semestre)):
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
        DataFull = DataFull[DataFull.customer_unique_id.isin(
            Data[k].customer_unique_id)].drop(columns='customer_unique_id')
        DataFull_fit = StandardScaler().fit(DataFull)
        ScaledDataFull = DataFull_fit.transform(DataFull)
        true_labels[k] = KMeans(n_clusters=5,
                                random_state=50).fit_predict(ScaledDataFull)
        DataP = Data[k].drop(columns='customer_unique_id')
        Data_fit = StandardScaler().fit(DataP)
        ScaledDataP = Data_fit.transform(DataP)
        pred_labels[k] = KMeans(n_clusters=5,
                                random_state=50).fit_predict(ScaledDataP)
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


# %%
figM = simulationMAJData(Customers, Payments, Orders, '2017-09', 'mensuelle')
figM.show(renderer='notebook')
if write_data is True:
    figM.write_image('./Figures/simMAJM.pdf')
    figM.write_image('./Figures/simMAJM.pdf')
# %%
figT = simulationMAJData(Customers, Payments, Orders, '2017-09',
                         'trimestrielle')
figT.show(renderer='notebook')
if write_data is True:
    figM.write_image('./Figures/simMAJT.pdf')
# %%
figS = simulationMAJData(Customers, Payments, Orders, '2017-09',
                         'semestrielle')
figS.show(renderer='notebook')
if write_data is True:
    figM.write_image('./Figures/simMAJS.pdf')
# %%
