# %% [markdown]
# Projet 5 : Segmentez des clients d'un site e-commerce

# %%
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
Customers = pd.read_csv('olist_customers_dataset.csv')
Geolocation = pd.read_csv('olist_geolocation_dataset.csv')
Items = pd.read_csv('olist_order_items_dataset.csv')
Payments = pd.read_csv('olist_order_payments_dataset.csv')
Reviews = pd.read_csv('olist_order_reviews_dataset.csv')
Orders = pd.read_csv('olist_orders_dataset.csv')
Products = pd.read_csv('olist_products_dataset.csv')
Sellers = pd.read_csv('olist_sellers_dataset.csv')
ProdNameTranslation = pd.read_csv('product_category_name_translation.csv')
# %% [markdown]
#### Clients
# %%
Customers.info()
# %% [markdown]
#- customer_id : identifiant clients par achat, clé commune avec les commandes
#- customer_unique_id : identifiant client unique permettant de voir les différents
#achats du client
#- customer_zip_code_prefix : clé commune avec les données de localisation
#- customer_city' : la ville du client
#- customer_state' : l'état (région) du client
# %%
Customers.describe()
# %%[markdown]
#### Localisation
# %%
Geolocation.info()
# %% [markdown]
#- geolocation_zip_code_prefix : colonne commune avec les données clients
#- geolocation_lat : latitude
#- geolocation_lng : longitude
#- geolocation_city : ville
#- geolocation_state : état
# %%
Geolocation.describe()
# %% [markdown]
#### Produits
# %%
Items.info()
# %% [markdown]
#- order_id : identifiant de la commande, clé commune avec les commandes
#- order_item_id : identifiant de l'object acheté au sein de la commande
#- product_id : identifiant du produit acheté
#- seller_id : identifiant du vendeur
#- shipping_limit_date : date limite pour transmettre la commande au transporteur
#- price : prix du produit
#- freight_value : frais de port
# %%
Items.describe()
# %% [markdown]
#### Paiement
# %%
Payments.info()
# %% [markdown]
#- order_id : identifiant de la commande, clé commune avec les commandes
#- payment_sequential : séquence de moyens de paiement utilisés par l'acheteur
#- payment_type : type de moyen de paiement
#- payment_installments : nombre de versements choisi par l'acheteur
#- payment_value : valeur de la transaction
# %%
Payments.describe()
# %% [markdown]
#### Note/commentaire

# %%
Reviews.info()
# %% [markdown]
#- review_id : identifiant de la note/commentaire
#- order_id : indentifiant de la commande, clé commune avec les commandes
#- review_score : note de satisfaction du client
#- review_comment_title : titre du commentaire
#- review_comment_message : commentaire
#- review_creation_date : date d'envoie du questionnaire de satisfaction
#- review_answer_timestamp : date de reception du questionnaire
# %%
Reviews.describe()
# %% [markdown]
#### Commandes
# %%
Orders.info()
# %% [markdown]
#- order_id : identifiant de la commande, clé commune avec les paiements, les notes
#et les objects achetés (items)
#- customer_id : indentifiant du client, clé commune avec les clients
#- order_status : status de la commande
#- order_purchase_timestamp : date de la commande
#- order_approved_at : date ou le paiement à été approuvé
#- order_delivered_carrier_date : date où la commande à été transmise au transporteur
#- order_delivered_customer_date : date de livraison
#- order_estimated_delivery_date : date de livraison estimée à la commande
# %%
Orders.describe()
# %% [markdown]
#### Produits
# %%
Products.info()
# %% [markdown]
#- product_id : identifiant du produit, clé commune avec les objects achetés (items)
#- product_category_name : catégorie du produit
#- product_name_lenght : nombre de caractères dans le nom du produit
#- product_description_lenght : nombre de caractère dans la description du produit
#- product_photos_qty : nombre de photos du produit
#- product_weight_g : poids du produit
#- product_length_cm : longueur du produit
#- product_height_cm : hauteur du produit
#- product_width_cm : largeur du produit
# %%
Products.describe()
# %% [markdown]
#### Vendeurs
# %%
Sellers.info()
# %% [markdown]
#- seller_id : identifiant du vendeur, clé commune avec les objects achetés (items)
#- seller_zip_code_prefix : clé commune avec les données de localisation
#- seller_city : ville du vendeur
#- seller_state : état (région) du vendeur
# %%
Sellers.describe()
# %% [markdown]
#### Traduction des noms de produits
# %%
ProdNameTranslation.info()
# %% [markdown]
#- product_category_name : noms des catégories de produits en portugais brésilien
#- product_category_name_english : noms des catégories de produits en anglais
# %%
ProdNameTranslation.describe()
# %% [markdown]
#Visualisation des liens entre les fichiers
#![Liens fichiers](https://i.imgur.com/HRhd2Y0.png "Visualisation des liens entre les
#fichiers")
# %% [markdown]
#### Analyse données commandes
# %%
# état de la commande
Orders.order_status.value_counts()
# %% [markdown]
# On ne va conserver que les commandes livrées et donc supprimer la colonne status. On
# conserve les colonnes n'ayant pas de valeurs manquantes. On conserve les données
# de date d'achat et de date de livraison estimée que l'on va mettre au format datetime
# %%
DelivOrders = Orders[Orders.order_status == 'delivered'].dropna(axis=1).drop(
    columns='order_status')
# %%
DelivOrders[['order_purchase_timestamp',
             'order_estimated_delivery_date']] = DelivOrders[[
                 'order_purchase_timestamp', 'order_estimated_delivery_date'
             ]].astype('datetime64[ns]')

# %%
# visualisation du nombre de commandes par jours
fig = px.line(DelivOrders.groupby(
    DelivOrders.order_purchase_timestamp.dt.date).count()['order_id'],
              title='Nombre de commandes par jours',
              labels=dict(value='Nombre de commandes',
                          order_purchase_timestamp='Date'))
fig.update_layout(showlegend=False)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/NbCommandesJ.pdf', height=350)
# %%
fig = px.line(DelivOrders.groupby(
    DelivOrders.order_purchase_timestamp.dt.date).count()['order_id'],
              title='Nombre de commandes par jours',
              labels=dict(value='Nombre de commandes',
                          order_purchase_timestamp='Date'))
fig.update_layout(showlegend=False)
if write_data is True:
    fig.write_image('./Figures/NbCommandesJ.pdf', height=350)

# %%
# visualisation du nombre de commandes par mois
DelivOrdersM = DelivOrders.groupby(
    DelivOrders.order_purchase_timestamp.dt.to_period('M')).count()['order_id']
DelivOrdersM.index = DelivOrdersM.index.astype('datetime64[M]')
DelivOrdersM = DelivOrdersM.reset_index()
fig = px.bar(DelivOrdersM,
             x='order_purchase_timestamp',
             y='order_id',
             title='Nombre de commandes par mois',
             labels=dict(order_id='Nombre de commandes',
                         order_purchase_timestamp='Date'),
             height=300,
             width=800)
fig.update_xaxes(dtick='M1', tickformat="%b\n%Y")
fig.update_layout(showlegend=False)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/NbCommandesM.pdf')

# %% [markdown]
#### Analyse données clients
# %%
# comptage du nombre de commandes par clients
OrdersNb = Customers.merge(DelivOrders, on='customer_id', how='right').groupby(
    'customer_unique_id').count()['customer_id'].reset_index().rename(
        columns={
            'customer_id': 'orders_number'
        }).merge(Customers[['customer_unique_id', 'customer_zip_code_prefix']],
                 on='customer_unique_id',
                 how='left').drop_duplicates('customer_unique_id')
# %%
OrdersNb['orders_number'].value_counts().sort_index()
if write_data is True:
    OrdersNb['orders_number'].value_counts().sort_index().reset_index(
    ).to_latex('./Tableaux/NbOrders.tex',
               header=['Nb commandes', 'Nb clients'],
               index=False,
               caption='Nombre de clients par nombre de commandes réalisées')
# %% [markdown]
# Au dessus de 4 commandes on a moins de 100 clients par nombre de commande
# on se limitera aux clients ayant fait 3 commandes maximum
# %%
OrdersNbClean = OrdersNb[OrdersNb.orders_number <= 3]
# %%
# visualisation de la proportion de clients ayant fait au moins 2 commandes
fig = px.pie(
    values=[
        OrdersNb[OrdersNb.orders_number >= 2].count()['customer_unique_id'],
        OrdersNb[OrdersNb.orders_number < 2].count()['customer_unique_id']
    ],
    names=['Au moins 2 commandes', '1 seule commande'],
    title=
    'Proportion de clients en fonction du nombre<br>de commandes effectuées')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/piePropNbOrders.pdf')
# %% [markdown]
# Seul 3% des clients effectuent au moins 2 commandes
# %%
# identifiant des clients ayant réalisés plus de 2 commandes livrées
CustomersMultID = Customers.merge(
    DelivOrders, on='customer_id',
    how='right').groupby('customer_unique_id').count()['customer_id'][
        Customers.merge(DelivOrders, on='customer_id', how='right').groupby(
            'customer_unique_id').count()['customer_id'] >= 2].reset_index(
            ).rename(columns={'customer_id': 'orders_number'})

# %%
# dataframe clients ayant réalisés plus de 2 commandes
CustomersMulti = CustomersMultID.merge(Customers,
                                       on='customer_unique_id',
                                       how='left')

# %% [markdown]
# calcul du temps moyen entre les commandes de chaques clients
# %%
# création d'une colonne par date de commande pour chaques clients
Date = CustomersMultID.merge(
    Customers[['customer_unique_id', 'customer_id']],
    on='customer_unique_id',
    how='left').merge(DelivOrders, on='customer_id').groupby(
        'customer_unique_id').order_purchase_timestamp.unique().apply(
            pd.Series).reset_index()

# temps entre deux commandes
for col in range(0, 14):
    Date['diff_date_order' + str(col)] = abs(
        Date.dropna(subset=[col + 1])[col].sub(
            Date.dropna(subset=[col + 1])[col + 1]))
# %%
# temps moyen entre les commandes d'un client
DateDiff = Date.drop(columns=[*range(0, 15)]).dropna(
    subset=['diff_date_order0']).set_index('customer_unique_id')
DateDiffMean = DateDiff.mean(axis=1).reset_index().rename(
    columns={0: 'date_order_mean_dif'})
DateDiffMean[
    'date_order_mean_dif_days'] = DateDiffMean.date_order_mean_dif.dt.round(
        '1d').dt.days
DateDiffMean.drop(columns=('date_order_mean_dif'), inplace=True)
DateDiffMean.head(3)
# temps moyen entre deux commande pour tous les clients
print('Temps moyen entre deux commandes {} jours'.format(
    round(DateDiffMean.date_order_mean_dif_days.mean(), 0)))
# %% [markdown]
# Il peut être intéressant de renouveller le modèle tout les 3 mois étant donné que
# c'est la durée moyenne entre 2 commandes
# %%
# calcul du temps depuis le dernier achat
LastPurchase = Customers[[
    'customer_unique_id', 'customer_id'
]].merge(DelivOrders, on='customer_id', how='right').groupby(
    'customer_unique_id').order_purchase_timestamp.max().reset_index().rename(
        columns={'order_purchase_timestamp': 'last_purchase_date'})
LastPurchase['last_purchase_days'] = -(
    LastPurchase.last_purchase_date -
    LastPurchase.last_purchase_date.max()) / (np.timedelta64(1, 'D'))
# %% [markdown]
#### Analyse des données de paiement
# %%
Payments.payment_type.unique()
# %%
# selection des paiements de commandes livrées
DelivPayments = DelivOrders[['order_id', 'customer_id']].merge(
    Payments, on='order_id',
    how='left').drop(columns={'payment_sequential', 'payment_installments'})
# %%
# groupement par commande et calcule du total payé et de la part payée en bon d'achat
DelivPaymentsVouch = DelivPayments.groupby('order_id').sum(
    'payment_value').reset_index().rename(columns={
        'payment_value': 'payment_total'
    }).merge(DelivPayments[DelivPayments.payment_type == 'voucher'].groupby(
        'order_id').sum('payment_value').reset_index().rename(
            columns={'payment_value': 'voucher_payment'}),
             on='order_id',
             how='right')
# calcul du pourcentage de la part des paiements en bon d'achat
DelivPaymentsVouch[
    'voucher_percent_part'] = DelivPaymentsVouch.voucher_payment.div(
        DelivPaymentsVouch.payment_total).mul(100)
# rassemblements des données de paiements et des calculs effectués
PaymentsGroup = DelivPayments.groupby('order_id').sum('payment_value').merge(
    DelivPaymentsVouch[['order_id', 'voucher_percent_part']],
    on='order_id',
    how='outer').fillna(0)
# %%
# groupement par client et calcul de la somme des ses achats et de la moyenne du
# pourcentage de la part payée par bon d'achat
PaymentsCust = Customers[['customer_unique_id', 'customer_id']].merge(
    DelivOrders[['customer_id', 'order_id']], on='customer_id',
    how='right').merge(PaymentsGroup, on='order_id',
                       how='left').groupby('customer_unique_id').agg({
                           'payment_value': {'sum', 'mean'},
                           'voucher_percent_part':
                           'mean'
                       }).reset_index()
PaymentsCust['total_payment'] = PaymentsCust.payment_value['sum']
PaymentsCust['mean_payment'] = PaymentsCust.payment_value['mean']
PaymentsCust.columns = PaymentsCust.columns.droplevel(1)
PaymentsCust.drop(columns='payment_value', inplace=True)
# %%
# nombre de clients qui dépensent plus de 3000
PaymentsCust[PaymentsCust.mean_payment < 3000].count()
# %% [markdown]
# Seul une cinquantaine de clients dépenses plus de 3000 nous allons donc les
# supprimer
# %%
PaymentsCustClean = PaymentsCust[PaymentsCust.mean_payment < 3000]
# %% [markdown]
#### Analyse données produits
# %%
# ajouts noms des produits en anglais aux données de produits
Products = Products.merge(
    ProdNameTranslation,
    on='product_category_name').drop(columns='product_category_name').rename(
        columns={'product_category_name_english': 'product_category_name'})
# %%
# certaines catégories sont des doublons et finissent avec _2
Products[Products.product_category_name.str.endswith(
    '_2')].product_category_name.unique()
# %%
# comfort corrigé par confort
Products.product_category_name = Products.product_category_name.str.replace(
    'comfort_2', 'confort')
# retrait du suffix
Products.product_category_name = Products.product_category_name.str.replace(
    '_2', '')
# %%
# figure du nombre de produits par catégories
fig = px.bar(x=Products.product_category_name.value_counts().index,
             y=Products.product_category_name.value_counts().values,
             labels=dict(x='Catégories de produits', y='Nombre de produits'),
             title='Nombre de produits par catégories',
             height=500,
             width=1200)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/NbProdCat.pdf')
# %%
subcat_str = [
    'fashio|luggage|leisure', 'health|beauty|perfum', 'toy|baby|diaper',
    'book|cd|dvd|media|music|audio|art|cine|stationery', 'grocer|food|drink',
    'phon|compu|tablet|electro|consol',
    'home|furnitur|garden|bath|house|applianc|cuisine|christmas|pet|market',
    'flow|gift|stuff', 'auto|tools', 'industry|security|condition'
]
cat_name = [
    'fashion_clothing_accessories', 'health_beauty', 'toys_baby', 'art_media',
    'groceries_food_drink', 'technology', 'home_furniture', 'flowers_gifts',
    'tools_car', 'industry_security'
]

Products10Cat = Products.copy()
# %%
for subcat, cat in zip(subcat_str, cat_name):
    Products10Cat.loc[Products10Cat.product_category_name.str.contains(subcat),
                      'product_category_name'] = cat
# %%
# liste des nouvelles catégories
Products10Cat.product_category_name.unique()

# %%
# figure du nombre de produits dans les nouvelles catégories
fig = px.bar(x=Products10Cat.product_category_name.value_counts().index,
             y=Products10Cat.product_category_name.value_counts().values,
             labels=dict(x='Catégories de produits', y='Nombre de produits'),
             title='Nombre de produits par catégories',
             height=400,
             width=600)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/NbProdCat10.pdf')

# %%
Products10CatDum = pd.get_dummies(Products10Cat,
                                  columns=['product_category_name'])
# %% [markdown]
#### Analyse des produits commandés
# %%
# merge items et données produits
ItemsProd = Items.merge(Products10CatDum, on='product_id', how='left')
# %%
# nombre d'objets par catégories achetés dans une commande et somme des prix et
#  des frais de port
ItemsMean = ItemsProd.groupby('order_id').agg({
    'order_item_id':
    'max',
    'price':
    'sum',
    'freight_value':
    'sum',
    'product_category_name_art_media':
    'sum',
    'product_category_name_fashion_clothing_accessories':
    'sum',
    'product_category_name_flowers_gifts':
    'sum',
    'product_category_name_groceries_food_drink':
    'sum',
    'product_category_name_health_beauty':
    'sum',
    'product_category_name_home_furniture':
    'sum',
    'product_category_name_industry_security':
    'sum',
    'product_category_name_technology':
    'sum',
    'product_category_name_tools_car':
    'sum',
    'product_category_name_toys_baby':
    'sum'
}).reset_index()

# %%
# nombre d'objets par catégories achetés par une personne et somme des prix
# et des frais de port
DelivItemCust = Customers[['customer_unique_id', 'customer_id']].merge(
    DelivOrders[['customer_id', 'order_id']], on='customer_id').merge(
        ItemsMean, on='order_id',
        how='left').groupby('customer_unique_id').agg({
            'order_item_id':
            'max',
            'price':
            'sum',
            'freight_value':
            'sum',
            'product_category_name_art_media':
            'sum',
            'product_category_name_fashion_clothing_accessories':
            'sum',
            'product_category_name_flowers_gifts':
            'sum',
            'product_category_name_groceries_food_drink':
            'sum',
            'product_category_name_health_beauty':
            'sum',
            'product_category_name_home_furniture':
            'sum',
            'product_category_name_industry_security':
            'sum',
            'product_category_name_technology':
            'sum',
            'product_category_name_tools_car':
            'sum',
            'product_category_name_toys_baby':
            'sum'
        }).reset_index().rename(
            columns={
                'order_item_id': 'max_items_order',
                'price': 'total_price',
                'freight_value': 'total_freight_value'
            })
# %% [markdown]
#### Analyse des données de localisation
# %%
# Conservation des colonnes latitude et longitude non présentent dans d'autres jeux
# de données. Moyenne des latitudes et longitude par préfix de code postale pour ne
# pas avoir de données redondantes
Geolocation = Geolocation.groupby(
    'geolocation_zip_code_prefix').mean().reset_index()
Geolocation.head(3)

# %% [markdown]
#### Analyse des notes/commentaires
# %%
# calcul de la moyenne des notes laissées par un client
NoteMean = Customers[['customer_unique_id', 'customer_id']].merge(
    DelivOrders[['customer_id', 'order_id']], on='customer_id',
    how='right').merge(Reviews, on='order_id', how='left').groupby(
        'customer_unique_id').mean('review_score').reset_index()
# %%
# agrégation des jeux de données entre eux
Data = OrdersNbClean.merge(DelivItemCust, on='customer_unique_id').merge(
    DateDiffMean, on='customer_unique_id', how='left').merge(
        LastPurchase[['customer_unique_id', 'last_purchase_days']],
        on='customer_unique_id',
        how='left').merge(NoteMean, on='customer_unique_id').merge(
            PaymentsCustClean, on='customer_unique_id').merge(
                Geolocation,
                left_on='customer_zip_code_prefix',
                right_on='geolocation_zip_code_prefix').drop(
                    columns='geolocation_zip_code_prefix')
Data.date_order_mean_dif_days.fillna(0, inplace=True)
# %%
Data['total_items'] = Data.loc[:,
                               Data.columns.str.
                               contains('product_category_name')].sum(axis=1)

# %%
# ACP des features retenues avec l'energystar score
numPCA = Data.select_dtypes('number').dropna().values
ScaledPCA = make_pipeline(StandardScaler(), PCA())
components = ScaledPCA.fit_transform(numPCA)
pca = ScaledPCA.named_steps['pca']
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
    fig.write_image('./Figures/ScreePlot.pdf', height=300)

# %%
# création des graphiques
for a1, a2 in [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
               [14, 15], [16, 17]]:
    fig = visuPCA(Data.select_dtypes('number').dropna(),
                  pca,
                  components,
                  loadings, [(a1, a2)],
                  color=None)
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/PCAF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=500,
                        height=500)
# %% [markdown]
# Il est difficile d'observer des variables qui auraient plus d'importances
# que d'autres ou d'en regrouper. Nous allons utiliser les méthodes de
# marketing traditionnelles : Recency (last_purchase_days),
# Frequency (orders_number), Monetary (mean_payment)
# %%
DataRFM = Data[['last_purchase_days', 'orders_number',
                'mean_payment']].dropna()

if write_data is True:
    DataRFM.to_csv('OlistDataRFM.csv', index=False)
# %%
DataRFMS = Data[[
    'last_purchase_days', 'orders_number', 'mean_payment', 'review_score'
]].dropna()
if write_data is True:
    DataRFMS.to_csv('OlistDataRFMS.csv', index=False)
# %%
# visualisation des données RFM
for col in DataRFM.columns:
    fig = px.histogram(DataRFM[col], marginal='box')
    fig.update_layout(
        title='Histogramme et diagramme en boite des données<br>de {}'.format(
            col))
    fig.show(renderer='notebook')
    if write_data is True:
        fig.write_image('./Figures/HistRFM{}.pdf'.format(col))

# %%
