# %% [markdown]
# Projet 5 : Segmentez des clients d'un site e-commerce

# %%
import pandas as pd
import numpy as numpy
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
# %% [markdown]
#### Commandes
# %%
Orders.info()
# %% [markdown]
#- order_id : identifiant de la commande, clé commune avec les payements, les notes
#et les objects achetés (items)
#- customer_id : indentifiant du client, clé commune avec les clients
#- order_status : status de la commande
#- order_purchase_timestamp : date de la commande
#- order_approved_at : date ou le paiement à été approuvé
#- order_delivered_carrier_date : date où la commande à été transmise au transporteur
#- order_delivered_customer_date : date de livraison
#- order_estimated_delivery_date : date de livraison estimée à la commande
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

# %% [markdown]
#### Vendeurs
# %%
Sellers.info()
# %% [markdown]
#- seller_id : identifiant du vendeur, clé commune avec les objects achetés (items)
#- seller_zip_code_prefix : clé commune avec les données de localisation
#- seller_city : ville du vendeur
#- seller_state : état (région) du vendeur

# %% [markdown]
#### Traduction des noms de produits
# %%
ProdNameTranslation.info()
# %% [markdown]
#- product_category_name : noms des catégories de produits en portugais brésilien
#- product_category_name_english : noms des catégories de produits en anglais
# %% [markdown]
#Visualisation des liens entre les fichiers
#![Liens fichiers](https://i.imgur.com/HRhd2Y0.png "Visualisation des liens entre les
#fichiers")
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
# %% [markdown]
#### Analyse des données de localisation
# %%
Geolocation = Geolocation.groupby(
    'geolocation_zip_code_prefix').mean().reset_index()
Geolocation.head(3)
# %% [markdown]
#### Analyse des données de paiement
# %%
Payments
# %% [markdown]
#### Analyse données commandes
# %%
# agrégation des autres jeux de données aux commandes pour selectionner les données
# intéressantes
Orders.merge(Items, on='order_id', how='left').merge(
    Reviews, on='order_id',
    how='left').merge(Customers, on='customer_id', how='left').merge(
        Geolocation,
        left_on='customer_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left').drop(columns='geolocation_zip_code_prefix').merge(
            Products, on='product_id', how='left')
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
    fig.write_image('./Figures/NbCommandesJ.pdf')
if write_data is True:
    fig.write_image('./Figures/NbCommandesJ.pdf')

# %% [markdown]
#### Analyse données clients
