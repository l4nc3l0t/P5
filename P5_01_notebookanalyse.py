# %% [markdown]
# Projet 5 : Segmentez des clients d'un site e-commerce

# %%
import pandas as pd
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
#- geolocation_city : ville
#- geolocation_state : état
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
#- order_estimated_delivery_date : date de livraison estimée à la commande
# %% [markdown]
#### Produits
# %%
Products.info()
# %%
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
#![Liens fichiers](https://i.imgur.com/HRhd2Y0.png "Visualisation des liens entre les fichiers")
# %%
