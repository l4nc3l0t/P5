1. il est conseillé de créer un nouvel environnement python pour éviter les conflits sinon effectuer uniquement les étape 3. 5. et 6. :

`python3 -m venv ./PyP5/`

2. activer l'environnement :

`source ./PyP5/bin/activate`

3. installer les dépendances:

`pip install -r requiremements.txt`

4. ajouter le noyau pour jupyter :

`./PyP5/bin/python -m ipykernel install --user --name 'PyP5'`

5. ouvrir le notebook de nettoyage

`jupyter notebook ./P5_01_notebookanalyse.ipynb`

6. changer le noyau dans jupyter : / Noyau / Changer le noyau / PyP5
