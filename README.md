# M1 - Brief 0 Entrainement de modèle

---

### Les éléments suivants ont été testés à partir du projet github :
https://github.com/DeVerMyst/OPCO-Module-1-Brief-0.git

---
### Installation
Mettre en place un environnement virtuel Python:
```bash
python -m venv .venv
```

Puis l'activer:

* **Windows (PowerShell) :**
    ```bash
    .\.venv\Scripts\Activate.ps1
    ```
* **Windows (CMD) :**
    ```bash
    .\.venv\Scripts\activate.bat
    ```
* **macOS / Linux :**
    ```bash
    source .venv/bin/activate
    ```

Puis, récupérer les librairies nécessaires au fonctionnement du projet:

```bash
pip install -r requirements.txt
```
Lancer la commande :

```bash
mlflow ui
```

Pour visualiser les différents résultats de test

---
## Dataset _Assurances_:
### 1. Utilisation du dataset `ds_old.csv`,

Génération d'un modèle permettant de prédire le montant total sinistre à partir des données suivantes :
- "age",
- "anciennete_contrat",
- "nombre_sinistres"
- "region",
  Les autres données n'ont pas été prise en compte, car jugées sans corrélation avec le calcul du montant de sinistre.

Apres entrainement avec multiples combinaisons de valeurs d'`epoch`, `test_size` et `batch_size`, le meilleur score **R²** obtenu
est de `0.76` avec:
- EPOCHS = 30
- BATCH_SIZE = 7
- TEST_SIZE = 0.6

### 2. Utilisation du dataset `ds_new.csv`,

Le modèle généré avec le dataset `ds_old.csv` est capable de faire des prédictions sur le dataset
`ds_new.csv` grace au code qui récupère les colonnes dont il a besoin en amont.

Pour entrainer un nouveau modèle à partir du dataset `ds_new.csv` seules les colonnes suivantes ont été retenues:
- age
- nombre_sinistres
- probabilite_sinistre
- region
- participation_prevention

**La colonne `montant_sinistre_estime` n'a pas été prise en compte du fait du caractère aléatoire qu'elle présentait.
Par exemple plusieurs lignes avec `nombre_sinistres` et  `montant_total_sinistres` à `0`, la valeur dans la colonne `montant_sinistre_estime` est supérieure à `0`**

Après entrainement avec plusieurs combinaisons de valeurs d'`epoch`, `test_size` et `batch_size`, le meilleur score **R²** obtenu
est de `0.68` avec:
- EPOCHS = 35
- BATCH_SIZE = 6
- TEST_SIZE = 0.7


## Modifications réalisées sur le code existant:

- Ajout d'une boucle permettant d'automatiser l'entrainement et les tests
- Ajout d'un aspect aléatoire contrôlé pour `EPOCHS`, `BATCH_SIZE` et `TEST_SIZE`
- Connexion avec ML FLOW pour visualiser les résultats et pouvoir trier par le paramètre R² afin d'obtenir les meilleurs paramètres
