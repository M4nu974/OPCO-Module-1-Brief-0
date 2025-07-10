import sys
import mlflow
import pandas as pd
import joblib
from os.path import join
import random

from modules.preprocess import preprocessing, split, preprocessing_new
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict

# Chargement des datasets
df_old = pd.read_csv(join('data','ds_old.csv'))
df_new = pd.read_csv(join('data','ds_new.csv'))

# Charger le préprocesseur
preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))

# Préprocesser les données
X, y, _ = preprocessing_new(df_new)
# X, y, _ = preprocessing(df_new)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Créer une nouvelle expérience MLflow
mlflow.set_experiment("RUNS Assurances NEW DATA")
# mlflow.set_experiment("RUNS Assurances OLD DATA")

for i in range(1):
    # Générer des hyperparamètres aléatoires dans les bornes spécifiées
    EPOCHS = round(random.randint(20, 500),10)
    BATCH_SIZE =int(EPOCHS/8)
    TEST_SIZE = round(random.uniform(0.5, 0.9),1)

    # Split des données avec la taille de test aléatoire
    X_train, X_test, y_train, y_test = split(X, y, test_size=TEST_SIZE, random_state=42)

    with mlflow.start_run():
        # Création et entraînement du modèle
        model = create_nn_model(X_train.shape[1])
        model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test,
                                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        # Sauvegarder le modèle
        model_path = join('models', f'model_sinistre_2025new.pkl')
        # model_path = join('models', f'model_sinistre_2025.pkl')
        # joblib.dump(model, model_path)

        # Charger le modèle sauvegardé
        model_loaded = joblib.load(model_path)

        # Prédiction sur train
        y_pred_train = model_predict(model_loaded, X_train)
        perf_train = evaluate_performance(y_train, y_pred_train)
        print(f'Run {i+1} - Entrainement - EPOCHS: {EPOCHS}, BATCH_SIZE: {BATCH_SIZE}, TEST_SIZE: {TEST_SIZE:.3f}')
        print_data(perf_train, exp_name="train")

        # Prédiction sur test
        y_pred_test = model_predict(model_loaded, X_test)
        perf_test = evaluate_performance(y_test, y_pred_test)
        print_data(perf_test, exp_name="tests")

        # Log des métriques dans MLflow
        mlflow.log_metric("epoch", EPOCHS)
        mlflow.log_metric("batch_size", BATCH_SIZE)
        mlflow.log_metric("test_size", TEST_SIZE)
        mlflow.log_metric("mse", perf_test["MSE"])
        mlflow.log_metric("mae", perf_test["MAE"])
        mlflow.log_metric("r2", perf_test["R²"])
        mlflow.sklearn.log_model(model_loaded, name=f"assurances_new_data")
        # mlflow.sklearn.log_model(model_loaded, name=f"assurances_old_data")

    # Optionnel : dessiner la courbe de perte pour chaque run
#draw_loss(hist)
