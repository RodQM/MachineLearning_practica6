import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

# Parte 1: Clasificador de Distancia Mínima
class MinimumDistanceClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = {cls: X[y == cls].mean(axis=0) for cls in self.classes_}
        return self

    def predict(self, X):
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids_.values()]).T
        predictions = np.array(list(self.centroids_.keys()))[np.argmin(distances, axis=1)]
        return predictions

# Función para realizar validaciones
def validate_model(clf, datasets, validation_types):
    scaler = StandardScaler()
    all_results = {}

    for name, (X, y) in datasets.items():
        print(f"Procesando {name}...")
        X = scaler.fit_transform(X)
        dataset_results = {}

        for validation_type in validation_types:
            if validation_type == "Hold-Out":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                dataset_results["Hold-Out"] = {"Accuracy": accuracy, "Confusion Matrix": conf_matrix}

            elif validation_type == "10-Fold CV":
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                y_pred = cross_val_predict(clf, X, y, cv=skf)
                accuracy = accuracy_score(y, y_pred)
                conf_matrix = confusion_matrix(y, y_pred)
                dataset_results["10-Fold CV"] = {"Accuracy": accuracy, "Confusion Matrix": conf_matrix}

            elif validation_type == "Leave-One-Out":
                loo = LeaveOneOut()
                y_pred = cross_val_predict(clf, X, y, cv=loo)
                accuracy = accuracy_score(y, y_pred)
                conf_matrix = confusion_matrix(y, y_pred)
                dataset_results["Leave-One-Out"] = {"Accuracy": accuracy, "Confusion Matrix": conf_matrix}

        all_results[name] = dataset_results

    return all_results

# Cargar datasets
breast_cancer = load_breast_cancer()
california_housing = fetch_california_housing()
covtype = fetch_covtype()

# Preparar los datasets
X_bc, y_bc = breast_cancer.data, breast_cancer.target
X_ch, y_ch = california_housing.data, california_housing.target
median_price = np.median(y_ch)
y_ch = (y_ch > median_price).astype(int)
X_cov, y_cov = covtype.data[:1000], covtype.target[:1000]

datasets = {
    "Breast Cancer": (X_bc, y_bc),
    "California Housing": (X_ch, y_ch),
    "Covertype": (X_cov, y_cov)
}

# Interfaz del programa
print("Seleccione la Parte a Ejecutar:")
print("1. Clasificador de Distancia Mínima")
print("2. Clasificador 1NN")

option = input("Ingrese 1 o 2: ")

validation_methods = ["Hold-Out", "10-Fold CV", "Leave-One-Out"]

if option == "1":
    print("Ejecutando la Parte 1...")
    clf = MinimumDistanceClassifier()
    results = validate_model(clf, datasets, validation_methods)
    for dataset, metrics in results.items():
        print(f"\nResultados para {dataset}:")
        for method, result in metrics.items():
            print(f"\n{method}:")
            print(f"Accuracy: {result['Accuracy']}")
            print(f"Matriz de Confusión:\n{result['Confusion Matrix']}")

elif option == "2":
    print("Ejecutando la Parte 2...")
    clf = KNeighborsClassifier(n_neighbors=1)
    results = validate_model(clf, datasets, validation_methods)
    for dataset, metrics in results.items():
        print(f"\nResultados para {dataset}:")
        for method, result in metrics.items():
            print(f"\n{method}:")
            print(f"Accuracy: {result['Accuracy']}")
            print(f"Matriz de Confusión:\n{result['Confusion Matrix']}")
else:
    print("Opción no válida.")
