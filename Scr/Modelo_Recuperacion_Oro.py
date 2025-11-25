import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


RANDOM_STATE = 12345


# -------------------------------------------------------------------
# Métrica sMAPE
# -------------------------------------------------------------------
def smape(y_true, y_pred, eps: float = 1e-9) -> float:
    """
    Calcula el Symmetric Mean Absolute Percentage Error (sMAPE) en %.

    sMAPE = 2 * |y_pred - y_true| / (|y_true| + |y_pred|)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_pred - y_true) / (denom + eps)) * 100.0


def smape_final(
    y_true_rougher,
    y_pred_rougher,
    y_true_final,
    y_pred_final,
) -> float:
    """
    sMAPE combinado:
    - 25% para rougher.output.recovery
    - 75% para final.output.recovery
    """
    s_r = smape(y_true_rougher, y_pred_rougher)
    s_f = smape(y_true_final, y_pred_final)
    return 0.25 * s_r + 0.75 * s_f


# -------------------------------------------------------------------
# Carga y preparación de datos
# -------------------------------------------------------------------
def load_data(
    train_path: str = "Data/gold_recovery_train.csv",
    test_path: str = "Data/gold_recovery_test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets de entrenamiento y prueba.
    La columna 'date' se usa como índice.
    """
    train = pd.read_csv(train_path, parse_dates=["date"], index_col="date")
    test = pd.read_csv(test_path, parse_dates=["date"], index_col="date")
    return train, test


def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """
    - Elimina columnas de salida (targets) como features.
    - Se queda con las columnas comunes entre train y test.
    - Devuelve X_train, X_test, y_rougher, y_final y la lista de features.
    """
    # Targets
    y_train_rougher = train["rougher.output.recovery"]
    y_train_final = train["final.output.recovery"]

    # Features candidatas: todas excepto las columnas de salida
    feature_candidates = [
        c
        for c in train.columns
        if not c.startswith("rougher.output")
        and not c.startswith("final.output")
    ]

    # Nos quedamos con las que existen también en test
    features = [c for c in feature_candidates if c in test.columns]

    X_train = train[features]
    X_test = test[features]

    return X_train, X_test, y_train_rougher, y_train_final, features


# -------------------------------------------------------------------
# Modelado y validación cruzada
# -------------------------------------------------------------------
def build_pipeline(model_name: str, rf_params: dict | None = None) -> Pipeline:
    """
    Construye un pipeline de preprocesamiento + modelo.

    model_name:
        - "linreg": LinearRegression
        - "rf": RandomForestRegressor
    """
    if model_name == "linreg":
        model = LinearRegression()
    elif model_name == "rf":
        default_params = dict(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        if rf_params is not None:
            default_params.update(rf_params)
        model = RandomForestRegressor(**default_params)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    return pipe


def cv_smape_final(
    model_name: str,
    X: pd.DataFrame,
    y_r: pd.Series,
    y_f: pd.Series,
    n_splits: int = 5,
    rf_params: dict | None = None,
    random_state: int = RANDOM_STATE,
):
    """
    Aplica validación cruzada KFold para dos targets (rougher y final) usando
    el mismo tipo de modelo/pipeline en ambos.

    Devuelve:
    - sMAPE rougher
    - sMAPE final
    - sMAPE combinado (25/75)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_r = y_r.reset_index(drop=True)
    y_f = y_f.reset_index(drop=True)
    X = X.reset_index(drop=True)

    y_pred_r = np.zeros_like(y_r.values, dtype=float)
    y_pred_f = np.zeros_like(y_f.values, dtype=float)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_r_tr, y_r_va = y_r.iloc[train_idx], y_r.iloc[val_idx]
        y_f_tr, y_f_va = y_f.iloc[train_idx], y_f.iloc[val_idx]

        pipe_r = build_pipeline(model_name, rf_params=rf_params)
        pipe_f = build_pipeline(model_name, rf_params=rf_params)

        pipe_r.fit(X_tr, y_r_tr)
        pipe_f.fit(X_tr, y_f_tr)

        y_pred_r[val_idx] = pipe_r.predict(X_va)
        y_pred_f[val_idx] = pipe_f.predict(X_va)

    s_r = smape(y_r.values, y_pred_r)
    s_f = smape(y_f.values, y_pred_f)
    s_fin = smape_final(y_r.values, y_pred_r, y_f.values, y_pred_f)

    return s_r, s_f, s_fin


def main():
    # 1. Cargar datos
    train, test = load_data()

    # 2. Preparar matrices de features y targets
    X_train, X_test, y_r_train, y_f_train, features = build_feature_matrix(train, test)

    print("Shape X_train:", X_train.shape)
    print("Shape X_test :", X_test.shape)
    print("Número de features:", len(features))

    # 3. Validación cruzada para dos tipos de modelo
    results = []
    for name in ["linreg", "rf"]:
        rf_params = dict(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1) if name == "rf" else None
        s_r, s_f, s_fin = cv_smape_final(
            name,
            X_train,
            y_r_train,
            y_f_train,
            n_splits=5,
            rf_params=rf_params,
        )
        results.append(
            {
                "modelo": name,
                "sMAPE_rougher": s_r,
                "sMAPE_final": s_f,
                "sMAPE_combinado": s_fin,
            }
        )

    res_df = pd.DataFrame(results).sort_values("sMAPE_combinado").reset_index(drop=True)
    print("\nResultados de CV (menor sMAPE es mejor):")
    print(res_df)

    best_name = res_df.iloc[0]["modelo"]
    print("\nMejor modelo según sMAPE combinado:", best_name)


if __name__ == "__main__":
    main()
