"""
data_generator.py
Genera datos sintéticos de demanda y plazos de entrega (lead times)
para un sistema de inventario con componentes de estacionalidad y tendencia.
"""

import numpy as np
import pandas as pd


def generate_dataset(
    n_weeks=500,
    demand_lambda=20,
    lead_time_low=2,
    lead_time_high=8,
    seasonality_amplitude=5,
    trend_slope=0.02,
    seed=42,
):
    """
    Genera un dataset sintético de demanda e inventario.

    Parámetros
    ----------
    n_weeks              : número de semanas a simular
    demand_lambda        : media de la distribución de Poisson (demanda base)
    lead_time_low/high   : rango uniforme del lead time en semanas
    seasonality_amplitude: amplitud del componente sinusoidal anual
    trend_slope          : incremento lineal por semana sobre la demanda base
    seed                 : semilla para reproducibilidad

    Retorna
    -------
    pd.DataFrame con columnas de demanda, lead time, features y target.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. GENERACIÓN DE LA DEMANDA SEMANAL
    # ------------------------------------------------------------------ #

    weeks = np.arange(n_weeks)

    # Componente base: ruido de Poisson con media demand_lambda
    base = rng.poisson(lam=demand_lambda, size=n_weeks)

    # Componente estacional: seno con periodo de 52 semanas
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * weeks / 52)

    # Componente de tendencia: crecimiento lineal suave
    trend = trend_slope * weeks

    # Demanda final: suma de los tres componentes, redondeada y >= 0
    demand = np.maximum(0, np.round(base + seasonality + trend)).astype(int)

    # ------------------------------------------------------------------ #
    # 2. GENERACIÓN DE LEAD TIMES
    # ------------------------------------------------------------------ #

    # Entero aleatorio uniforme en [lead_time_low, lead_time_high] por semana
    lead_times = rng.integers(low=lead_time_low, high=lead_time_high + 1, size=n_weeks)

    # ------------------------------------------------------------------ #
    # 3. CONSTRUCCIÓN DEL DATAFRAME BASE
    # ------------------------------------------------------------------ #

    df = pd.DataFrame({
        "demand": demand,
        "lead_time": lead_times,
        "week_of_year": weeks,  # índice de semana dentro del episodio (0 a n_weeks-1)
    })

    # ------------------------------------------------------------------ #
    # 4. FEATURES DERIVADAS
    # ------------------------------------------------------------------ #

    # Lags: demanda de las semanas anteriores
    for lag in [1, 2, 3, 4]:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # Media móvil de las últimas 4 semanas
    df["rolling_mean_4"] = df["demand"].shift(1).rolling(window=4).mean()

    # Desviación estándar de las últimas 4 semanas
    df["rolling_std_4"] = df["demand"].shift(1).rolling(window=4).std()

    # Feature de interacción: demanda actual × lead time de esa semana
    df["demand_x_leadtime"] = df["demand"] * df["lead_time"]

    # ------------------------------------------------------------------ #
    # 5. TARGET: demanda acumulada de las próximas 5 semanas
    # ------------------------------------------------------------------ #

    # Suma de demand en t+1, t+2, t+3, t+4, t+5 (horizonte = media del lead time)
    df["target"] = sum(df["demand"].shift(-i) for i in range(1, 6))

    # ------------------------------------------------------------------ #
    # 6. LIMPIEZA: eliminar filas con NaN (por lags y por el target al final)
    # ------------------------------------------------------------------ #

    df = df.dropna().reset_index(drop=True)

    # Convertir columnas enteras que pandas convirtió a float por los NaN
    int_cols = ["demand", "lead_time", "lag_1", "lag_2", "lag_3", "lag_4", "target"]
    df[int_cols] = df[int_cols].astype(int)

    return df


# ---------------------------------------------------------------------- #
# BLOQUE PRINCIPAL: ejecución directa del script
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    dataset = generate_dataset()

    print("=== Primeras 5 filas ===")
    print(dataset.head())

    print("\n=== Estadísticas descriptivas ===")
    print(dataset.describe())
