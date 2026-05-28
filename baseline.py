"""
baseline.py
Línea base de ML supervisado para gestión de inventario.

Flujo:
  1. Genera datos sintéticos con data_generator.py
  2. Entrena un LightGBM para predecir la demanda acumulada de las próximas 5 semanas
  3. Usa esa predicción dentro de una política (s, S) clásica
  4. Simula un episodio de 52 semanas en InventoryEnv y reporta métricas
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importamos el generador de datos y el entorno de inventario (misma carpeta)
from data_generator import generate_dataset
from inventory_env import InventoryEnv


# ====================================================================== #
# 1. ENTRENAMIENTO DEL MODELO
# ====================================================================== #

def train_model(df, test_size=0.2):
    """
    Entrena un LGBMRegressor para predecir la demanda acumulada de las
    próximas 5 semanas a partir de features históricas.

    Parámetros
    ----------
    df        : DataFrame devuelto por generate_dataset()
    test_size : fracción de filas reservadas para test (al final de la serie)

    Retorna
    -------
    model   : modelo LightGBM ya entrenado
    X_test  : features del conjunto de test
    y_test  : target real del conjunto de test
    """

    # ------------------------------------------------------------------ #
    # Definición de features y target
    # ------------------------------------------------------------------ #

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_4",   # demanda de semanas previas
        "rolling_mean_4",                        # media móvil últimas 4 semanas
        "rolling_std_4",                         # dispersión últimas 4 semanas
        "week_of_year",                          # posición en la serie
        "demand_x_leadtime",                     # interacción demanda × lead time
    ]
    target_col = "target"  # demanda acumulada de las próximas 5 semanas

    X = df[feature_cols]
    y = df[target_col]

    # ------------------------------------------------------------------ #
    # División temporal train / test (sin shuffle: respetar el orden)
    # ------------------------------------------------------------------ #

    split = int(len(df) * (1 - test_size))   # índice de corte
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

    # ------------------------------------------------------------------ #
    # Entrenamiento del modelo LightGBM
    # ------------------------------------------------------------------ #

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1,          # silenciar logs de LightGBM
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Evaluación sobre el conjunto de test
    # ------------------------------------------------------------------ #

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5   # sklearn>=1.4: sqrt(mse)

    print(f"MAE  en test: {mae:.2f} unidades")
    print(f"RMSE en test: {rmse:.2f} unidades")

    return model, X_test, y_test


# ====================================================================== #
# 2. POLÍTICA (s, S) ALIMENTADA POR EL MODELO
# ====================================================================== #

class BaselinePolicy:
    """
    Política clásica de inventario (s, S):
      - Si el stock actual cae por debajo del punto de reorden s, lanza
        un pedido hasta alcanzar el nivel objetivo S.
      - Los valores de s y S se calculan a partir de la predicción del
        modelo LightGBM para el horizonte de lead time.
    """

    def __init__(
        self,
        model,
        service_level=0.95,
        lead_time_mean=5,
        holding_cost=1.0,
        stockout_cost=10.0,
        fixed_order_cost=10.0,
    ):
        self.model            = model
        self.service_level    = service_level
        self.lead_time_mean   = lead_time_mean
        self.holding_cost     = holding_cost
        self.stockout_cost    = stockout_cost
        self.fixed_order_cost = fixed_order_cost

        # Factor z para nivel de servicio del 95 %
        self.z = 1.65

    # ------------------------------------------------------------------ #
    # Cálculo de los parámetros s y S
    # ------------------------------------------------------------------ #

    def compute_s_S(self, predicted_demand, demand_std):
        """
        Fórmulas estándar de la política (s, S):

          s = predicted_demand + z * demand_std * sqrt(lead_time_mean)
          S = s + predicted_demand

        predicted_demand : predicción del modelo (demanda acumulada en 5 semanas)
        demand_std       : desviación estándar reciente de la demanda (rolling_std_4)
        """

        # Punto de reorden: cubre la demanda esperada más el stock de seguridad
        s = predicted_demand + self.z * demand_std * np.sqrt(self.lead_time_mean)

        # Nivel objetivo: el punto de reorden más una cantidad adicional igual a la demanda esperada
        S = s + predicted_demand

        # Redondeamos y garantizamos que sean positivos
        s = max(1, int(round(s)))
        S = max(1, int(round(S)))

        return s, S

    # ------------------------------------------------------------------ #
    # Decisión de pedido
    # ------------------------------------------------------------------ #

    def decide_order(self, current_stock, features):
        """
        Dado el stock actual y el vector de features, genera la predicción
        del modelo y decide cuántas unidades pedir.

        Parámetros
        ----------
        current_stock : nivel de stock actual
        features      : array 1-D [lag_1, lag_2, lag_3, lag_4,
                                    rolling_mean_4, rolling_std_4,
                                    week_of_year, demand_x_leadtime]

        Retorna
        -------
        qty : número de unidades a pedir (0 si no hace falta reponer)
        """

        # Envolver en DataFrame con los nombres usados en el entrenamiento
        # para evitar warnings de sklearn sobre feature names
        feature_cols = [
            "lag_1", "lag_2", "lag_3", "lag_4",
            "rolling_mean_4", "rolling_std_4",
            "week_of_year", "demand_x_leadtime",
        ]
        features_df = pd.DataFrame([features], columns=feature_cols)

        # Predicción del modelo: demanda acumulada en las próximas 5 semanas
        predicted_demand = self.model.predict(features_df)[0]

        # Desviación estándar reciente (rolling_std_4 está en la posición 5)
        demand_std = features[5]

        # Calculamos el punto de reorden s y el nivel objetivo S
        s, S = self.compute_s_S(predicted_demand, demand_std)

        # Regla de la política (s, S)
        if current_stock < s:
            return max(0, S - int(current_stock))   # pedimos hasta alcanzar S
        else:
            return 0                                  # no es necesario pedir


# ====================================================================== #
# 3. EJECUCIÓN PRINCIPAL
# ====================================================================== #

if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    # Paso 1: Generar el dataset
    # ------------------------------------------------------------------ #

    print("=== Generando dataset ===")
    df = generate_dataset()
    print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas\n")

    # ------------------------------------------------------------------ #
    # Paso 2: Entrenar el modelo
    # ------------------------------------------------------------------ #

    print("=== Entrenando modelo LightGBM ===")
    model, X_test, y_test = train_model(df)
    print()

    # ------------------------------------------------------------------ #
    # Paso 3: Instanciar la política baseline
    # ------------------------------------------------------------------ #

    policy = BaselinePolicy(model=model)

    # ------------------------------------------------------------------ #
    # Paso 4: Simulación de 52 semanas en InventoryEnv
    # ------------------------------------------------------------------ #

    print("=== Simulando 52 semanas con la política baseline (s, S) ===")

    env = InventoryEnv()
    obs, _ = env.reset(seed=42)

    # Métricas a acumular durante la simulación
    total_cost       = 0.0
    weeks_no_stockout = 0
    n_orders          = 0
    done              = False

    while not done:

        # -- Construir el vector de features desde el estado del entorno --

        # demand_history es una deque [oldest, ..., newest] con las 4 últimas demandas
        demands = list(env.demand_history)   # [lag_4, lag_3, lag_2, lag_1]

        lag_1 = demands[-1]   # semana más reciente
        lag_2 = demands[-2]
        lag_3 = demands[-3]
        lag_4 = demands[-4]   # semana más antigua de las 4

        rolling_mean_4 = np.mean(demands)
        # ddof=1 para consistencia con pandas; si todas son iguales, std → 0
        rolling_std_4  = float(np.std(demands, ddof=1)) if len(set(demands)) > 1 else 0.0

        week_of_year       = float(env.week)

      # demand_x_leadtime usa la media del lead time (5) como proxy del valor real,
# que no está disponible en tiempo de inferencia. El generador usó el valor real;
# el sesgo resultante es pequeño dado que la media es 5 y la varianza del lead time [2,8] es moderada.
        demand_x_leadtime  = lag_1 * policy.lead_time_mean   # proxy: lag_1 × lead_time_mean

        features = [
            lag_1, lag_2, lag_3, lag_4,
            rolling_mean_4, rolling_std_4,
            week_of_year, demand_x_leadtime,
        ]

        # -- Decisión de la política: cantidad a pedir --

        qty_to_order = policy.decide_order(env.stock, features)

        # -- Convertir cantidad a índice de acción del entorno --
        # El entorno mapea acción i → i * 5 unidades (máximo 100, 21 opciones)
        action = min(20, max(0, round(qty_to_order / 5)))

        if action > 0:
            n_orders += 1

        # -- Ejecutar paso en el entorno --

        obs, reward, terminated, truncated, info = env.step(action)

        total_cost += info["cost"]

        # Contamos semana sin rotura si no hubo unidades no cubiertas
        if info["units_short"] == 0:
            weeks_no_stockout += 1

        done = terminated or truncated

    # ------------------------------------------------------------------ #
    # Paso 5: Resultados finales
    # ------------------------------------------------------------------ #

    service_level_pct = 100.0 * weeks_no_stockout / env.episode_length

    print(f"\n{'='*45}")
    print(f"Coste total acumulado : {total_cost:.2f} €")
    print(f"Nivel de servicio     : {service_level_pct:.1f} %  "
          f"({weeks_no_stockout}/{env.episode_length} semanas sin rotura)")
    print(f"Pedidos lanzados      : {n_orders}")
    print(f"{'='*45}")
