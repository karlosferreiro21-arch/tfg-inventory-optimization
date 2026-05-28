"""
evaluate.py
Evaluación comparativa entre la política baseline (s,S) + LightGBM
y el agente PPO entrenado con Deep Reinforcement Learning.

Flujo:
  1. Genera datos y entrena el modelo LightGBM para la baseline
  2. Carga el mejor modelo PPO guardado en ppo_best_model.zip
  3. Evalúa ambas políticas en 100 episodios independientes
  4. Imprime tabla comparativa y genera figura comparativa
  5. Guarda los resultados en un CSV
"""

import os
import ctypes

# ── FIX DE COMPATIBILIDAD macOS ──────────────────────────────────────── #
# LightGBM y PyTorch (usado internamente por SB3) cargan cada uno su      #
# propia versión de libomp (OpenMP). Tener dos runtimes de OpenMP en el   #
# mismo proceso produce un SIGSEGV en macOS.                               #
# Solución: precargar libomp UNA SOLA VEZ con ctypes antes de que         #
# cualquier librería lo haga, de modo que ambas reutilicen la misma copia. #
_LIBOMP = os.path.expanduser("~/libomp_tmp/libomp_lib/libomp.dylib")
if os.path.exists(_LIBOMP):
    ctypes.CDLL(_LIBOMP)
# ─────────────────────────────────────────────────────────────────────── #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

# Importamos los módulos del proyecto (misma carpeta)
from inventory_env import InventoryEnv
from data_generator import generate_dataset
from baseline import train_model, BaselinePolicy

# ====================================================================== #
# RUTAS
# ====================================================================== #

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, "ppo_best_model.zip")
FIGURE_PATH     = os.path.join(BASE_DIR, "comparativa.png")
CSV_PATH        = os.path.join(BASE_DIR, "resultados_comparativa.csv")


# ====================================================================== #
# 1. FUNCIÓN CENTRAL DE EVALUACIÓN
# ====================================================================== #

def evaluate_policy(policy_fn, n_episodes=100, base_seed=0):
    """
    Ejecuta n_episodes episodios completos y recoge métricas por episodio.

    Parámetros
    ----------
    policy_fn  : función con firma  action = policy_fn(obs, env)
                 - obs    : observación actual del entorno (array de 7 floats)
                 - env    : instancia de InventoryEnv (para acceder al estado interno)
                 - action : entero entre 0 y 20 (índice de acción del entorno)
    n_episodes : número de episodios a simular
    base_seed  : semilla del episodio i = base_seed + i, para reproducibilidad

    Retorna
    -------
    dict con listas:
      "costs"          : coste total acumulado de cada episodio
      "service_levels" : nivel de servicio (%) de cada episodio
      "n_orders"       : número de pedidos lanzados en cada episodio
    """

    costs, service_levels, n_orders_list = [], [], []

    env = InventoryEnv()   # reutilizamos la misma instancia entre episodios

    for i in range(n_episodes):

        # Cada episodio tiene su propia semilla → demanda y lead times distintos
        obs, _ = env.reset(seed=base_seed + i)

        episode_cost      = 0.0
        weeks_no_stockout = 0
        episode_orders    = 0
        done              = False

        while not done:
            # La política decide la acción a partir de la observación y el entorno
            action = policy_fn(obs, env)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_cost += info["cost"]

            if info["units_short"] == 0:
                weeks_no_stockout += 1

            if action > 0:
                episode_orders += 1

            done = terminated or truncated

        # Nivel de servicio: % de semanas sin rotura en este episodio
        service_pct = 100.0 * weeks_no_stockout / env.episode_length

        costs.append(episode_cost)
        service_levels.append(service_pct)
        n_orders_list.append(episode_orders)

    return {
        "costs":          costs,
        "service_levels": service_levels,
        "n_orders":       n_orders_list,
    }


# ====================================================================== #
# 2. ADAPTADORES DE POLÍTICA → FUNCIÓN COMPATIBLE CON evaluate_policy
# ====================================================================== #

def make_baseline_fn(policy):
    """
    Envuelve una instancia de BaselinePolicy en una función compatible
    con evaluate_policy.

    La observación del entorno (7 floats normalizados) no contiene
    directamente las features que necesita el modelo LightGBM, así que
    las reconstruimos a partir del estado interno del entorno (demand_history,
    week, stock) igual que hacemos en baseline.py.
    """

    def baseline_fn(obs, env):
        # Extraemos el historial de demanda del entorno
        # demand_history es deque [oldest, …, newest] con las 4 últimas demandas
        demands = list(env.demand_history)  # [lag_4, lag_3, lag_2, lag_1]

        lag_1 = demands[-1]
        lag_2 = demands[-2]
        lag_3 = demands[-3]
        lag_4 = demands[-4]

        rolling_mean_4 = np.mean(demands)
        rolling_std_4  = float(np.std(demands, ddof=1)) if len(set(demands)) > 1 else 0.0

        week_of_year      = float(env.week)
      # demand_x_leadtime usa la media del lead time (5) como proxy del valor real,
# que no está disponible en tiempo de inferencia. El generador usó el valor real;
# el sesgo resultante es pequeño dado que la media es 5 y la varianza del lead time [2,8] es moderada.
        demand_x_leadtime = lag_1 * policy.lead_time_mean  # proxy lag_1 × lead_time_mean

        features = [
            lag_1, lag_2, lag_3, lag_4,
            rolling_mean_4, rolling_std_4,
            week_of_year, demand_x_leadtime,
        ]

        # La política devuelve unidades a pedir; lo convertimos al índice de acción
        qty = policy.decide_order(env.stock, features)
        action = min(20, max(0, round(qty / 5)))
        return action

    return baseline_fn


def make_agent_fn(model):
    """
    Envuelve el modelo PPO en una función compatible con evaluate_policy.

    PPO recibe directamente la observación normalizada del entorno
    (el mismo vector de 7 floats que Gymnasium devuelve en reset/step),
    por lo que no necesitamos ninguna transformación adicional.
    """

    def agent_fn(obs, env):
        # deterministic=True → usamos la política greedy, sin exploración
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return agent_fn


# ====================================================================== #
# 3. VISUALIZACIÓN COMPARATIVA
# ====================================================================== #

def plot_comparison(baseline_results, agent_results):
    """
    Genera una figura con tres histogramas superpuestos (uno por métrica)
    y la guarda en FIGURE_PATH.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Evaluación comparativa: Baseline vs. Agente PPO (100 episodios)",
        fontsize=13,
    )

    # Colores fijos para las dos políticas
    COLOR_BASELINE = "darkorange"
    COLOR_AGENT    = "steelblue"
    ALPHA          = 0.6   # transparencia para ver la superposición

    # ------------------------------------------------------------------ #
    # Subplot 1: Coste total por episodio
    # ------------------------------------------------------------------ #
    ax = axes[0]
    ax.hist(baseline_results["costs"], bins=20, color=COLOR_BASELINE,
            alpha=ALPHA, label="Baseline (s,S)")
    ax.hist(agent_results["costs"],    bins=20, color=COLOR_AGENT,
            alpha=ALPHA, label="Agente PPO")
    ax.set_title("Coste total por episodio")
    ax.set_xlabel("Coste acumulado (€)")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # ------------------------------------------------------------------ #
    # Subplot 2: Nivel de servicio por episodio
    # ------------------------------------------------------------------ #
    ax = axes[1]
    ax.hist(baseline_results["service_levels"], bins=20, color=COLOR_BASELINE,
            alpha=ALPHA, label="Baseline (s,S)")
    ax.hist(agent_results["service_levels"],    bins=20, color=COLOR_AGENT,
            alpha=ALPHA, label="Agente PPO")
    ax.set_title("Nivel de servicio por episodio")
    ax.set_xlabel("Nivel de servicio (%)")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # ------------------------------------------------------------------ #
    # Subplot 3: Número de pedidos por episodio
    # ------------------------------------------------------------------ #
    ax = axes[2]
    ax.hist(baseline_results["n_orders"], bins=20, color=COLOR_BASELINE,
            alpha=ALPHA, label="Baseline (s,S)")
    ax.hist(agent_results["n_orders"],    bins=20, color=COLOR_AGENT,
            alpha=ALPHA, label="Agente PPO")
    ax.set_title("Pedidos lanzados por episodio")
    ax.set_xlabel("Número de pedidos")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Figura comparativa guardada en: {FIGURE_PATH}")


# ====================================================================== #
# 4. TABLA RESUMEN
# ====================================================================== #

def print_summary(baseline_results, agent_results):
    """
    Imprime una tabla comparativa con media ± desviación estándar
    de las tres métricas, más la reducción porcentual de coste.
    """

    def stats(values):
        arr = np.array(values)
        return arr.mean(), arr.std()

    b_cost_mean, b_cost_std   = stats(baseline_results["costs"])
    a_cost_mean, a_cost_std   = stats(agent_results["costs"])

    b_sl_mean, b_sl_std       = stats(baseline_results["service_levels"])
    a_sl_mean, a_sl_std       = stats(agent_results["service_levels"])

    b_ord_mean, _             = stats(baseline_results["n_orders"])
    a_ord_mean, _             = stats(agent_results["n_orders"])

    # Reducción de coste del agente respecto a la baseline (positivo = mejora)
    cost_reduction = 100.0 * (b_cost_mean - a_cost_mean) / b_cost_mean

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"{'MÉTRICA':<30} {'BASELINE':>10}  {'AGENTE PPO':>10}")
    print(sep)
    print(f"{'Coste medio (€)':<30} {b_cost_mean:>10.1f}  {a_cost_mean:>10.1f}")
    print(f"{'Desv. std. coste (€)':<30} {b_cost_std:>10.1f}  {a_cost_std:>10.1f}")
    print(f"{'Nivel de servicio medio (%)':<30} {b_sl_mean:>10.1f}  {a_sl_mean:>10.1f}")
    print(f"{'Desv. std. servicio (%)':<30} {b_sl_std:>10.1f}  {a_sl_std:>10.1f}")
    print(f"{'Pedidos medios por episodio':<30} {b_ord_mean:>10.1f}  {a_ord_mean:>10.1f}")
    print(sep)
    print(f"Reducción de coste del agente vs. baseline: {cost_reduction:+.1f} %")
    print(sep)


# ====================================================================== #
# BLOQUE PRINCIPAL
# ====================================================================== #

if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    # Paso 1: Preparar la política baseline
    # ------------------------------------------------------------------ #

    print("=== Preparando la política baseline ===")
    df    = generate_dataset()
    model_lgbm, _, _ = train_model(df)
    baseline_policy  = BaselinePolicy(model=model_lgbm)
    print()

    # ------------------------------------------------------------------ #
    # Paso 2: Cargar el agente PPO entrenado
    # ------------------------------------------------------------------ #

    print(f"=== Cargando modelo PPO desde {BEST_MODEL_PATH} ===")
    ppo_model = PPO.load(BEST_MODEL_PATH)
    print("Modelo PPO cargado.\n")

    # ------------------------------------------------------------------ #
    # Paso 3: Construir las funciones de política compatibles con evaluate_policy
    # ------------------------------------------------------------------ #

    baseline_fn = make_baseline_fn(baseline_policy)
    agent_fn    = make_agent_fn(ppo_model)

    # ------------------------------------------------------------------ #
    # Paso 4: Evaluación de ambas políticas en 100 episodios
    # ------------------------------------------------------------------ #

    print("=== Evaluando política baseline (100 episodios) ===")
    baseline_results = evaluate_policy(baseline_fn, n_episodes=100, base_seed=0)

    print("=== Evaluando agente PPO (100 episodios) ===")
    agent_results = evaluate_policy(agent_fn, n_episodes=100, base_seed=0)

    # Usamos base_seed=0 para ambas políticas: cada episodio i ve exactamente
    # la misma secuencia de demanda, lo que hace la comparación más justa.

    # ------------------------------------------------------------------ #
    # Paso 5: Resultados
    # ------------------------------------------------------------------ #

    print_summary(baseline_results, agent_results)
    plot_comparison(baseline_results, agent_results)

    # ------------------------------------------------------------------ #
    # Paso 6: Guardar CSV con resultados por episodio
    # ------------------------------------------------------------------ #

    results_df = pd.DataFrame({
        "episodio":          range(1, 101),
        "coste_baseline":    baseline_results["costs"],
        "servicio_baseline": baseline_results["service_levels"],
        "pedidos_baseline":  baseline_results["n_orders"],
        "coste_agente":      agent_results["costs"],
        "servicio_agente":   agent_results["service_levels"],
        "pedidos_agente":    agent_results["n_orders"],
    })

    results_df.to_csv(CSV_PATH, index=False)
    print(f"\nResultados por episodio guardados en: {CSV_PATH}")
