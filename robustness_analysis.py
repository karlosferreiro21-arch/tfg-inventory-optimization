"""
robustness_analysis.py
Análisis de robustez de ambas políticas ante distintos niveles de
variabilidad en el lead time.

Idea: mantenemos la media del lead time aproximadamente constante pero
aumentamos el rango (varianza), y medimos cómo cambia el rendimiento
de la baseline (s,S) y del agente PPO en cada escenario.

Escenarios:
  1. Baja variabilidad : lead time uniforme en [4, 6]  → media = 5,  rango = 2
  2. Media (base)      : lead time uniforme en [2, 8]  → media = 5,  rango = 6
  3. Alta variabilidad : lead time uniforme en [1, 12] → media = 6.5, rango = 11
"""

import os
import ctypes

# ── FIX OpenMP macOS (mismo que en evaluate.py) ───────────────────────── #
# Precargar libomp ANTES de importar LightGBM o PyTorch para evitar el    #
# conflicto entre los dos runtimes de OpenMP que causa SIGSEGV.            #
_LIBOMP = os.path.expanduser("~/libomp_tmp/libomp_lib/libomp.dylib")
if os.path.exists(_LIBOMP):
    ctypes.CDLL(_LIBOMP)
# ─────────────────────────────────────────────────────────────────────── #

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from inventory_env import InventoryEnv
from data_generator import generate_dataset
from baseline import train_model, BaselinePolicy

# ====================================================================== #
# RUTAS
# ====================================================================== #

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, "ppo_best_model.zip")
FIGURE_PATH     = os.path.join(BASE_DIR, "robustez.png")

# ====================================================================== #
# DEFINICIÓN DE ESCENARIOS
# ====================================================================== #

# Cada escenario es un diccionario con los parámetros que necesitamos
# para configurar el entorno y la baseline.
SCENARIOS = [
    {
        "name":           "Baja [4,6]",
        "lead_time_low":  4,
        "lead_time_high": 6,
        "lead_time_mean": 5.0,    # media del uniforme [4,6] = (4+6)/2
    },
    {
        "name":           "Media [2,8]",
        "lead_time_low":  2,
        "lead_time_high": 8,
        "lead_time_mean": 5.0,    # media del uniforme [2,8] = (2+8)/2
    },
    {
        "name":           "Alta [1,12]",
        "lead_time_low":  1,
        "lead_time_high": 12,
        "lead_time_mean": 6.5,    # media del uniforme [1,12] = (1+12)/2
    },
]


# ====================================================================== #
# FUNCIÓN DE EVALUACIÓN POR ESCENARIO
# ====================================================================== #

def evaluate_scenario(scenario, policy_fn, n_episodes=50):
    """
    Evalúa una política en un escenario concreto de lead time.

    El entorno se crea con los parámetros del escenario sobreescribiendo
    los atributos lead_time_low y lead_time_high de la instancia.

    Parámetros
    ----------
    scenario   : dict con 'lead_time_low', 'lead_time_high', 'lead_time_mean'
    policy_fn  : función  action = policy_fn(obs, env)
    n_episodes : número de episodios a simular

    Retorna
    -------
    dict con listas 'costs', 'service_levels', 'n_orders'
    """

    costs, service_levels, n_orders_list = [], [], []

    # Creamos el entorno y sobreescribimos los parámetros del escenario
    env = InventoryEnv()
    env.lead_time_low  = scenario["lead_time_low"]
    env.lead_time_high = scenario["lead_time_high"]

    for i in range(n_episodes):
        obs, _ = env.reset(seed=i)   # semilla i para reproducibilidad

        episode_cost      = 0.0
        weeks_no_stockout = 0
        episode_orders    = 0
        done              = False

        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_cost += info["cost"]
            if info["units_short"] == 0:
                weeks_no_stockout += 1
            if action > 0:
                episode_orders += 1

            done = terminated or truncated

        costs.append(episode_cost)
        service_levels.append(100.0 * weeks_no_stockout / env.episode_length)
        n_orders_list.append(episode_orders)

    return {
        "costs":          costs,
        "service_levels": service_levels,
        "n_orders":       n_orders_list,
    }


# ====================================================================== #
# ADAPTADORES DE POLÍTICA
# ====================================================================== #

def make_baseline_fn(scenario):
    """
    Construye la función de política baseline para un escenario dado.

    Para cada escenario:
      1. Generamos datos sintéticos con el rango de lead time del escenario
      2. Entrenamos un LGBMRegressor sobre esos datos
      3. Instanciamos BaselinePolicy con lead_time_mean del escenario

    Así la baseline siempre está calibrada para el escenario que evaluamos.
    """

    print(f"  Entrenando LightGBM para escenario '{scenario['name']}'...")

    # Generamos datos con el lead time correcto para este escenario
    df = generate_dataset(
        lead_time_low=scenario["lead_time_low"],
        lead_time_high=scenario["lead_time_high"],
    )

    # Entrenamos el modelo (silenciamos los prints de métricas con redirect)
    lgbm, _, _ = train_model(df)

    # Instanciamos la política con la media de lead time del escenario
    policy = BaselinePolicy(
        model=lgbm,
        lead_time_mean=scenario["lead_time_mean"],
    )

    # Construimos y devolvemos la función compatible con evaluate_scenario
    def baseline_fn(obs, env):
        demands = list(env.demand_history)   # [lag_4, lag_3, lag_2, lag_1]

        lag_1 = demands[-1]
        lag_2 = demands[-2]
        lag_3 = demands[-3]
        lag_4 = demands[-4]

        rolling_mean_4 = np.mean(demands)
        rolling_std_4  = float(np.std(demands, ddof=1)) if len(set(demands)) > 1 else 0.0

        week_of_year      = float(env.week)
        demand_x_leadtime = lag_1 * policy.lead_time_mean

        features = [
            lag_1, lag_2, lag_3, lag_4,
            rolling_mean_4, rolling_std_4,
            week_of_year, demand_x_leadtime,
        ]

        qty    = policy.decide_order(env.stock, features)
        action = min(20, max(0, round(qty / 5)))
        return action

    return baseline_fn


def make_agent_fn(model):
    """
    Envuelve el modelo PPO en una función compatible con evaluate_scenario.

    El mismo modelo se usa para los tres escenarios sin reentrenar:
    queremos ver cómo generaliza ante condiciones de lead time distintas
    a las del entrenamiento (escenario base [2,8]).
    """

    def agent_fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return agent_fn


# ====================================================================== #
# FIGURA DE RESULTADOS
# ====================================================================== #

def plot_robustness(all_results):
    """
    Genera una figura con dos subplots de barras agrupadas:
      - Subplot izquierdo : coste medio ± desv. std. por escenario
      - Subplot derecho   : nivel de servicio medio por escenario

    all_results : lista de dicts, uno por escenario, con claves
                  'scenario', 'baseline', 'agent'
    """

    scenario_names = [r["scenario"]["name"] for r in all_results]
    n_scenarios    = len(scenario_names)

    # Extraemos estadísticas
    b_cost_means = [np.mean(r["baseline"]["costs"])          for r in all_results]
    b_cost_stds  = [np.std(r["baseline"]["costs"])           for r in all_results]
    a_cost_means = [np.mean(r["agent"]["costs"])             for r in all_results]
    a_cost_stds  = [np.std(r["agent"]["costs"])              for r in all_results]

    b_sl_means   = [np.mean(r["baseline"]["service_levels"]) for r in all_results]
    a_sl_means   = [np.mean(r["agent"]["service_levels"])    for r in all_results]

    x      = np.arange(n_scenarios)   # posiciones en el eje x
    width  = 0.35                     # anchura de cada barra

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Análisis de robustez: impacto de la variabilidad del lead time",
        fontsize=13,
    )

    # ------------------------------------------------------------------ #
    # Subplot 1: Coste medio con barras de error
    # ------------------------------------------------------------------ #
    ax = axes[0]

    bars_b = ax.bar(x - width / 2, b_cost_means, width,
                    yerr=b_cost_stds, capsize=4,
                    color="darkorange", alpha=0.8, label="Baseline (s,S)")
    bars_a = ax.bar(x + width / 2, a_cost_means, width,
                    yerr=a_cost_stds, capsize=4,
                    color="steelblue",  alpha=0.8, label="Agente PPO")

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names)
    ax.set_xlabel("Escenario de lead time")
    ax.set_ylabel("Coste medio (€)")
    ax.set_title("Coste medio por escenario")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ------------------------------------------------------------------ #
    # Subplot 2: Nivel de servicio medio
    # ------------------------------------------------------------------ #
    ax = axes[1]

    ax.bar(x - width / 2, b_sl_means, width,
           color="darkorange", alpha=0.8, label="Baseline (s,S)")
    ax.bar(x + width / 2, a_sl_means, width,
           color="steelblue",  alpha=0.8, label="Agente PPO")

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names)
    ax.set_xlabel("Escenario de lead time")
    ax.set_ylabel("Nivel de servicio medio (%)")
    ax.set_title("Nivel de servicio medio por escenario")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"\nFigura guardada en: {FIGURE_PATH}")


# ====================================================================== #
# TABLA DE RESULTADOS
# ====================================================================== #

def print_table(all_results):
    """Imprime una tabla con los resultados de cada combinación política-escenario."""

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"{'ESCENARIO':<16} {'POLÍTICA':<16} {'COSTE MEDIO':>12} {'DESV.STD':>10} {'SERVICIO%':>10}")
    print(sep)

    for r in all_results:
        name = r["scenario"]["name"]

        b_cost = np.mean(r["baseline"]["costs"])
        b_std  = np.std(r["baseline"]["costs"])
        b_sl   = np.mean(r["baseline"]["service_levels"])

        a_cost = np.mean(r["agent"]["costs"])
        a_std  = np.std(r["agent"]["costs"])
        a_sl   = np.mean(r["agent"]["service_levels"])

        reduction = 100.0 * (b_cost - a_cost) / b_cost

        print(f"{name:<16} {'Baseline (s,S)':<16} {b_cost:>12.1f} {b_std:>10.1f} {b_sl:>9.1f}%")
        print(f"{'':<16} {'Agente PPO':<16} {a_cost:>12.1f} {a_std:>10.1f} {a_sl:>9.1f}%")
        print(f"{'':<16} {'Reducción coste':<16} {reduction:>+11.1f}%")
        print("-" * 72)

    print(sep)


# ====================================================================== #
# BLOQUE PRINCIPAL
# ====================================================================== #

if __name__ == "__main__":

    # Cargamos el modelo PPO una sola vez (se reutiliza en los 3 escenarios)
    print(f"Cargando modelo PPO desde {BEST_MODEL_PATH}...")
    ppo_model  = PPO.load(BEST_MODEL_PATH)
    agent_fn   = make_agent_fn(ppo_model)
    print("Modelo PPO cargado.\n")

    all_results = []

    for scenario in SCENARIOS:
        print(f"{'='*50}")
        print(f"Escenario: {scenario['name']}")
        print(f"{'='*50}")

        # Baseline: reentrenada con datos del escenario
        baseline_fn = make_baseline_fn(scenario)

        print(f"  Evaluando baseline ({50} episodios)...")
        baseline_res = evaluate_scenario(scenario, baseline_fn, n_episodes=50)

        # PPO: mismo modelo para los tres escenarios
        print(f"  Evaluando agente PPO ({50} episodios)...")
        agent_res = evaluate_scenario(scenario, agent_fn, n_episodes=50)

        all_results.append({
            "scenario": scenario,
            "baseline": baseline_res,
            "agent":    agent_res,
        })

        print(f"  Baseline  → coste: {np.mean(baseline_res['costs']):.1f} € | "
              f"servicio: {np.mean(baseline_res['service_levels']):.1f}%")
        print(f"  Agente PPO→ coste: {np.mean(agent_res['costs']):.1f} € | "
              f"servicio: {np.mean(agent_res['service_levels']):.1f}%")
        print()

    # Tabla y figura
    print_table(all_results)
    plot_robustness(all_results)
