"""
train_agent.py
Entrenamiento de un agente de Deep Reinforcement Learning con PPO
sobre el entorno de inventario InventoryEnv.

Flujo:
  1. Crea y envuelve los entornos (entrenamiento y evaluación)
  2. Instancia el agente PPO con Stable-Baselines3
  3. Entrena durante 500 000 pasos con evaluación periódica (EvalCallback)
  4. Guarda el modelo final y genera la curva de aprendizaje
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results

# Importamos nuestro entorno de inventario
from inventory_env import InventoryEnv

# ====================================================================== #
# RUTAS DE SALIDA
# ====================================================================== #

# Directorio donde está este script (carpeta código)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Archivo de logs del Monitor (carpeta temporal dentro de código)
LOG_DIR = os.path.join(BASE_DIR, "monitor_logs")

# Rutas de los modelos guardados
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "ppo_inventory_agent")   # .zip lo añade SB3
BEST_MODEL_PATH  = os.path.join(BASE_DIR, "ppo_best_model")        # ídem

# Ruta de la figura de la curva de aprendizaje
CURVE_PATH = os.path.join(BASE_DIR, "curva_aprendizaje.png")


# ====================================================================== #
# 1. CREACIÓN DE ENTORNOS
# ====================================================================== #

def make_train_env(seed=42):
    """
    Crea el entorno de entrenamiento envuelto con Monitor.

    Monitor registra la recompensa acumulada y la duración de cada episodio
    en un fichero CSV dentro de LOG_DIR. Esto nos permite leer la curva
    de aprendizaje al terminar el entrenamiento.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    env = InventoryEnv()
    env = Monitor(env, filename=os.path.join(LOG_DIR, "train"))
    return env


def make_eval_env(seed=123):
    """
    Crea un entorno de evaluación separado (sin Monitor para no mezclar logs).
    Usamos una semilla distinta para que las evaluaciones sean independientes
    del entorno de entrenamiento.
    """
    env = InventoryEnv()
    return env


# ====================================================================== #
# 2. CONFIGURACIÓN Y ENTRENAMIENTO DEL AGENTE PPO
# ====================================================================== #

def train_ppo(total_timesteps=500_000):
    """
    Instancia el agente PPO, registra el callback de evaluación y entrena.

    Hiperparámetros PPO elegidos:
      - MlpPolicy      : red neuronal densa estándar (2 capas de 64 neuronas)
      - learning_rate  : velocidad de actualización de los pesos (3e-4 es el default de SB3)
      - n_steps        : pasos que recolecta por entorno antes de cada actualización
      - batch_size     : tamaño del mini-batch para la actualización por gradiente
      - n_epochs       : veces que reutiliza cada batch recolectado antes de descartarlo
      - gamma          : factor de descuento (cuánto valoramos recompensas futuras)
      - gae_lambda     : lambda para Generalized Advantage Estimation (reduce varianza)
      - clip_range     : recorte del ratio de política (evita actualizaciones demasiado grandes)
    """

    # -- Entornos --
    train_env = make_train_env(seed=42)
    eval_env  = make_eval_env(seed=123)

    # -- Agente PPO --
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=42,
    )

    # ------------------------------------------------------------------ #
    # Callback de evaluación periódica
    #
    # EvalCallback evalúa el agente cada eval_freq pasos usando n_eval_episodes
    # episodios completos en eval_env. Si la recompensa media mejora, guarda
    # automáticamente el modelo en best_model_path.
    # ------------------------------------------------------------------ #
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=BASE_DIR,   # guardará ppo_best_model.zip aquí
        log_path=LOG_DIR,                # log de evaluaciones en eval.npz
        eval_freq=10_000,                # evalúa cada 10 000 pasos
        n_eval_episodes=5,               # 5 episodios por evaluación
        deterministic=True,              # política determinista en evaluación
        render=False,
    )

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #
    print(f"\nIniciando entrenamiento PPO durante {total_timesteps:,} pasos...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    # -- Guardar el modelo final --
    model.save(FINAL_MODEL_PATH)
    print(f"\nModelo final guardado en: {FINAL_MODEL_PATH}.zip")

    # EvalCallback siempre guarda el mejor modelo como "best_model.zip".
    # Lo renombramos al nombre que especifica el proyecto.
    sb3_best = os.path.join(BASE_DIR, "best_model.zip")
    if os.path.exists(sb3_best):
        os.rename(sb3_best, BEST_MODEL_PATH + ".zip")
        print(f"Mejor modelo renombrado a: {BEST_MODEL_PATH}.zip")

    return model


# ====================================================================== #
# 3. CURVA DE APRENDIZAJE
# ====================================================================== #

def plot_learning_curve():
    """
    Lee los registros del Monitor, calcula una media móvil y guarda
    la figura de la curva de aprendizaje.

    El Monitor guarda un CSV con columnas [r, l, t]:
      r = recompensa acumulada del episodio
      l = duración del episodio (pasos)
      t = tiempo transcurrido (segundos)
    """

    # load_results de SB3 lee todos los CSVs del Monitor en el directorio dado
    results = load_results(LOG_DIR)
    rewards = results["r"].values   # recompensas brutas por episodio

    # Media móvil de ventana 50 para suavizar la curva
    window = 50
    if len(rewards) >= window:
        # np.convolve calcula la media móvil con kernel uniforme
        kernel       = np.ones(window) / window
        smooth_rewards = np.convolve(rewards, kernel, mode="valid")
        # El eje x de la media móvil empieza en el episodio 'window - 1'
        smooth_x = np.arange(window - 1, len(rewards))
    else:
        # Si hay pocos episodios, mostramos sólo los datos brutos
        smooth_rewards = rewards
        smooth_x = np.arange(len(rewards))

    # -- Figura --
    fig, ax = plt.subplots(figsize=(10, 5))

    # Recompensa bruta (gris claro, transparente para no tapar la media móvil)
    ax.plot(
        np.arange(len(rewards)), rewards,
        color="lightgray", linewidth=0.8, alpha=0.7, label="Recompensa por episodio",
    )

    # Media móvil (azul oscuro)
    ax.plot(
        smooth_x, smooth_rewards,
        color="steelblue", linewidth=2.0, label=f"Media móvil ({window} ep.)",
    )

    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa acumulada")
    ax.set_title("Curva de aprendizaje del agente PPO")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=150)
    plt.close(fig)   # liberar memoria

    print(f"Curva de aprendizaje guardada en: {CURVE_PATH}")


# ====================================================================== #
# BLOQUE PRINCIPAL
# ====================================================================== #

if __name__ == "__main__":

    # 1. Entrenar el agente PPO y guardar el modelo final
    model = train_ppo(total_timesteps=500_000)

    # 2. Generar y guardar la curva de aprendizaje
    plot_learning_curve()

    print("\nEntrenamiento completado. Modelo guardado en ppo_inventory_agent.zip")
