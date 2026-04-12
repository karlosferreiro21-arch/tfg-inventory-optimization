# tfg-inventory-optimization

# Optimización de Inventarios con Machine Learning y Deep Reinforcement Learning

Trabajo Fin de Grado — Grado en Ingeniería de las Tecnologías Industriales  
Escuela Técnica Superior de Ingeniería, Universidad Loyola  
Autor: Carlos Ferreiro Vera | Tutor: Bernardo Ronquillo Japón

## Descripción

Este repositorio contiene el código fuente del TFG "Optimización de Inventarios 
en Cadenas de Suministro con Tiempos de Entrega Estocásticos: Un Estudio 
Comparativo entre Machine Learning Supervisado y Deep Reinforcement Learning (DRL)".

El proyecto implementa y compara dos enfoques de inteligencia artificial para 
la gestión de inventarios bajo incertidumbre en los plazos de entrega.

## Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `inventory_env.py` | Gemelo Digital — entorno de simulación Gymnasium |
| `data_generator.py` | Generador de datos sintéticos de demanda y lead times |
| `baseline.py` | Línea base LightGBM + política (s,S) |
| `train_agent.py` | Entrenamiento del agente PPO con Stable-Baselines3 |
| `evaluate.py` | Evaluación comparativa baseline vs. agente PPO |
| `robustness_analysis.py` | Análisis de robustez ante distintos escenarios de lead time |

## Orden de ejecución

```bash
# 1. Generar y explorar los datos sintéticos
python data_generator.py

# 2. Entrenar y evaluar la línea base
python baseline.py

# 3. Entrenar el agente PPO (tarda ~2 minutos)
python train_agent.py

# 4. Evaluación comparativa (100 episodios)
python evaluate.py

# 5. Análisis de robustez (3 escenarios)
python robustness_analysis.py
```

## Requisitos

```bash
pip install gymnasium stable-baselines3 lightgbm scikit-learn numpy pandas matplotlib
```

## Resultados principales

| Métrica | Baseline (s,S) | Agente PPO |
|---|---|---|
| Coste medio anual (€) | 6.804 | 3.858 |
| Nivel de servicio (%) | 96,6 | 75,1 |
| Reducción de coste | — | 43,3% |
