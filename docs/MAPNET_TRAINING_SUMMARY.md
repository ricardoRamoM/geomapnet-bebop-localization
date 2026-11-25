# Resumen general de entrenamientos MapNet — Parrot Bebop

## 1. Objetivo

Este documento recopila los experimentos de entrenamiento realizados con MapNet para la estimación de poses 6-DoF del dron Parrot Bebop.

Se incluyen los entrenamientos realizados sobre tres versiones del dataset:

- **v2_0**
- **v2_2**
- **v2_3**

El objetivo es comparar el desempeño de las diferentes configuraciones de entrenamiento, analizar los errores geométricos de posición y orientación, y estudiar el efecto de las distintas representaciones de las poses.

---

## 2. Aclaraciones sobre las versiones del dataset

Las tres versiones utilizan diferentes convenciones y procesos de representación de las poses.

| Versión | Convención | Descripción |
|---|---|---|
| v2_0 | W2C → C2W | Durante todo el proceso de entrenamiento se utilizan poses W2C. Posteriormente, las predicciones se convierten de W2C a C2W para obtener el resumen de evaluación presentado en `summary.csv`. |
| v2_2 | W2C | Las poses se representan mediante la transformación World-to-Camera. |
| v2_3 | C2W | Las poses se representan mediante la transformación Camera-to-World. |

**Importante:** Para v2_0, los resultados presentados corresponden a las poses después de realizar la conversión de W2C a C2W. Son los resultados recopilados en el archivo `summary.csv`.

Las convenciones indicadas para v2_2 y v2_3 son las utilizadas realmente en cada versión, independientemente de las etiquetas que pueda imprimir el código de evaluación.

---

# 3. Experimentos v2_0

## 3.1. Entrenamientos realizados

Se realizaron siete experimentos con diferentes configuraciones:

1. `baseline`
2. `500`
3. `lr5e-5`
4. `seed42`
5. `skip20`
6. `skip5`
7. `wd1e-4`

Los resultados se obtuvieron del archivo `summary.csv`.

El resumen contiene dos grupos de estadísticas:

- **All:** resultados considerando todas las imágenes evaluadas en cada experimento.
- **Common:** resultados considerando únicamente las imágenes compartidas entre los experimentos.

La comparación mediante imágenes comunes permite reducir las diferencias ocasionadas por utilizar distintas cantidades de imágenes durante la evaluación.

## 3.2. Resultados generales — All images

| Experimento | N | Pos. media (m) | Pos. mediana (m) | RMSE (m) | P95 (m) | Orient. media (°) | Orient. mediana (°) | P95 (°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 284 | 0.6455 | 0.5159 | 0.8262 | 1.5109 | 16.8080 | 9.0867 | 68.2083 |
| 500 | 284 | 0.6411 | 0.5053 | 0.8454 | 1.6118 | 16.8976 | 9.3150 | 87.3570 |
| lr5e-5 | 284 | 0.6723 | 0.5368 | 0.8657 | 1.6047 | 17.2992 | 10.4027 | 71.9262 |
| seed42 | 284 | 0.7279 | 0.5414 | 1.0132 | 2.2068 | 17.6409 | 9.4461 | 99.2671 |
| skip20 | 264 | 0.7538 | 0.5734 | 1.0061 | 2.3334 | 20.0015 | 9.9128 | 117.9302 |
| skip5 | 294 | 0.7562 | 0.5998 | 0.9387 | 1.9335 | 18.7620 | 9.7984 | 81.0421 |
| wd1e-4 | 284 | 0.6317 | 0.4956 | 0.8807 | 1.5725 | 17.9671 | 9.7571 | 94.8276 |

## 3.3. Resultados sobre imágenes comunes — Common images

| Experimento | N común | Pos. media (m) | Pos. mediana (m) | RMSE (m) | P95 (m) | Orient. media (°) | Orient. mediana (°) | P95 (°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 264 | 0.6616 | 0.5264 | 0.8476 | 1.5384 | 17.4570 | 9.2568 | 75.7808 |
| 500 | 264 | 0.6563 | 0.5096 | 0.8679 | 1.6646 | 17.5994 | 9.5130 | 96.6529 |
| lr5e-5 | 264 | 0.6892 | 0.5432 | 0.8879 | 1.6456 | 17.7895 | 10.3962 | 74.7047 |
| seed42 | 264 | 0.7517 | 0.5609 | 1.0441 | 2.3282 | 18.2783 | 9.4849 | 105.2210 |
| skip20 | 264 | 0.7538 | 0.5734 | 1.0061 | 2.3334 | 20.0015 | 9.9128 | 117.9302 |
| skip5 | 264 | 0.7757 | 0.6022 | 0.9663 | 2.1827 | 19.7590 | 9.7984 | 108.1235 |
| wd1e-4 | 264 | 0.6515 | 0.5143 | 0.9071 | 1.7539 | 18.6149 | 9.9206 | 106.1070 |

## 3.4. Mejores resultados v2_0

Considerando las imágenes comunes:

| Métrica | Mejor experimento | Resultado |
|---|---|---:|
| Posición media | wd1e-4 | 0.6515 m |
| Posición mediana | 500 | 0.5096 m |
| Posición RMSE | baseline | 0.8476 m |
| Posición P95 | baseline | 1.5384 m |
| Orientación media | baseline | 17.4570° |
| Orientación mediana | baseline | 9.2568° |
| Orientación P95 | lr5e-5 | 74.7047° |

**Observaciones:**

- `wd1e-4` consigue el menor error medio de posición.
- `500` obtiene la menor mediana de error de posición.
- `baseline` presenta el mejor RMSE y P95 de posición.
- `baseline` también consigue los mejores resultados de orientación media y mediana.
- `lr5e-5` consigue el menor P95 angular.
- Los experimentos `skip5` y `skip20` presentan resultados considerablemente peores en posición y orientación.
- El experimento `seed42` también muestra un deterioro respecto a baseline.

En general, v2_0 presenta errores geométricos superiores a los obtenidos posteriormente con v2_2 y v2_3.

---

# 4. Experimentos v2_2 — W2C

## 4.1. Entrenamientos realizados

Se realizaron seis ejecuciones de entrenamiento:

1. `bebop_mapnet_baseline`
2. `bebop_mapnet_500`
3. `bebop_mapnet_lr5e-5`
4. `bebop_mapnet_wd1e-4`
5. `bebop_mapnet_wd1e-4_2000`
6. `bebop_mapnet_smoke`

El experimento `bebop_mapnet_smoke` corresponde a una prueba de funcionamiento y no se considera un entrenamiento válido para la comparación de resultados.

Por lo tanto:

- Entrenamientos incluidos: 6.
- Entrenamientos con artefactos de evaluación: 5.
- Smoke tests excluidos: 1.

## 4.2. Resultados de evaluación

Todos los experimentos evaluados utilizan el split `val`, con 508 imágenes evaluadas y 508 imágenes comunes.

| Experimento | Imágenes | Pos. media (m) | RMSE (m) | P95 (m) | Orient. media (°) | P95 (°) | Best val_loss | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 508 | 0.4021 | 0.4760 | 0.9329 | 10.9574 | 24.4409 | 0.614461 | 277 |
| 500 | 508 | 0.3571 | 0.4172 | 0.7831 | 9.9699 | 27.3341 | 0.547697 | 495 |
| lr5e-5 | 508 | 0.4234 | 0.5004 | 0.9433 | 11.3278 | 34.4832 | 0.660161 | 290 |
| wd1e-4 | 508 | 0.4006 | 0.4638 | 0.8714 | 11.3753 | 26.9230 | 0.604704 | 296 |
| wd1e-4_2000 | 508 | 0.3477 | 0.4151 | 0.7985 | 8.3944 | 28.2830 | 0.554829 | 1311 |

## 4.3. Mejores resultados v2_2

| Métrica | Mejor experimento | Resultado |
|---|---|---:|
| Posición media | wd1e-4_2000 | 0.3477 m |
| Posición RMSE | wd1e-4_2000 | 0.4151 m |
| Posición P95 | 500 | 0.7831 m |
| Orientación media | wd1e-4_2000 | 8.3944° |
| Orientación P95 | baseline | 24.4409° |

**Observaciones:**

- `wd1e-4_2000` obtiene los mejores resultados de posición media, RMSE y orientación media.
- `500` consigue el menor P95 de posición.
- `baseline` presenta el menor P95 angular.
- `lr5e-5` es la configuración con los peores resultados generales.
- `wd1e-4` por sí solo no consigue superar consistentemente a baseline.
- La combinación de weight decay y el entrenamiento prolongado de `wd1e-4_2000` produce una mejora importante en la orientación.

El mejor resultado general de v2_2 corresponde a `bebop_mapnet_wd1e-4_2000`, especialmente por su error medio de orientación de 8.3944°.

---

# 5. Experimentos v2_3 — C2W

## 5.1. Entrenamientos realizados

Se realizaron cinco ejecuciones de entrenamiento:

1. `bebop_mapnet_baseline`
2. `bebop_mapnet_500`
3. `bebop_mapnet_lr5e-5`
4. `bebop_mapnet_wd1e-4`
5. `bebop_mapnet_wd1e-4_2000`

Se excluyó el smoke test `bebop_mapnet_queue_smoke`.

Por lo tanto:

- Entrenamientos incluidos: 5.
- Entrenamientos con artefactos de evaluación: 5.
- Smoke tests excluidos: 1.

## 5.2. Resultados de evaluación

Todos los experimentos evaluados utilizan el split `val`, con 508 imágenes evaluadas y 508 imágenes comunes.

| Experimento | Imágenes | Pos. media (m) | RMSE (m) | P95 (m) | Orient. media (°) | P95 (°) | Best val_loss | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 508 | 0.3475 | 0.4150 | 0.8006 | 12.8885 | 45.7472 | 0.670173 | 290 |
| 500 | 508 | 0.3322 | 0.4018 | 0.8283 | 10.9258 | 28.5743 | 0.622661 | 482 |
| lr5e-5 | 508 | 0.3769 | 0.4449 | 0.8332 | 12.7299 | 36.3677 | 0.728753 | 284 |
| wd1e-4 | 508 | 0.3681 | 0.4326 | 0.8391 | 12.0160 | 34.3833 | 0.678797 | 259 |
| wd1e-4_2000 | 508 | 0.3764 | 0.4468 | 0.8969 | 12.8091 | 43.8819 | 0.669946 | 232 |

## 5.3. Mejores resultados v2_3

| Métrica | Mejor experimento | Resultado |
|---|---|---:|
| Posición media | 500 | 0.3322 m |
| Posición RMSE | 500 | 0.4018 m |
| Posición P95 | baseline | 0.8006 m |
| Orientación media | 500 | 10.9258° |
| Orientación P95 | 500 | 28.5743° |

**Observaciones:**

- `500` consigue los mejores resultados en posición media, RMSE y orientación media.
- `baseline` presenta el menor P95 de posición.
- `500` también consigue el menor P95 de orientación.
- `lr5e-5` presenta los peores resultados de posición y el mayor valor de `val_loss`.
- `wd1e-4` no consigue superar a `500`.
- `wd1e-4_2000` tampoco consigue mejorar los resultados de `500`.

El mejor entrenamiento general de v2_3 corresponde a `bebop_mapnet_500`.

---

# 6. Comparación general entre versiones

Para comparar las tres versiones se consideran los mejores experimentos de cada una según sus resultados geométricos.

| Métrica | v2_0 | v2_2 W2C | v2_3 C2W |
|---|---:|---:|---:|
| Mejor posición media (m) | 0.6515 | 0.3477 | **0.3322** |
| Mejor posición mediana (m) | 0.5096 | — | — |
| Mejor posición RMSE (m) | 0.8476 | 0.4151 | **0.4018** |
| Mejor posición P95 (m) | 1.5384 | 0.7831 | **0.8006** |
| Mejor orientación media (°) | 17.4570 | **8.3944** | 10.9258 |
| Mejor orientación mediana (°) | 9.2568 | — | — |
| Mejor orientación P95 (°) | 74.7047 | **24.4409** | 28.5743 |

**Nota:** Para v2_0 se utilizan los resultados sobre imágenes comunes. Para v2_2 y v2_3 se utilizan las estadísticas de sus respectivos análisis sobre las 508 imágenes compartidas. Las medianas no están incluidas en los resúmenes de v2_2 y v2_3.

## 6.1. Mejores modelos identificados

| Versión | Mejor modelo | Principal fortaleza |
|---|---|---|
| v2_0 | baseline | Orientación media y RMSE de posición |
| v2_2 W2C | wd1e-4_2000 | Mejor orientación media y posición media |
| v2_3 C2W | 500 | Mejor posición media y RMSE |

## 6.2. Interpretación general

Los resultados muestran una mejora importante entre v2_0 y las versiones posteriores v2_2 y v2_3.

Las versiones v2_2 y v2_3 presentan errores medios de posición considerablemente menores y también reducen los errores de orientación.

Entre v2_2 y v2_3 se observan diferencias interesantes:

- **v2_3 (C2W)** consigue los mejores resultados de posición media y RMSE.
- **v2_2 (W2C)** consigue los mejores resultados de orientación media y P95 angular.
- Las diferencias entre ambas versiones no permiten establecer que una convención sea universalmente superior.
- Los resultados también dependen de la configuración de entrenamiento y de la optimización.

Es importante recordar que v2_0 utiliza un proceso diferente: las poses se entrenan en W2C y posteriormente se convierten a C2W para obtener las estadísticas de evaluación.

Por lo tanto, la comparación entre versiones debe interpretarse considerando tanto las convenciones de representación como los cambios realizados en el dataset y el pipeline.

---

# 7. Consideraciones sobre las métricas

Las métricas utilizadas tienen los siguientes significados:

| Métrica | Descripción |
|---|---|
| Position mean | Promedio del error euclidiano de posición en XYZ, expresado en metros. |
| Position median | Mediana del error euclidiano de posición. |
| Position RMSE | Raíz del error cuadrático medio de posición; penaliza más los errores grandes. |
| Position P95 | Percentil 95 del error de posición. |
| Orientation mean | Promedio del error angular de orientación, expresado en grados. |
| Orientation median | Mediana del error angular de orientación. |
| Orientation P95 | Percentil 95 del error angular de orientación. |
| Best val_loss | Menor pérdida de validación registrada durante el entrenamiento. |
| Best epoch | Época en la que se obtuvo el mejor `val_loss`. |

Las pérdidas de entrenamiento y validación son métricas de optimización de la red y no deben confundirse con los errores geométricos de posición y orientación.

Los resultados geométricos se obtienen de los artefactos generados por `eval.py` y de los análisis posteriores.

---

# 8. Trabajo pendiente y siguientes experimentos

Para continuar con el análisis de MapNet se identifican las siguientes tareas:

1. Evaluar checkpoints históricos para estudiar la evolución del error geométrico a lo largo del entrenamiento.
2. Analizar las trayectorias reales y predichas de los mejores modelos.
3. Comparar los errores por secuencia para identificar posibles zonas o recorridos problemáticos.
4. Evaluar la estabilidad de los mejores experimentos utilizando diferentes semillas.
5. Realizar el experimento equivalente `wd1e-4_2000` en v2_3 para completar la comparación con v2_2.
6. Verificar las conversiones W2C/C2W y asegurar que las evaluaciones comparativas utilizan exactamente el mismo sistema de coordenadas.

---

# 9. Conclusiones

Hasta el momento se han realizado experimentos con tres versiones del dataset Bebop y distintas configuraciones de entrenamiento de MapNet.

Los resultados más destacados son:

- **v2_0:** el modelo baseline presenta el mejor equilibrio general en las métricas de las imágenes comunes, aunque los errores siguen siendo relativamente elevados.
- **v2_2 (W2C):** el experimento `wd1e-4_2000` consigue el mejor error medio de orientación de todos los entrenamientos analizados: **8.3944°**.
- **v2_3 (C2W):** el experimento `500` consigue el mejor error medio de posición: **0.3322 m**, y el mejor RMSE: **0.4018 m**.

En conjunto, los resultados de v2_2 y v2_3 representan una mejora importante respecto a v2_0.

La comparación entre W2C y C2W todavía requiere experimentos adicionales y verificaciones del pipeline para determinar qué representación ofrece un comportamiento más consistente para la estimación de poses 6-DoF del Parrot Bebop.

**Mejores resultados actuales:**

- Mejor posición: v2_3 — `bebop_mapnet_500`.
- Mejor orientación: v2_2 — `bebop_mapnet_wd1e-4_2000`.

