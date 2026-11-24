# Adaptación del dataset Bebop para NVIDIA GeoMapNet / MapNet

**Bitácora técnica del proceso**
**Fecha de corte:** 18 de noviembre de 2026
**Proyecto:** Estimación de pose 6-DoF de un dron Parrot Bebop 2
**Framework:** NVIDIA GeoMapNet / MapNet
**Origen de las poses:** COLMAP
**Hardware de entrenamiento:** NVIDIA RTX 3500 Ada Laptop GPU de 12 GB
**Entorno:** Conda `mapnet-py311`

---

# 1. Objetivo

El objetivo de este trabajo fue adaptar el dataset propio del proyecto de estimación de pose 6-DoF de un dron Parrot Bebop 2 para poder utilizarlo con la implementación de **MapNet incluida en NVIDIA GeoMapNet**.

El dataset original se había preparado para el proyecto de estimación de pose mediante una red neuronal propia. Sus imágenes estaban acompañadas de poses calculadas mediante COLMAP y transformadas al sistema de coordenadas físico del dron.

Sin embargo, GeoMapNet no utiliza directamente ese formato. La implementación original de MapNet espera:

- Imágenes organizadas por secuencia.
- Archivos de poses asociados a cada imagen.
- Poses expresadas como transformaciones **World-to-Camera (W2C)**.
- Archivos que indiquen qué secuencias pertenecen a entrenamiento y validación.
- Estadísticas de las poses.
- Estadísticas de normalización de imágenes RGB.
- Una estructura de datos compatible con sus propios `dataset_loaders`.

Por tanto, el trabajo consistió en adaptar el dataset al formato esperado por GeoMapNet, manteniendo la información geométrica original y evitando modificar innecesariamente la implementación oficial de MapNet.

## 1.1. Criterios que se mantuvieron durante la adaptación

Se establecieron las siguientes condiciones:

1. Utilizar exclusivamente imágenes que tengan una pose registrada por COLMAP.
2. No interpolar poses para imágenes que no tengan correspondencia.
3. Excluir las 15 imágenes de referencia utilizadas para la alineación de COLMAP.
4. Mantener las secuencias independientes para evitar formar clips que mezclen recorridos diferentes.
5. Convertir las poses al formato W2C esperado por MapNet.
6. Mantener la convención de coordenadas física FLU utilizada por el proyecto original.
7. Utilizar MapNet convencional, no MapNet++.
8. No utilizar odometría ni posicionamiento externo como entrada de la red.
9. Utilizar la implementación original del modelo y adaptar el dataset a ella.

---

# 2. Repositorios y estructura de trabajo

El repositorio utilizado fue:

```text
/data/research/mapnet/geomapnet/
```

El entorno Conda utilizado fue:

```text
mapnet-py311
```

Los datos originales y los datos preparados se organizaron fuera del repositorio:

```text
/data/research/mapnet/
├── bebop_raw/
├── bebop_data/
└── geomapnet/
```

La estructura relevante del repositorio GeoMapNet quedó organizada de la siguiente manera:

```text
geomapnet/
├── dataset_loaders/
│   ├── composite.py
│   ├── seven_scenes.py
│   ├── robotcar.py
│   └── bebop.py
│
├── scripts/
│   ├── configs/
│   │   ├── mapnet.ini
│   │   ├── mapnet_smoke.ini
│   │   ├── ...
│   │   ├── bebop_mapnet.ini
│   │   ├── bebop_mapnet_skip5.ini
│   │   ├── bebop_mapnet_skip20.ini
│   │   ├── bebop_mapnet_seed42.ini
│   │   └── bebop_mapnet_500.ini
│   │
│   ├── train.py
│   ├── eval.py
│   └── ...
│
└── ...
```

La ruta de datos que utiliza GeoMapNet para encontrar el dataset Bebop es:

```text
../data/deepslam_data/Bebop/bebop/
```

La ubicación relativa es importante porque el entrenamiento se ejecuta desde el directorio `scripts/`.

---

# 3. Dataset original de Bebop

## 3.1. Recorridos utilizados

El dataset se compone de cinco recorridos grabados con el Parrot Bebop 2.

Las imágenes se extrajeron de los videos y posteriormente se procesaron mediante COLMAP para obtener las poses de cámara.

La distribución original fue:

| Secuencia | Tipo | Frames extraídos | Imágenes registradas en COLMAP |
|---|---|---:|---:|
| `bebop_001_val` | Validación | 325 | 304 |
| `bebop_002_train` | Entrenamiento | 165 | 138 |
| `bebop_003_train` | Entrenamiento | 222 | 217 |
| `bebop_004_train` | Entrenamiento | 268 | 234 |
| `bebop_005_train` | Entrenamiento | 319 | 267 |
| **Total** | | **1299** | **1160** |

Por tanto:

- Se extrajeron 1299 imágenes.
- COLMAP registró 1160 imágenes.
- Las imágenes sin pose registrada quedaron fuera del dataset utilizable por MapNet.
- Las 15 imágenes de referencia también se excluyeron.

## 3.2. Distribución final

El conjunto quedó dividido de la siguiente manera:

| Conjunto | Secuencias | Imágenes |
|---|---|---:|
| Train | `bebop_002_train` | 138 |
| Train | `bebop_003_train` | 217 |
| Train | `bebop_004_train` | 234 |
| Train | `bebop_005_train` | 267 |
| **Train total** | **4 secuencias** | **856** |
| Validation/Test | `bebop_001_val` | **304** |
| **Total utilizable** | **5 secuencias** | **1160** |

La secuencia `bebop_001_val` se reservó para validación y evaluación.

Las otras cuatro secuencias se utilizaron para entrenamiento.

**Importante:** aunque el archivo se denomina `TestSplit.txt` en la configuración de datos, durante el entrenamiento se utiliza como conjunto de validación. Por tanto, sus resultados no deben confundirse automáticamente con una evaluación final independiente.

---

# 4. Convenciones de coordenadas

Esta fue una de las partes más importantes de la adaptación.

El proyecto original utiliza el sistema físico FLU del dron:

- X: Forward, hacia delante.
- Y: Left, hacia la izquierda.
- Z: Up, hacia arriba.

COLMAP utiliza un sistema de coordenadas de cámara óptica:

- X: hacia la derecha de la imagen.
- Y: hacia abajo de la imagen.
- Z: hacia delante, en dirección de observación.

Por tanto, no era correcto utilizar directamente las poses de cámara de COLMAP como si fueran poses del cuerpo del dron.

## 4.1. Transformación de cámara óptica a cuerpo FLU

Se utilizó la matriz:

```python
R_BODY_FROM_CAMERA = np.array([
    [0., 0., 1.],
    [-1., 0., 0.],
    [0., -1., 0.]
])
```

Esta matriz representa la transformación de orientación entre el sistema óptico de la cámara y el sistema FLU del cuerpo del dron.

En términos de ejes:

```text
Cámara óptica                  Cuerpo FLU

X cámara (derecha)   --->      -Y FLU
Y cámara (abajo)     --->      -Z FLU
Z cámara (adelante)  --->       X FLU
```

La matriz se utilizó al convertir las poses para mantener la orientación física correcta.

## 4.2. Transformación de poses

Las poses originales de COLMAP se habían alineado con las posiciones físicas de referencia y transformado al sistema FLU mediante el procesamiento posterior de COLMAP.

El archivo utilizado como fuente de poses fue el CSV generado por el pipeline del proyecto:

```text
poses_split.csv
```

Este archivo contiene la información de posición y orientación asociada a las imágenes, incluyendo:

- Nombre de imagen.
- Posición X, Y, Z.
- Cuaternión de orientación.
- Información de secuencia y partición.

El cuaternión se maneja en orden:

```text
qx, qy, qz, qw
```

Para MapNet fue necesario transformar estas poses al formato W2C.

## 4.3. ¿Por qué World-to-Camera?

La implementación original de GeoMapNet/MapNet procesa matrices de transformación World-to-Camera.

Una transformación de cámara a mundo y una transformación de mundo a cámara no son intercambiables.

Si se representa una transformación homogénea:

\[
T_{W\leftarrow C} =
\begin{bmatrix}
R_{W\leftarrow C} & t_{W\leftarrow C}\\
0 & 1
\end{bmatrix}
\]

su inversa es:

\[
T_{C\leftarrow W} =
T_{W\leftarrow C}^{-1}
=
\begin{bmatrix}
R^T & -R^Tt\\
0 & 1
\end{bmatrix}
\]

La matriz W2C es la que se necesita para expresar las coordenadas del mundo en el sistema de la cámara.

Durante la adaptación se verificó matemáticamente la transformación de las poses. No fue necesario invertir matrices adicionales ni modificar el código oficial de `process_poses()`.

---

# 5. Preparación de los archivos para GeoMapNet

## 5.1. Script de preparación

Se utilizó el proceso de preparación del dataset, implementado en:

```text
prepare_bebop.py
```

Su función fue generar la estructura y los archivos que GeoMapNet necesita a partir de los datos originales.

Entre las operaciones realizadas se encuentran:

1. Organizar las imágenes por secuencia.
2. Asociar cada imagen con su pose.
3. Transformar las poses al formato W2C.
4. Escribir los archivos `poses.txt`.
5. Generar los archivos de partición.
6. Mantener únicamente las imágenes registradas y válidas.

La estructura final esperada por GeoMapNet es:

```text
Bebop/
└── bebop/
    ├── bebop_001_val/
    │   ├── frame_XXXX.jpg
    │   └── ...
    │
    ├── bebop_002_train/
    │   ├── frame_XXXX.jpg
    │   └── ...
    │
    ├── bebop_003_train/
    │   ├── frame_XXXX.jpg
    │   └── ...
    │
    ├── bebop_004_train/
    │   ├── frame_XXXX.jpg
    │   └── ...
    │
    ├── bebop_005_train/
    │   ├── frame_XXXX.jpg
    │   └── ...
    │
    ├── TrainSplit.txt
    ├── TestSplit.txt
    ├── pose_stats.txt
    └── stats.txt
```

Los nombres concretos de las imágenes se conservan en los archivos de poses y en los archivos de partición.

## 5.2. Archivo `poses.txt`

Cada secuencia tiene su propio archivo:

```text
poses.txt
```

Cada línea contiene:

```text
nombre_imagen tx ty tz r11 r12 r13 r21 r22 r23 r31 r32 r33
```

Es decir:

```text
filename + 12 valores de la matriz W2C
```

La matriz se almacena utilizando sus primeras tres filas:

\[
T_{W2C} =
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x\\
r_{21} & r_{22} & r_{23} & t_y\\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
\]

La matriz completa tiene 3 filas y 4 columnas.

Los 12 valores se escriben en orden por filas (*row-major*).

Esto es importante porque el cargador original de GeoMapNet espera precisamente este formato.

### Decisión importante

No se incluyeron imágenes que carecían de pose.

Tampoco se generaron poses interpoladas.

Por tanto, el archivo `poses.txt` representa exclusivamente las imágenes para las cuales existe una transformación válida.

## 5.3. Archivos de partición

Se utilizaron:

```text
TrainSplit.txt
TestSplit.txt
```

El archivo `TrainSplit.txt` contiene las cuatro secuencias de entrenamiento:

```text
bebop_002_train
bebop_003_train
bebop_004_train
bebop_005_train
```

El archivo `TestSplit.txt` contiene:

```text
bebop_001_val
```

Las 15 imágenes de referencia no se incluyeron.

Esto fue suficiente para resolver el problema de las referencias: al no aparecer en `TrainSplit.txt` ni en las secuencias utilizadas por el cargador, no participan en el entrenamiento.

**No fue necesario volver a generar `poses_split.csv` ni repetir el procesamiento de COLMAP.**

---

# 6. Funcionamiento de `process_poses()` y representación interna de MapNet

Una vez que las poses W2C están disponibles, GeoMapNet las procesa mediante su implementación original.

La función:

```python
process_poses()
```

utiliza la matriz W2C y obtiene una representación de seis valores:

\[
[t_x,t_y,t_z,q_{\log x},q_{\log y},q_{\log z}]
\]

Los primeros tres valores corresponden a la traslación.

Los otros tres corresponden a la representación logarítmica de la rotación utilizada por la implementación.

Esto significa que MapNet no trabaja directamente con el vector de siete valores:

```text
[x, y, z, qx, qy, qz, qw]
```

que se utiliza en el proyecto original.

La representación interna de seis valores es parte del diseño de GeoMapNet.

**No se modificó `process_poses()` para introducir el formato de cuaterniones del proyecto original**, porque la adaptación correcta consiste en proporcionar las matrices W2C que la implementación espera.

---

# 7. Implementación del cargador `bebop.py`

Una vez preparado el dataset, se integró una clase específica para cargarlo.

Archivo:

```text
dataset_loaders/bebop.py
```

El objetivo fue adaptar el comportamiento de los cargadores existentes de GeoMapNet, manteniendo la estructura esperada por el resto del framework.

## 7.1. Responsabilidades del cargador

El cargador Bebop se encarga de:

- Leer `TrainSplit.txt` y `TestSplit.txt`.
- Localizar las secuencias correspondientes.
- Leer los archivos `poses.txt`.
- Asociar nombres de imágenes con sus matrices W2C.
- Procesar las poses mediante la lógica compatible con MapNet.
- Mantener los rangos de cada secuencia.
- Cargar las imágenes.
- Aplicar las transformaciones de imagen.
- Entregar los datos con las dimensiones esperadas por el modelo.

## 7.2. No mezclar secuencias

Un aspecto importante fue mantener los límites entre recorridos.

Las imágenes de:

```text
bebop_002_train
bebop_003_train
bebop_004_train
bebop_005_train
```

no deben mezclarse al formar clips.

Cada secuencia tiene su propio rango de imágenes.

Los rangos correspondientes al conjunto de entrenamiento son:

| Secuencia | Inicio | Fin exclusivo | Imágenes |
|---|---:|---:|---:|
| `bebop_002_train` | 0 | 138 | 138 |
| `bebop_003_train` | 138 | 355 | 217 |
| `bebop_004_train` | 355 | 589 | 234 |
| `bebop_005_train` | 589 | 856 | 267 |

Estos límites permiten que el cargador forme clips dentro de una secuencia sin conectar el último frame de un recorrido con el primero de otro.

Para validación se utiliza la secuencia independiente de 304 imágenes.

## 7.3. Parámetros de secuencia

La configuración principal utiliza:

```ini
steps = 3
skip = 10
variable_skip = no
```

Esto significa que MapNet trabaja con clips de tres imágenes y utiliza un salto de diez posiciones entre imágenes consecutivas del clip.

El comportamiento exacto de los clips depende de los límites de cada secuencia.

Por ejemplo, el cargador no debe formar un clip que empiece en `bebop_002_train` y termine en `bebop_003_train`.

---

# 8. Integración con `composite.py`

GeoMapNet utiliza un cargador compuesto para combinar los datasets disponibles.

Archivo:

```text
dataset_loaders/composite.py
```

Se integró Bebop en este mecanismo para que el entrenamiento pudiera utilizarlo de manera similar a los datasets originales.

El objetivo fue que el resto del pipeline pudiera solicitar:

```text
Dataset = Bebop
Scene = bebop
```

sin tener que cambiar la arquitectura de entrenamiento.

La integración permite que el framework obtenga los conjuntos de entrenamiento y validación a través de su interfaz habitual.

## 8.1. Resultado de las pruebas del cargador

Se ejecutaron pruebas independientes antes de iniciar el entrenamiento.

Los resultados fueron:

| Prueba | Resultado |
|---|---|
| Carga del dataset Bebop | Correcta |
| Imágenes de entrenamiento | 856 |
| Imágenes de validación | 304 |
| Poses de entrenamiento | `(856, 6)` |
| Poses de validación | `(304, 6)` |
| Clips de entrenamiento | 776 |
| Clips de validación | 284 |
| Límites entre secuencias | Correctos |
| Carga de imágenes | Correcta |
| Forward de MapNet | Correcto |
| Backward | Correcto |
| Evaluación del criterio de pérdida | Correcta |

La prueba también verificó que los tensores de imágenes y poses tenían las dimensiones esperadas.

Durante la prueba inicial con imágenes de 224 × 224 se obtuvieron tensores como:

```text
Images: [2, 3, 3, 224, 224]
Targets: [2, 3, 6]
```

La interpretación es:

```text
[batch_size, steps, channels, height, width]
```

y:

```text
[batch_size, steps, pose_dimension]
```

En este caso:

- Batch de 2 clips.
- 3 imágenes por clip.
- 3 canales RGB.
- Imágenes de 224 × 224.
- 6 valores por pose.

**Resultado:** la integración de `bebop.py` con `composite.py` quedó funcional.

---

# 9. Estadísticas del dataset

GeoMapNet necesita estadísticas para normalizar tanto las imágenes como las poses.

Se generaron dos archivos diferentes:

```text
stats.txt
pose_stats.txt
```

No deben confundirse porque representan información distinta.

## 9.1. `pose_stats.txt`

Este archivo contiene las estadísticas de las poses de entrenamiento.

Se calcularon utilizando exclusivamente las 856 poses del conjunto Train.

Los valores obtenidos fueron:

### Media

```text
[-0.46176825, 0.87096691, -0.30300037]
```

### Desviación estándar

```text
[1.67781889, 0.57006439, 1.55531743]
```

Estas estadísticas corresponden a la representación de pose utilizada internamente por el cargador de MapNet, no a las coordenadas FLU originales de siete componentes.

Se utiliza la media y desviación estándar del conjunto de entrenamiento para normalizar las poses.

No se deben calcular las estadísticas utilizando validación, porque eso introduciría información del conjunto de evaluación en el proceso de entrenamiento.

## 9.2. `stats.txt`

Este archivo contiene las estadísticas de las imágenes RGB.

Se calculó a partir de las imágenes de entrenamiento.

Los valores obtenidos fueron:

### Media RGB

```text
0.5114418369 0.4854273536 0.4521671474
```

### Varianza RGB

```text
0.0406564952 0.0431380892 0.0448696309
```

La información corresponde a:

| Canal | Media | Varianza |
|---|---:|---:|
| R | 0.5114418369 | 0.0406564952 |
| G | 0.4854273536 | 0.0431380892 |
| B | 0.4521671474 | 0.0448696309 |

El entrenamiento utiliza estas estadísticas para normalizar las imágenes.

---

# 10. Integración de Bebop en `scripts/train.py`

Una vez que el cargador y las estadísticas funcionaron, se integró el dataset en el punto de entrada del entrenamiento.

Archivo:

```text
scripts/train.py
```

## 10.1. Problema inicial

Originalmente, el parser de GeoMapNet aceptaba los datasets:

```text
7Scenes
RobotCar
```

Bebop todavía no estaba contemplado.

Además, el entrenamiento cargaba las estadísticas utilizando una ruta relativa basada en el dataset y la escena.

Por tanto, era necesario agregar Bebop al mecanismo de selección y asegurar que sus estadísticas fueran localizadas correctamente.

## 10.2. Integración realizada

Se incorporó el reconocimiento del dataset:

```text
Bebop
```

La ruta de estadísticas utilizada es:

```text
../data/deepslam_data/Bebop/bebop/stats.txt
```

También se mantuvo la restricción de que Bebop se utiliza con:

```text
--model mapnet
```

No se integró Bebop con MapNet++.

La razón es que el trabajo de adaptación se realizó para la implementación convencional de MapNet y su correspondiente cargador.

## 10.3. Arquitectura utilizada

La configuración utiliza:

- MapNet.
- Backbone ResNet34.
- Pesos preentrenados.
- GPU CUDA.
- Criterio de pérdida original de MapNet.

No se sustituyó la arquitectura por una red propia.

Tampoco se modificó el funcionamiento del criterio de pérdida para adaptar artificialmente los resultados a Bebop.

---

# 11. Configuración inicial `bebop_mapnet.ini`

Se creó una configuración específica para Bebop:

```text
scripts/configs/bebop_mapnet.ini
```

La configuración se basó en la configuración original de MapNet y se adaptó al dataset.

El contenido relevante fue:

```ini
[training]
n_epochs = 300
batch_size = 20
do_val = yes
seed = 7
shuffle = yes
num_workers = 4
snapshot = 10
val_freq = 1
max_grad_norm = 0

[optimization]
opt = adam
lr = 1e-4
weight_decay = 0.0005

[logging]
visdom = no
print_freq = 20

[hyperparameters]
beta = -3.0
gamma = -3.0
dropout = 0.5
skip = 10
variable_skip = no
real = no
steps = 3
color_jitter = 0.0
```

## 11.1. Justificación de los parámetros

| Parámetro | Valor | Función |
|---|---:|---|
| `n_epochs` | 300 | Número de épocas |
| `batch_size` | 20 | Clips por batch |
| `do_val` | yes | Activar validación |
| `seed` | 7 | Semilla inicial |
| `shuffle` | yes | Mezclar datos |
| `num_workers` | 4 | Procesos de carga |
| `snapshot` | 10 | Guardar checkpoints cada 10 épocas |
| `val_freq` | 1 | Validar cada época |
| `opt` | adam | Optimizador |
| `lr` | 1e-4 | Learning rate |
| `weight_decay` | 0.0005 | Regularización |
| `beta` | -3.0 | Hiperparámetro del criterio MapNet |
| `gamma` | -3.0 | Hiperparámetro del criterio MapNet |
| `dropout` | 0.5 | Dropout |
| `skip` | 10 | Salto entre imágenes del clip |
| `steps` | 3 | Imágenes por clip |
| `variable_skip` | no | Salto fijo |
| `real` | no | Configuración de datos no real |
| `color_jitter` | 0.0 | Sin variación artificial de color |

### Nota sobre `color_jitter`

La configuración original de MapNet tenía un valor de `color_jitter` de 0.7.

Para el primer experimento de Bebop se decidió utilizar:

```ini
color_jitter = 0.0
```

De esta manera se comenzó con una configuración sin perturbaciones artificiales de color.

El objetivo era establecer una línea base antes de experimentar con otros parámetros.

---

# 12. Resolución de problemas durante la integración

Durante la adaptación aparecieron varios problemas de compatibilidad y configuración.

## 12.1. El cargador no reconocía el dataset Bebop

**Problema:**

La implementación original solo contemplaba los datasets de 7Scenes y RobotCar.

**Solución:**

Se creó `dataset_loaders/bebop.py` y se integró en `dataset_loaders/composite.py`.

Posteriormente se agregó Bebop al mecanismo de selección de `scripts/train.py`.

## 12.2. Diferencia entre las poses del proyecto y las de MapNet

**Problema:**

El proyecto original utilizaba posiciones y cuaterniones, mientras que MapNet espera matrices W2C.

**Solución:**

Se transformaron las poses al formato W2C y se generaron los archivos `poses.txt`.

Se mantuvo la función original `process_poses()`.

No se intentó alimentar directamente a MapNet con el formato de siete valores.

## 12.3. Imágenes sin poses registradas

**Problema:**

No todas las imágenes extraídas de los videos fueron registradas por COLMAP.

**Solución:**

Se decidió utilizar exclusivamente las imágenes con poses válidas.

No se interpolaron las poses faltantes.

Esto redujo el dataset de 1299 imágenes extraídas a 1160 imágenes utilizables.

## 12.4. Imágenes de referencia

**Problema:**

Las 15 imágenes de referencia utilizadas para la alineación de COLMAP no debían aparecer en el entrenamiento.

**Solución:**

Se excluyeron de `TrainSplit.txt` y de las secuencias que utiliza el cargador.

Una vez realizado esto, no fue necesario volver a generar los CSV ni repetir la preparación de poses.

## 12.5. Formación de clips entre secuencias

**Problema:**

MapNet trabaja con clips de imágenes consecutivas. Si no se respetan los límites de cada recorrido, podrían generarse clips que mezclen secuencias diferentes.

**Solución:**

Se mantuvieron los rangos de cada secuencia en el cargador Bebop.

La prueba confirmó que los clips se forman dentro de cada recorrido.

## 12.6. Estadísticas de poses

**Problema:**

El framework necesita estadísticas de las poses para su normalización.

**Solución:**

Se recalculó `pose_stats.txt` utilizando las poses del conjunto de entrenamiento.

Resultado:

```text
mean:
-0.46176825 0.87096691 -0.30300037

std:
1.67781889 0.57006439 1.55531743
```

## 12.7. Estadísticas RGB

**Problema:**

GeoMapNet espera un archivo `stats.txt` con las estadísticas de las imágenes.

**Solución:**

Se calcularon las estadísticas RGB del conjunto de entrenamiento.

Resultado:

```text
mean:
0.5114418369 0.4854273536 0.4521671474

variance:
0.0406564952 0.0431380892 0.0448696309
```

## 12.8. Diferencia de resolución entre las pruebas y el entrenamiento

Durante las pruebas iniciales del cargador se observaron imágenes de:

```text
224 × 224
```

Sin embargo, el pipeline de entrenamiento de GeoMapNet aplica su propia transformación:

```python
Resize(256)
```

Conservando la relación de aspecto de las imágenes, la resolución de entrada resulta aproximadamente:

```text
456 × 256
```

Por tanto, no debe interpretarse la prueba inicial con imágenes de 224 × 224 como la resolución definitiva de entrenamiento.

El modelo utiliza la transformación definida por el pipeline original.

## 12.9. Advertencias durante la ejecución

Durante las primeras ejecuciones aparecieron mensajes como:

```text
TORCH_MODEL_ZOO is deprecated
```

y:

```text
No se encontró crop_size.txt; se continuará sin él
```

Se identificaron como advertencias informativas.

No impidieron la carga del modelo ni el entrenamiento.

## 12.10. Ejecución desde el directorio incorrecto

Se identificó que la ejecución debía realizarse desde:

```text
/data/research/mapnet/geomapnet/scripts/
```

Esto es importante porque el framework utiliza rutas relativas e imports basados en el directorio de ejecución.

El comando correcto se ejecuta desde `scripts/`, no desde la raíz del repositorio.

---

# 13. Prueba final de integración antes del entrenamiento

Antes de iniciar el entrenamiento completo se comprobó que todo el pipeline funcionara.

Los resultados fueron:

```text
Train:
856 imágenes
776 clips

Validation:
304 imágenes
284 clips
```

Las dimensiones observadas durante las pruebas fueron:

```text
Images:
[2, 3, 3, 224, 224]

Targets:
[2, 3, 6]
```

También se comprobó:

- Lectura de imágenes.
- Lectura de poses.
- Conversión de poses.
- Normalización.
- Formación de clips.
- Separación de secuencias.
- Forward del modelo.
- Backward.
- Evaluación del criterio de pérdida.

**Conclusión:** el dataset Bebop quedó correctamente integrado en el pipeline de MapNet y estaba listo para entrenarse.

---

# 14. Primer entrenamiento de MapNet

## 14.1. Comando utilizado

El entrenamiento se ejecutó desde:

```bash
cd /data/research/mapnet/geomapnet/scripts
```

Comando:

```bash
python train.py \
    --dataset Bebop \
    --scene bebop \
    --config_file configs/bebop_mapnet.ini \
    --model mapnet \
    --device 0
```

La GPU utilizada fue:

```text
NVIDIA RTX 3500 Ada Laptop GPU
```

Con aproximadamente 12 GB de VRAM.

## 14.2. Directorio de resultados

El experimento inicial generó su directorio en:

```text
scripts/logs/Bebop_bebop_mapnet_bebop_mapnet/
```

El archivo de registro fue:

```text
scripts/logs/Bebop_bebop_mapnet_bebop_mapnet/log.txt
```

Los checkpoints se guardan en el directorio del experimento.

Los nombres siguen el formato:

```text
epoch_XXX.pth.tar
```

La configuración tiene:

```ini
snapshot = 10
```

Por lo tanto, se guardan checkpoints periódicos cada diez épocas, además del checkpoint final.

## 14.3. Primeros resultados de validación observados

Durante una ejecución inicial se observaron los siguientes valores de `val_loss`:

| Época | Validation loss |
|---:|---:|
| 0 | 6.869299 |
| 1 | 4.610822 |
| 2 | 2.415968 |
| 3 | 2.637463 |
| 4 | 1.904449 |
| 5 | 1.813661 |
| 6 | 1.947875 |
| 7 | 1.721691 |
| 8 | 1.658232 |
| 9 | 1.631327 |
| 10 | 1.623218 |
| 11 | 1.605870 |
| 12 | 1.554749 |
| 13 | 1.514855 |
| 14 | 1.539233 |
| 15 | 1.472930 |
| 16 | 1.453597 |
| 17 | 1.468567 |
| 18 | 1.452769 |
| 19 | 1.395638 |

El mejor valor observado hasta esa época fue:

```text
Epoch 19
val_loss = 1.395638
```

Estos valores documentan la evolución inicial de esa ejecución. No deben confundirse con una tabla de los mejores resultados finales de los cinco experimentos.

## 14.4. Loss negativo

Durante el entrenamiento también se observaron valores negativos en la pérdida de entrenamiento.

Esto no significa necesariamente que exista un error.

MapNet utiliza un criterio de pérdida propio con parámetros como:

```ini
beta = -3.0
gamma = -3.0
```

Por ello, la pérdida interna no debe interpretarse como un error cuadrático medio convencional que obligatoriamente tenga que ser positivo.

Lo importante es revisar el comportamiento de la validación y, posteriormente, evaluar las poses predichas mediante métricas geométricas.

---

# 15. Consideraciones de hardware y temperatura

Durante el entrenamiento se revisó el estado de la GPU mediante:

```bash
nvidia-smi
```

El equipo reportó:

```text
GPU: NVIDIA RTX 3500 Ada Laptop GPU
VRAM total: aproximadamente 12 GB
VRAM utilizada: aproximadamente 6838 MiB
GPU utilization: 100 %
Temperatura: 87 °C
Potencia: aproximadamente 90 W / 115 W
```

El proceso de entrenamiento utilizaba aproximadamente:

```text
6612 MiB
```

La GPU estaba trabajando a plena utilización.

Se identificó que 87 °C es una temperatura elevada para una ejecución prolongada.

Por eso se recomendó:

- Mantener el equipo conectado a la corriente.
- No permitir que entre en suspensión.
- Mantener una buena ventilación.
- Vigilar la temperatura.
- Evitar ejecutar varios entrenamientos simultáneamente.

La cola de experimentos se diseñó para ejecutar un entrenamiento a la vez, evitando competir por los recursos de la GPU.

---

# 16. Preparación de experimentos adicionales

Después de establecer la configuración base se decidió realizar experimentos para analizar la influencia de:

- El parámetro `skip`.
- La semilla aleatoria.
- El número de épocas.

Se prepararon cuatro configuraciones adicionales.

Todas se crearon a partir de:

```text
configs/bebop_mapnet.ini
```

## 16.1. Experimento con `skip = 5`

Comandos:

```bash
cp configs/bebop_mapnet.ini configs/bebop_mapnet_skip5.ini

sed -i \
's/^n_epochs = 300$/n_epochs = 150/; s/^skip = 10$/skip = 5/' \
configs/bebop_mapnet_skip5.ini
```

Configuración:

```text
Épocas: 150
Seed: 7
Skip: 5
Steps: 3
```

El menor salto produce clips con imágenes más cercanas temporalmente.

Número de clips:

```text
Train: 816
Validation: 294
```

## 16.2. Experimento con `skip = 20`

Comandos:

```bash
cp configs/bebop_mapnet.ini configs/bebop_mapnet_skip20.ini

sed -i \
's/^n_epochs = 300$/n_epochs = 150/; s/^skip = 10$/skip = 20/' \
configs/bebop_mapnet_skip20.ini
```

Configuración:

```text
Épocas: 150
Seed: 7
Skip: 20
Steps: 3
```

El mayor salto genera clips con una separación temporal mayor.

Número de clips:

```text
Train: 696
Validation: 264
```

## 16.3. Experimento con `seed = 42`

Comandos:

```bash
cp configs/bebop_mapnet.ini configs/bebop_mapnet_seed42.ini

sed -i \
's/^seed = 7$/seed = 42/' \
configs/bebop_mapnet_seed42.ini
```

Configuración:

```text
Épocas: 300
Seed: 42
Skip: 10
Steps: 3
```

Este experimento permite observar la variabilidad introducida por la semilla aleatoria.

El número de clips se mantiene:

```text
Train: 776
Validation: 284
```

## 16.4. Experimento de 500 épocas

Comandos:

```bash
cp configs/bebop_mapnet.ini configs/bebop_mapnet_500.ini

sed -i \
's/^n_epochs = 300$/n_epochs = 500/' \
configs/bebop_mapnet_500.ini
```

Configuración:

```text
Épocas: 500
Seed: 7
Skip: 10
Steps: 3
```

Este experimento permite observar si continuar entrenando durante más épocas mejora el desempeño o si aparece sobreajuste.

Número de clips:

```text
Train: 776
Validation: 284
```

## 16.5. Verificación de las configuraciones

Se revisaron los archivos con:

```bash
grep -E 'n_epochs|seed|skip|steps' configs/bebop_mapnet*.ini
```

Los valores preparados fueron:

| Configuración | Épocas | Seed | Skip | Steps |
|---|---:|---:|---:|---:|
| `bebop_mapnet.ini` | 300 | 7 | 10 | 3 |
| `bebop_mapnet_skip5.ini` | 150 | 7 | 5 | 3 |
| `bebop_mapnet_skip20.ini` | 150 | 7 | 20 | 3 |
| `bebop_mapnet_seed42.ini` | 300 | 42 | 10 | 3 |
| `bebop_mapnet_500.ini` | 500 | 7 | 10 | 3 |

---

# 17. Automatización de la cola de entrenamientos

Para evitar tener que iniciar manualmente cada experimento, se creó un script:

```text
scripts/run_bebop_queue.sh
```

El propósito fue ejecutar los experimentos secuencialmente.

## 17.1. Creación del script

Se utilizó:

```bash
nano run_bebop_queue.sh
```

Posteriormente:

```bash
chmod +x run_bebop_queue.sh
```

## 17.2. Lógica del script

El script realiza las siguientes operaciones:

1. Cambia al directorio `scripts/`.
2. Crea el directorio de control de la cola.
3. Guarda un registro de ejecución.
4. Espera a que termine el entrenamiento inicial.
5. Ejecuta `skip5`.
6. Ejecuta `skip20`.
7. Ejecuta `seed42`.
8. Ejecuta `extended500`.
9. Detiene la cola si algún entrenamiento termina con código distinto de cero.
10. Registra cuándo termina cada experimento.

La espera inicial utiliza el PID del entrenamiento original:

```bash
CURRENT_PID=1111721
```

La comprobación se realiza con:

```bash
while kill -0 "$CURRENT_PID" 2>/dev/null; do
    CMD=$(ps -p "$CURRENT_PID" -o args= 2>/dev/null)
    case "$CMD" in
        *"python train.py"*) sleep 30 ;;
        *) break ;;
    esac
done
```

Así se evita iniciar el siguiente experimento mientras el proceso original siga ejecutándose.

## 17.3. Función de ejecución

Cada experimento se ejecuta con el comando:

```bash
python train.py \
    --dataset Bebop \
    --scene bebop \
    --config_file "configs/$CONFIG" \
    --model mapnet \
    --device 0
```

La salida de consola se guarda en archivos independientes:

```text
logs/queue_control/skip5_console.log
logs/queue_control/skip20_console.log
logs/queue_control/seed42_console.log
logs/queue_control/extended500_console.log
```

El registro general se guarda en:

```text
logs/queue_control/queue.log
```

Y el registro de `nohup`:

```text
logs/queue_control/nohup.log
```

## 17.4. Error al iniciar la cola

El primer intento se realizó con:

```bash
nohup ./run_bebop_queue.sh \
    > logs/queue_control/nohup.log 2>&1 &
```

Pero se obtuvo:

```text
bash: logs/queue_control/nohup.log:
No existe el archivo o el directorio
```

La causa fue que el directorio:

```text
logs/queue_control/
```

todavía no existía.

Aunque el script contenía:

```bash
mkdir -p logs/queue_control
```

el shell intenta abrir el archivo de redirección antes de ejecutar el script.

Por eso, el `mkdir` interno no podía solucionar ese problema.

### Solución

Se creó el directorio antes de ejecutar `nohup`:

```bash
mkdir -p logs/queue_control
```

Después se inició nuevamente:

```bash
nohup ./run_bebop_queue.sh \
    > logs/queue_control/nohup.log 2>&1 &
```

Y se consultó el PID:

```bash
echo "QUEUE PID: $!"
```

El resultado fue:

```text
QUEUE PID: 1128757
```

## 17.5. Comprobación de la cola

Se revisó el registro:

```bash
cat logs/queue_control/queue.log
```

Resultado:

```text
========================================
Cola iniciada: mié 18 nov 2026 03:05:01 CST
Esperando al entrenamiento actual PID 1111721
```

También se verificaron los procesos:

```bash
ps -p 1111721,1127774 -o pid,stat,etime,cmd
```

Resultado:

```text
PID       STAT     ELAPSED   CMD
1111721   Sl+      26:36     python train.py ...
```

Esto confirmó que:

- El entrenamiento inicial seguía activo.
- La cola estaba esperando correctamente.
- El intento anterior fallido ya no estaba activo.

La cola se inició correctamente.

---

# 18. Resultado de los cinco entrenamientos

Posteriormente se confirmó que los cinco entrenamientos habían terminado correctamente.

La cola finalizó a las:

```text
08:17:41 CST
```

Los experimentos terminaron con código de salida:

```text
0
```

Esto confirma que los procesos finalizaron sin reportar errores de ejecución.

## 18.1. Resumen de los experimentos completados

| Experimento | Épocas | Seed | Skip | Train clips | Validation clips | Estado |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 300 | 7 | 10 | 776 | 284 | Terminado |
| Skip 5 | 150 | 7 | 5 | 816 | 294 | Terminado |
| Skip 20 | 150 | 7 | 20 | 696 | 264 | Terminado |
| Seed 42 | 300 | 42 | 10 | 776 | 284 | Terminado |
| Extended 500 | 500 | 7 | 10 | 776 | 284 | Terminado |

Todos utilizaron:

```text
Dataset: Bebop
Scene: bebop
Model: MapNet
Backbone: ResNet34
Optimizer: Adam
Learning rate: 1e-4
Weight decay: 0.0005
Batch size: 20
Steps: 3
```

También se mantuvieron:

```text
beta = -3.0
gamma = -3.0
dropout = 0.5
do_val = yes
val_freq = 1
```

## 18.2. Qué se puede concluir de esta etapa

Se completó una primera serie de cinco entrenamientos con diferentes configuraciones.

Los experimentos permiten estudiar tres aspectos:

### A. Influencia de `skip`

Comparación:

```text
skip = 5
skip = 10
skip = 20
```

Esta comparación permite investigar cómo afecta la separación temporal entre imágenes al aprendizaje de la pose.

### B. Influencia de la semilla

Comparación:

```text
seed = 7
seed = 42
```

Se puede estudiar la variabilidad del entrenamiento al cambiar la inicialización y otros procesos aleatorios.

### C. Influencia del número de épocas

Comparación:

```text
300 épocas
500 épocas
```

Permite investigar si entrenar durante más tiempo mejora la validación o produce sobreajuste.

**Importante:** que los cinco procesos hayan terminado con código 0 no significa que ya se haya identificado el mejor modelo. Para determinarlo hay que revisar las curvas de validación, seleccionar los checkpoints correspondientes y ejecutar la evaluación de poses.

No se dispone en esta bitácora de una tabla confirmada con la mejor época y el mejor `val_loss` final de cada uno de los cinco experimentos. Por ello, no se asignan valores inventados.

---

# 19. Archivos y resultados que se deben conservar

Los principales archivos de la adaptación son:

| Archivo | Función |
|---|---|
| `prepare_bebop.py` | Preparación del dataset y generación del formato compatible |
| `poses.txt` | Poses W2C por secuencia |
| `TrainSplit.txt` | Secuencias de entrenamiento |
| `TestSplit.txt` | Secuencia de validación |
| `pose_stats.txt` | Estadísticas de las poses |
| `stats.txt` | Estadísticas RGB |
| `dataset_loaders/bebop.py` | Cargador específico de Bebop |
| `dataset_loaders/composite.py` | Integración del cargador |
| `scripts/train.py` | Entrada del entrenamiento |
| `scripts/configs/bebop_mapnet.ini` | Configuración base |
| `scripts/configs/bebop_mapnet_skip5.ini` | Configuración skip 5 |
| `scripts/configs/bebop_mapnet_skip20.ini` | Configuración skip 20 |
| `scripts/configs/bebop_mapnet_seed42.ini` | Configuración seed 42 |
| `scripts/configs/bebop_mapnet_500.ini` | Configuración 500 épocas |
| `scripts/run_bebop_queue.sh` | Ejecución secuencial |

Los registros de entrenamiento se encuentran bajo:

```text
scripts/logs/
```

Los registros de la cola:

```text
scripts/logs/queue_control/
```

Los checkpoints se encuentran en los directorios individuales de cada experimento.

---

# 20. Estado final de la adaptación

A fecha del 18 de noviembre de 2026, el proceso de adaptación del dataset Bebop a MapNet puede considerarse completado.

| Etapa | Estado |
|---|---|
| Extracción de imágenes | Completada |
| Registro de imágenes mediante COLMAP | Completado |
| Selección de imágenes con pose válida | Completada |
| Exclusión de referencias | Completada |
| Conversión de poses a W2C | Completada |
| Generación de `poses.txt` | Completada |
| Generación de archivos de split | Completada |
| Implementación de `bebop.py` | Completada |
| Integración en `composite.py` | Completada |
| Generación de `pose_stats.txt` | Completada |
| Generación de `stats.txt` | Completada |
| Integración en `scripts/train.py` | Completada |
| Prueba del DataLoader | Superada |
| Prueba forward/backward | Superada |
| Configuración de MapNet para Bebop | Completada |
| Entrenamiento baseline | Completado |
| Entrenamiento skip 5 | Completado |
| Entrenamiento skip 20 | Completado |
| Entrenamiento seed 42 | Completado |
| Entrenamiento de 500 épocas | Completado |

---

# 21. Siguiente etapa: evaluación de los modelos

La siguiente fase ya no consiste en adaptar el dataset ni en volver a entrenar desde cero.

Ahora corresponde evaluar los modelos generados.

GeoMapNet dispone de:

```text
scripts/eval.py
```

La evaluación debe realizarse utilizando la configuración correspondiente a cada experimento y su checkpoint.

El objetivo es obtener predicciones de pose y compararlas con las poses de referencia.

## 21.1. Qué debe medirse

Para el proyecto de estimación de pose 6-DoF interesa obtener, como mínimo:

### Error de posición

\[
e_t = \|\mathbf{t}_{pred}-\mathbf{t}_{gt}\|_2
\]

Donde:

- \(\mathbf{t}_{pred}\) es la posición predicha.
- \(\mathbf{t}_{gt}\) es la posición de referencia.

Se debe reportar en metros.

Métricas recomendadas:

- Error medio.
- Error mediano.
- RMSE.
- Error máximo.

### Error de orientación

Para dos cuaterniones normalizados:

\[
q_{pred}, q_{gt}
\]

el error angular puede calcularse mediante:

\[
e_R =
2\arccos
\left(
\left|
q_{pred}\cdot q_{gt}
\right|
\right)
\]

El resultado se expresa en grados.

Métricas recomendadas:

- Error angular medio.
- Error angular mediano.
- Error angular máximo.

### Tiempo de inferencia

También conviene medir:

- Tiempo por imagen.
- FPS.
- Memoria utilizada.
- Tamaño del checkpoint.

## 21.2. Precaución con las métricas

MapNet utiliza una representación interna de pose diferente de la representación FLU del proyecto.

Por tanto, antes de calcular los errores finales es necesario asegurarse de que:

1. Las predicciones se transformen correctamente a la convención física del proyecto.
2. Las posiciones estén expresadas en las mismas unidades.
3. Las orientaciones utilicen el mismo marco de coordenadas.
4. Se respete el orden de los cuaterniones.
5. No se compare directamente una pose W2C contra una pose body-to-world.

La evaluación debe realizarse en un sistema de coordenadas común.

Además, la validación utilizada durante el entrenamiento no debe presentarse como un test independiente.

---

# 22. Conclusión general

La adaptación del dataset Bebop a NVIDIA GeoMapNet/MapNet requirió resolver principalmente tres problemas:

1. **Compatibilidad geométrica:** convertir las poses obtenidas mediante COLMAP y alineadas al sistema FLU al formato World-to-Camera que utiliza MapNet.
2. **Compatibilidad de datos:** implementar un cargador específico que leyera las secuencias, poses y particiones sin mezclar recorridos.
3. **Compatibilidad con el framework:** integrar el dataset en `composite.py` y `train.py`, generar las estadísticas necesarias y crear una configuración propia.

La solución consistió en adaptar el dataset y sus cargadores sin reemplazar la arquitectura original de MapNet.

El resultado es un dataset con:

```text
5 secuencias
1160 imágenes registradas
856 imágenes de entrenamiento
304 imágenes de validación
776 clips train con skip=10
284 clips validation con skip=10
```

El cargador fue probado, el modelo pudo realizar forward y backward, y se completaron cinco entrenamientos con diferentes configuraciones.

Por tanto, el trabajo de integración está terminado y el proyecto se encuentra en la etapa de análisis y evaluación de los modelos entrenados.

**La siguiente decisión importante debe basarse en los errores reales de posición y orientación, no únicamente en el valor de la función de pérdida de MapNet.**