# **Bitácora técnica: registro de imágenes de test, filtrado de reconstrucción COLMAP y exportación de poses FLU — v2_1**

Proceso completado

COLMAP 3.9.1

TXT y BIN validados

El objetivo de este procedimiento fue incorporar las nuevas imágenes de test de `v2_1` a la reconstrucción 3D previamente obtenida con `v2_0`, sin reconstruir el mapa desde cero, obtener las poses de las imágenes registradas en coordenadas métricas FLU y generar una versión limpia de la reconstrucción que excluyera las imágenes correspondientes a `1er Intento`.

El resultado final es una reconstrucción con **2,115 imágenes registradas, 107,703 puntos 3D y un error medio de reproyección de 0.966145 px**. Además, se obtuvieron las poses de las 926 imágenes de test que lograron registrarse.

A continuación dejo la bitácora organizada en el orden del trabajo realizado.

# **1. Estado inicial: reconstrucción v2_0**

Antes de incorporar las nuevas imágenes, ya teníamos una reconstrucción COLMAP correspondiente al dataset `v2_0`.

**Ubicación del proyecto:**

```
cd /data/drone-6dof-pose-estimation
```

**Reconstrucción original:**

```
results_colmap/v2_0/
├── database.db
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── alignment/
    ├── similarity_transform_colmap_to_FLU.txt
    └── poses_drone_FLU.csv
```

La reconstrucción original tenía las siguientes características:

| Métrica                     | v2_0          |
| --------------------------- | ------------- |
| Cámaras                     | 1             |
| Imágenes registradas        | 1,189         |
| Puntos 3D                   | 107,703       |
| Observaciones               | 551,502       |
| Error medio de reproyección | 0.966145 px   |
| Resolución                  | 856 × 480     |
| Modelo de cámara            | SIMPLE_RADIAL |

La reconstrucción de `v2_0` fue nuestra referencia geométrica. La idea era utilizarla como mapa fijo para registrar las nuevas imágenes, evitando ejecutar nuevamente `mapper`.

**Decisión importante:** no queríamos modificar la geometría original ni generar otra reconstrucción independiente. Queríamos localizar las nuevas imágenes dentro del mapa ya existente.

# **2. Preparación del nuevo dataset v2_1**

Se incorporó un conjunto de 11 secuencias nuevas de test:

```
dataset/v2_1/
└── frames_all/
    ├── bebop_test_001/
    ├── bebop_test_002/
    ├── bebop_test_003/
    ├── bebop_test_004/
    ├── bebop_test_005/
    ├── bebop_test_006/
    ├── bebop_test_007/
    ├── bebop_test_008/
    ├── bebop_test_009/
    ├── bebop_test_010/
    └── bebop_test_011/
```

La inspección inicial dio los siguientes resultados:

| Característica                  | Resultado |
| ------------------------------- | --------- |
| Imágenes                        | 1,123     |
| Resolución                      | 856 × 480 |
| Secuencias                      | 11        |
| Imágenes con errores de lectura | 0         |

Las imágenes tenían la misma resolución que las utilizadas para construir el mapa `v2_0`, lo que permitió utilizar la cámara COLMAP existente.

# **3. Preparación de la base de datos para el registro**

Se preparó una base de datos independiente:

```
results_colmap/v2_1_registration/database.db
```

La intención fue conservar la base de datos original de `v2_0` y trabajar sobre una copia destinada al nuevo registro.

Se incorporaron las características de las imágenes de test y se realizó matching mediante el método basado en Vocabulary Tree.

COLMAP se encontraba instalado en su versión 3.9.1, sin soporte CUDA para sus operaciones SIFT, por lo que las operaciones de extracción y matching se realizaron con CPU.

La comprobación de la ayuda del registrador fue:

```
colmap image_registrator -h 2>&1 | tee \
    results_colmap/v2_1_registration/logs/image_registrator_help.txt
```

## **3.1. Resultado del matching**

El proceso terminó correctamente:

```
Matching image [1123/1123]
Elapsed time: 170.967 [minutes]
```

Es decir, aproximadamente 2 horas y 51 minutos.

La comprobación de las correspondencias verificadas dio:

| Tipo de correspondencia | Pares     |
| ----------------------- | --------- |
| Original–Original       | 1,919,820 |
| Test–Original           | 28,942    |
| Test–Test               | 62,012    |
| Total                   | 2,010,774 |

Además:

**Conectividad de las imágenes de test**

# **1,093 / 1,123**

97.33 % de las imágenes de test tenían al menos una correspondencia geométrica verificada con imágenes originales.

Esto fue importante porque demostró que la mayoría de las nuevas imágenes tenían correspondencias suficientes para intentar ubicarse dentro de la reconstrucción existente.

Hay que distinguir este resultado del registro final: tener correspondencias verificadas no garantiza que una imagen termine registrada en el modelo 3D.

# **4. Registro de las imágenes de test sobre el mapa v2_0**

Una vez preparada la base de datos, se ejecutó `image_registrator`.

La operación utilizó:

- La base de datos preparada para `v2_1`.
- El modelo sparse de `v2_0` como referencia.
- Una salida independiente para `v2_1`.

Las rutas utilizadas fueron:

```
Base de datos:
results_colmap/v2_1_registration/database.db

Modelo de referencia:
results_colmap/v2_0/sparse/0/

Salida del registro:
results_colmap/v2_1_registration/sparse/0/
```

La operación tuvo un primer intento fallido por un problema con el directorio de salida. Se corrigió la ruta y se volvió a ejecutar el registro, que terminó generando el modelo ampliado.

El resultado se guardó inicialmente en:

```
results_colmap/v2_1_registration/
```

Posteriormente, la carpeta se renombró a:

```
results_colmap/v2_1/
```

**Importante sobre la reproducibilidad:** el historial disponible confirma las rutas, el resultado y la operación realizada, pero no conserva íntegra la línea de comandos definitiva de `image_registrator` ni todos sus parámetros efectivos. Por eso no voy a presentar como comando ejecutado una combinación de parámetros que solamente se había propuesto durante la preparación.

## **4.1. Resultado del registro**

El modelo generado contenía:

| Procedencia                           | Imágenes  |
| ------------------------------------- | --------- |
| Imágenes originales de v2_0           | 1,189     |
| Imágenes adicionales de `1er Intento` | 161       |
| Nuevas imágenes de test registradas   | 926       |
| **Total**                             | **2,276** |

La tasa de registro de las nuevas imágenes de test fue:

\\[ \frac{926}{1123}\times100=82.46\\% \\]

**Resultado del registro de test**

# **926 / 1,123**

Imágenes registradas en el mapa v2_0.

Registradas: 82.46 %

No registradas: 17.54 %

## **4.2. Resultado por secuencia**

Este fue el resultado de las 11 secuencias:

| Secuencia      | Total     | Registradas | No registradas |
| -------------- | --------- | ----------- | -------------- |
| bebop_test_001 | 80        | 80          | 0              |
| bebop_test_002 | 91        | 91          | 0              |
| bebop_test_003 | 74        | 74          | 0              |
| bebop_test_004 | 140       | 111         | 29             |
| bebop_test_005 | 88        | 66          | 22             |
| bebop_test_006 | 57        | 43          | 14             |
| bebop_test_007 | 112       | 100         | 12             |
| bebop_test_008 | 119       | 86          | 33             |
| bebop_test_009 | 101       | 77          | 24             |
| bebop_test_010 | 115       | 88          | 27             |
| bebop_test_011 | 146       | 110         | 36             |
| **Total**      | **1,123** | **926**     | **197**        |

Las primeras tres secuencias tuvieron registro completo. En las demás hubo imágenes que no lograron localizarse en el mapa.

Esto no significa necesariamente que esas imágenes sean incorrectas: puede deberse a falta de correspondencias suficientes, cambios de perspectiva, desenfoque, diferencias de iluminación o falta de características visuales comunes con el mapa original.

# **5. Conversión del modelo registrado a TXT para inspección**

Después del registro se trabajó con una versión TXT para poder analizar los nombres y las rutas de todas las imágenes.

La carpeta utilizada fue:

```
results_colmap/v2_1/sparse/0_txt/
```

Esta exportación permitió inspeccionar el contenido de `images.txt` y distinguir las imágenes originales, las nuevas imágenes de test y las imágenes de `1er Intento`.

La clasificación se realizó con el script:

```
src/post_colmap/test_registration/analyze_v2_1_images.py
```

Su función fue analizar las rutas completas de las imágenes registradas, no solamente sus nombres de archivo.

## **5.1. Resultado de la clasificación**

El análisis encontró:

| Categoría               | Imágenes  |
| ----------------------- | --------- |
| Originales de v2_0      | 1,189     |
| Nuevas imágenes de test | 926       |
| `1er Intento`           | 161       |
| Otras imágenes          | 0         |
| **Total**               | **2,276** |

Las 1,189 imágenes originales se componían de:

| Secuencia original | Imágenes  |
| ------------------ | --------- |
| bebop_001_val      | 305       |
| bebop_002_train    | 138       |
| bebop_003_train    | 217       |
| bebop_004_train    | 238       |
| bebop_005_train    | 276       |
| Referencias        | 15        |
| **Total**          | **1,189** |

Las 161 imágenes de `1er Intento` eran un conjunto que no queríamos conservar en la reconstrucción final.

Un detalle importante fue que algunos nombres de archivo coincidían con los de las secuencias válidas. Por eso no era suficiente filtrar únicamente por el nombre del archivo: había que comprobar la ruta completa.

# **6. Filtrado de la reconstrucción**

Esta fue una de las etapas más importantes del procedimiento.

El objetivo era generar un modelo que contuviera:

- Las 1,189 imágenes originales de `v2_0`.
- Las 926 imágenes nuevas de test registradas.
- Ninguna imagen de `1er Intento`.

Por tanto:

\\[ 1189+926=2115 \\]

El modelo final debía contener exactamente **2,115 imágenes**.

## **6.1. Script utilizado**

Se utilizó:

```
src/post_colmap/filter/filter_colmap_reconstruction.py
```

Configuración principal:

```
PROJECT_ROOT = Path(__file__).resolve().parents[3]INPUT_MODEL = (    PROJECT_ROOT / "results_colmap/v2_1/sparse/0_txt/")OUTPUT_DIR = (    PROJECT_ROOT / "results_colmap/v2_1/sparse/1_txt/")EXPECTED_IMAGES = 2115MIN_TRACK_LENGTH = 2
```

También se definieron los conteos esperados por secuencia:

```
VIDEO_COUNTS = {    "bebop_001_val": 305,    "bebop_002_train": 138,    "bebop_003_train": 217,    "bebop_004_train": 238,    "bebop_005_train": 276,    "bebop_test_001": 80,    "bebop_test_002": 91,    "bebop_test_003": 74,    "bebop_test_004": 111,    "bebop_test_005": 66,    "bebop_test_006": 43,    "bebop_test_007": 100,    "bebop_test_008": 86,    "bebop_test_009": 77,    "bebop_test_010": 88,    "bebop_test_011": 110,}
```

El patrón de selección fue:

```
IMAGE_PATTERN = re.compile(    r"^(?:ref_frame\d+|"    r"bebop_00[1-5]_(?:val|train)_frame_\d+|"    r"bebop_test_0\d+_frame_\d+)\.jpg$")
```

## **6.2. Exclusión explícita de&#x20;**`1er Intento`

Se modificó `select_images()` para comprobar primero la ruta completa:

```
def select_images(images):    selected = {}    counts = Counter()    for image_id, image in images.items():        full_name = image["name"]        path_parts = Path(full_name).parts        # Excluir explícitamente todas las imágenes de 1er Intento.        if "1er Intento" in path_parts:            counts["excluded_1er_intento"] += 1            continue        name = Path(full_name).name        if IMAGE_PATTERN.fullmatch(name):            selected[image_id] = image            if name.startswith("ref_frame"):                counts["references"] += 1            else:                video = "_".join(name.split("_")[:3])                counts[video] += 1    return selected, counts
```

Este cambio fue fundamental.

Si se hubiera utilizado solamente el nombre del archivo, algunas imágenes de `1er Intento` podrían haber pasado el filtro porque compartían patrones de nombres con las secuencias válidas.

## **6.3. Filtrado de puntos 3D**

Se definió un mínimo de dos observaciones por punto:

```
MIN_TRACK_LENGTH = 2
```

La función `filter_points()` fue:

```
def filter_points(points, selected_image_ids):    filtered = {}    removed = 0    for point_id, point in points.items():        valid_track = [            (image_id, point2D_idx)            for image_id, point2D_idx in point["track"]            if image_id in selected_image_ids        ]        if len(valid_track) >= MIN_TRACK_LENGTH:            filtered[point_id] = {                "prefix": point["prefix"],                "track": valid_track,            }        else:            removed += 1    return filtered, removed
```

La función realiza lo siguiente:

1. Recorre cada punto 3D.
2. Recupera sus observaciones.
3. Conserva únicamente las observaciones asociadas a las imágenes seleccionadas.
4. Comprueba que el track tenga al menos dos observaciones.
5. Conserva el punto si cumple la condición.
6. Descarta el punto si no cumple el mínimo.

Después se actualizaron las referencias de `images.txt`, asignando `-1` a los puntos 3D que ya no estuvieran presentes en el modelo filtrado.

También se realizaron validaciones de referencias cruzadas para comprobar que los índices de las observaciones fueran consistentes entre `images.txt` y `points3D.txt`.

## **6.4. Ejecución del filtro**

El script se ejecutó con:

```
python src/post_colmap/filter/filter_colmap_reconstruction.py
```

Los archivos generados fueron:

```
results_colmap/v2_1/
├── filtering_report.txt
└── sparse/
    ├── 0/
    ├── 0_txt/
    └── 1_txt/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt
```

## **6.5. Resultado del filtrado**

El reporte confirmó:

| Métrica                              | Resultado |
| ------------------------------------ | --------- |
| Imágenes originales                  | 2,276     |
| Imágenes seleccionadas               | 2,115     |
| Imágenes excluidas                   | 161       |
| Imágenes `1er Intento` excluidas     | 161       |
| Imágenes adicionales no clasificadas | 0         |
| Puntos 3D originales                 | 107,703   |
| Puntos 3D conservados                | 107,703   |
| Puntos 3D eliminados                 | 0         |
| Track mínimo                         | 2         |

El hecho de que no se eliminara ningún punto 3D significa que, después de excluir las imágenes no deseadas, todos los puntos conservaron al menos dos observaciones.

### **Comprobación de imágenes**

Se ejecutó:

```
grep -cE '^[0-9]+ ' \
    results_colmap/v2_1/sparse/1_txt/images.txt
```

Resultado:

```
2115
```

### **Comprobación de puntos 3D**

```
grep -cE '^[0-9]+ ' \
    results_colmap/v2_1/sparse/1_txt/points3D.txt
```

Resultado:

```
107703
```

### **Comprobación adicional de los tracks**

También se inspeccionó directamente la longitud de los tracks mediante AWK:

```
awk '
    $1 ~ /^[0-9]+$/ && NF >= 8 {
        n = (NF - 8) / 2
        hist[n]++

        if (n < 2) {
            bad++
            if (bad <= 10) {
                print "TRACK INFERIOR A 2:", $0
            }
        }

        total++
    }

    END {
        print "\n========== RESUMEN TRACKS =========="
        print "Total puntos:", total
        print "Puntos con 1 observación:", hist[1]
        print "Puntos con 2 observaciones:", hist[2]
        print "Puntos con menos de 2:", bad
    }
' results_colmap/v2_1/sparse/1_txt/points3D.txt
```

La salida mostró:

```
Total puntos: 107703
Puntos con 1 observación:
Puntos con 2 observaciones: 49991
Puntos con menos de 2:
```

Los valores vacíos de los contadores de una observación y menos de dos corresponden a cero.

Por tanto, el filtro cumplió su criterio.

# **7. Conversión de TXT a BIN**

Una vez validado el modelo TXT, se convirtió a formato binario de COLMAP.

La carpeta de salida debía existir previamente. En el primer intento, COLMAP falló porque no podía abrir `cameras.bin` dentro de un directorio que todavía no existía.

El error fue:

```
Check failed: file.is_open()
results_colmap/v2_1/sparse/1/cameras.bin
```

La solución fue crear el directorio:

```
mkdir -p results_colmap/v2_1/sparse/1
```

Después se ejecutó:

```
colmap model_converter \
    --input_path results_colmap/v2_1/sparse/1_txt \
    --output_path results_colmap/v2_1/sparse/1 \
    --output_type BIN
```

Esta vez la conversión terminó sin errores.

Los archivos generados fueron:

```
results_colmap/v2_1/sparse/1/
├── cameras.bin
├── images.bin
└── points3D.bin
```

Tamaños observados:

| Archivo      | Tamaño     |
| ------------ | ---------- |
| cameras.bin  | 64 bytes   |
| images.bin   | 132 MB     |
| points3D.bin | 12 MB      |
| **Total**    | **143 MB** |

# **8. Validación final de la reconstrucción BIN**

Para comprobar que el modelo binario era legible y contenía los datos correctos, se ejecutó:

```
colmap model_analyzer \
    --path results_colmap/v2_1/sparse/1
```

La salida obtenida fue:

```
Cameras: 1
Images: 2115
Registered images: 2115
Points: 107703
Observations: 760269
Mean track length: 7.058940
Mean observations per image: 359.465248
Mean reprojection error: 0.966145px
```

**Validación final**

Correcta

# **2,115**

Imágenes registradas

# **107,703**

Puntos 3D

# **760,269**

Observaciones

# **0.966145 px**

Error medio de reproyección

El modelo BIN es legible por COLMAP, tiene todas las imágenes seleccionadas registradas y conserva la calidad geométrica del modelo original.

# **9. Exportación de las poses de las nuevas imágenes de test a FLU y metros**

Esta etapa es independiente del filtrado del modelo.

El objetivo era obtener las coordenadas de las imágenes de test registradas en el sistema físico utilizado por el proyecto:

- X: adelante.
- Y: izquierda.
- Z: arriba.

Además, las posiciones debían expresarse en metros y las orientaciones mediante cuaterniones.

Se utilizó el script:

```
src/post_colmap/test_registration/export_test_poses.py
```

El archivo de salida se encuentra bajo:

```
results_colmap/v2_1/alignment/poses_test_FLU.csv
```

La carpeta originalmente se llamaba `v2_1_registration` y posteriormente se renombró a `v2_1`.

## **9.1. Transformación COLMAP a FLU**

Para mantener el mismo sistema de referencia físico de `v2_0`, se reutilizó la transformación de alineación previamente calculada con las 15 referencias.

La transformación es:

\\[ p\_{FLU}=sR\_{align}p\_{COLMAP}+t \\]

Donde:

- \\(s\\) es el factor de escala.
- \\(R\_{align}\\) es la rotación de alineación.
- \\(t\\) es la traslación.
- \\(p\_{COLMAP}\\) es la posición de la cámara en el sistema de COLMAP.
- \\(p\_{FLU}\\) es la posición en el sistema físico FLU.

La matriz de transformación guardada en el proyecto fue:

```
0.015268508489  0.020831721855  0.393915598933  1.277068304361
-0.394464332302 -0.000354928716 0.015308547818 -1.007751350712
0.001162006550 -0.394211238373 0.020802315991 0.794436357393
0.000000000000 0.000000000000 0.000000000000 1.000000000000
```

Esta transformación es especialmente importante porque permite que las poses nuevas sean comparables con las coordenadas físicas de las imágenes originales.

**No se debe volver a aplicar la transformación al CSV exportado**, porque las coordenadas ya están en metros y FLU.

## **9.2. Resultado de la exportación**

Se obtuvieron las poses de las **926 imágenes de test registradas**.

Las 197 imágenes que no lograron registrarse no tienen una pose COLMAP recuperable a través de este procedimiento, por lo que no forman parte de las poses exportadas.

El CSV contiene las posiciones y orientaciones de las imágenes registradas, con la salida de pose en el formato del proyecto:

\\[ [x,y,z,q_x,q_y,q_z,q_w] \\]

Donde:

- \\(x,y,z\\): posición en metros, sistema FLU.
- \\(q_x,q_y,q_z,q_w\\): orientación en cuaternión normalizado.

Esto permite utilizar las nuevas imágenes como datos de evaluación con coordenadas de referencia métricas.

# **10. Estructura final de resultados**

Después de completar el proceso, la organización relevante del proyecto quedó así:

```
drone-6dof-pose-estimation/
│
├── dataset/
│   └── v2_1/
│       └── frames_all/
│           ├── bebop_test_001/
│           ├── bebop_test_002/
│           ├── bebop_test_003/
│           ├── bebop_test_004/
│           ├── bebop_test_005/
│           ├── bebop_test_006/
│           ├── bebop_test_007/
│           ├── bebop_test_008/
│           ├── bebop_test_009/
│           ├── bebop_test_010/
│           └── bebop_test_011/
│
├── results_colmap/
│   │
│   ├── v2_0/
│   │   ├── database.db
│   │   ├── sparse/
│   │   │   └── 0/
│   │   └── alignment/
│   │       ├── similarity_transform_colmap_to_FLU.txt
│   │       └── poses_drone_FLU.csv
│   │
│   └── v2_1/
│       ├── database.db
│       ├── filtering_report.txt
│       │
│       ├── sparse/
│       │   ├── 0/
│       │   │   └── Modelo registrado original
│       │   │
│       │   ├── 0_txt/
│       │   │   └── Modelo registrado en TXT
│       │   │
│       │   ├── 1_txt/
│       │   │   ├── cameras.txt
│       │   │   ├── images.txt
│       │   │   └── points3D.txt
│       │   │
│       │   └── 1/
│       │       ├── cameras.bin
│       │       ├── images.bin
│       │       └── points3D.bin
│       │
│       └── alignment/
│           └── poses_test_FLU.csv
│
└── src/
    └── post_colmap/
        ├── filter/
        │   └── filter_colmap_reconstruction.py
        │
        └── test_registration/
            ├── analyze_v2_1_images.py
            └── export_test_poses.py
```

Esta estructura representa los archivos principales del flujo. Los directorios también pueden contener logs y otros archivos auxiliares.

# **11. Comparación general: antes y después**

Esta tabla resume el resultado de todo el procedimiento.

| Característica                  | v2_0 original           | v2_1 filtrado |
| ------------------------------- | ----------------------- | ------------- |
| Imágenes registradas            | 1,189                   | 2,115         |
| Puntos 3D                       | 107,703                 | 107,703       |
| Observaciones                   | 551,502                 | 760,269       |
| Error medio                     | 0.966145 px             | 0.966145 px   |
| Imágenes nuevas de test         | 0                       | 926           |
| Imágenes de `1er Intento`       | 0 en el modelo original | 0             |
| Poses nuevas de test exportadas | 0                       | 926           |

La ampliación de la reconstrucción permitió incorporar las nuevas imágenes sin cambiar la geometría del mapa original.

El número de observaciones aumentó porque ahora existen más imágenes asociadas a los mismos puntos 3D.

# **12. Comandos principales del procedimiento**

Esta es una recopilación de los comandos más importantes que se ejecutaron o utilizaron para inspeccionar y validar el resultado.

**Inspeccionar la ayuda del registrador:**

```
colmap image_registrator -h 2>&1 | tee \
    results_colmap/v2_1_registration/logs/image_registrator_help.txt
```

**Ejecutar el análisis de imágenes:**

```
python src/post_colmap/test_registration/analyze_v2_1_images.py
```

**Ejecutar el filtrado:**

```
python src/post_colmap/filter/filter_colmap_reconstruction.py
```

**Contar imágenes seleccionadas:**

```
grep -cE '^[0-9]+ ' \
    results_colmap/v2_1/sparse/1_txt/images.txt
```

**Contar puntos 3D:**

```
grep -cE '^[0-9]+ ' \
    results_colmap/v2_1/sparse/1_txt/points3D.txt
```

**Crear directorio BIN:**

```
mkdir -p results_colmap/v2_1/sparse/1
```

**Convertir TXT a BIN:**

```
colmap model_converter \
    --input_path results_colmap/v2_1/sparse/1_txt \
    --output_path results_colmap/v2_1/sparse/1 \
    --output_type BIN
```

**Analizar la reconstrucción final:**

```
colmap model_analyzer \
    --path results_colmap/v2_1/sparse/1
```

**Exportar poses de test:**

```
python src/post_colmap/test_registration/export_test_poses.py
```

# **13. Conclusiones técnicas**

El procedimiento permitió completar cuatro objetivos fundamentales.

**1. Extender el mapa original.**

Se incorporaron 926 imágenes nuevas de test a la reconstrucción de `v2_0`, utilizando `image_registrator` en lugar de reconstruir la escena desde cero.

**2. Obtener un conjunto de evaluación con poses de referencia.**

Las 926 imágenes registradas tienen coordenadas expresadas en metros y en el sistema FLU. Esto permite compararlas con las predicciones de los modelos de estimación de pose.

**3. Limpiar la reconstrucción.**

Se identificaron y excluyeron las 161 imágenes de `1er Intento`. El modelo final conserva las imágenes originales válidas y las imágenes nuevas de test registradas.

**4. Validar la integridad del modelo.**

La reconstrucción binaria fue analizada correctamente con COLMAP y conserva los 107,703 puntos 3D y el error medio de reproyección de 0.966145 px.

# **14. Estado del proyecto y siguiente etapa**

**Estado del flujo COLMAP**

1. **Reconstrucción v2_0**

   Completado

   Mapa original de 1,189 imágenes.
2. **Preparación y matching v2_1**

   Completado

   1,123 imágenes procesadas y 1,093 conectadas mediante correspondencias verificadas.
3. **Registro de nuevas imágenes**

   Completado

   926 imágenes registradas.
4. **Exportación de poses FLU**

   Completado

   Poses de test en metros y cuaterniones.
5. **Filtrado de reconstrucción**

   Completado

   2,115 imágenes, sin `1er Intento`.
6. **Conversión TXT a BIN y validación**

   Completado

   Modelo analizado correctamente con COLMAP.

## **¿Qué sigue para el proyecto de estimación de pose 6-DoF?**

La siguiente etapa natural es utilizar las poses nuevas como ground truth de evaluación para los modelos de aprendizaje profundo.

En particular, el conjunto de test `v2_1` puede utilizarse para:

1. Evaluar el modelo ResNet-18 entrenado con las imágenes de `v2_0`.
2. Evaluar MapNet/GeoMapNet sobre las mismas imágenes registradas.
3. Comparar error de posición en metros y error angular de orientación.
4. Analizar el desempeño por secuencia, no solamente con una métrica global.
5. Comparar los resultados de ambos métodos usando exactamente las mismas imágenes y poses de referencia.

**Una consideración metodológica:** las 926 imágenes de test tienen poses de referencia porque pudieron registrarse contra el mapa COLMAP. Las 197 restantes deben mantenerse identificadas como no registradas y no mezclarse con las métricas de evaluación que requieren ground truth.

En resumen, ya tienes una reconstrucción ampliada y filtrada, junto con un conjunto de 926 imágenes de test con poses métricas FLU. Esto constituye una base sólida para la evaluación cuantitativa de tu estimador de pose 6-DoF.