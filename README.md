# Sistema de Detección de Emociones en Audio (SER) 🎙️🤖

Este repositorio aloja la documentación y el código fuente de la **Práctica 3: Detección de Emociones mediante Características Acústicas del Habla** para la asignatura de Procesamiento del Habla.

El objetivo principal es desarrollar un sistema capaz de clasificar el estado emocional de un hablante basándose exclusivamente en la **física del sonido** (intensidad, tono, MFCCs), sin utilizar transcripción de texto (ASR) ni procesamiento de lenguaje natural (NLP).

> [!important]
> Documentación: ver [aquí](docs/Documentacion.md)

***

## 📂 Estructura del Repositorio

```text
LM-Speech-Emotion-Recognition/
│
├── data/
│   ├── processed/             # CSVs con características extraídas (MFCCs)
│   │   ├── ssi_custom_features.xlsx  # Dataset de Entrenamiento (Source)
│   │   ├── testing_ravdess.csv      # Dataset de Control (RAVDESS)
│   │   └── real_tests.xlsx           # Dataset Experimental (Voces propias)
│   └── models/                 # modelos guardados de los flujos orange
├── src/                       # Código fuente Python
│   ├── training_extractor/    # Extractor para el dataset principal
│   │   ├── main.py
│   │   └── requirements.txt
│   │
│   └── Datasets_extractor.ipynb  # Cuaderno colab para crear los 3 datasets necesarios
│
├── orange_workflow/           # Flujos de Orange Data Mining (.ows)
│   ├── Practica3_3_Emociones_Agrupadas.ows    # Estrategia de agrupación (3 emociones)
│   ├── Practica3_4_Emociones.ows   # Estrategia de selección (4 emociones)
│   └── capturas/                        # Esquemas visuales del flujo
│
└── docs/                      # Documentación adicional y enunciados
    ├── Documentacion.md       # Documentación en formato Markdown
    ├── Documentacion.pdf         # Documentación en pdf
    └── assets/                 # Archivos de apoyo para documentación
```

***

## ⚙️ Metodología y Flujo de Trabajo

El proyecto implementa un pipeline híbrido que combina la extracción de características en Python con el modelado predictivo en Orange Data Mining.

### 1. Extracción de Características (Python)
Utilizando la librería `librosa`, transformamos los audios crudos (.wav) en vectores numéricos. La característica más determinante seleccionada ha sido los **MFCCs (Mel-frequency cepstral coefficients)**, calculando la media de 40 coeficientes por audio.

### 2. Estrategia de Validación (Orange Data Mining)
A diferencia de los enfoques tradicionales, hemos diseñado una validación en dos niveles para medir tanto la robustez estadística como la generalización real (**Cross-Corpus Validation**).

#### A. Validación Interna (Cross-Validation)
* **Objetivo:** Evitar el *sesgo de partición* y el sobreajuste al dataset de entrenamiento.
* **Método:** Utilizamos el widget *Test & Score* con **k-fold cross-validation** sobre el archivo `ssi_custom_features.csv`. Esto asegura que el modelo es estable matemáticamente dentro del dominio de datos original.

#### B. Validación Externa (Inferencia)
* **Objetivo:** Evaluar la capacidad del modelo para generalizar ante condiciones acústicas desconocidas (Domain Shift).
* **Método:** Utilizamos el widget *Predictions*. Entrenamos el modelo con la totalidad del dataset principal y lanzamos predicciones sobre dos fuentes externas:
    1.  **RAVDESS:** Dataset de actores profesionales (Audio limpio, actuación arquetípica).
    2.  **Voces Propias:** Grabaciones con equipo rudimentario (Micrófonos no profesionales, ruido ambiente) con el objetivo de validar el rendimiento ante la ausencia de acondicionamiento acústico (ruido de fondo, eco y micrófonos estándar).

***

## 🧪 Experimentación

Se han diseñado dos flujos de trabajo en Orange (`.ows`) para probar distintas hipótesis de modelado:

| Experimento | Archivo `.ows` | Descripción |
| :--- | :--- | :--- |
| **Estrategia de Agrupación** | `Practica_3.ows` | Se agrupan emociones semánticamente cercanas en **3 macrounidades**. Busca maximizar el *Accuracy* global reduciendo la granularidad del problema. |
| **Estrategia de Selección** | `Practica_3_4Emociones.ows` | Se filtran y conservan únicamente las **4 emociones básicas** (Ira, Tristeza, Felicidad, Neutral). Evalúa el rendimiento en el esquema estándar de Paul Ekman. |

***

## 📊 Discusión de Resultados

Tras el análisis de las matrices de confusión en ambas estrategias, hemos observado un fenómeno notable:

### La Paradoja de RAVDESS
El modelo obtiene un rendimiento significativamente superior en el dataset externo **RAVDESS** (88-94% de acierto) comparado con el propio dataset de entrenamiento (~70%) o las voces propias.

**Conclusiones técnicas:**
1.  **Arquetipos Emocionales:** El modelo ha aprendido eficazmente a detectar emociones "de caricatura" o de alta intensidad (propias de actores entrenados). Al ser RAVDESS un dataset de actuación extrema, las características acústicas son muy separables.
2.  **Gap de Producción:** El bajo rendimiento en las **voces propias** indica que el modelo es sensible a las condiciones del canal (micrófono, ruido) y a la falta de entrenamiento actoral de los sujetos de prueba.
3.  **No es Overfitting clásico:** El hecho de que funcione bien en RAVDESS descarta un sobreajuste simple; el modelo *sabe* detectar emociones, pero requiere una limpieza de señal y una expresividad que no siempre se da en entornos naturales.

***

## 🛠️ Instalación y Uso

### Obtención de datasets

Ejecutar scripts de python localmente o abrir el cuaderno de Google Colab

#### Local

1. **Requisitos de Python:**
   ```bash
   pip install librosa numpy pandas kagglehub
   ```

#### Google Colab


### **Ejecución de Orange:**
   * Instalar [Orange Data Mining](https://orangedatamining.com/).
   * Abrir los archivos `.ows` situados en `orange_workflow/`.
   * **Importante:** Es posible que debas re-vincular la ruta de los archivos CSV en los widgets "File" al descargarlos en tu máquina local.