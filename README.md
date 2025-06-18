# TFM - Detección de Fraude Vehicular con Deep Learning

Este repositorio contiene el Trabajo de Fin de Máster (TFM) enfocado en la **detección automática de fraude en aperturas de cuentas bancarias** a partir de imágenes, usando técnicas de redes neuronales profundas y despliegue vía APIs.

---

## 📁 Estructura del Proyecto
├── API/ # Backend de inferencia
│ ├── inference_api.py # Código principal de la API
│ ├── Dockerfile # Imagen Docker para la API
│ ├── requirements.txt # Dependencias para la API
│ └── artifacts/ # Pesos del modelo entrenado y encoder
│ ├── model.pt
│ └── encoder.pkl
│ └── explainer.pkl
│
├── inference_API/ # Interfaz cliente para consumir la API
│ ├── web_app.py # Interfaz de usuario
│ ├── Dockerfile # Imagen Docker para frontend
│ └── requirements.txt # Dependencias para el frontend
│
├── tfm_modelo.ipynb # Notebook completo de entrenamiento
├── docker-compose.yml # Orquestación de los contenedores
└── README.md # Este archivo

## 🔍 Descripción

Este sistema está compuesto por:

- 🧠 **Modelo de detección de fraude/daños** entrenado con registros tabulares.
- 🔁 **API de inferencia** en FastAPI para servir el modelo.
- 💻 **Interfaz visual** que permite subir registros y obtener predicciones y explicaciones (con los explainers de SHAP) de las mismas en forma de lenguaje natural usando LLM.
- 🐳 Contenedores Docker para un despliegue completo con `docker-compose`.

---

## 🚀 Instrucciones de uso

### 🔧 Requisitos

- Docker
- Docker Compose

### 🐳 Despliegue con Docker Compose

Desde la raíz del proyecto:

```bash
docker-compose up --build
Esto levanta:

api: servicio de inferencia (API/)

inference_api: interfaz de usuario (inference_API/)

Accede a la interfaz desde tu navegador en http://localhost:8000 (o el puerto definido).

🏋️‍♀️ Entrenamiento del modelo
Todo el proceso de entrenamiento, validación y pruebas de hiperparámetros está documentado en:

📓 tfm_training.ipynb

Incluye:
Análisis exploratorio de variables
Pruebas de MLPs con entrenamiento en detección de fraude
Explicabilidad
Selección del final

Exportación de pesos (model.pt)
🧪 Artifacts
Los modelos entrenados y transformadores necesarios para la inferencia están guardados en:

📁 API/artifacts/

model.pt: pesos del modelo

encoder.pkl: encoder para clases o preprocesamiento

explainer.pkl: explainer para la explicabilidad

📦 API Endpoints

POST /predict_fraud: recibe un registro y devuelve predicción

GET /docs: documentación interactiva para incluir el registro

