# Calorflow - ANCAP DataChallenge 2025

## 🔬 Proyecto de Predicción de PCI y H2

Sistema de Machine Learning para predecir PCI (Poder Calorífico Inferior) y H2 (contenido de hidrógeno) en procesos de refinería FCC y CCR.

## 📁 Estructura del Proyecto

```
calorflow/
├── backend/                # Backend API (FastAPI)
│   ├── app/
│   │   ├── api/           # Endpoints
│   │   ├── models/        # Schemas Pydantic
│   │   ├── services/      # Lógica de negocio
│   │   └── config.py      # Configuración
│   ├── main.py            # Aplicación principal
│   └── requirements.txt
│
├── frontend/              # Frontend (React + TypeScript + Vite)
│   ├── src/
│   │   ├── components/    # Componentes React
│   │   ├── services/      # Cliente API
│   │   └── App.tsx        # Aplicación principal
│   └── package.json
│
├── src/                   # Módulos de ML (compartidos)
│   ├── trainer.py         # Entrenamiento de modelos
│   ├── predictor.py       # Predicciones
│   ├── features.py        # Feature engineering
│   ├── data_utils.py      # Utilidades de datos
│   └── api/               # Helpers para API
│
├── data/                  # Datos del proyecto
│   ├── processed/         # Datos preprocesados
│   └── FCC - Cracking Catalítico/
│   └── CCR - Reforming Catalítico/
│
├── models/                # Modelos entrenados
│   ├── FCC/
│   └── CCR/
│
├── notebooks/             # Jupyter notebooks
│   └── train_competition.ipynb
│
├── config/                # Configuración compartida
├── logs/                  # Logs de la aplicación
├── docker-compose.yml     # Orquestación Docker
└── README.md
```

## 🚀 Inicio Rápido

### Opción 1: Con Docker (Recomendado)

```bash
# Iniciar todo el stack (backend + frontend)
docker-compose up --build

# Backend estará en: http://localhost:8000
# Frontend estará en: http://localhost:5173
```

### Opción 2: Desarrollo Local

#### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📊 Uso

### 1. Dashboard
Visualiza métricas y distribuciones de los datos de entrenamiento:
- Distribuciones de PCI y H2
- Series temporales
- Métricas de los modelos

### 2. Predicciones
Realiza predicciones desde archivos CSV:
- Selecciona proceso (FCC o CCR)
- Sube archivo CSV con datos operacionales
- Obtén predicciones de PCI y H2

### 3. Entrenamiento
Entrena nuevos modelos:
- Configura hiperparámetros
- Selecciona número de trials de Optuna
- Monitorea el progreso

## 🔧 API Endpoints

```
GET  /api/v1/models                       # Listar modelos disponibles
GET  /api/v1/metrics/{process}            # Obtener métricas
GET  /api/v1/visualizations/{process}     # Datos para visualizaciones

POST /api/v1/predict                      # Predicción desde JSON
POST /api/v1/predict/csv                  # Predicción desde CSV
POST /api/v1/train                        # Entrenar modelo
```

## 📈 Visualizaciones

El frontend incluye gráficas interactivas basadas en el notebook:
- Histogramas de distribución
- Series temporales
- Métricas de rendimiento
- Comparaciones FCC vs CCR

## 🛠️ Tecnologías

### Backend
- FastAPI
- Python 3.11+
- scikit-learn, XGBoost, LightGBM, CatBoost
- Pandas, NumPy

### Frontend
- React 18
- TypeScript
- Vite
- Plotly.js (gráficas)
- React Router

## 📝 Scripts Útiles

```bash
# Entrenar modelos desde línea de comandos
python train.py

# Ejecutar notebook de exploración
jupyter notebook notebooks/train_competition.ipynb

# Ver documentación de la API
# http://localhost:8000/docs
```

## 👥 Equipo

**Team Never be Frog**
- Felipe Cabrera
- Stefano Francolino

## 📄 Licencia

ANCAP DataChallenge 2025
