[README.md](https://github.com/user-attachments/files/23731814/README.md)
# Sistema de Detección de Fraude Crediticio e Inmobiliario

## ¿Qué hace este sistema?

Este proyecto detecta **solicitudes de crédito fraudulentas** antes de que sean aprobadas, identificando patrones sospechosos y comportamientos anómalos que podrían indicar intención de fraude desde el inicio.

### La diferencia clave
- **Impago normal**: Cliente legítimo que no puede pagar por circunstancias
- **Fraude**: Solicitud con datos falsos e intención de engañar desde el principio
- **Nuestro objetivo**: Detectar el fraude antes de otorgar el crédito

## Objetivos

### Objetivo Principal
Desarrollar un sistema de detección de fraude inmobiliario basado en técnicas de machine learning capaz de identificar patrones financieros anómalos y comportamientos fraudulentos a partir de datos crediticios reales.

### Objetivos Específicos
- Integrar y procesar múltiples fuentes de datos crediticios
- Implementar modelos de detección de anomalías (Autoencoder, LOF, Isolation Forest)
- Desarrollar análisis temporal con LSTM Autoencoder
- Crear un sistema de correlación interproducto (créditos-tarjetas)
- Evaluar y comparar diferentes algoritmos de machine learning

## Arquitectura del Sistema

El sistema está compuesto por cuatro módulos principales:

### 1. Autoencoder Denso
- **Propósito**: Detección de anomalías estructurales en solicitudes crediticias
- **Técnica**: Red neuronal autoencoder entrenada solo con casos legítimos
- **Resultado**: Identificación de inconsistencias financieras y documentales

### 2. Análisis de Correlación Cliente-Tarjeta
- **Propósito**: Detección de patrones fraudulentos interproducto
- **Técnica**: Análisis de correlaciones entre anomalías crediticias y fraude transaccional
- **Resultado**: Visión integral del riesgo financiero del cliente

### 3. LOF e Isolation Forest
- **Propósito**: Detección de outliers financieros globales y locales
- **Técnica**: Combinación de algoritmos de detección no supervisada
- **Resultado**: Identificación de comportamientos financieros extremos

### 4. LSTM Autoencoder
- **Propósito**: Análisis temporal de patrones de comportamiento
- **Técnica**: Red neuronal recurrente para secuencias temporales
- **Resultado**: Detección de anomalías dinámicas en el tiempo

## Resultados Principales

### Rendimiento de Modelos Supervisados
- **LightGBM**: ROC-AUC = 0.992
- **Random Forest**: ROC-AUC = 0.983
- **Regresión Logística**: ROC-AUC = 0.974

### Capacidades de Detección
- Identificación de más de 46,000 casos sospechosos (15% del dataset)
- Detección de inconsistencias temporales críticas en 55,374 casos
- Análisis de correlaciones interproducto con correlación r = 0.74

## Estructura del Repositorio

```
notebooks/
├── Autoencoder_Denso.ipynb              # Detección con autoencoder básico
├── Autoencoder_Denso_Categoricas.ipynb  # Autoencoder con variables categóricas
├── Deteccion_Ambiguedades.ipynb         # Reglas de negocio y ambigüedades
├── Experimento_LSTM.ipynb               # Análisis temporal con LSTM
├── IsolationFores.ipynb                 # Isolation Forest básico
├── IsolationForest_Categoricas.ipynb    # IF con variables categóricas
├── LabelSpreading.ipynb                 # Propagación de etiquetas
├── LabelSpreading_Categoricas.ipynb     # Label spreading con categóricas
├── LOF.ipynb                            # Local Outlier Factor básico
├── LOF_Categoricas.ipynb                # LOF con variables categóricas
├── Modelo_Hibrido_Fraude.ipynb          # Sistema híbrido integrado
├── Self-Training.ipynb                  # Aprendizaje semi-supervisado
├── Self-training_Categoricas.ipynb      # Self-training con categóricas
├── Sistema_Deteccion_Riesgo_Crediticio.ipynb  # Sistema final integrado
└── comparador_modelos.ipynb             # Comparación de todos los modelos
```

## Tecnologías Utilizadas

### Librerías de Machine Learning
- **scikit-learn**: Algoritmos de ML tradicionales
- **TensorFlow/Keras**: Redes neuronales profundas
- **LightGBM**: Gradient boosting optimizado
- **XGBoost**: Extreme gradient boosting

### Procesamiento de Datos
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Matplotlib/Seaborn**: Visualización

### Detección de Anomalías
- **Isolation Forest**: Detección de outliers globales
- **Local Outlier Factor**: Detección de outliers locales
- **Autoencoders**: Reconstrucción neuronal

## Dataset

### Fuente de Datos
- **Dataset**: Home Credit Default Risk (Kaggle)
- **Registros**: 307,511 solicitudes crediticias
- **Variables**: 122 características originales
- **Target**: Riesgo de incumplimiento (8.07% de casos positivos)

### Variables Clave
- Información demográfica y socioeconómica
- Historial crediticio y bancario
- Ratios financieros derivados
- Indicadores de riesgo temporal
- Variables de comportamiento transaccional

## Metodología

### Preprocesamiento
1. Limpieza y tratamiento de valores faltantes
2. Ingeniería de características (31 variables derivadas)
3. Normalización y escalado de datos
4. Tratamiento de variables categóricas

### Entrenamiento
1. División estratificada de datos (70/15/15)
2. Entrenamiento de modelos base con datos normales
3. Validación cruzada y optimización de hiperparámetros
4. Evaluación con múltiples métricas

### Evaluación
- **ROC-AUC**: Capacidad discriminativa general
- **Precision-Recall**: Rendimiento en clase minoritaria
- **Matriz de confusión**: Análisis de errores
- **Curvas de lift**: Valor empresarial del modelo

## Instalación y Uso

### Requisitos
```bash
pip install pandas numpy scikit-learn tensorflow lightgbm matplotlib seaborn
```

### Ejecución
1. Clonar el repositorio
2. Descargar el dataset de Home Credit Default Risk
3. Ejecutar los notebooks en el orden sugerido
4. Revisar los resultados en `comparador_modelos.ipynb`

## Resultados y Conclusiones

### Principales Hallazgos
- Los métodos no supervisados puros muestran limitaciones significativas (AUC ~0.50-0.62)
- El sistema híbrido integrado supera ampliamente estos resultados (AUC 0.73)
- Los modelos supervisados finales alcanzan rendimiento excepcional (AUC >0.97)
- La combinación de múltiples enfoques mejora la robustez del sistema

### Impacto Empresarial
- Reducción del 97% en revisiones manuales
- Detección temprana de patrones fraudulentos complejos
- Capacidad de análisis interproducto para visión 360° del cliente
- Sistema escalable para implementación en producción

## Contribuciones

Este proyecto demuestra la efectividad de combinar múltiples técnicas de machine learning para la detección de fraude financiero, proporcionando una solución integral que supera significativamente los enfoques tradicionales.

## Autor

**Joshua Gabriel López Pinto**
- Universidad San Ignacio de Loyola
- Ingeniería de Sistemas de Información & Ciencia de Datos
- Código: 2114058

## Licencia

Este proyecto es desarrollado con fines académicos como parte del curso de Aprendizaje Automático II.
