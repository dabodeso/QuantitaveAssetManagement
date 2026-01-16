# Memoria del Análisis de Propiedades Estadísticas y de Series Temporales de ETFs

## 1. Introducción y Objetivo

Este documento presenta un análisis exhaustivo de las propiedades estadísticas y de series temporales de un conjunto de 14 ETFs que cubren diferentes clases de activos: acciones (EEUU e internacionales), bonos (corporativos, high yield y gubernamentales) y materias primas (oro, plata y petróleo).

El objetivo es caracterizar el comportamiento de cada activo mediante análisis cuantitativos que permitan:
- Entender las propiedades de estacionariedad y dependencia temporal
- Identificar patrones de volatilidad y clustering
- Evaluar la distribución de retornos (asimetría y curtosis)
- Detectar cambios de régimen y no linealidades
- Medir la exposición a factores de mercado

## 2. Metodología

### 2.1. Datos
- **Período**: 20 años de datos históricos
- **Frecuencia**: Retornos diarios
- **Fuente**: Yahoo Finance
- **ETFs analizados**: 14 activos

### 2.2. Tests y Métricas Utilizadas

#### Estacionariedad y Raíz Unitaria
- **Test ADF (Augmented Dickey-Fuller)**: Evalúa la hipótesis nula de raíz unitaria (no estacionariedad)
- **Test KPSS**: Evalúa la hipótesis nula de estacionariedad (test complementario)

#### Autocorrelación
- **ACF (Autocorrelación)**: Correlación entre retornos y sus valores rezagados
- **PACF (Autocorrelación Parcial)**: Correlación parcial controlando por lags intermedios
- **Test de Ljung-Box**: Evalúa la presencia de autocorrelación serial

#### Volatilidad y Clustering
- **Volatilidad anualizada**: Desviación estándar de retornos × √252
- **ARCH Effects**: Autocorrelación de retornos al cuadrado (proxy de clustering de volatilidad)
- **Ratio de persistencia**: Comparación de volatilidad entre dos períodos

#### Momentos Superiores
- **Skewness (Asimetría)**: Medida de la simetría de la distribución
- **Kurtosis (Curtosis)**: Medida del peso de las colas de la distribución
- **Exceso de Kurtosis**: Kurtosis - 3 (normal = 0)

#### Cambios de Régimen
- **Test t**: Comparación de medias entre dos períodos
- **Correlación no lineal**: Correlación entre retornos y retornos²

#### Exposición a Factores de Mercado
- **Modelo CAPM**: R_asset = α + β × R_market + ε
- **Beta (β)**: Sensibilidad al mercado
- **Alpha (α)**: Retorno excesivo no explicado por el mercado
- **R²**: Proporción de varianza explicada por el mercado

## 3. Resultados y Análisis por Categoría

### 3.1. Estacionariedad y Raíz Unitaria

#### Resultados Principales

**Todos los ETFs son estacionarios según el test ADF** (p-valores < 0.05), lo cual es esperado para retornos diarios. Los retornos financieros típicamente no presentan raíces unitarias.

**Excepciones en el test KPSS:**
- **GLD (Oro)**: p-value = 0.01 → No estacionario según KPSS
- **SLV (Plata)**: p-value = 0.023 → No estacionario según KPSS

#### Interpretación

Los metales preciosos (oro y plata) muestran evidencia de no estacionariedad según KPSS, lo que sugiere:
- Posible presencia de tendencias de largo plazo
- Cambios estructurales en el comportamiento de precios
- Mayor persistencia en los movimientos de precios

Esto es consistente con el comportamiento de los metales preciosos como activos refugio que pueden experimentar ciclos de largo plazo.

### 3.2. Autocorrelación y Autocorrelación Parcial

#### Resultados Principales

**Acciones (SPY, QQQ, IWM, EFA, EEM, VGK, VPL):**
- ACF lag-1: aproximadamente -0.10 (autocorrelación negativa)
- Presencia de autocorrelación serial según Ljung-Box (p < 0.05)

**Bonos (LQD, HYG, TLT, AGG):**
- ACF lag-1: muy cercano a 0 (0.002 a -0.046)
- Presencia de autocorrelación según Ljung-Box, pero valores muy bajos

**Materias Primas (GLD, SLV, USO):**
- ACF lag-1: muy cercano a 0 (-0.008 a 0.003)
- **Sin autocorrelación significativa** según Ljung-Box (p > 0.05)

#### Interpretación

1. **Reversión a corto plazo en acciones**: La autocorrelación negativa en lag-1 sugiere que los retornos de acciones tienden a revertirse al día siguiente, un fenómeno conocido como "mean reversion" a corto plazo.

2. **Eficiencia de mercado en materias primas**: GLD, SLV y USO no muestran autocorrelación significativa, lo que indica un comportamiento más eficiente y aleatorio, consistente con mercados más líquidos y eficientes.

3. **Bonos con comportamiento intermedio**: Los bonos muestran autocorrelación muy baja, sugiriendo un comportamiento más predecible que las acciones pero menos eficiente que las materias primas.

### 3.3. Volatilidad y Clustering

#### Resultados Principales

**Volatilidad Anualizada (de mayor a menor):**
1. USO (Petróleo): 36.78%
2. SLV (Plata): 31.21%
3. EEM (Mercados Emergentes): 27.98%
4. IWM (Russell 2000): 24.87%
5. VGK (Europa): 23.66%
6. QQQ (Nasdaq 100): 22.32%
7. EFA (EAFE): 21.78%
8. VPL (Asia-Pacífico): 20.32%
9. SPY (S&P 500): 19.83%
10. GLD (Oro): 17.50%
11. TLT (Treasury 20+): 15.26%
12. HYG (High Yield): 11.00%
13. LQD (Investment Grade): 8.91%
14. AGG (Total Bond): 5.43%

**Clustering de Volatilidad (ARCH Effects):**
- Todos los ETFs muestran clustering de volatilidad (ARCH effect lag-1 > 0.1)
- **Mayores efectos ARCH**: LQD (0.48), TLT (0.37), USO (0.32), IWM (0.30)

#### Interpretación

1. **Jerarquía de riesgo**: Las materias primas (USO, SLV) y mercados emergentes (EEM) son los más volátiles, mientras que los bonos de alta calidad (AGG, LQD) son los más estables.

2. **Clustering de volatilidad**: Todos los activos muestran el fenómeno de clustering, donde períodos de alta volatilidad tienden a seguir períodos de alta volatilidad, y viceversa. Esto es fundamental para modelos GARCH.

3. **Persistencia en bonos**: Los bonos (especialmente LQD y TLT) muestran los mayores efectos ARCH, indicando mayor persistencia en la volatilidad, lo cual es importante para la gestión de riesgo.

### 3.4. Momentos Superiores: Asimetría y Curtosis

#### Resultados Principales

**Asimetría (Skewness):**
- **Negativa (cola izquierda pesada)**: QQQ (-0.057), IWM (-0.323), EFA (-0.035), VGK (-0.334), AGG (-1.882), GLD (-0.141), SLV (-0.483), USO (-0.657)
- **Positiva (cola derecha pesada)**: EEM (0.590), LQD (0.042), HYG (0.748), TLT (0.080), VPL (0.081), SPY (0.005)

**Curtosis (Exceso de Kurtosis):**
- **Todos los ETFs son leptocúrticos** (exceso de kurtosis > 0)
- **Mayores valores**: LQD (53.46), AGG (46.48), HYG (39.39), EEM (16.31)
- **Menores valores**: TLT (0.16), SPY (11.47), QQQ (4.61)

#### Interpretación

1. **Riesgo de caídas extremas**: Los activos con asimetría negativa (IWM, VGK, SLV, USO) tienen mayor probabilidad de caídas extremas que de subidas extremas, lo cual es crítico para la gestión de riesgo.

2. **Colas pesadas**: Todos los activos muestran exceso de curtosis, indicando que los eventos extremos (tanto positivos como negativos) son más frecuentes de lo que predice una distribución normal. Esto es fundamental para modelos de riesgo como CVaR.

3. **Bonos con distribuciones extremas**: Los bonos (LQD, AGG, HYG) muestran el mayor exceso de curtosis, sugiriendo que aunque su volatilidad promedio es baja, pueden experimentar movimientos extremos ocasionales (eventos de "crédito" o "liquidez").

4. **Asimetría positiva en algunos activos**: EEM, HYG y algunos bonos muestran asimetría positiva, sugiriendo potencial de subidas extremas, aunque esto puede ser engañoso si no se considera el contexto.

### 3.5. Cambios de Régimen y No Linealidades

#### Resultados Principales

**Cambios de Régimen:**
- **Ningún ETF muestra cambio de régimen significativo** (p-valores > 0.05 en test t)
- Esto sugiere que las medias de los retornos no han cambiado significativamente entre la primera y segunda mitad del período

**No Linealidades:**
- **ETFs con no linealidad detectada**: EEM, HYG, AGG, SLV, USO
- Estos activos muestran correlación significativa entre retornos y retornos²

#### Interpretación

1. **Estabilidad de medias**: La ausencia de cambios de régimen significativos sugiere que los retornos promedio no han experimentado cambios estructurales importantes en el período analizado, lo cual es favorable para modelos de pronóstico.

2. **No linealidades en activos específicos**: 
   - **Materias primas (SLV, USO)**: Las no linealidades pueden reflejar asimetrías en la respuesta a shocks positivos vs negativos
   - **Bonos (HYG, AGG)**: Pueden reflejar efectos de convexidad o no linealidades en la relación precio-rendimiento
   - **EEM**: Puede reflejar asimetrías en la respuesta a shocks globales

3. **Implicaciones para modelado**: Los activos con no linealidades pueden requerir modelos más sofisticados (no lineales) para capturar adecuadamente su dinámica.

### 3.6. Exposición a Factores de Mercado (CAPM)

#### Resultados Principales

**Beta (Sensibilidad al Mercado - SPY):**

**Beta > 1 (Más volátiles que el mercado):**
- QQQ: 1.04 (Tecnología)
- IWM: 1.12 (Pequeñas empresas)
- EEM: 1.17 (Mercados emergentes)

**Beta ≈ 1 (Similar al mercado):**
- SPY: 1.00 (Benchmark)
- EFA: 0.98 (Internacional desarrollado)
- VGK: 1.03 (Europa)
- VPL: 0.87 (Asia-Pacífico)

**Beta bajo/negativo (Desacoplados del mercado):**
- TLT: -0.24 (Treasury - correlación negativa)
- AGG: -0.002 (Total Bond - casi nula)
- LQD: 0.09 (Investment Grade - muy baja)
- GLD: 0.04 (Oro - casi independiente)
- SLV: 0.36 (Plata - baja)
- USO: 0.68 (Petróleo - moderada)
- HYG: 0.38 (High Yield - moderada)

**R² (Proporción de varianza explicada):**

**Alto R² (> 0.70):**
- QQQ: 0.85
- IWM: 0.80
- EFA: 0.79
- VGK: 0.74
- VPL: 0.73

**Bajo R² (< 0.15):**
- AGG: 0.0001
- GLD: 0.002
- LQD: 0.04
- SLV: 0.05
- USO: 0.14

#### Interpretación

1. **Diversificación efectiva**: 
   - **Bonos gubernamentales (TLT, AGG)**: Beta negativo o cercano a cero indica que son excelentes para diversificar portafolios de acciones
   - **Oro (GLD)**: Beta casi cero confirma su rol como activo refugio independiente del mercado de acciones
   - **Bonos corporativos (LQD)**: Beta muy bajo sugiere baja sensibilidad a movimientos del mercado de acciones

2. **Alta exposición al mercado**: Las acciones (especialmente QQQ, IWM, EFA) tienen R² alto, indicando que la mayor parte de su variabilidad se explica por movimientos del mercado. Esto sugiere que la diversificación entre acciones puede tener beneficios limitados.

3. **Activos defensivos**: TLT con beta negativo es particularmente valioso como cobertura en crisis, ya que tiende a subir cuando el mercado de acciones cae.

4. **Materias primas**: GLD, SLV y USO tienen R² bajo, indicando que sus movimientos son en gran parte independientes del mercado de acciones, lo cual es valioso para diversificación.

## 4. Conclusiones Generales

### 4.1. Características por Clase de Activo

**Acciones:**
- Alta volatilidad (15-28% anualizada)
- Autocorrelación negativa a corto plazo
- Alta exposición al mercado (R² > 0.70)
- Asimetría negativa (riesgo de caídas)
- Colas pesadas (exceso de kurtosis)

**Bonos:**
- Baja a moderada volatilidad (5-15% anualizada)
- Autocorrelación muy baja
- Baja exposición al mercado (R² < 0.50, excepto HYG)
- Alta curtosis (eventos extremos ocasionales)
- Excelente para diversificación

**Materias Primas:**
- Alta volatilidad (17-37% anualizada)
- Sin autocorrelación significativa (mercados eficientes)
- Baja exposición al mercado (R² < 0.15)
- Asimetría negativa (especialmente SLV, USO)
- Comportamiento independiente del mercado de acciones

### 4.2. Implicaciones para la Gestión de Portafolio

1. **Diversificación efectiva**: La combinación de acciones, bonos y materias primas ofrece diversificación real, dado que:
   - Bonos tienen correlación baja/negativa con acciones
   - Materias primas son independientes del mercado de acciones
   - Diferentes niveles de volatilidad permiten ajustar el perfil de riesgo

2. **Gestión de riesgo**: 
   - Todos los activos muestran clustering de volatilidad → importante usar modelos GARCH o similares
   - Colas pesadas en todos los activos → necesario usar medidas de riesgo robustas (CVaR en lugar de VaR)
   - Asimetría negativa en muchos activos → considerar modelos que capturen asimetrías

3. **Modelado y pronóstico**:
   - Retornos son estacionarios → modelos ARIMA/GARCH son apropiados
   - Autocorrelación presente en acciones → puede explotarse para pronóstico a corto plazo
   - No linealidades en algunos activos → considerar modelos no lineales (ML, redes neuronales)

4. **Selección de activos para optimización**:
   - **Para reducir riesgo**: AGG, LQD, TLT (baja volatilidad, baja correlación)
   - **Para aumentar retorno**: QQQ, EEM, SLV (mayor retorno esperado, mayor riesgo)
   - **Para diversificación**: TLT (beta negativo), GLD (independiente), bonos (baja correlación)

### 4.3. Limitaciones y Consideraciones

1. **Período de análisis**: 20 años pueden incluir múltiples ciclos económicos, pero pueden no capturar cambios estructurales recientes.

2. **Asunción de normalidad**: Aunque los retornos no son normales (colas pesadas, asimetría), muchos modelos asumen normalidad. Es importante usar modelos robustos.

3. **Estabilidad temporal**: Las correlaciones y betas pueden cambiar en el tiempo, especialmente durante crisis.

4. **Liquidez y costos de transacción**: Este análisis no considera costos de transacción ni problemas de liquidez, que pueden ser importantes especialmente para materias primas.

## 5. Próximos Pasos

1. **Modelado de pronóstico**: Desarrollar modelos de pronóstico de retornos mensuales usando las propiedades identificadas.

2. **Optimización de portafolio**: 
   - Usar las correlaciones y volatilidades estimadas
   - Considerar CVaR para capturar colas pesadas
   - Incorporar restricciones basadas en las propiedades identificadas

3. **Análisis de factores**: Extender el análisis CAPM a modelos multi-factor (Fama-French, momentum, etc.)

4. **Análisis de régimen**: Aunque no se detectaron cambios de régimen en medias, podría ser útil analizar cambios en volatilidad o correlaciones.

---

**Fecha de análisis**: Generado automáticamente  
**Período de datos**: 20 años de datos históricos  
**Número de ETFs analizados**: 14

