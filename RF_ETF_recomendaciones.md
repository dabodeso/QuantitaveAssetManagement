# Recomendaciones para ETF de Tasa Libre de Riesgo (RF)

## ¿Qué es la Tasa Libre de Riesgo?

La tasa libre de riesgo (Risk-Free Rate, RF) es el retorno teórico de una inversión sin riesgo de incumplimiento. En la práctica, se usa como proxy el retorno de bonos gubernamentales de corto plazo del país de referencia (típicamente Estados Unidos).

## ETFs Actuales - Análisis

De los ETFs actuales en el portafolio, **ninguno es apropiado** como proxy de RF:

- **TLT** (Treasury 20+ años): Es de **largo plazo**, tiene volatilidad (15.26% anualizada) y sensibilidad a cambios en tasas de interés. No es apropiado.
- **AGG** (Total Bond Market): Incluye bonos corporativos y de diferentes vencimientos. No es puro riesgo libre.
- **LQD** (Investment Grade Corporativos): Son bonos corporativos, tienen riesgo crediticio. No es apropiado.
- **HYG** (High Yield): Tiene riesgo crediticio alto. No es apropiado.

## ETFs Recomendados para RF

### Opción 1: **BIL** (Recomendado) ⭐
- **Nombre**: SPDR Bloomberg 1-3 Month T-Bill ETF
- **Vencimiento**: 1-3 meses (muy corto plazo)
- **Características**:
  - Volatilidad muy baja (~0.5-1% anualizada)
  - Prácticamente sin riesgo crediticio (bonos del Tesoro EEUU)
  - Retorno cercano a la tasa de fondos federales
  - **Más cercano a la tasa libre de riesgo teórica**

### Opción 2: **SHY**
- **Nombre**: iShares 1-3 Year Treasury Bond ETF
- **Vencimiento**: 1-3 años
- **Características**:
  - Volatilidad baja (~2-3% anualizada)
  - Bonos del Tesoro EEUU (sin riesgo crediticio)
  - Mayor duración que BIL, por lo tanto más sensible a cambios en tasas
  - Buena alternativa si BIL no tiene suficiente historia

### Opción 3: **SGOV**
- **Nombre**: iShares 0-3 Month Treasury Bond ETF
- **Vencimiento**: 0-3 meses
- **Características**:
  - Similar a BIL
  - Volatilidad muy baja
  - Excelente proxy de RF

### Opción 4: **VGSH**
- **Nombre**: Vanguard Short-Term Treasury ETF
- **Vencimiento**: 1-3 años
- **Características**:
  - Similar a SHY
  - Bajo costo de gestión
  - Buena alternativa

## Recomendación Final

**Usar BIL o SGOV** como proxy de RF porque:
1. Son los más cercanos a la tasa libre de riesgo teórica (vencimiento muy corto)
2. Tienen volatilidad mínima
3. Representan bonos del Tesoro EEUU (sin riesgo crediticio)
4. Son ampliamente aceptados en la literatura académica como proxy de RF

**Si BIL no tiene suficiente historia de datos (20 años)**, usar **SHY** como alternativa.

## Uso en el Proyecto

Para calcular **retornos en exceso** (excess returns):
```
Excess Return = Asset Return - RF Return
```

Para modelos **CAPM**:
```
Expected Return = RF + Beta × (Market Return - RF)
```

Para **optimización de portafolio**:
- Usar RF como activo libre de riesgo en la frontera eficiente
- Calcular Sharpe Ratio: (Return - RF) / Volatility

