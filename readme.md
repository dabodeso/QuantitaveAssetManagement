# Proyecto de Gestión Cuantitativa de Activos

Este proyecto implementa un flujo de trabajo completo de inversión cuantitativa, incluyendo:
1. Análisis de series temporales de activos financieros
2. Modelos de pronóstico de retornos
3. Optimización de portafolio

## Instalación

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

- `download_etf_data.py`: Script para descargar datos de ETFs de Yahoo Finance
- `etf_returns_dict.pkl`: Diccionario con retornos diarios de los ETFs (generado)
- `requirements.txt`: Dependencias del proyecto

## Uso

### Paso 1: Descargar datos de ETFs

Ejecuta el script para descargar 20 años de datos históricos de ETFs:

```bash
python download_etf_data.py
```

El script:
- Descarga datos de ETFs que cubren acciones, bonos y materias primas
- Calcula retornos diarios
- Crea un diccionario con el nombre de cada ETF y sus retornos
- Guarda los datos en formato pickle y CSV en el directorio `data/`
- Genera gráficos comparativos de todos los ETFs

### ETFs Seleccionados

#### Acciones - EEUU
1. **SPY** - S&P 500 (EEUU)
2. **QQQ** - Nasdaq 100 (Tecnología EEUU)
3. **IWM** - Russell 2000 (Pequeñas Empresas EEUU)

#### Acciones - Internacional
4. **EFA** - EAFE - Europa, Asia, Lejano Oriente
5. **EEM** - Mercados Emergentes
6. **VGK** - Europa (Vanguard)
7. **VPL** - Asia-Pacífico (Vanguard)

#### Bonos
8. **LQD** - Bonos Corporativos Investment Grade
9. **HYG** - Bonos High Yield (Basura)
10. **TLT** - Bonos del Tesoro EEUU 20+ años
11. **AGG** - Total Bond Market (Bonos Agregados)

#### Tasa Libre de Riesgo (RF)
12. **SHY** - Treasury 1-3 años (iShares) - Proxy de tasa libre de riesgo

#### Materias Primas
13. **GLD** - Oro (SPDR Gold Trust)
14. **SLV** - Plata (iShares Silver Trust)
15. **USO** - Petróleo (United States Oil Fund)

**Nota**: SHY se usa como proxy de la tasa libre de riesgo (RF) para calcular retornos en exceso y en modelos CAPM. Ver `RF_ETF_recomendaciones.md` para más detalles.

## Próximos Pasos

- Análisis de propiedades de series temporales
- Desarrollo de modelos de pronóstico
- Optimización de portafolio
