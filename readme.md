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

Ejecuta el script para descargar 10 años de datos históricos de 10 ETFs:

```bash
python download_etf_data.py
```

El script:
- Descarga datos de 10 ETFs que cubren diferentes regiones del mundo
- Calcula retornos diarios
- Crea un diccionario con el nombre de cada ETF y sus retornos
- Guarda los datos en formato pickle y CSV

### ETFs Seleccionados

1. **SPY** - S&P 500 (EEUU)
2. **EFA** - EAFE - Europa, Asia, Lejano Oriente
3. **EEM** - Mercados Emergentes
4. **VEA** - Mercados Desarrollados Ex-EEUU
5. **VWO** - Mercados Emergentes (Vanguard)
6. **IWM** - Russell 2000 (Pequeñas Empresas EEUU)
7. **QQQ** - Nasdaq 100 (Tecnología EEUU)
8. **VGK** - Europa (Vanguard)
9. **VPL** - Asia-Pacífico (Vanguard)
10. **VTI** - Total Stock Market (EEUU)

## Próximos Pasos

- Análisis de propiedades de series temporales
- Desarrollo de modelos de pronóstico
- Optimización de portafolio
