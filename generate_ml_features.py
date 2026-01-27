"""
Script para generar features de Machine Learning para modelo de decisión geográfica

Este script genera:
1. Features técnicas por ETF/activo
2. Features de geografía (agrupación por región)
3. Variable objetivo: Sharpe Ratio futuro por geografía
4. Datos estructurados para entrenar modelo ML que decide qué geografías comprar
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Definir ETFs y activos por geografía
# ESTRUCTURA: Quitamos EFA, mantenemos VGK y VPL para tener control granular por geografía
ETFS_BY_GEOGRAPHY = {
    'USA': {
        'SPY': 'S&P 500 (EEUU)',
        'QQQ': 'Nasdaq 100 (Tecnología EEUU)',
        'IWM': 'Russell 2000 (Pequeñas Empresas EEUU)'
    },
    'EUROPA': {
        'VGK': 'Europa (Vanguard)'
    },
    'ASIA_PACIFICO': {
        'VPL': 'Asia-Pacífico (Vanguard)'
    },
    'EMERGENTES': {
        'EEM': 'Mercados Emergentes'
    },
    'BONOS': {
        'TLT': 'Bonos del Tesoro EEUU 20+ años',
        'LQD': 'Bonos Corporativos Investment Grade',
        'HYG': 'Bonos High Yield (Basura)',
        'SHY': 'Tasa Libre de Riesgo - Treasury 1-3 años'
    },
    'MATERIAS_PRIMAS': {
        'GLD': 'Oro (SPDR Gold Trust)',
        'USO': 'Petróleo (United States Oil Fund)',
        'DJP': 'Bloomberg Commodity Index (iPath)'
    },
    'REAL_ESTATE': {
        'VNQ': 'REITs EEUU (Vanguard Real Estate ETF)'
    },
    'DIVISAS': {
        'EURUSD': 'Euro/Dólar (Forex)'
    }
}

# ETFs adicionales recomendados
ADDITIONAL_ETFS = {
    'VEA': 'Europa, Asia-Pacífico Desarrollado (alternativa a EFA)',
    'DJP': 'Bloomberg Commodity Index',
    'DBA': 'Agricultura (DB Agriculture Fund)',
    'VNQ': 'REITs EEUU (Real Estate)',
    'EFA': 'EAFE (mantener como referencia, pero no usar en modelo)'
}

def download_forex_data(symbol='EURUSD=X', years=20):
    """Descarga datos de Forex desde Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"  [WARNING]  No se encontraron datos para {symbol}")
            return None
        
        # Calcular retornos diarios
        data['Returns'] = data['Close'].pct_change() * 100
        returns = data['Returns'].dropna()
        
        # Normalizar índice (eliminar zona horaria, duplicados)
        if isinstance(returns.index, pd.DatetimeIndex):
            if returns.index.tz is not None:
                returns.index = returns.index.tz_localize(None)
            returns.index = returns.index.normalize()
            returns = returns[~returns.index.duplicated(keep='first')]
        
        print(f"  [OK] {symbol}: {len(returns)} dias de retornos")
        return returns
    except Exception as e:
        print(f"  [ERROR] Error descargando {symbol}: {str(e)}")
        return None

def load_returns_data(data_dir='data', filename='etf_returns_dict.pkl'):
    """Carga los datos de retornos desde el archivo pickle y normaliza índices"""
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            returns_dict = pickle.load(f)
        
        # Normalizar índices de todas las Series (eliminar zona horaria, duplicados)
        normalized_dict = {}
        for symbol, series in returns_dict.items():
            if isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex):
                series_normalized = series.copy()
                # Eliminar zona horaria si existe
                if series_normalized.index.tz is not None:
                    series_normalized.index = series_normalized.index.tz_localize(None)
                # Normalizar a medianoche
                series_normalized.index = series_normalized.index.normalize()
                # Eliminar duplicados
                series_normalized = series_normalized[~series_normalized.index.duplicated(keep='first')]
                normalized_dict[symbol] = series_normalized
            else:
                normalized_dict[symbol] = series
        
        return normalized_dict
    else:
        print(f"[WARNING] Archivo {filepath} no encontrado. Necesitas ejecutar download_etf_data.py primero.")
        return None

def download_additional_data(data_dir='data', returns_dict=None):
    """Descarga datos adicionales: VIX, DXY, EUR/USD, Credit Spreads, FRED, etc."""
    print("\n" + "=" * 80)
    print("DESCARGANDO DATOS ADICIONALES")
    print("=" * 80)
    
    additional_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 365)
    
    # 1. VIX (Volatilidad)
    print("\n1. Descargando VIX (Volatilidad)...")
    try:
        vix = yf.Ticker("^VIX").history(start=start_date, end=end_date)
        if not vix.empty:
            vix_returns = vix['Close'].pct_change() * 100
            vix_returns = vix_returns.dropna()
            # Normalizar índice
            if isinstance(vix_returns.index, pd.DatetimeIndex):
                if vix_returns.index.tz is not None:
                    vix_returns.index = vix_returns.index.tz_localize(None)
                vix_returns.index = vix_returns.index.normalize()
                vix_returns = vix_returns[~vix_returns.index.duplicated(keep='first')]
            additional_data['VIX'] = vix_returns
            print(f"   [OK] VIX: {len(additional_data['VIX'])} observaciones")
        else:
            print("   [WARNING]  VIX no disponible")
    except Exception as e:
        print(f"   [ERROR] Error descargando VIX: {e}")
    
    # 2. DXY (Dollar Index)
    print("\n2. Descargando DXY (Dollar Index)...")
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(start=start_date, end=end_date)
        if dxy.empty:
            # Intentar con otro símbolo
            dxy = yf.Ticker("^DXY").history(start=start_date, end=end_date)
        if not dxy.empty:
            dxy_returns = dxy['Close'].pct_change() * 100
            dxy_returns = dxy_returns.dropna()
            # Normalizar índice
            if isinstance(dxy_returns.index, pd.DatetimeIndex):
                if dxy_returns.index.tz is not None:
                    dxy_returns.index = dxy_returns.index.tz_localize(None)
                dxy_returns.index = dxy_returns.index.normalize()
                dxy_returns = dxy_returns[~dxy_returns.index.duplicated(keep='first')]
            additional_data['DXY'] = dxy_returns
            print(f"   [OK] DXY: {len(additional_data['DXY'])} observaciones")
        else:
            print("   [WARNING]  DXY no disponible")
    except Exception as e:
        print(f"   [ERROR] Error descargando DXY: {e}")
    
    # 3. EUR/USD
    print("\n3. Descargando EUR/USD...")
    eurusd = download_forex_data('EURUSD=X', years=20)
    if eurusd is not None:
        additional_data['EURUSD'] = eurusd
    
    # 4. Yield Curve (10Y - 3M como proxy)
    print("\n4. Descargando Yield Curve (10Y Treasury)...")
    try:
        tnx = yf.Ticker("^TNX").history(start=start_date, end=end_date)
        if not tnx.empty:
            tnx_series = tnx['Close'].copy()
            # Normalizar índice
            if isinstance(tnx_series.index, pd.DatetimeIndex):
                if tnx_series.index.tz is not None:
                    tnx_series.index = tnx_series.index.tz_localize(None)
                tnx_series.index = tnx_series.index.normalize()
                tnx_series = tnx_series[~tnx_series.index.duplicated(keep='first')]
            additional_data['TNX_10Y'] = tnx_series
            print(f"   [OK] TNX (10Y): {len(additional_data['TNX_10Y'])} observaciones")
    except Exception as e:
        print(f"   [ERROR] Error descargando TNX: {e}")
    
    try:
        irx = yf.Ticker("^IRX").history(start=start_date, end=end_date)
        if not irx.empty:
            irx_series = irx['Close'].copy()
            # Normalizar índice
            if isinstance(irx_series.index, pd.DatetimeIndex):
                if irx_series.index.tz is not None:
                    irx_series.index = irx_series.index.tz_localize(None)
                irx_series.index = irx_series.index.normalize()
                irx_series = irx_series[~irx_series.index.duplicated(keep='first')]
            additional_data['IRX_3M'] = irx_series
            # Calcular spread
            if 'TNX_10Y' in additional_data:
                aligned = pd.DataFrame({
                    'TNX': additional_data['TNX_10Y'],
                    'IRX': additional_data['IRX_3M']
                }).dropna()
                if len(aligned) > 0:
                    spread = aligned['TNX'] - aligned['IRX']
                    additional_data['YIELD_SPREAD'] = spread
                    print(f"   [OK] Yield Spread: {len(additional_data['YIELD_SPREAD'])} observaciones")
    except Exception as e:
        print(f"   [ERROR] Error descargando IRX: {e}")
    
    # 5. Credit Spread (HYG - LQD)
    print("\n5. Calculando Credit Spread (HYG - LQD)...")
    try:
        # Cargar retornos de HYG y LQD
        if returns_dict is None:
            # Intentar cargar desde archivo
            try:
                with open(os.path.join(data_dir, 'etf_returns_dict.pkl'), 'rb') as f:
                    returns_dict = pickle.load(f)
            except:
                returns_dict = None
        
        if returns_dict is not None and 'HYG' in returns_dict and 'LQD' in returns_dict:
            # Los índices ya están normalizados en load_returns_data()
            aligned_credit = pd.DataFrame({
                'HYG': returns_dict['HYG'],
                'LQD': returns_dict['LQD']
            }).dropna()
            
            if len(aligned_credit) > 0:
                # Credit spread como diferencia de retornos (proxy de riesgo crediticio)
                # Spread alto = mayor riesgo crediticio
                credit_spread = aligned_credit['HYG'] - aligned_credit['LQD']
                additional_data['CREDIT_SPREAD'] = credit_spread
                print(f"   [OK] Credit Spread (HYG-LQD): {len(additional_data['CREDIT_SPREAD'])} observaciones")
        else:
            print("   [WARNING] HYG o LQD no disponibles para calcular credit spread")
    except Exception as e:
        print(f"   [ERROR] Error calculando Credit Spread: {e}")
    
    # 6. Indicadores de FRED (opcional, requiere API key)
    print("\n6. Intentando descargar indicadores de FRED...")
    try:
        # Intentar importar fredapi
        try:
            from fredapi import Fred
            fred_api_key = os.getenv('FRED_API_KEY')
            
            if fred_api_key:
                fred = Fred(api_key=fred_api_key)
                
                # Indicadores clave
                fred_indicators = {
                    'FEDFUNDS': 'Fed Funds Rate',
                    'CPIAUCSL': 'CPI (Inflación)',
                    'UNRATE': 'Unemployment Rate',
                    'DGS10': '10-Year Treasury Rate',
                    'DGS2': '2-Year Treasury Rate'
                }
                
                for indicator, description in fred_indicators.items():
                    try:
                        data = fred.get_series(indicator, start=start_date, end=end_date)
                        if not data.empty:
                            # Convertir a cambios diarios
                            if indicator in ['FEDFUNDS', 'CPIAUCSL', 'UNRATE']:
                                # Para estos, usar cambio porcentual
                                data_changes = data.pct_change() * 100
                            else:
                                # Para tasas (DGS10, DGS2), usar cambio absoluto
                                data_changes = data.diff()
                            
                            data_changes = data_changes.dropna()
                            if len(data_changes) > 0:
                                additional_data[f'FRED_{indicator}'] = data_changes
                                print(f"   [OK] FRED {indicator} ({description}): {len(additional_data[f'FRED_{indicator}'])} observaciones")
                    except Exception as e:
                        print(f"   [WARNING] No se pudo descargar {indicator}: {e}")
                
                # Calcular yield curve spread de FRED si tenemos ambas tasas
                if 'FRED_DGS10' in additional_data and 'FRED_DGS2' in additional_data:
                    aligned_fred = pd.DataFrame({
                        'DGS10': additional_data['FRED_DGS10'],
                        'DGS2': additional_data['FRED_DGS2']
                    }).dropna()
                    
                    if len(aligned_fred) > 0:
                        # Spread como diferencia (no retornos, sino niveles)
                        # Necesitamos los niveles originales, no los retornos
                        try:
                            dgs10_levels = fred.get_series('DGS10', start=start_date, end=end_date)
                            dgs2_levels = fred.get_series('DGS2', start=start_date, end=end_date)
                            aligned_levels = pd.DataFrame({
                                'DGS10': dgs10_levels,
                                'DGS2': dgs2_levels
                            }).dropna()
                            
                            if len(aligned_levels) > 0:
                                additional_data['FRED_YIELD_SPREAD'] = aligned_levels['DGS10'] - aligned_levels['DGS2']
                                print(f"   [OK] FRED Yield Spread (10Y-2Y): {len(additional_data['FRED_YIELD_SPREAD'])} observaciones")
                        except:
                            pass
            else:
                print("   [INFO] FRED_API_KEY no configurada. Para usar indicadores de FRED:")
                print("         1. Obtén una API key gratuita en: https://fred.stlouisfed.org/docs/api/api_key.html")
                print("         2. Configura la variable de entorno: $env:FRED_API_KEY='tu_api_key'")
                print("         3. O instala fredapi: pip install fredapi")
        except ImportError:
            print("   [INFO] fredapi no instalado. Para usar indicadores de FRED:")
            print("         pip install fredapi")
            print("         Y configura FRED_API_KEY como variable de entorno")
    except Exception as e:
        print(f"   [WARNING] Error con FRED: {e}")
    
    # 7. Factores de Fama-French
    print("\n7. Calculando/Descargando Factores de Fama-French...")
    try:
        # Cargar retornos si no están disponibles
        if returns_dict is None:
            try:
                with open(os.path.join(data_dir, 'etf_returns_dict.pkl'), 'rb') as f:
                    returns_dict = pickle.load(f)
            except:
                returns_dict = None
        
        if returns_dict is not None:
            # 7.1 Mkt-Rf (Market Risk Premium): SPY - SHY
            if 'SPY' in returns_dict and 'SHY' in returns_dict:
                spy_returns = returns_dict['SPY']
                shy_returns = returns_dict['SHY']
                # Normalizar índices
                if isinstance(spy_returns.index, pd.DatetimeIndex):
                    spy_returns.index = spy_returns.index.tz_localize(None) if spy_returns.index.tz else spy_returns.index
                    spy_returns.index = spy_returns.index.normalize()
                if isinstance(shy_returns.index, pd.DatetimeIndex):
                    shy_returns.index = shy_returns.index.tz_localize(None) if shy_returns.index.tz else shy_returns.index
                    shy_returns.index = shy_returns.index.normalize()
                
                # Alinear fechas
                aligned = pd.DataFrame({'SPY': spy_returns, 'SHY': shy_returns}).dropna()
                if len(aligned) > 0:
                    ff_mkt_rf = aligned['SPY'] - aligned['SHY']
                    additional_data['FF_MKT_RF'] = ff_mkt_rf
                    print(f"   [OK] FF_MKT_RF (Market Risk Premium): {len(ff_mkt_rf)} observaciones")
            
            # 7.2 SMB (Small Minus Big): IWM - SPY
            if 'IWM' in returns_dict and 'SPY' in returns_dict:
                iwm_returns = returns_dict['IWM']
                spy_returns = returns_dict['SPY']
                # Normalizar índices
                if isinstance(iwm_returns.index, pd.DatetimeIndex):
                    iwm_returns.index = iwm_returns.index.tz_localize(None) if iwm_returns.index.tz else iwm_returns.index
                    iwm_returns.index = iwm_returns.index.normalize()
                if isinstance(spy_returns.index, pd.DatetimeIndex):
                    spy_returns.index = spy_returns.index.tz_localize(None) if spy_returns.index.tz else spy_returns.index
                    spy_returns.index = spy_returns.index.normalize()
                
                # Alinear fechas
                aligned = pd.DataFrame({'IWM': iwm_returns, 'SPY': spy_returns}).dropna()
                if len(aligned) > 0:
                    ff_smb = aligned['IWM'] - aligned['SPY']
                    additional_data['FF_SMB'] = ff_smb
                    print(f"   [OK] FF_SMB (Small Minus Big): {len(ff_smb)} observaciones")
            
            # 7.3 HML (High Minus Low / Value vs Growth): Intentar con ETFs VTV/VUG o descargar factores oficiales
            # Intentar primero con ETFs value/growth
            try:
                vtv = yf.Ticker("VTV").history(start=start_date, end=end_date)  # Value ETF
                vug = yf.Ticker("VUG").history(start=start_date, end=end_date)  # Growth ETF
                
                if not vtv.empty and not vug.empty:
                    vtv_returns = vtv['Close'].pct_change() * 100
                    vug_returns = vug['Close'].pct_change() * 100
                    vtv_returns = vtv_returns.dropna()
                    vug_returns = vug_returns.dropna()
                    
                    # Normalizar índices
                    if isinstance(vtv_returns.index, pd.DatetimeIndex):
                        vtv_returns.index = vtv_returns.index.tz_localize(None) if vtv_returns.index.tz else vtv_returns.index
                        vtv_returns.index = vtv_returns.index.normalize()
                    if isinstance(vug_returns.index, pd.DatetimeIndex):
                        vug_returns.index = vug_returns.index.tz_localize(None) if vug_returns.index.tz else vug_returns.index
                        vug_returns.index = vug_returns.index.normalize()
                    
                    # Alinear fechas
                    aligned = pd.DataFrame({'VTV': vtv_returns, 'VUG': vug_returns}).dropna()
                    if len(aligned) > 0:
                        ff_hml = aligned['VTV'] - aligned['VUG']  # Value - Growth
                        additional_data['FF_HML'] = ff_hml
                        print(f"   [OK] FF_HML (High Minus Low - proxy VTV/VUG): {len(ff_hml)} observaciones")
            except Exception as e:
                print(f"   [WARNING] No se pudo calcular HML con ETFs: {e}")
                # Intentar descargar factores oficiales de French Data Library
                try:
                    import requests
                    from io import StringIO
                    
                    # Descargar factores Fama-French 3 factores (diarios)
                    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        import zipfile
                        from io import BytesIO
                        
                        zip_file = zipfile.ZipFile(BytesIO(response.content))
                        # El archivo dentro del zip se llama "F-F_Research_Data_Factors_daily.CSV"
                        csv_content = zip_file.read(zip_file.namelist()[0]).decode('utf-8')
                        
                        # Leer CSV (tiene header y footer que hay que limpiar)
                        lines = csv_content.split('\n')
                        # Encontrar donde empiezan los datos (después de la línea de header)
                        data_start = None
                        for i, line in enumerate(lines):
                            if line.strip().startswith('19') or line.strip().startswith('20'):
                                data_start = i
                                break
                        
                        if data_start is not None:
                            # Leer datos
                            data_lines = [line for line in lines[data_start:] if line.strip() and not line.strip().startswith('Copyright')]
                            csv_clean = '\n'.join(['Date,Mkt-RF,SMB,HML,RF'] + data_lines[:-1])  # Última línea suele ser vacía o copyright
                            
                            ff_data = pd.read_csv(StringIO(csv_clean), parse_dates=['Date'], index_col='Date')
                            ff_data.index = pd.to_datetime(ff_data.index)
                            
                            # Convertir a porcentaje diario (los datos vienen en porcentaje)
                            # Filtrar por rango de fechas
                            ff_data = ff_data[(ff_data.index >= start_date) & (ff_data.index <= end_date)]
                            
                            if len(ff_data) > 0:
                                # HML ya está en el archivo
                                ff_hml_official = ff_data['HML']  # Ya está en porcentaje
                                # Convertir índice a datetime sin timezone
                                ff_hml_official.index = ff_hml_official.index.tz_localize(None) if ff_hml_official.index.tz else ff_hml_official.index
                                ff_hml_official.index = ff_hml_official.index.normalize()
                                
                                additional_data['FF_HML'] = ff_hml_official
                                print(f"   [OK] FF_HML (High Minus Low - oficial): {len(ff_hml_official)} observaciones")
                                
                                # También descargar UMD si está disponible
                                # UMD está en un archivo separado
                                try:
                                    url_umd = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
                                    response_umd = requests.get(url_umd, timeout=10)
                                    if response_umd.status_code == 200:
                                        zip_file_umd = zipfile.ZipFile(BytesIO(response_umd.content))
                                        csv_content_umd = zip_file_umd.read(zip_file_umd.namelist()[0]).decode('utf-8')
                                        
                                        lines_umd = csv_content_umd.split('\n')
                                        data_start_umd = None
                                        for i, line in enumerate(lines_umd):
                                            if line.strip().startswith('19') or line.strip().startswith('20'):
                                                data_start_umd = i
                                                break
                                        
                                        if data_start_umd is not None:
                                            data_lines_umd = [line for line in lines_umd[data_start_umd:] if line.strip() and not line.strip().startswith('Copyright')]
                                            csv_clean_umd = '\n'.join(['Date,UMD'] + data_lines_umd[:-1])
                                            
                                            ff_umd_data = pd.read_csv(StringIO(csv_clean_umd), parse_dates=['Date'], index_col='Date')
                                            ff_umd_data.index = pd.to_datetime(ff_umd_data.index)
                                            ff_umd_data = ff_umd_data[(ff_umd_data.index >= start_date) & (ff_umd_data.index <= end_date)]
                                            
                                            if len(ff_umd_data) > 0:
                                                ff_umd = ff_umd_data['UMD']
                                                ff_umd.index = ff_umd.index.tz_localize(None) if ff_umd.index.tz else ff_umd.index
                                                ff_umd.index = ff_umd.index.normalize()
                                                
                                                additional_data['FF_UMD'] = ff_umd
                                                print(f"   [OK] FF_UMD (Momentum - oficial): {len(ff_umd)} observaciones")
                                except Exception as e_umd:
                                    print(f"   [WARNING] No se pudo descargar UMD: {e_umd}")
                                
                                # Intentar descargar factores de 5 factores (RMW, CMA)
                                try:
                                    url_5f = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
                                    response_5f = requests.get(url_5f, timeout=10)
                                    if response_5f.status_code == 200:
                                        zip_file_5f = zipfile.ZipFile(BytesIO(response_5f.content))
                                        csv_content_5f = zip_file_5f.read(zip_file_5f.namelist()[0]).decode('utf-8')
                                        
                                        lines_5f = csv_content_5f.split('\n')
                                        data_start_5f = None
                                        for i, line in enumerate(lines_5f):
                                            if line.strip().startswith('19') or line.strip().startswith('20'):
                                                data_start_5f = i
                                                break
                                        
                                        if data_start_5f is not None:
                                            data_lines_5f = [line for line in lines_5f[data_start_5f:] if line.strip() and not line.strip().startswith('Copyright')]
                                            csv_clean_5f = '\n'.join(['Date,Mkt-RF,SMB,HML,RMW,CMA,RF'] + data_lines_5f[:-1])
                                            
                                            ff_5f_data = pd.read_csv(StringIO(csv_clean_5f), parse_dates=['Date'], index_col='Date')
                                            ff_5f_data.index = pd.to_datetime(ff_5f_data.index)
                                            ff_5f_data = ff_5f_data[(ff_5f_data.index >= start_date) & (ff_5f_data.index <= end_date)]
                                            
                                            if len(ff_5f_data) > 0:
                                                # RMW (Robust Minus Weak)
                                                ff_rmw = ff_5f_data['RMW']
                                                ff_rmw.index = ff_rmw.index.tz_localize(None) if ff_rmw.index.tz else ff_rmw.index
                                                ff_rmw.index = ff_rmw.index.normalize()
                                                additional_data['FF_RMW'] = ff_rmw
                                                print(f"   [OK] FF_RMW (Profitability - oficial): {len(ff_rmw)} observaciones")
                                                
                                                # CMA (Conservative Minus Aggressive)
                                                ff_cma = ff_5f_data['CMA']
                                                ff_cma.index = ff_cma.index.tz_localize(None) if ff_cma.index.tz else ff_cma.index
                                                ff_cma.index = ff_cma.index.normalize()
                                                additional_data['FF_CMA'] = ff_cma
                                                print(f"   [OK] FF_CMA (Investment - oficial): {len(ff_cma)} observaciones")
                                except Exception as e_5f:
                                    print(f"   [WARNING] No se pudo descargar factores 5F (RMW, CMA): {e_5f}")
                except Exception as e_ff:
                    print(f"   [WARNING] No se pudo descargar factores oficiales de Fama-French: {e_ff}")
        else:
            print("   [WARNING] No se pudieron cargar retornos para calcular factores Fama-French")
    except Exception as e:
        print(f"   [ERROR] Error calculando/descargando factores Fama-French: {e}")
    
    return additional_data

def generate_technical_features_ml(returns_dict, additional_data=None, windows=[60, 252]):
    """
    Genera features técnicas para ML, incluyendo features de mercado adicionales.
    
    Returns:
    --------
    dict: Diccionario con features por ETF/activo
    """
    print("\n" + "=" * 80)
    print("GENERANDO FEATURES TÉCNICAS PARA ML")
    print("=" * 80)
    
    # Convertir a DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    # Normalizar índices: eliminar zona horaria y duplicados
    if isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = returns_df.index.tz_localize(None)  # Eliminar zona horaria
        returns_df.index = returns_df.index.normalize()  # Normalizar a medianoche
        returns_df = returns_df[~returns_df.index.duplicated(keep='first')]  # Eliminar duplicados
    
    # Agregar datos adicionales
    if additional_data:
        for key, series in additional_data.items():
            if isinstance(series, pd.Series):
                # Normalizar índice de series adicionales también
                series_normalized = series.copy()
                if isinstance(series_normalized.index, pd.DatetimeIndex):
                    series_normalized.index = series_normalized.index.tz_localize(None)
                    series_normalized.index = series_normalized.index.normalize()
                    series_normalized = series_normalized[~series_normalized.index.duplicated(keep='first')]
                # Reindexar al índice de returns_df para alinear fechas
                # Esto creará NaN donde no hay datos, pero es esperado
                returns_df[key] = series_normalized.reindex(returns_df.index)
    
    # Alinear todas las fechas (usar intersección, no eliminar todas las filas)
    # Mantener solo fechas donde hay datos de al menos algunos activos
    returns_df = returns_df.dropna(how='all')  # Eliminar solo filas donde TODOS son NaN
    returns_df = returns_df.sort_index()  # Ordenar por fecha
    
    rf_returns = returns_df['SHY'] if 'SHY' in returns_df.columns else None
    
    features_dict = {}
    
    # Procesar cada activo
    for symbol in returns_df.columns:
        if symbol in ['VIX', 'DXY', 'TNX_10Y', 'IRX_3M', 'YIELD_SPREAD', 'FF_MKT_RF', 'FF_SMB', 'FF_HML', 'FF_UMD', 'FF_RMW', 'FF_CMA']:
            # Para indicadores de mercado, generar features más simples (solo value)
            print(f"\nGenerando features para {symbol} (indicador de mercado)...")
            returns = returns_df[symbol].dropna()  # Eliminar NaN
            
            if len(returns) == 0:
                print(f"  [WARNING] {symbol}: No hay datos válidos")
                continue
            
            # Normalizar índice
            if isinstance(returns.index, pd.DatetimeIndex):
                returns.index = returns.index.tz_localize(None)
                returns.index = returns.index.normalize()
            
            features = pd.DataFrame(index=returns.index)
            features['value'] = returns  # Solo mantener el valor, sin std ni zscore
            
            # No eliminar columnas con NaN (son normales al inicio por ventanas rolling)
            features = features.loc[:, ~features.isna().all()]
            features_dict[symbol] = features
            print(f"  [OK] {symbol}: {features_dict[symbol].shape[1]} features")
            continue
        
        # SOLO generar features para SPY (benchmark del mercado)
        # Eliminar features de todos los demás ETFs individuales
        if symbol != 'SPY':
            continue
        
        print(f"\nGenerando features para {symbol} (benchmark del mercado)...")
        returns = returns_df[symbol].dropna()  # Eliminar NaN del retorno específico
        
        if len(returns) == 0:
            print(f"  [WARNING] {symbol}: No hay datos válidos")
            continue
        
        # Normalizar índice
        if isinstance(returns.index, pd.DatetimeIndex):
            returns.index = returns.index.tz_localize(None)
            returns.index = returns.index.normalize()
        
        features = pd.DataFrame(index=returns.index)
        features['returns'] = returns
        
        # 1. RETURNS (eliminado momentum, mantener solo return)
        for window in windows:
            if window <= len(returns):
                features[f'return_{window}d'] = ((1 + returns/100).rolling(window=window).apply(lambda x: x.prod()) - 1) * 100
        
        # 2. VOLATILIDAD
        for window in windows:
            if window <= len(returns):
                # Usar min_periods para reducir NaN al inicio
                min_periods = max(1, int(window * 0.5))  # Al menos 50% de la ventana
                features[f'volatility_{window}d'] = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)
        
        # 3. SHARPE RATIO
        for window in windows:
            if window <= len(returns):
                # Usar min_periods para reducir NaN al inicio
                min_periods = max(1, int(window * 0.5))  # Al menos 50% de la ventana
                mean_return = returns.rolling(window=window, min_periods=min_periods).mean() * 252
                vol = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)
                
                # Usar RF si está disponible
                if rf_returns is not None:
                    # Alinear rf_returns con returns
                    rf_aligned = rf_returns.reindex(returns.index)
                    rf_mean = rf_aligned.rolling(window=window).mean() * 252
                    excess_return = mean_return - rf_mean
                    sharpe = excess_return / vol
                    # Reemplazar infinitos y división por cero con NaN
                    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
                    # Solo asignar si hay al menos algunos valores válidos
                    if sharpe.notna().any():
                        features[f'sharpe_{window}d'] = sharpe
                else:
                    # Si no hay RF, calcular Sharpe sin RF
                    sharpe = mean_return / vol
                    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
                    # Solo asignar si hay al menos algunos valores válidos
                    if sharpe.notna().any():
                        features[f'sharpe_{window}d'] = sharpe
        
        # 4. DRAWDOWN (solo max_drawdown, eliminar drawdown actual)
        cumulative = (1 + returns/100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        # Usar min_periods para reducir NaN
        features['max_drawdown_60d'] = drawdown.rolling(window=60, min_periods=30).min()
        
        # Eliminadas: return_vol_ratio, autocorr, skewness, kurtosis, beta, corr_market, zscore
        # Eliminadas: features de mercado agregadas (vix_level, dxy_level, etc.)
        
        # No eliminar columnas - las NaN al inicio son normales por ventanas rolling
        # Solo verificar que hay al menos una columna con datos
        if features.shape[1] > 0:
            features_dict[symbol] = features
            print(f"  [OK] {symbol}: {features.shape[1]} features generadas")
        else:
            print(f"  [WARNING] {symbol}: No se generaron features")
            features_dict[symbol] = pd.DataFrame(index=returns.index)
    
    return features_dict

def generate_geography_features(features_dict, returns_dict):
    """
    Genera features agregadas por geografía.
    
    Returns:
    --------
    dict: Features agregadas por geografía
    """
    print("\n" + "=" * 80)
    print("GENERANDO FEATURES POR GEOGRAFÍA")
    print("=" * 80)
    
    returns_df = pd.DataFrame(returns_dict)
    
    # Normalizar índices: eliminar zona horaria y duplicados
    if isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = returns_df.index.tz_localize(None)  # Eliminar zona horaria
        returns_df.index = returns_df.index.normalize()  # Normalizar a medianoche
        returns_df = returns_df[~returns_df.index.duplicated(keep='first')]  # Eliminar duplicados
    
    returns_df = returns_df.dropna(how='all')
    returns_df = returns_df.sort_index()  # Ordenar por fecha
    
    geography_features = {}
    
    for geo, etfs in ETFS_BY_GEOGRAPHY.items():
        if geo == 'DIVISAS':
            continue  # EUR/USD se maneja por separado
        
        print(f"\nProcesando geografía: {geo}")
        available_etfs = [etf for etf in etfs.keys() if etf in returns_df.columns]
        
        if len(available_etfs) == 0:
            print(f"  [WARNING]  No hay ETFs disponibles para {geo}")
            continue
        
        # Features agregadas de la geografía
        geo_returns = returns_df[available_etfs]
        
        # Retorno promedio de la geografía (pesos iguales)
        geo_features = pd.DataFrame(index=returns_df.index)
        geo_avg_return = geo_returns.mean(axis=1, skipna=True)
        geo_features[f'{geo}_avg_return'] = geo_avg_return
        
        # Volatilidad: si hay solo 1 ETF, usar su volatilidad individual
        # Si hay múltiples ETFs, usar std entre ellos
        if len(available_etfs) == 1:
            # Solo 1 ETF: calcular volatilidad rolling del ETF individual
            single_etf = geo_returns.iloc[:, 0]
            # Usar min_periods para evitar NaN excesivos al inicio
            geo_vol = single_etf.rolling(window=60, min_periods=30).std() * np.sqrt(252)
            geo_features[f'{geo}_volatility'] = geo_vol
        else:
            # Múltiples ETFs: calcular volatilidad de la cartera (promedio ponderado)
            # Primero calcular retorno promedio de la cartera
            portfolio_return = geo_returns.mean(axis=1, skipna=True)
            # Luego calcular volatilidad rolling del retorno de la cartera
            geo_vol = portfolio_return.rolling(window=60, min_periods=30).std() * np.sqrt(252)
            geo_features[f'{geo}_volatility'] = geo_vol
        
        # Sharpe promedio
        if 'SHY' in returns_df.columns:
            rf = returns_df['SHY']
            # Alinear fechas
            rf_aligned = rf.reindex(geo_features.index)
            avg_return_aligned = geo_features[f'{geo}_avg_return']
            vol_aligned = geo_features[f'{geo}_volatility']
            
            # Calcular excess return (en porcentaje diario)
            excess_return = avg_return_aligned - rf_aligned
            
            # Sharpe: (excess return anualizado) / (volatilidad anualizada)
            # excess_return está en % diario, multiplicar por 252 para anualizar
            # vol_aligned ya está anualizada
            sharpe = (excess_return * 252) / vol_aligned
            # Reemplazar infinitos y NaN con NaN
            sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
            # Solo asignar donde hay datos válidos de volatilidad
            sharpe = sharpe.where(vol_aligned.notna(), np.nan)
            geo_features[f'{geo}_sharpe'] = sharpe
        
        # Correlación promedio entre ETFs de la geografía
        if len(available_etfs) > 1:
            for window in [60, 252]:
                if window <= len(geo_returns):
                    avg_corr_series = []
                    for i in range(window-1, len(geo_returns)):
                        window_data = geo_returns.iloc[i-window+1:i+1]
                        if len(window_data) == window:
                            corr_matrix = window_data.corr()
                            # Obtener solo el triángulo superior (sin diagonal)
                            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                            avg_corr = corr_matrix.where(mask).stack().mean()
                            avg_corr_series.append(avg_corr)
                        else:
                            avg_corr_series.append(np.nan)
                    
                    # Crear serie con índices correctos
                    values = [np.nan] * (window-1) + avg_corr_series
                    geo_features[f'{geo}_avg_correlation_{window}d'] = pd.Series(values, index=geo_returns.index)
        
        geography_features[geo] = geo_features
        print(f"  [OK] {geo}: {geo_features.shape[1]} features")
    
    return geography_features

def generate_target_by_geography(returns_dict, rf_symbol='SHY', forward_window=20):
    """
    Genera variable objetivo: Sharpe Ratio futuro por geografía.
    
    IMPORTANTE: Elimina las últimas 'forward_window' filas porque no tienen datos futuros
    (evita data leakage - no podemos usar datos futuros para predecir).
    
    Returns:
    --------
    dict: Sharpe futuro por geografía
    """
    print("\n" + "=" * 80)
    print("GENERANDO VARIABLE OBJETIVO POR GEOGRAFÍA")
    print("=" * 80)
    print(f"[INFO] Eliminando últimas {forward_window} filas (no tienen datos futuros - evita data leakage)")
    
    returns_df = pd.DataFrame(returns_dict)
    
    # Normalizar índices: eliminar zona horaria y duplicados
    if isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = returns_df.index.tz_localize(None)  # Eliminar zona horaria
        returns_df.index = returns_df.index.normalize()  # Normalizar a medianoche
        returns_df = returns_df[~returns_df.index.duplicated(keep='first')]  # Eliminar duplicados
    
    # NO hacer dropna() completo - elimina demasiadas filas
    # Solo eliminar filas donde TODOS los ETFs de la geografía son NaN
    returns_df = returns_df.dropna(how='all')
    returns_df = returns_df.sort_index()  # Ordenar por fecha
    
    rf_returns = returns_df[rf_symbol] if rf_symbol in returns_df.columns else None
    
    target_by_geo = {}
    
    for geo, etfs in ETFS_BY_GEOGRAPHY.items():
        if geo == 'DIVISAS':
            continue
        
        available_etfs = [etf for etf in etfs.keys() if etf in returns_df.columns]
        
        if len(available_etfs) == 0:
            print(f"  [WARNING] {geo}: No hay ETFs disponibles")
            continue
        
        print(f"\nGenerando target para {geo}...")
        
        # Retorno promedio de la geografía (solo donde hay datos de al menos un ETF)
        geo_returns = returns_df[available_etfs].mean(axis=1, skipna=True)
        
        # Eliminar NaN del geo_returns para tener solo fechas válidas
        geo_returns = geo_returns.dropna()
        
        if len(geo_returns) < forward_window + 1:
            print(f"  [WARNING] {geo}: No hay suficientes datos ({len(geo_returns)} < {forward_window + 1})")
            target_by_geo[geo] = pd.Series(dtype=float)
            continue
        
        # Calcular Sharpe futuro
        # IMPORTANTE: Solo calcular hasta len - forward_window porque después no hay datos futuros
        future_sharpe = pd.Series(index=geo_returns.index, dtype=float)
        
        # Calcular targets solo hasta donde tenemos datos futuros
        max_idx = len(geo_returns) - forward_window
        
        # Alinear rf_returns con geo_returns si está disponible
        if rf_returns is not None:
            # Alinear fechas
            rf_aligned = rf_returns.reindex(geo_returns.index)
        else:
            rf_aligned = None
        
        for i in range(max_idx):
            # Datos futuros (de i+1 a i+forward_window)
            future_returns = geo_returns.iloc[i+1:i+forward_window+1]
            
            # Verificar que tenemos suficientes datos no-NaN
            valid_returns = future_returns.dropna()
            
            if len(valid_returns) >= forward_window * 0.8:  # Al menos 80% de datos
                mean_return = valid_returns.mean() * 252
                vol = valid_returns.std() * np.sqrt(252)
                
                if vol > 0:
                    if rf_aligned is not None:
                        rf_future = rf_aligned.iloc[i+1:i+forward_window+1].dropna()
                        if len(rf_future) > 0:
                            rf_mean = rf_future.mean() * 252
                            excess_return = mean_return - rf_mean
                            sharpe = excess_return / vol
                        else:
                            sharpe = mean_return / vol
                    else:
                        sharpe = mean_return / vol
                    
                    future_sharpe.iloc[i] = sharpe
        
        # ELIMINAR las últimas forward_window filas (no tienen datos futuros - data leakage)
        # Estas filas no deberían usarse para entrenar porque "conocemos el futuro"
        if len(future_sharpe) > forward_window:
            future_sharpe.iloc[-forward_window:] = np.nan
            print(f"  [INFO] Eliminadas últimas {forward_window} filas (evita data leakage)")
        
        target_by_geo[geo] = future_sharpe
        non_null_count = future_sharpe.notna().sum()
        print(f"  [OK] {geo}: {non_null_count} observaciones válidas de {len(future_sharpe)}")
    
    return target_by_geo

def create_ml_dataset(features_dict, geography_features, target_by_geo, additional_data=None):
    """
    Crea dataset final para ML con todas las features y targets.
    
    Returns:
    --------
    pd.DataFrame: Dataset completo para ML
    """
    print("\n" + "=" * 80)
    print("CREANDO DATASET PARA ML")
    print("=" * 80)
    
    # Obtener índice común (fechas)
    # Priorizar fechas de targets (son las más importantes)
    all_indices = set()
    
    # Primero agregar índices de targets (son los más importantes)
    for target in target_by_geo.values():
        all_indices.update(target.index)
    
    # Luego agregar índices de features (para tener todas las fechas disponibles)
    for features in features_dict.values():
        all_indices.update(features.index)
    for features in geography_features.values():
        all_indices.update(features.index)
    
    common_index = sorted(list(all_indices))
    ml_dataset = pd.DataFrame(index=common_index)
    
    # Agregar features individuales de ETFs
    print("\nAgregando features individuales...")
    for symbol, features in features_dict.items():
        if symbol in ['VIX', 'DXY', 'TNX_10Y', 'IRX_3M', 'YIELD_SPREAD']:
            # Features de indicadores de mercado
            for col in features.columns:
                ml_dataset[f'{symbol}_{col}'] = features[col]
        else:
            # Features de ETFs
            for col in features.columns:
                ml_dataset[f'{symbol}_{col}'] = features[col]
    
    # Agregar features de geografía
    print("Agregando features de geografía...")
    for geo, features in geography_features.items():
        for col in features.columns:
            ml_dataset[col] = features[col]
    
    # Agregar targets
    print("Agregando variables objetivo...")
    for geo, target in target_by_geo.items():
        ml_dataset[f'target_{geo}_sharpe'] = target
    
    # Eliminar filas sin target (las últimas forward_window filas ya fueron eliminadas en generate_target_by_geography)
    # NO eliminar filas con NaN en features - eso es normal para ventanas rolling
    target_cols = [col for col in ml_dataset.columns if col.startswith('target_')]
    if target_cols:
        # Eliminar solo filas donde TODOS los targets son NaN
        # Esto elimina las últimas forward_window filas (ya marcadas como NaN) y cualquier otra fila sin target
        rows_before = len(ml_dataset)
        ml_dataset = ml_dataset.dropna(subset=target_cols, how='all')
        rows_after = len(ml_dataset)
        print(f"  Filas eliminadas (sin target o sin datos futuros): {rows_before - rows_after} filas")
        print(f"  Filas restantes con al menos un target: {rows_after} filas")
    else:
        print("[WARNING] No se encontraron columnas de target")
    
    # NO hacer dropna() general - las NaN en features son esperadas y normales
    # El modelo ML (XGBoost, LightGBM) puede manejar NaN automáticamente
    
    print(f"\n[OK] Dataset creado: {ml_dataset.shape[0]} filas, {ml_dataset.shape[1]} columnas")
    if ml_dataset.shape[0] == 0:
        print("[WARNING] Dataset ML está vacío. Verifica que:")
        print("  - Los features tienen fechas comunes")
        print("  - Los índices están alineados correctamente")
        print("  - No se eliminaron todas las filas por tener demasiados NaN")
    else:
        print(f"  Features: {ml_dataset.shape[1] - len(target_by_geo)}")
        print(f"  Targets: {len(target_by_geo)}")
    
    return ml_dataset

def save_ml_data(features_dict, geography_features, target_by_geo, ml_dataset, data_dir='data'):
    """Guarda todos los datos generados"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Guardar features individuales
    with open(os.path.join(data_dir, 'ml_features_dict.pkl'), 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"\n[OK] Features individuales guardadas en {data_dir}/ml_features_dict.pkl")
    
    # Guardar features de geografía
    with open(os.path.join(data_dir, 'geography_features_dict.pkl'), 'wb') as f:
        pickle.dump(geography_features, f)
    print(f"[OK] Features de geografía guardadas en {data_dir}/geography_features_dict.pkl")
    
    # Guardar targets
    with open(os.path.join(data_dir, 'target_by_geography_dict.pkl'), 'wb') as f:
        pickle.dump(target_by_geo, f)
    print(f"[OK] Targets por geografía guardadas en {data_dir}/target_by_geography_dict.pkl")
    
    # Guardar dataset completo
    ml_dataset.to_csv(os.path.join(data_dir, 'ml_dataset.csv'))
    print(f"[OK] Dataset completo guardado en {data_dir}/ml_dataset.csv")
    
    # Guardar también en pickle (más rápido)
    with open(os.path.join(data_dir, 'ml_dataset.pkl'), 'wb') as f:
        pickle.dump(ml_dataset, f)
    print(f"[OK] Dataset completo guardado en {data_dir}/ml_dataset.pkl")

def main():
    """Función principal"""
    data_dir = 'data'
    
    print("=" * 80)
    print("GENERACIÓN DE FEATURES PARA ML - MODELO DE DECISIÓN GEOGRÁFICA")
    print("=" * 80)
    
    # 1. Cargar datos de retornos
    print("\n1. Cargando datos de retornos...")
    returns_dict = load_returns_data(data_dir)
    if returns_dict is None:
        print("❌ No se pueden cargar los datos. Ejecuta download_etf_data.py primero.")
        return
    
    print(f"   [OK] {len(returns_dict)} ETFs cargados")
    
    # 2. Descargar datos adicionales
    print("\n2. Descargando datos adicionales (VIX, DXY, EUR/USD, Yield Curve, Credit Spreads, FRED)...")
    additional_data = download_additional_data(data_dir, returns_dict)
    
    # Agregar EUR/USD a returns_dict si está disponible
    if 'EURUSD' in additional_data:
        returns_dict['EURUSD'] = additional_data['EURUSD']
    
    # 3. Generar features técnicas
    print("\n3. Generando features técnicas...")
    features_dict = generate_technical_features_ml(returns_dict, additional_data)
    
    # 4. Generar features de geografía
    print("\n4. Generando features de geografía...")
    geography_features = generate_geography_features(features_dict, returns_dict)
    
    # 5. Generar targets por geografía
    print("\n5. Generando variables objetivo...")
    target_by_geo = generate_target_by_geography(returns_dict, forward_window=20)
    
    # 6. Crear dataset ML
    print("\n6. Creando dataset para ML...")
    ml_dataset = create_ml_dataset(features_dict, geography_features, target_by_geo, additional_data)
    
    # 7. Guardar todo
    print("\n7. Guardando datos...")
    save_ml_data(features_dict, geography_features, target_by_geo, ml_dataset, data_dir)
    
    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)
    print("\nDataset listo para entrenar modelo ML.")
    print("El modelo puede predecir Sharpe Ratio futuro por geografía.")
    print("Estructura: Features → Target por geografía (USA, EUROPA, ASIA_PACIFICO, etc.)")
    
    return {
        'features_dict': features_dict,
        'geography_features': geography_features,
        'target_by_geo': target_by_geo,
        'ml_dataset': ml_dataset
    }

if __name__ == "__main__":
    results = main()
