# -*- coding: utf-8 -*-
"""
=============================================================================
OBSERVATORIO INTEGRADO DE INDICADORES SOCIOECONÓMICOS DE COLOMBIA
FASE 4 – Aplicación interactiva (Streamlit)
=============================================================================
Integra los productos de las fases previas:
  • Fase 1 – Diagnóstico estructural y de calidad del libro Excel fuente.
  • Fase 2 – Panel longitudinal homologado (panel_observatorio_fase2.parquet)
             y catálogo maestro (catalogo_indicadores.csv).
  • Fase 3 – EDA, rankings, correlaciones, IVR y brechas urbano-rurales.
  • Fase 4 – Este tablero: EDA, PCA, econometría espacial, VAR/VEC, ML y XAI.

Ejecución:
    streamlit run app.py

Estructura esperada:
    app.py
    data/panel_observatorio_fase2.parquet
    data/catalogo_indicadores.csv
    data/colombia_departamentos.geojson      (opcional: respaldo offline del mapa)

Dependencias principales:
    streamlit plotly pandas numpy pyarrow scikit-learn scipy statsmodels
Opcionales (con respaldo interno si faltan):
    libpysal esda geopandas shapely shap requests
=============================================================================
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dependencias opcionales (la app degrada con elegancia si no están)
# ---------------------------------------------------------------------------
try:
    from scipy import stats as sps
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    HAS_SM = True
except Exception:  # pragma: no cover
    HAS_SM = False

try:
    import libpysal
    from esda.moran import Moran, Moran_Local
    HAS_PYSAL = True
except Exception:  # pragma: no cover
    HAS_PYSAL = False

try:
    import geopandas as gpd
    from shapely.geometry import shape as shp_shape
    HAS_GPD = True
except Exception:  # pragma: no cover
    HAS_GPD = False
    try:
        from shapely.geometry import shape as shp_shape
        HAS_SHAPELY = True
    except Exception:
        HAS_SHAPELY = False
else:
    HAS_SHAPELY = True

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False

# ---------------------------------------------------------------------------
# Configuración de página (debe ser la primera llamada a Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Observatorio Socioeconómico de Colombia · Fase 4",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTES
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIRS = [APP_DIR / "data", Path.cwd() / "data", APP_DIR, Path.cwd()]
PANEL_FILE = "panel_observatorio_fase2.parquet"
PANEL_FILE_CSV = "panel_observatorio_fase2.csv"
CATALOGO_FILE = "catalogo_indicadores.csv"
GEOJSON_LOCAL = "colombia_departamentos.geojson"
GEOJSON_URL_OFICIAL = "https://raw.githubusercontent.com/mledoze/countries/master/data/col.geo.json"
GEOJSON_URL_DEPTOS = (
    "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"
)

TOTAL_NACIONAL = "Total nacional"
DESAG_TOTAL = "Total"
DESAG_CAB = "Cabeceras"
DESAG_RURAL = "Centros poblados y rural disperso"

AZUL = "#1E3A8A"
AZUL_OSC = "#0F172A"
AZUL_MED = "#2563EB"
PIZARRA = "#334155"
PIZARRA_CL = "#64748B"
ESMERALDA = "#10B981"
DORADO = "#D4A017"
ROJO = "#DC2626"
PALETA = ["#1E3A8A", "#10B981", "#D4A017", "#DC2626", "#7C3AED", "#0EA5E9",
          "#F97316", "#334155", "#14B8A6", "#DB2777", "#65A30D", "#A16207"]

# Metadatos de indicadores: (etiqueta corta, polaridad, transformación sugerida)
# polaridad: +1 → mayor es mejor ; -1 → mayor es peor ; 0 → neutro (volúmenes)
META_IND = {
    "PIB_volumen_encadenado": ("PIB volumen encadenado", +1, "log"),
    "PIB_tasa_crecimiento": ("Crecimiento del PIB (%)", +1, None),
    "PIB_contribucion_crecimiento": ("Contribución al crecimiento (pp)", +1, None),
    "PIB_per_capita": ("PIB per cápita", +1, "log"),
    "IDC": ("Índice Dptal. de Competitividad", +1, None),
    "incidencia_pobreza_monetaria": ("Pobreza monetaria (%)", -1, None),
    "personas_pobreza_monetaria": ("Personas en pobreza (miles)", -1, "log"),
    "coeficiente_gini": ("Coeficiente de Gini", -1, None),
    "IPM": ("Pobreza multidimensional IPM (%)", -1, None),
    "Tasa de Desocupación (TD)": ("Desempleo – TD (%)", -1, None),
    "Tasa de Ocupación (TO)": ("Ocupación – TO (%)", +1, None),
    "Tasa Global de Participación (TGP)": ("Participación – TGP (%)", +1, None),
    "Tasa de Subocupación (TS)": ("Subocupación – TS (%)", -1, None),
    "% población en edad de trabajar": ("% Población en edad de trabajar", 0, None),
    "Población total": ("Población total (miles)", 0, "log"),
    "Población en edad de trabajar (PET)": ("PET (miles)", 0, "log"),
    "Fuerza de trabajo": ("Fuerza de trabajo (miles)", 0, "log"),
    "Población ocupada": ("Población ocupada (miles)", 0, "log"),
    "Población desocupada": ("Población desocupada (miles)", 0, "log"),
    "Población fuera de la fuerza de trabajo": ("Fuera de la fuerza de trabajo (miles)", 0, "log"),
    "Población subocupada": ("Población subocupada (miles)", 0, "log"),
    "Fuerza de trabajo potencial": ("Fuerza de trabajo potencial (miles)", 0, "log"),
}

DESC_INDICADORES = {
    "PIB_volumen_encadenado": "PIB departamental en series encadenadas de volumen, año de referencia 2015",
    "PIB_tasa_crecimiento": "Tasa de crecimiento anual del PIB en volumen",
    "PIB_contribucion_crecimiento": "Contribución del departamento al crecimiento del PIB nacional",
    "PIB_per_capita": "PIB departamental por habitante a precios corrientes",
    "IDC": "Índice Departamental de Competitividad (Consejo Privado de Competitividad)",
    "incidencia_pobreza_monetaria": "Porcentaje de personas en situación de pobreza monetaria",
    "personas_pobreza_monetaria": "Número de personas en pobreza monetaria (miles)",
    "coeficiente_gini": "Coeficiente de Gini de desigualdad del ingreso (escala 0-1)",
    "IPM": "Incidencia de pobreza multidimensional",
    "% población en edad de trabajar": "Porcentaje de la población de 15 años y más",
    "Tasa Global de Participación (TGP)": "Fuerza de trabajo / PET × 100",
    "Tasa de Ocupación (TO)": "Ocupados / PET × 100",
    "Tasa de Desocupación (TD)": "Desocupados / Fuerza de trabajo × 100",
    "Tasa de Subocupación (TS)": "Subocupados / Fuerza de trabajo × 100",
    "Población total": "Población total (miles)",
    "Población en edad de trabajar (PET)": "Población de 15 años y más (miles)",
    "Fuerza de trabajo": "Ocupados + desocupados (miles)",
    "Población ocupada": "Personas ocupadas (miles)",
    "Población desocupada": "Personas desocupadas (miles)",
    "Población fuera de la fuerza de trabajo": "PET − Fuerza de trabajo (miles)",
    "Población subocupada": "Personas subocupadas (miles)",
    "Fuerza de trabajo potencial": "Fuerza de trabajo potencial (miles)",
}

INDICADORES_CLAVE = [
    "PIB_per_capita", "PIB_tasa_crecimiento", "IDC", "incidencia_pobreza_monetaria",
    "coeficiente_gini", "IPM", "Tasa de Desocupación (TD)", "Tasa de Ocupación (TO)",
    "Tasa Global de Participación (TGP)",
]
DEPTOS_REFERENCIA = ["Bogotá D.C.", "Antioquia", "Valle del Cauca", "Atlántico", "Chocó",
                     "La Guajira", "Vaupés", "Vichada"]

# Homologación nombre canónico → código DIVIPOLA (Fase 2) y centroides aproximados (lat, lon)
DIVIPOLA = {
    "Total nacional": "00", "Amazonas": "91", "Antioquia": "05", "Arauca": "81", "Atlántico": "08",
    "Bogotá D.C.": "11", "Bolívar": "13", "Boyacá": "15", "Caldas": "17", "Caquetá": "18",
    "Casanare": "85", "Cauca": "19", "Cesar": "20", "Chocó": "27", "Córdoba": "23",
    "Cundinamarca": "25", "Guainía": "94", "Guaviare": "95", "Huila": "41", "La Guajira": "44",
    "Magdalena": "47", "Meta": "50", "Nariño": "52", "Norte de Santander": "54", "Putumayo": "86",
    "Quindío": "63", "Risaralda": "66", "Archipiélago de San Andrés": "88", "Santander": "68",
    "Sucre": "70", "Tolima": "73", "Valle del Cauca": "76", "Vaupés": "97", "Vichada": "99",
}
CODIGO_A_NOMBRE = {v: k for k, v in DIVIPOLA.items()}
CENTROIDES = {
    "05": (6.70, -75.50), "08": (10.70, -74.95), "11": (4.45, -74.20), "13": (8.70, -74.50),
    "15": (5.55, -73.35), "17": (5.30, -75.30), "18": (0.90, -73.80), "19": (2.40, -76.80),
    "20": (9.35, -73.55), "23": (8.35, -75.80), "25": (4.95, -74.10), "27": (5.70, -76.65),
    "41": (2.55, -75.55), "44": (11.35, -72.50), "47": (10.25, -74.20), "50": (3.30, -73.00),
    "52": (1.45, -77.70), "54": (7.95, -72.85), "63": (4.45, -75.70), "66": (5.10, -75.95),
    "68": (6.70, -73.30), "70": (9.00, -75.10), "73": (3.95, -75.20), "76": (3.85, -76.50),
    "81": (6.55, -71.00), "85": (5.35, -71.80), "86": (0.45, -75.90), "88": (12.55, -81.72),
    "91": (-1.40, -71.60), "94": (2.70, -68.80), "95": (2.05, -72.30), "97": (0.60, -70.60),
    "99": (4.80, -69.30),
}
REGIONES = {
    "Caribe": ["08", "13", "20", "23", "44", "47", "70", "88"],
    "Andina": ["05", "11", "15", "17", "25", "41", "54", "63", "66", "68", "73"],
    "Pacífica": ["19", "27", "52", "76"],
    "Orinoquía": ["50", "81", "85", "99"],
    "Amazonía": ["18", "86", "91", "94", "95", "97"],
}
CODIGO_A_REGION = {c: r for r, cs in REGIONES.items() for c in cs}

MODULOS = [
    "🏛️ Bienvenida y Marco del Observatorio",
    "📊 Explorador Territorial, Rankings y Brechas",
    "📈 Evolución Temporal y Comparativa",
    "🧪 EDA y Reducción de Dimensionalidad (PCA)",
    "🗺️ Econometría Espacial y Cartografía",
    "⏳ Series de Tiempo (VAR / VEC)",
    "🤖 Machine Learning y Minería de Datos",
    "🎛️ Simulador de Política e XAI",
    "📋 Ficha Departamental y Descargas",
]

# =============================================================================
# ESTILO: CSS + PLANTILLA PLOTLY
# =============================================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}

/* ---------- Barra lateral ---------- */
section[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {AZUL_OSC} 0%, {AZUL} 100%);
}}
section[data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    padding: .42rem .6rem; border-radius: 10px; margin-bottom: 2px; transition: background .2s ease;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{ background: rgba(255,255,255,.09); }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.18); }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
    background: rgba(255,255,255,.06); border: 1px dashed rgba(255,255,255,.35);
}}
.sb-brand {{ text-align:center; padding: .4rem 0 .9rem 0; }}
.sb-brand .flag {{ font-size: 2.3rem; line-height: 1; }}
.sb-brand .t1 {{ font-weight: 800; font-size: 1.05rem; letter-spacing: .3px; color:#FFFFFF !important; }}
.sb-brand .t2 {{ font-size: .78rem; color: #FCD34D !important; letter-spacing: 1.6px; text-transform: uppercase; }}
.sb-status {{ font-size:.78rem; background: rgba(255,255,255,.07); border-radius: 10px; padding:.6rem .75rem; }}

/* ---------- Hero ---------- */
.hero {{
    background: radial-gradient(1200px 400px at 85% -20%, rgba(212,160,23,.35), transparent 60%),
                linear-gradient(120deg, {AZUL_OSC} 0%, {AZUL} 55%, #1D4ED8 100%);
    color: #FFFFFF; border-radius: 22px; padding: 2.1rem 2.4rem; margin-bottom: 1.3rem;
    box-shadow: 0 18px 40px -18px rgba(15,23,42,.55); position: relative; overflow: hidden;
}}
.hero h1 {{ color:#FFFFFF; font-size: 2.05rem; font-weight: 800; margin: 0 0 .35rem 0; line-height:1.15; }}
.hero p {{ color: #DBEAFE; font-size: 1.02rem; margin: 0; max-width: 950px; }}
.hero .badge {{
    display:inline-block; background: rgba(252,211,77,.16); color:#FCD34D; border:1px solid rgba(252,211,77,.45);
    border-radius: 999px; padding: .18rem .75rem; font-size:.75rem; font-weight:700; letter-spacing:1.2px;
    text-transform: uppercase; margin-bottom:.8rem;
}}

/* ---------- Encabezado de módulo ---------- */
.mod-header {{
    display:flex; align-items:center; gap: 1rem; padding: 1.05rem 1.35rem; border-radius: 18px;
    background: linear-gradient(110deg, {AZUL} 0%, #1D4ED8 70%, #2563EB 100%); color:#fff;
    margin-bottom: 1.1rem; box-shadow: 0 12px 28px -16px rgba(30,58,138,.65);
}}
.mod-header .ico {{ font-size: 2.1rem; background: rgba(255,255,255,.12); border-radius: 14px; padding: .35rem .6rem; }}
.mod-header h2 {{ color:#fff; margin:0; font-size: 1.45rem; font-weight: 800; }}
.mod-header p {{ color:#DBEAFE; margin:.15rem 0 0 0; font-size:.92rem; }}

/* ---------- Tarjetas métricas ---------- */
.metric-card {{
    background: #FFFFFF; border-radius: 16px; padding: 1rem 1.1rem .9rem 1.1rem;
    border: 1px solid #E2E8F0; border-left: 5px solid {AZUL};
    box-shadow: 0 6px 18px -10px rgba(15,23,42,.30); transition: transform .18s ease, box-shadow .18s ease;
    height: 100%; min-height: 118px;
}}
.metric-card:hover {{ transform: translateY(-3px); box-shadow: 0 14px 28px -14px rgba(15,23,42,.40); }}
.metric-card .mc-top {{ display:flex; justify-content:space-between; align-items:flex-start; }}
.metric-card .mc-title {{ color:{PIZARRA_CL}; font-size:.74rem; font-weight:700; letter-spacing:.6px; text-transform:uppercase; }}
.metric-card .mc-ico {{ font-size:1.25rem; opacity:.9; }}
.metric-card .mc-value {{ color:{AZUL_OSC}; font-size:1.65rem; font-weight:800; margin:.25rem 0 .1rem 0; line-height:1.1; word-break: break-word; }}
.metric-card .mc-sub {{ color:{PIZARRA_CL}; font-size:.78rem; }}
.metric-card .mc-delta {{ display:inline-block; font-size:.76rem; font-weight:700; border-radius:999px; padding:.08rem .55rem; margin-top:.3rem; }}
.mc-good {{ background:#D1FAE5; color:#047857; }}
.mc-bad {{ background:#FEE2E2; color:#B91C1C; }}
.mc-neutral {{ background:#E2E8F0; color:#334155; }}
.accent-emerald {{ border-left-color: {ESMERALDA}; }}
.accent-gold {{ border-left-color: {DORADO}; }}
.accent-red {{ border-left-color: {ROJO}; }}
.accent-slate {{ border-left-color: {PIZARRA}; }}

/* ---------- Callouts ---------- */
.callout {{ border-radius: 14px; padding: .85rem 1.05rem; margin: .4rem 0 .9rem 0; font-size: .9rem;
           border: 1px solid; line-height: 1.5; }}
.callout-info {{ background:#EFF6FF; border-color:#BFDBFE; color:#1E3A8A; }}
.callout-warn {{ background:#FFFBEB; border-color:#FDE68A; color:#92400E; }}
.callout-ok {{ background:#ECFDF5; border-color:#A7F3D0; color:#065F46; }}
.callout b {{ font-weight: 700; }}

/* ---------- Flujo metodológico ---------- */
.phase-wrap {{ display:flex; gap:.7rem; align-items:stretch; flex-wrap:wrap; }}
.phase {{ flex:1 1 210px; background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:1rem 1.05rem;
         box-shadow: 0 6px 16px -10px rgba(15,23,42,.25); position:relative; }}
.phase .num {{ display:inline-flex; width:2rem; height:2rem; border-radius:999px; align-items:center; justify-content:center;
              background:{AZUL}; color:#fff; font-weight:800; margin-bottom:.45rem; }}
.phase.active {{ border:2px solid {DORADO}; background: linear-gradient(180deg,#FFFBEB 0%, #FFFFFF 60%); }}
.phase.active .num {{ background:{DORADO}; }}
.phase h4 {{ margin:.1rem 0 .35rem 0; color:{AZUL_OSC}; font-size:1rem; }}
.phase ul {{ margin:0; padding-left:1.05rem; color:{PIZARRA}; font-size:.84rem; }}
.phase-arrow {{ align-self:center; color:{DORADO}; font-size:1.6rem; font-weight:800; }}

/* ---------- Pestañas ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 2px solid #E2E8F0; }}
.stTabs [data-baseweb="tab"] {{
    background:#F1F5F9; border-radius: 12px 12px 0 0; padding: .55rem 1rem; font-weight: 600; color:{PIZARRA};
}}
.stTabs [aria-selected="true"] {{ background:{AZUL} !important; color:#FFFFFF !important; }}
.stTabs [aria-selected="true"] p {{ color:#FFFFFF !important; }}

/* ---------- Gráficos y tablas como tarjetas ---------- */
[data-testid="stPlotlyChart"] {{
    background:#FFFFFF; border-radius:16px; border:1px solid #E2E8F0; padding:.35rem;
    box-shadow: 0 6px 18px -12px rgba(15,23,42,.28);
}}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
.stDownloadButton button, .stButton button {{
    border-radius: 12px; font-weight: 700; border: 1px solid {AZUL};
}}
.stDownloadButton button {{ background:{ESMERALDA}; color:#fff; border-color:{ESMERALDA}; }}
.stDownloadButton button:hover {{ background:#059669; color:#fff; }}
.foot {{ color:{PIZARRA_CL}; font-size:.78rem; text-align:center; margin-top:2.2rem; }}
.pill {{ display:inline-block; background:#E0E7FF; color:{AZUL}; border-radius:999px; padding:.1rem .6rem;
         font-size:.75rem; font-weight:700; margin:.1rem .15rem; }}
</style>
"""


def configurar_plotly() -> None:
    """Registra una plantilla Plotly coherente con la identidad visual."""
    pio.templates["observatorio"] = go.layout.Template(
        layout=dict(
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1E293B", size=12.5),
            colorway=PALETA,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            title=dict(font=dict(size=16, color=AZUL_OSC), x=0.01, xanchor="left"),
            hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#CBD5E1", font=dict(size=12, color="#0F172A")),
            xaxis=dict(gridcolor="#EEF2F7", zerolinecolor="#CBD5E1", linecolor="#CBD5E1"),
            yaxis=dict(gridcolor="#EEF2F7", zerolinecolor="#CBD5E1", linecolor="#CBD5E1"),
            legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0", borderwidth=1,
                        font=dict(size=11)),
            margin=dict(l=20, r=20, t=64, b=24),
            modebar=dict(orientation="v", bgcolor="rgba(255,255,255,0.6)"),
            coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=14)),
        )
    )
    pio.templates.default = "plotly_white+observatorio"


# =============================================================================
# UTILIDADES DE PRESENTACIÓN
# =============================================================================
def _version_streamlit() -> tuple:
    nums = re.findall(r"\d+", st.__version__)
    return tuple(int(x) for x in nums[:2]) if nums else (0, 0)


_ANCHO_NUEVO = _version_streamlit() >= (1, 50)


def mostrar_fig(fig: go.Figure, key: str | None = None) -> None:
    """Muestra una figura Plotly con zoom, barra de herramientas y plantilla propia."""
    config = {"displaylogo": False, "scrollZoom": True,
              "toImageButtonOptions": {"format": "png", "scale": 2}}
    if _ANCHO_NUEVO:
        st.plotly_chart(fig, width="stretch", theme=None, key=key, config=config)
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key, config=config)


def mostrar_tabla(df: pd.DataFrame, **kwargs) -> None:
    kwargs.setdefault("hide_index", True)
    if _ANCHO_NUEVO:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


def fmt_num(v, dec: int | None = None, unidad: str = "") -> str:
    """Formato numérico es-CO (miles con punto, decimales con coma)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "—"
    v = float(v)
    if dec is None:
        av = abs(v)
        dec = 0 if av >= 1000 else (1 if av >= 100 else (2 if av >= 1 else 3))
    if abs(v) >= 1e12:
        txt = f"{v / 1e12:,.2f} B"
    elif abs(v) >= 1e9:
        txt = f"{v / 1e9:,.2f} mil M"
    elif abs(v) >= 1e6:
        txt = f"{v / 1e6:,.2f} M"
    else:
        txt = f"{v:,.{dec}f}"
    txt = txt.replace(",", "§").replace(".", ",").replace("§", ".")
    return f"{txt}{(' ' + unidad) if unidad else ''}"


def fmt_pct(v, dec: int = 1, signo: bool = False) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "—"
    txt = f"{v * 100:{'+' if signo else ''}.{dec}f}".replace(".", ",")
    return f"{txt} %"


def etiqueta(ind: str) -> str:
    return META_IND.get(ind, (ind, 0, None))[0]


def polaridad(ind: str) -> int:
    if ind in META_IND:
        return META_IND[ind][1]
    t = str(ind).lower()
    if any(k in t for k in ["desocup", "pobreza", "gini", "ipm", "subocup"]):
        return -1
    if any(k in t for k in ["pib", "ocupación", "competitiv", "participación"]):
        return +1
    return 0


def transformacion_sugerida(ind: str):
    return META_IND.get(ind, (ind, 0, None))[2]


def normalizar_texto(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9 ]", " ", s.upper()).strip()


def encabezado(icono: str, titulo: str, subtitulo: str) -> None:
    st.markdown(
        f"""<div class="mod-header"><div class="ico">{icono}</div>
        <div><h2>{titulo}</h2><p>{subtitulo}</p></div></div>""",
        unsafe_allow_html=True,
    )


def callout(texto: str, tipo: str = "info") -> None:
    st.markdown(f'<div class="callout callout-{tipo}">{texto}</div>', unsafe_allow_html=True)


def tarjeta_html(titulo: str, valor: str, sub: str = "", icono: str = "", delta: float | None = None,
                 delta_txt: str | None = None, polar: int = 0, acento: str = "") -> str:
    """Tarjeta métrica con indicador de tendencia. polar=+1: subir es bueno; -1: subir es malo."""
    delta_html = ""
    if delta is not None and np.isfinite(delta):
        flecha = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
        if polar == 0 or delta == 0:
            clase = "mc-neutral"
        else:
            clase = "mc-good" if (delta > 0) == (polar > 0) else "mc-bad"
        delta_html = f'<span class="mc-delta {clase}">{flecha} {delta_txt or fmt_num(delta)}</span>'
    return f"""<div class="metric-card {acento}">
        <div class="mc-top"><div class="mc-title">{titulo}</div><div class="mc-ico">{icono}</div></div>
        <div class="mc-value">{valor}</div>
        <div class="mc-sub">{sub}</div>{delta_html}</div>"""


def fila_tarjetas(tarjetas: list[dict], n_cols: int | None = None) -> None:
    if not tarjetas:
        return
    cols = st.columns(n_cols or len(tarjetas))
    for c, t in zip(cols, tarjetas):
        with c:
            st.markdown(tarjeta_html(**t), unsafe_allow_html=True)


def escala_color(ind: str) -> str:
    p = polaridad(ind)
    return "YlOrRd" if p < 0 else ("Tealgrn" if p > 0 else "Blues")


def descargar_csv(df: pd.DataFrame, nombre: str, etiqueta_btn: str = "⬇️ Descargar CSV", key=None) -> None:
    st.download_button(etiqueta_btn, df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=nombre, mime="text/csv", key=key)


# =============================================================================
# CARGA Y NORMALIZACIÓN DE DATOS
# =============================================================================
COLUMNAS_PANEL = ["departamento_canonico", "codigo_divipola", "anio", "indicador",
                  "desagregacion", "valor", "unidad", "fuente", "marca_calidad"]


def _normalizar_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    faltan = [c for c in ["departamento_canonico", "anio", "indicador", "valor"] if c not in df.columns]
    if faltan:
        raise ValueError(f"El panel no contiene las columnas obligatorias: {faltan}")
    for c in COLUMNAS_PANEL:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["departamento_canonico", "indicador", "desagregacion", "unidad", "fuente", "marca_calidad"]:
        df[c] = df[c].astype("object").where(df[c].notna(), None)
        df[c] = df[c].map(lambda x: str(x).strip() if x is not None else None)
    df["desagregacion"] = df["desagregacion"].fillna(DESAG_TOTAL)
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df[df["anio"].notna() & df["indicador"].notna() & df["departamento_canonico"].notna()].copy()
    df["anio"] = df["anio"].astype(int)
    # Código DIVIPOLA como texto de 2 dígitos (recupera desde el nombre si falta)
    cod = df["codigo_divipola"].astype("object")
    cod = cod.map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() in
                  ("", "nan", "None", "<NA>") else re.sub(r"\.0$", "", str(x).strip()).zfill(2))
    cod = cod.where(cod.notna(), df["departamento_canonico"].map(DIVIPOLA))
    df["codigo_divipola"] = cod
    # Duplicados exactos de clave (se conserva la primera observación no nula)
    df = (df.sort_values("valor", na_position="last")
            .drop_duplicates(subset=["departamento_canonico", "anio", "indicador", "desagregacion"], keep="first")
            .sort_values(["indicador", "departamento_canonico", "desagregacion", "anio"])
            .reset_index(drop=True))
    return df


def _catalogo_desde_panel(panel: pd.DataFrame) -> pd.DataFrame:
    cat = panel.groupby("indicador").agg(
        unidad=("unidad", "first"), fuente=("fuente", "first"), n_obs=("valor", "count"),
        n_territorios=("departamento_canonico", "nunique"), anio_min=("anio", "min"),
        anio_max=("anio", "max"),
    ).reset_index()
    cat["descripcion"] = cat["indicador"].map(DESC_INDICADORES)
    return cat


def _normalizar_catalogo(cat: pd.DataFrame | None, panel: pd.DataFrame) -> pd.DataFrame:
    base = _catalogo_desde_panel(panel)
    if cat is None or cat.empty or "indicador" not in cat.columns:
        out = base
    else:
        cat = cat.copy()
        cat.columns = [str(c).strip().lstrip("﻿") for c in cat.columns]
        out = cat.merge(base, on="indicador", how="outer", suffixes=("", "_panel"))
        for c in ["unidad", "fuente", "n_territorios", "anio_min", "anio_max", "n_obs", "descripcion"]:
            if f"{c}_panel" in out.columns:
                if c in out.columns:
                    out[c] = out[c].where(out[c].notna(), out[f"{c}_panel"])
                else:
                    out[c] = out[f"{c}_panel"]
                out = out.drop(columns=[f"{c}_panel"])
    if "descripcion" not in out.columns:
        out["descripcion"] = None
    out["descripcion"] = out["descripcion"].where(out["descripcion"].notna(), out["indicador"].map(DESC_INDICADORES))
    out["descripcion"] = out["descripcion"].fillna("—")
    out["institucion"] = out["fuente"].fillna("").map(
        lambda f: "CPC" if "competitividad" in f.lower() or "cpc" in f.lower()
        else ("DANE" if "dane" in f.lower() else (f.split("-")[0].strip() or "Otra")))
    out["etiqueta"] = out["indicador"].map(etiqueta)
    for c in ["anio_min", "anio_max", "n_territorios", "n_obs"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    cols = ["indicador", "etiqueta", "descripcion", "unidad", "fuente", "institucion",
            "anio_min", "anio_max", "n_territorios", "n_obs"]
    return out[[c for c in cols if c in out.columns]].sort_values("indicador").reset_index(drop=True)


@st.cache_data(show_spinner="Leyendo panel del observatorio…")
def leer_panel_ruta(ruta: str, _mtime: float) -> pd.DataFrame:
    p = Path(ruta)
    raw = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, encoding="utf-8-sig")
    return _normalizar_panel(raw)


@st.cache_data(show_spinner="Leyendo panel cargado…")
def leer_panel_bytes(contenido: bytes, nombre: str) -> pd.DataFrame:
    import io
    bio = io.BytesIO(contenido)
    raw = pd.read_parquet(bio) if nombre.lower().endswith(".parquet") else pd.read_csv(bio, encoding="utf-8-sig")
    return _normalizar_panel(raw)


@st.cache_data(show_spinner=False)
def leer_catalogo(ruta: str | None, contenido: bytes | None, _panel_firma: int, panel: pd.DataFrame) -> pd.DataFrame:
    import io
    cat = None
    try:
        if contenido is not None:
            cat = pd.read_csv(io.BytesIO(contenido), encoding="utf-8-sig")
        elif ruta is not None:
            cat = pd.read_csv(ruta, encoding="utf-8-sig")
    except Exception:
        cat = None
    return _normalizar_catalogo(cat, panel)


def _buscar_archivo(nombres: list[str]) -> Path | None:
    for d in DATA_DIRS:
        for n in nombres:
            p = d / n
            if p.exists():
                return p
    return None


def _codigo_desde_props(props: dict) -> str | None:
    for k in ["codigo_divipola", "DPTO_CCDGO", "DPTO", "dpto", "DPTO_CODE", "COD_DEPTO", "cod_depto", "id"]:
        if k in props and props[k] not in (None, ""):
            v = re.sub(r"\.0$", "", str(props[k]).strip())
            if v.isdigit() and len(v) <= 2:
                return v.zfill(2)
    for k in ["DPTO_CNMBR", "NOMBRE_DPT", "nombre", "name", "NAME_1", "departamento"]:
        if k in props and props[k]:
            n = normalizar_texto(props[k])
            if "BOGOTA" in n:
                return "11"
            if "SAN ANDRES" in n:
                return "88"
            for canon, cod in DIVIPOLA.items():
                if normalizar_texto(canon) == n:
                    return cod
    return None


def _preparar_geojson(gj: dict) -> dict | None:
    """Valida que el GeoJSON sea departamental y añade properties.codigo_divipola."""
    if not isinstance(gj, dict) or "features" not in gj:
        return None
    feats = []
    for f in gj["features"]:
        props = dict(f.get("properties") or {})
        cod = _codigo_desde_props(props)
        if cod is None or cod not in CODIGO_A_NOMBRE:
            continue
        props["codigo_divipola"] = cod
        props["departamento_canonico"] = CODIGO_A_NOMBRE[cod]
        feats.append({"type": "Feature", "geometry": f.get("geometry"), "properties": props})
    if len(feats) < 20:
        return None
    return {"type": "FeatureCollection", "features": feats}


def _descargar_json(url: str, timeout: int = 25) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Observatorio-Colombia/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner="Cargando cartografía departamental…", ttl=86400)
def cargar_geojson() -> tuple[dict | None, str, list[str]]:
    """Orden de búsqueda: archivo local → URL indicada (mledoze) → MGN DANE 2018 (GitHub)."""
    notas: list[str] = []
    local = _buscar_archivo([GEOJSON_LOCAL])
    if local is not None:
        try:
            gj = _preparar_geojson(json.loads(local.read_text(encoding="utf-8")))
            if gj:
                return gj, f"Archivo local ({local.name})", notas
            notas.append("El GeoJSON local no tiene geometrías departamentales reconocibles.")
        except Exception as e:
            notas.append(f"No se pudo leer el GeoJSON local: {e}")
    gj_raw = _descargar_json(GEOJSON_URL_OFICIAL)
    if gj_raw is None:
        notas.append("No fue posible descargar el GeoJSON de mledoze/countries.")
    else:
        gj = _preparar_geojson(gj_raw)
        if gj:
            return gj, "mledoze/countries", notas
        n = len(gj_raw.get("features", []))
        notas.append(f"El GeoJSON de mledoze/countries contiene {n} geometría(s) (contorno nacional), "
                     "no polígonos departamentales; se usa el Marco Geoestadístico Nacional (DANE, 2018).")
    gj_raw = _descargar_json(GEOJSON_URL_DEPTOS, timeout=40)
    if gj_raw is not None:
        gj = _preparar_geojson(gj_raw)
        if gj:
            return gj, "DANE – MGN 2018 (vía GitHub)", notas
    notas.append("Sin cartografía poligonal disponible: se usará un mapa de burbujas por centroides.")
    return None, "Centroides aproximados", notas


def obtener_datos():
    """Carga panel y catálogo desde /data; si no existen, habilita la carga manual."""
    ruta_panel = _buscar_archivo([PANEL_FILE, PANEL_FILE_CSV])
    ruta_cat = _buscar_archivo([CATALOGO_FILE])
    panel, origen = None, ""
    if ruta_panel is not None:
        try:
            panel = leer_panel_ruta(str(ruta_panel), ruta_panel.stat().st_mtime)
            origen = f"data/{ruta_panel.name}"
        except Exception as e:
            st.sidebar.error(f"Error leyendo {ruta_panel.name}: {e}")
    cat_bytes = None
    if panel is None:
        with st.sidebar:
            st.markdown("#### 📂 Cargar datos")
            up = st.file_uploader("Panel (parquet o csv)", type=["parquet", "csv"], key="up_panel")
            upc = st.file_uploader("Catálogo de indicadores (csv)", type=["csv"], key="up_cat")
        if up is not None:
            try:
                panel = leer_panel_bytes(up.getvalue(), up.name)
                origen = f"Carga manual ({up.name})"
            except Exception as e:
                st.sidebar.error(f"Archivo no válido: {e}")
        if upc is not None:
            cat_bytes = upc.getvalue()
    if panel is None:
        return None, None, ""
    firma = int(pd.util.hash_pandas_object(panel[["indicador", "anio"]].head(500), index=False).sum() % 1e12) + len(panel)
    catalogo = leer_catalogo(str(ruta_cat) if (ruta_cat is not None and cat_bytes is None) else None,
                             cat_bytes, firma, panel)
    return panel, catalogo, origen


# =============================================================================
# FUNCIONES ANALÍTICAS COMUNES
# =============================================================================
def solo_deptos(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[(panel["departamento_canonico"] != TOTAL_NACIONAL) & panel["departamento_canonico"].notna()]


def nacional(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["departamento_canonico"] == TOTAL_NACIONAL]


def ordenar_indicadores(inds) -> list[str]:
    inds = list(dict.fromkeys(inds))
    clave = [i for i in INDICADORES_CLAVE if i in inds]
    return clave + sorted([i for i in inds if i not in clave])


def indicadores_con_datos(df: pd.DataFrame) -> list[str]:
    return ordenar_indicadores(df.dropna(subset=["valor"])["indicador"].unique().tolist())


def desags_de(df: pd.DataFrame, ind: str) -> list[str]:
    d = df[(df["indicador"] == ind) & df["valor"].notna()]["desagregacion"].dropna().unique().tolist()
    orden = [DESAG_TOTAL, DESAG_CAB, DESAG_RURAL, "Hombre", "Mujer"]
    return [x for x in orden if x in d] + sorted([x for x in d if x not in orden])


def anios_de(df: pd.DataFrame, ind: str, desag: str = DESAG_TOTAL) -> list[int]:
    s = df[(df["indicador"] == ind) & (df["desagregacion"] == desag) & df["valor"].notna()]["anio"]
    return sorted(s.unique().tolist())


def unidad_de(panel: pd.DataFrame, ind: str) -> str:
    u = panel.loc[panel["indicador"] == ind, "unidad"].dropna()
    return str(u.iloc[0]) if len(u) else ""


def corte(df: pd.DataFrame, ind: str, anio: int, desag: str = DESAG_TOTAL) -> pd.DataFrame:
    return df[(df["indicador"] == ind) & (df["anio"] == anio) & (df["desagregacion"] == desag)
              & df["valor"].notna()].copy()


def anio_con_mejor_cobertura(deptos: pd.DataFrame, inds: list[str], min_deptos: int = 15) -> int:
    sub = deptos[deptos["indicador"].isin(inds) & (deptos["desagregacion"] == DESAG_TOTAL) & deptos["valor"].notna()]
    if sub.empty:
        return int(deptos["anio"].max())
    cnt = sub.groupby(["anio", "indicador"])["departamento_canonico"].nunique().unstack(fill_value=0)
    score = (cnt >= min_deptos).sum(axis=1) + cnt.sum(axis=1) / 1e4
    return int(score.idxmax())


def matriz_departamental(deptos: pd.DataFrame, anio: int, inds: list[str], desag: str = DESAG_TOTAL,
                         tolerancia: int = 2, imputar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriz departamento × indicador para un año.
    - Pivota con pivot_table.
    - Si un indicador no existe en el año, toma el año disponible más cercano (± tolerancia).
    - Imputa faltantes por la media del indicador en ese año (media transversal).
    Devuelve (matriz, bitácora_de_imputación)."""
    bitacora = []
    columnas = {}
    for ind in inds:
        disponibles = anios_de(deptos, ind, desag)
        if not disponibles:
            bitacora.append({"indicador": ind, "año usado": None, "deptos observados": 0,
                             "imputados (media del año)": 0, "nota": "sin datos – excluido"})
            continue
        if anio in disponibles:
            a = anio
            nota = "año solicitado"
        else:
            a = min(disponibles, key=lambda x: (abs(x - anio), -x))
            if abs(a - anio) > tolerancia:
                bitacora.append({"indicador": ind, "año usado": a, "deptos observados": 0,
                                 "imputados (media del año)": 0,
                                 "nota": f"año más cercano ({a}) fuera de la tolerancia ±{tolerancia} – excluido"})
                continue
            nota = f"sustituido por el año más cercano ({a})"
        sub = corte(deptos, ind, a, desag)
        columnas[ind] = sub.pivot_table(index="departamento_canonico", values="valor", aggfunc="mean")["valor"]
        bitacora.append({"indicador": ind, "año usado": a, "deptos observados": int(columnas[ind].notna().sum()),
                         "imputados (media del año)": 0, "nota": nota})
    if not columnas:
        return pd.DataFrame(), pd.DataFrame(bitacora)
    X = pd.DataFrame(columnas)
    X = X.reindex(sorted(deptos["departamento_canonico"].unique()))
    X = X.dropna(how="all")
    bit = pd.DataFrame(bitacora)
    if imputar:
        for c in X.columns:
            n_na = int(X[c].isna().sum())
            if n_na:
                X[c] = X[c].fillna(X[c].mean())
                bit.loc[bit["indicador"] == c, "imputados (media del año)"] = n_na
    return X, bit


def panel_ancho_dept_anio(deptos: pd.DataFrame, inds: list[str], desag: str = DESAG_TOTAL) -> pd.DataFrame:
    sub = deptos[deptos["indicador"].isin(inds) & (deptos["desagregacion"] == desag)]
    return sub.pivot_table(index=["departamento_canonico", "anio"], columns="indicador",
                           values="valor", aggfunc="mean")


def minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng and np.isfinite(rng) else s * 0 + 0.5


def ols_simple(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0:
        return {}
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b0 + b1 * x
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    p = np.nan
    if HAS_SCIPY:
        p = sps.linregress(x, y).pvalue
    return {"b0": b0, "b1": b1, "r2": r2, "p": p, "n": len(x), "r": np.corrcoef(x, y)[0, 1]}


# =============================================================================
# MÓDULO 1 – BIENVENIDA, CONTEXTUALIZACIÓN Y MARCO DEL OBSERVATORIO
# =============================================================================
def modulo_bienvenida(panel: pd.DataFrame, catalogo: pd.DataFrame) -> None:
    deptos = solo_deptos(panel)
    a_min, a_max = int(panel["anio"].min()), int(panel["anio"].max())
    st.markdown(
        f"""<div class="hero"><span class="badge">Fase 4 · Tablero analítico integrado</span>
        <h1>🇨🇴 Observatorio Integrado de Indicadores Socioeconómicos de Colombia</h1>
        <p>Plataforma de inteligencia territorial que integra cuentas nacionales, mercado laboral, pobreza
        monetaria y multidimensional, desigualdad y competitividad para los 32 departamentos y Bogotá D.C.
        ({a_min}–{a_max}). Combina análisis exploratorio, econometría espacial y de series de tiempo,
        aprendizaje automático e interpretabilidad para apoyar la formulación de política pública basada en evidencia.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    n_val = int(panel["valor"].notna().sum())
    fila_tarjetas([
        dict(titulo="Registros del panel", valor=fmt_num(len(panel), 0), icono="🗂️",
             sub=f"{fmt_num(n_val, 0)} con valor observado ({fmt_pct(n_val / max(len(panel), 1), 0)})"),
        dict(titulo="Cobertura temporal", valor=f"{a_min}–{a_max}", icono="📅",
             sub=f"{a_max - a_min + 1} años en el panel", acento="accent-gold"),
        dict(titulo="Territorios", valor=str(deptos["departamento_canonico"].nunique()), icono="🗺️",
             sub="departamentos + Bogotá D.C." + (" · y Total nacional" if TOTAL_NACIONAL in
                                                  panel["departamento_canonico"].values else ""),
             acento="accent-emerald"),
        dict(titulo="Indicadores", valor=str(panel["indicador"].nunique()), icono="📐",
             sub=f"{panel['fuente'].nunique()} fuentes · {panel['desagregacion'].nunique()} desagregaciones",
             acento="accent-slate"),
    ])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["🎯 Propósito y alcance", "🔄 Flujo metodológico",
                              "📚 Catálogo técnico de indicadores", "🧭 Cobertura del panel"])

    with t1:
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("#### Propósito del Observatorio")
            st.markdown(
                "El Observatorio consolida en un **panel longitudinal homologado** las principales estadísticas "
                "socioeconómicas territoriales de Colombia, con trazabilidad metodológica completa (fuente, "
                "unidad, marca de calidad). Su objetivo es **medir, comparar y explicar** las disparidades "
                "territoriales para orientar decisiones de inversión y focalización.")
            st.markdown("#### Objetivos de desarrollo territorial")
            st.markdown(
                "- **Convergencia regional:** cerrar brechas de ingreso (PIB per cápita) y productividad.\n"
                "- **Reducción de la pobreza:** monetaria y multidimensional (IPM), con énfasis en la ruralidad.\n"
                "- **Equidad:** disminuir la desigualdad del ingreso (Gini) y las brechas por sexo.\n"
                "- **Trabajo decente:** elevar la ocupación y reducir el desempleo y la subocupación.\n"
                "- **Competitividad:** fortalecer instituciones, capital humano e innovación (IDC).")
        with c2:
            st.markdown("#### Alcance analítico")
            alcance = pd.DataFrame({
                "Dimensión": ["Económica", "Laboral", "Pobreza y equidad", "Competitividad", "Espacial",
                              "Dinámica", "Predictiva"],
                "Herramientas": ["PIB volumen, crecimiento, PIB per cápita", "TD, TO, TGP, TS (GEIH)",
                                 "Pobreza monetaria, IPM, Gini, brechas", "IDC (CPC)",
                                 "Mapas, I de Moran, LISA", "VAR, VEC, Granger, IRF",
                                 "Ridge, LASSO, SVR, SHAP, simulación"],
            })
            mostrar_tabla(alcance)
            callout("<b>Advertencias metodológicas permanentes.</b> Pobreza monetaria y Gini sólo 2021–2025 "
                    "(no empalmables con series previas) · GEIH cubre 23 departamentos + Bogotá · IPM 2020–2021 "
                    "con notas operativas · IDC con escala propia (CPC) · PIB per cápita en precios corrientes "
                    "vs. PIB en volumen · Las correlaciones son ecológicas (nivel departamental).", "warn")

    with t2:
        st.markdown(
            """<div class="phase-wrap">
            <div class="phase"><div class="num">1</div><h4>Inspección y diagnóstico</h4><ul>
              <li>Reconocimiento de 8 pestañas del libro fuente</li><li>Bloques, encabezados multinivel y notas</li>
              <li>Alertas de calidad (p, pr, **, ***)</li></ul></div>
            <div class="phase-arrow">➜</div>
            <div class="phase"><div class="num">2</div><h4>Estandarización</h4><ul>
              <li>Homologación territorial + DIVIPOLA</li><li>Transformación wide → long</li>
              <li>Panel integrado y catálogo maestro</li></ul></div>
            <div class="phase-arrow">➜</div>
            <div class="phase"><div class="num">3</div><h4>EDA y compuestos</h4><ul>
              <li>Rankings y estadísticas descriptivas</li><li>Correlaciones y dispersión</li>
              <li>IVR exploratorio y brecha IPM rural–urbana</li></ul></div>
            <div class="phase-arrow">➜</div>
            <div class="phase active"><div class="num">4</div><h4>Observatorio interactivo</h4><ul>
              <li>PCA e índice sintético</li><li>Moran, VAR/VEC, ML</li>
              <li>Simulador de política y XAI (SHAP)</li></ul></div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        # Sankey: fuente → familia de indicador → panel integrado (número de observaciones)
        flujo = panel.dropna(subset=["valor"]).groupby(["fuente", "indicador"]).size().reset_index(name="n")
        flujo["fuente"] = flujo["fuente"].fillna("Sin fuente")
        flujo["familia"] = flujo["indicador"].map(_familia_indicador)
        f1 = flujo.groupby(["fuente", "familia"])["n"].sum().reset_index()
        f2 = flujo.groupby("familia")["n"].sum().reset_index()
        nodos = list(f1["fuente"].unique()) + list(f2["familia"].unique()) + ["Panel integrado (Fase 2)"]
        idx = {n: i for i, n in enumerate(nodos)}
        src = [idx[a] for a in f1["fuente"]] + [idx[a] for a in f2["familia"]]
        tgt = [idx[b] for b in f1["familia"]] + [idx["Panel integrado (Fase 2)"]] * len(f2)
        val = list(f1["n"]) + list(f2["n"])
        colores_nodo = [PALETA[i % len(PALETA)] for i in range(len(nodos))]
        colores_nodo[-1] = DORADO
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(label=nodos, pad=18, thickness=18, color=colores_nodo, line=dict(width=0)),
            link=dict(source=src, target=tgt, value=val, color="rgba(30,58,138,0.18)",
                      hovertemplate="%{source.label} → %{target.label}<br>%{value:,} observaciones<extra></extra>"),
        ))
        fig.update_layout(title="Trazabilidad de los datos: fuente → familia de indicadores → panel", height=430)
        mostrar_fig(fig, key="sankey")

    with t3:
        c1, c2, c3 = st.columns([1, 1.3, 1.2])
        inst = c1.multiselect("Institución", sorted(catalogo["institucion"].dropna().unique()),
                              key="cat_inst")
        fuentes = c2.multiselect("Fuente", sorted(catalogo["fuente"].dropna().unique()), key="cat_fuente")
        unidades = c3.multiselect("Unidad de medida", sorted(catalogo["unidad"].dropna().unique()), key="cat_unid")
        q = st.text_input("🔎 Buscar indicador o descripción", key="cat_q", placeholder="p. ej. pobreza, PIB, tasa…")
        cat = catalogo.copy()
        if inst:
            cat = cat[cat["institucion"].isin(inst)]
        if fuentes:
            cat = cat[cat["fuente"].isin(fuentes)]
        if unidades:
            cat = cat[cat["unidad"].isin(unidades)]
        if q:
            m = cat["indicador"].str.contains(q, case=False, na=False) | \
                cat["descripcion"].str.contains(q, case=False, na=False)
            cat = cat[m]
        st.caption(f"{len(cat)} de {len(catalogo)} indicadores")
        max_terr = int(catalogo["n_territorios"].max()) if catalogo["n_territorios"].notna().any() else 34
        mostrar_tabla(cat, column_config={
            "indicador": st.column_config.TextColumn("Indicador", width="medium"),
            "etiqueta": st.column_config.TextColumn("Nombre corto"),
            "descripcion": st.column_config.TextColumn("Descripción", width="large"),
            "anio_min": st.column_config.NumberColumn("Desde", format="%d"),
            "anio_max": st.column_config.NumberColumn("Hasta", format="%d"),
            "n_territorios": st.column_config.ProgressColumn("Territorios", min_value=0, max_value=max_terr,
                                                             format="%d"),
            "n_obs": st.column_config.NumberColumn("Obs.", format="%d"),
        })
        if not cat.empty and cat["anio_min"].notna().any():
            g = cat.dropna(subset=["anio_min", "anio_max"]).sort_values("anio_min", ascending=False)
            fig = go.Figure(go.Bar(
                y=g["etiqueta"], x=(g["anio_max"] - g["anio_min"] + 1).astype(float),
                base=g["anio_min"].astype(float) - 0.5, orientation="h",
                marker=dict(color=[PALETA[["DANE", "CPC"].index(i) if i in ["DANE", "CPC"] else 2]
                                   for i in g["institucion"]]),
                customdata=np.stack([g["anio_min"], g["anio_max"], g["fuente"], g["unidad"]], axis=1),
                hovertemplate="<b>%{y}</b><br>%{customdata[0]}–%{customdata[1]}<br>%{customdata[2]}"
                              "<br>Unidad: %{customdata[3]}<extra></extra>",
            ))
            fig.update_layout(title="Ventana temporal por indicador (azul = DANE · verde = CPC)",
                              height=max(320, 26 * len(g) + 90), xaxis_title="Año", yaxis_title=None)
            mostrar_fig(fig, key="gantt_cat")
        descargar_csv(cat, "catalogo_indicadores_filtrado.csv", "⬇️ Descargar catálogo filtrado", key="dl_cat")

    with t4:
        desag_cov = st.radio("Desagregación", [DESAG_TOTAL] + [d for d in [DESAG_CAB, DESAG_RURAL, "Hombre", "Mujer"]
                                                          if d in panel["desagregacion"].unique()],
                             horizontal=True, key="cov_desag")
        cov = (deptos[(deptos["desagregacion"] == desag_cov) & deptos["valor"].notna()]
               .groupby(["indicador", "anio"])["departamento_canonico"].nunique().unstack("anio"))
        if cov.empty:
            st.warning("No hay datos para esta desagregación.")
        else:
            cov.index = [etiqueta(i) for i in cov.index]
            fig = px.imshow(cov, aspect="auto", color_continuous_scale="Blues", text_auto=True,
                            labels=dict(color="Deptos."))
            fig.update_layout(title="Número de departamentos con dato por indicador y año",
                              height=max(360, 28 * len(cov) + 120), xaxis_title=None, yaxis_title=None)
            fig.update_xaxes(dtick=1, tickangle=-45)
            mostrar_fig(fig, key="cov_heat")
        marcas = panel["marca_calidad"].fillna("sin marca").value_counts().reset_index()
        marcas.columns = ["marca_calidad", "observaciones"]
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("##### Marcas de calidad heredadas de la Fase 2")
            mostrar_tabla(marcas)
        with c2:
            fig = px.pie(marcas, names="marca_calidad", values="observaciones", hole=.55,
                         color_discrete_sequence=PALETA)
            fig.update_layout(title="Distribución de marcas de calidad", height=330, showlegend=True)
            mostrar_fig(fig, key="pie_marcas")


def _familia_indicador(ind: str) -> str:
    i = str(ind)
    if i.startswith("PIB"):
        return "Cuentas nacionales (PIB)"
    if i == "IDC":
        return "Competitividad"
    if "pobreza" in i.lower() or i == "coeficiente_gini":
        return "Pobreza monetaria y desigualdad"
    if i == "IPM":
        return "Pobreza multidimensional"
    return "Mercado laboral y demografía"


# =============================================================================
# MÓDULO 2 – EXPLORADOR TERRITORIAL, RANKINGS Y BRECHAS
# =============================================================================
def modulo_explorador(panel: pd.DataFrame) -> None:
    encabezado("📊", "Explorador Territorial, Rankings y Brechas",
               "Indicadores por año, departamento y desagregación · rankings dinámicos · brechas urbano-rurales y por sexo")
    deptos = solo_deptos(panel)
    inds = indicadores_con_datos(deptos)
    c1, c2, c3 = st.columns([2, 1.4, 1])
    ind = c1.selectbox("Indicador", inds, format_func=etiqueta, key="ex_ind")
    dsg_opts = desags_de(deptos, ind)
    desag = c2.selectbox("Desagregación", dsg_opts, key="ex_desag")
    anios = anios_de(deptos, ind, desag)
    if not anios:
        st.warning("No hay años con datos para esta combinación.")
        return
    anio = c3.selectbox("Año", anios[::-1], key="ex_anio")
    sub = corte(deptos, ind, anio, desag)
    unidad = unidad_de(panel, ind)
    pol = polaridad(ind)

    # ---- Tarjetas ----
    media = sub["valor"].mean()
    prev = corte(deptos, ind, anio - 1, desag)
    delta_media = media - prev["valor"].mean() if not prev.empty else None
    if pol >= 0:
        mejor = sub.loc[sub["valor"].idxmax()]; peor = sub.loc[sub["valor"].idxmin()]
    else:
        mejor = sub.loc[sub["valor"].idxmin()]; peor = sub.loc[sub["valor"].idxmax()]
    nac = corte(nacional(panel), ind, anio, desag)
    cv = sub["valor"].std() / abs(media) if media else np.nan
    fila_tarjetas([
        dict(titulo="Promedio departamental", valor=fmt_num(media), icono="📍",
             sub=f"{len(sub)} territorios · {unidad}", delta=delta_media,
             delta_txt=f"{fmt_num(delta_media)} vs {anio - 1}" if delta_media is not None else None, polar=pol),
        dict(titulo="Valor nacional", valor=fmt_num(nac["valor"].iloc[0]) if not nac.empty else "—", icono="🇨🇴",
             sub="Total nacional" if not nac.empty else "No reportado para este indicador", acento="accent-gold"),
        dict(titulo="Mejor desempeño", valor=fmt_num(mejor["valor"]), icono="🏆",
             sub=mejor["departamento_canonico"], acento="accent-emerald"),
        dict(titulo="Mayor rezago", valor=fmt_num(peor["valor"]), icono="⚠️",
             sub=peor["departamento_canonico"], acento="accent-red"),
        dict(titulo="Dispersión (CV)", valor=fmt_pct(cv), icono="📏",
             sub=f"Brecha máx/mín: {fmt_num(sub['valor'].max() / sub['valor'].min()) if sub['valor'].min() > 0 else '—'}×",
             acento="accent-slate"),
    ])
    if pol == 0:
        callout("Este indicador es de <b>volumen/neutro</b>: 'mejor' y 'rezago' se refieren a valores máximo y mínimo.")
    st.write("")
    t1, t2, t3, t4 = st.tabs(["🏆 Rankings dinámicos", "📊 Distribución y evolución",
                              "🏙️ Brechas urbano-rurales", "👥 Brechas por sexo"])

    with t1:
        c1, c2, c3 = st.columns([1.2, 1.2, 1.4])
        vista = c1.radio("Vista", ["Top N", "Últimos N", "Todos"], horizontal=True, key="ex_vista")
        orden = c2.radio("Ordenamiento", ["Descendente", "Ascendente"], horizontal=True, key="ex_orden")
        n = c3.slider("N departamentos", 5, 20, 10, key="ex_n", disabled=(vista == "Todos"))
        asc = orden == "Ascendente"
        r = sub.sort_values("valor", ascending=asc).reset_index(drop=True)
        r["puesto"] = np.arange(1, len(r) + 1)
        if vista == "Top N":
            r = r.head(n)
        elif vista == "Últimos N":
            r = r.tail(n)
        r_plot = r.iloc[::-1]
        fig = px.bar(r_plot, x="valor", y="departamento_canonico", orientation="h", color="valor",
                     color_continuous_scale=escala_color(ind), text=r_plot["valor"].map(lambda v: fmt_num(v)),
                     custom_data=["puesto", "codigo_divipola", "marca_calidad"])
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="<b>%{y}</b> (DIVIPOLA %{customdata[1]})<br>Puesto: %{customdata[0]}"
                                        "<br>Valor: %{x:,.3f}<br>Marca: %{customdata[2]}<extra></extra>")
        if not nac.empty:
            fig.add_vline(x=float(nac["valor"].iloc[0]), line_dash="dash", line_color=DORADO,
                          annotation_text="Nacional", annotation_position="top right",
                          annotation_font_color=DORADO)
        fig.add_vline(x=float(media), line_dash="dot", line_color=PIZARRA, annotation_text="Promedio",
                      annotation_position="top left", annotation_font_color=PIZARRA)
        vmin, vmax = float(min(r["valor"].min(), 0)), float(max(r["valor"].max(), 0))
        fig.update_xaxes(range=[vmin * 1.15 if vmin < 0 else 0, vmax * 1.15 if vmax > 0 else 0])
        fig.update_layout(title=f"{etiqueta(ind)} · {desag} · {anio} — {vista} ({orden.lower()})",
                          height=max(380, 30 * len(r) + 120), coloraxis_showscale=False,
                          xaxis_title=unidad, yaxis_title=None)
        mostrar_fig(fig, key="rank_bar")
        callout(f"Polaridad del indicador: <b>{'mayor es mejor' if pol > 0 else ('mayor es peor' if pol < 0 else 'neutral')}</b>. "
                "Use el ordenamiento para ver los extremos que le interesen.")
        tabla = r[["puesto", "departamento_canonico", "codigo_divipola", "valor", "marca_calidad"]]
        mostrar_tabla(tabla, column_config={"valor": st.column_config.NumberColumn("Valor", format="%.3f")})
        descargar_csv(tabla, f"ranking_{ind}_{anio}.csv", key="dl_rank")

    with t2:
        hist = deptos[(deptos["indicador"] == ind) & (deptos["desagregacion"] == desag) & deptos["valor"].notna()]
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig = px.box(hist, x="anio", y="valor", points="all", hover_name="departamento_canonico",
                         color_discrete_sequence=[AZUL])
            nac_h = nacional(panel)
            nac_h = nac_h[(nac_h["indicador"] == ind) & (nac_h["desagregacion"] == desag)].dropna(subset=["valor"])
            if not nac_h.empty:
                fig.add_trace(go.Scatter(x=nac_h["anio"], y=nac_h["valor"], mode="lines+markers", name="Nacional",
                                         line=dict(color=DORADO, width=3)))
            fig.update_layout(title="Distribución departamental por año", height=430,
                              xaxis_title=None, yaxis_title=unidad)
            mostrar_fig(fig, key="box_hist")
        with c2:
            fig = px.histogram(sub, x="valor", nbins=15, color_discrete_sequence=[ESMERALDA], marginal="rug",
                               hover_name="departamento_canonico")
            fig.update_layout(title=f"Histograma {anio}", height=430, xaxis_title=unidad, yaxis_title="Departamentos")
            mostrar_fig(fig, key="hist_anio")
        disp = hist.groupby("anio")["valor"].agg(["mean", "std", "min", "max", "count"])
        disp["cv"] = disp["std"] / disp["mean"].abs()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=disp.index, y=disp["mean"], name="Promedio", mode="lines+markers",
                                 line=dict(color=AZUL, width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=disp.index, y=disp["max"], name="Máximo", mode="lines",
                                 line=dict(color=PALETA[1], dash="dot")), secondary_y=False)
        fig.add_trace(go.Scatter(x=disp.index, y=disp["min"], name="Mínimo", mode="lines",
                                 line=dict(color=PALETA[3], dash="dot"), fill="tonexty",
                                 fillcolor="rgba(30,58,138,0.06)"), secondary_y=False)
        fig.add_trace(go.Bar(x=disp.index, y=disp["cv"], name="Coef. de variación (σ-convergencia)",
                             marker_color="rgba(212,160,23,0.45)"), secondary_y=True)
        fig.update_layout(title="Convergencia territorial: promedio, rango y coeficiente de variación", height=420)
        fig.update_yaxes(title_text=unidad, secondary_y=False)
        fig.update_yaxes(title_text="CV", secondary_y=True, tickformat=".0%")
        mostrar_fig(fig, key="sigma_conv")

    with t3:
        _brechas_urbano_rurales(panel, deptos)

    with t4:
        _brechas_sexo(panel, deptos)


def _brechas_urbano_rurales(panel: pd.DataFrame, deptos: pd.DataFrame) -> None:
    cand = [i for i in deptos["indicador"].unique()
            if {DESAG_CAB, DESAG_RURAL}.issubset(set(desags_de(deptos, i)))]
    if not cand:
        st.warning("No hay indicadores con desagregación Cabeceras / Centros poblados y rural disperso.")
        return
    c1, c2 = st.columns([1.5, 1])
    ind = c1.selectbox("Indicador con desagregación urbano-rural", ordenar_indicadores(cand),
                       format_func=etiqueta, key="br_ind")
    anios = sorted(set(anios_de(deptos, ind, DESAG_CAB)) & set(anios_de(deptos, ind, DESAG_RURAL)))
    anio = c2.selectbox("Año", anios[::-1], key="br_anio")
    sub = deptos[(deptos["indicador"] == ind) & (deptos["anio"] == anio)]
    b = sub.pivot_table(index="departamento_canonico", columns="desagregacion", values="valor")
    if DESAG_CAB not in b or DESAG_RURAL not in b:
        st.warning("Faltan desagregaciones para el año seleccionado.")
        return
    b = b.dropna(subset=[DESAG_CAB, DESAG_RURAL])
    b["brecha_pp"] = b[DESAG_RURAL] - b[DESAG_CAB]
    b["razon"] = b[DESAG_RURAL] / b[DESAG_CAB].replace(0, np.nan)
    b = b.sort_values("brecha_pp")
    fila_tarjetas([
        dict(titulo="Brecha promedio (rural − cabecera)", valor=f"{fmt_num(b['brecha_pp'].mean())} pp", icono="⚖️",
             sub=f"{len(b)} departamentos · {anio}", acento="accent-red"),
        dict(titulo="Mayor brecha", valor=f"{fmt_num(b['brecha_pp'].max())} pp", icono="🔺",
             sub=b["brecha_pp"].idxmax(), acento="accent-gold"),
        dict(titulo="Menor brecha", valor=f"{fmt_num(b['brecha_pp'].min())} pp", icono="🔻",
             sub=b["brecha_pp"].idxmin(), acento="accent-emerald"),
        dict(titulo="Razón rural / urbano (mediana)", valor=f"{fmt_num(b['razon'].median())}×", icono="➗",
             sub="veces la incidencia urbana", acento="accent-slate"),
    ])
    st.write("")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        fig = go.Figure()
        for dep, row in b.iterrows():
            fig.add_trace(go.Scatter(x=[row[DESAG_CAB], row[DESAG_RURAL]], y=[dep, dep], mode="lines",
                                     line=dict(color="#CBD5E1", width=3), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=b[DESAG_CAB], y=b.index, mode="markers", name="Cabeceras",
                                 marker=dict(color=AZUL, size=11),
                                 hovertemplate="<b>%{y}</b><br>Cabeceras: %{x:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=b[DESAG_RURAL], y=b.index, mode="markers", name="Rural disperso",
                                 marker=dict(color=DORADO, size=11, symbol="diamond"),
                                 customdata=b["brecha_pp"],
                                 hovertemplate="<b>%{y}</b><br>Rural: %{x:.1f}<br>Brecha: %{customdata:.1f} pp<extra></extra>"))
        fig.update_layout(title=f"Diagrama de mancuerna: {etiqueta(ind)} {anio}", height=max(450, 22 * len(b) + 120),
                          xaxis_title=unidad_de(panel, ind), legend=dict(orientation="h", y=1.04, x=0))
        mostrar_fig(fig, key="dumbbell")
    with c2:
        fig = px.bar(b.reset_index(), x="brecha_pp", y="departamento_canonico", orientation="h",
                     color="brecha_pp", color_continuous_scale="OrRd")
        fig.update_layout(title="Brecha (puntos porcentuales)", height=max(450, 22 * len(b) + 120),
                          coloraxis_showscale=False, yaxis_title=None, xaxis_title="pp")
        mostrar_fig(fig, key="brecha_bar")
    # Evolución de la brecha
    ev = deptos[(deptos["indicador"] == ind) & deptos["desagregacion"].isin([DESAG_CAB, DESAG_RURAL])]
    ev = ev.pivot_table(index=["departamento_canonico", "anio"], columns="desagregacion", values="valor").dropna()
    ev["brecha"] = ev[DESAG_RURAL] - ev[DESAG_CAB]
    ev_m = ev.groupby("anio")[["brecha", DESAG_CAB, DESAG_RURAL]].mean().reset_index()
    nac_ev = nacional(panel)
    nac_ev = nac_ev[(nac_ev["indicador"] == ind) & nac_ev["desagregacion"].isin([DESAG_CAB, DESAG_RURAL])]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ev_m["anio"], y=ev_m[DESAG_CAB], name="Cabeceras (prom. dptal.)",
                             line=dict(color=AZUL, width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=ev_m["anio"], y=ev_m[DESAG_RURAL], name="Rural (prom. dptal.)",
                             line=dict(color=DORADO, width=3)), secondary_y=False)
    fig.add_trace(go.Bar(x=ev_m["anio"], y=ev_m["brecha"], name="Brecha (pp)", marker_color="rgba(220,38,38,.35)"),
                  secondary_y=True)
    if not nac_ev.empty:
        nv = nac_ev.pivot_table(index="anio", columns="desagregacion", values="valor").dropna()
        if not nv.empty:
            fig.add_trace(go.Scatter(x=nv.index, y=nv[DESAG_RURAL] - nv[DESAG_CAB], name="Brecha nacional (pp)",
                                     mode="lines+markers", line=dict(color=ROJO, dash="dash")), secondary_y=True)
    fig.update_layout(title="Evolución de la brecha urbano-rural", height=400)
    fig.update_yaxes(title_text="%", secondary_y=False)
    fig.update_yaxes(title_text="pp", secondary_y=True)
    mostrar_fig(fig, key="brecha_ev")
    callout("Pobreza monetaria y Gini se publican sólo con desagregación <b>Total</b> en el panel; la brecha "
            "urbano-rural se analiza con el IPM (DANE – ENCV). IPM 2020 y 2021 tienen notas operativas.", "info")


def _brechas_sexo(panel: pd.DataFrame, deptos: pd.DataFrame) -> None:
    cand = [i for i in deptos["indicador"].unique() if {"Hombre", "Mujer"}.issubset(set(desags_de(deptos, i)))]
    if not cand:
        st.info("No hay indicadores con desagregación por sexo en el panel.")
        return
    c1, c2 = st.columns([1.5, 1])
    ind = c1.selectbox("Indicador con desagregación por sexo", ordenar_indicadores(cand),
                       format_func=etiqueta, key="sx_ind")
    anios = sorted(set(anios_de(deptos, ind, "Hombre")) & set(anios_de(deptos, ind, "Mujer")))
    anio = c2.selectbox("Año", anios[::-1], key="sx_anio")
    s = deptos[(deptos["indicador"] == ind) & (deptos["anio"] == anio)]
    b = s.pivot_table(index="departamento_canonico", columns="desagregacion", values="valor")[["Hombre", "Mujer"]].dropna()
    b["brecha"] = b["Mujer"] - b["Hombre"]
    b = b.sort_values("brecha")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.scatter(b.reset_index(), x="Hombre", y="Mujer", hover_name="departamento_canonico",
                         color="brecha", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                         size=b["brecha"].abs().values + 0.5)
        lo, hi = float(b[["Hombre", "Mujer"]].min().min()), float(b[["Hombre", "Mujer"]].max().max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Paridad",
                                 line=dict(color=PIZARRA, dash="dash")))
        fig.update_layout(title=f"Mujeres vs. hombres – {etiqueta(ind)} {anio}", height=460)
        mostrar_fig(fig, key="sx_scatter")
    with c2:
        fig = px.bar(b.reset_index(), x="brecha", y="departamento_canonico", orientation="h", color="brecha",
                     color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
        fig.update_layout(title="Brecha (Mujer − Hombre, pp)", height=460, coloraxis_showscale=False, yaxis_title=None)
        mostrar_fig(fig, key="sx_bar")
    callout(f"En {anio}, la incidencia femenina supera a la masculina en <b>{int((b['brecha'] > 0).sum())}</b> de "
            f"{len(b)} departamentos (brecha promedio {fmt_num(b['brecha'].mean())} pp).")


# =============================================================================
# MÓDULO 3 – EVOLUCIÓN TEMPORAL Y COMPARATIVA DEPARTAMENTAL
# =============================================================================
def modulo_evolucion(panel: pd.DataFrame) -> None:
    encabezado("📈", "Evolución Temporal y Comparativa Departamental",
               "Trayectorias históricas multivariables (hasta 6 departamentos) y perfiles multidimensionales en radar")
    deptos = solo_deptos(panel)
    todos = sorted(deptos["departamento_canonico"].unique())
    t1, t2 = st.tabs(["📉 Series de tiempo multivariables", "🕸️ Radar multidimensional"])

    with t1:
        c1, c2 = st.columns([2, 1])
        sel = c1.multiselect("Departamentos (máx. 6)", todos, max_selections=6, key="ev_deps",
                             default=[d for d in ["Bogotá D.C.", "Antioquia", "Valle del Cauca", "Chocó",
                                                  "La Guajira", "Vaupés"] if d in todos][:6])
        inds = c2.multiselect("Indicadores (1–4)", indicadores_con_datos(deptos), max_selections=4,
                              default=[i for i in ["PIB_per_capita", "IPM"] if i in deptos["indicador"].unique()],
                              format_func=etiqueta, key="ev_inds")
        c3, c4, c5 = st.columns(3)
        desag = c3.selectbox("Desagregación", [DESAG_TOTAL, DESAG_CAB, DESAG_RURAL, "Hombre", "Mujer"], key="ev_desag")
        base100 = c4.toggle("Índice base 100 (primer año común)", key="ev_b100")
        ref_nac = c5.toggle("Incluir referencia nacional", value=True, key="ev_nac")
        if not sel or not inds:
            st.info("Seleccione al menos un departamento y un indicador.")
        else:
            fig = make_subplots(rows=len(inds), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=[etiqueta(i) + (" (base 100)" if base100 else "") for i in inds])
            colores = {d: PALETA[k % len(PALETA)] for k, d in enumerate(sel)}
            resumen = []
            for r, ind in enumerate(inds, start=1):
                sub = panel[(panel["indicador"] == ind) & (panel["desagregacion"] == desag) & panel["valor"].notna()]
                piv = sub.pivot_table(index="anio", columns="departamento_canonico", values="valor")
                if piv.empty:
                    continue
                cols = [d for d in sel if d in piv.columns]
                if ref_nac and TOTAL_NACIONAL in piv.columns:
                    cols.append(TOTAL_NACIONAL)
                piv = piv[cols]
                if base100:
                    comun = piv.dropna()
                    if not comun.empty:
                        piv = piv / comun.iloc[0] * 100
                for d in cols:
                    s = piv[d].dropna()
                    if s.empty:
                        continue
                    es_nac = d == TOTAL_NACIONAL
                    fig.add_trace(go.Scatter(
                        x=s.index, y=s.values, mode="lines+markers", name=d, legendgroup=d, showlegend=(r == 1),
                        line=dict(color=DORADO if es_nac else colores[d], width=3 if es_nac else 2.2,
                                  dash="dash" if es_nac else "solid"),
                        marker=dict(size=5),
                        hovertemplate=f"<b>{d}</b><br>{etiqueta(ind)}<br>Año %{{x}}: %{{y:,.2f}}<extra></extra>"),
                        row=r, col=1)
                    if len(s) >= 2 and not es_nac:
                        n_a = s.index[-1] - s.index[0]
                        tcac = ((s.iloc[-1] / s.iloc[0]) ** (1 / n_a) - 1) if (n_a > 0 and s.iloc[0] > 0 and s.iloc[-1] > 0) else np.nan
                        resumen.append({"indicador": etiqueta(ind), "departamento": d,
                                        "año inicial": int(s.index[0]), "valor inicial": s.iloc[0],
                                        "año final": int(s.index[-1]), "valor final": s.iloc[-1],
                                        "variación absoluta": s.iloc[-1] - s.iloc[0], "TCAC": tcac})
            fig.update_layout(height=320 * len(inds) + 80, hovermode="x unified",
                              legend=dict(orientation="h", y=1.02 + 0.02 * (4 - len(inds)), x=0),
                              title="Trayectorias históricas comparadas")
            fig.update_xaxes(dtick=2)
            mostrar_fig(fig, key="ev_lines")
            if resumen:
                st.markdown("##### Resumen de cambio (primer vs. último año disponible)")
                mostrar_tabla(pd.DataFrame(resumen), column_config={
                    "valor inicial": st.column_config.NumberColumn(format="%.2f"),
                    "valor final": st.column_config.NumberColumn(format="%.2f"),
                    "variación absoluta": st.column_config.NumberColumn(format="%.2f"),
                    "TCAC": st.column_config.NumberColumn("TCAC (tasa de crecimiento anual compuesta)", format="percent"),
                })

    with t2:
        cand = [i for i in indicadores_con_datos(deptos) if polaridad(i) != 0 or i in INDICADORES_CLAVE]
        c1, c2 = st.columns([1, 2.2])
        anio_def = anio_con_mejor_cobertura(deptos, INDICADORES_CLAVE)
        anios = sorted(deptos["anio"].unique())
        anio = c1.selectbox("Año de referencia", anios[::-1], index=anios[::-1].index(anio_def), key="rd_anio")
        inds = c2.multiselect("Dimensiones (≥3)", cand, format_func=etiqueta, key="rd_inds",
                              default=[i for i in INDICADORES_CLAVE if i in cand][:7])
        c3, c4 = st.columns([2.2, 1])
        sel = c3.multiselect("Departamentos (máx. 6)", todos, max_selections=6, key="rd_deps",
                             default=[d for d in ["Bogotá D.C.", "Antioquia", "Chocó", "La Guajira"] if d in todos])
        invertir = c4.toggle("Invertir indicadores negativos (1 = mejor)", value=True, key="rd_inv")
        if len(inds) < 3 or not sel:
            st.info("Seleccione al menos 3 dimensiones y 1 departamento.")
            return
        X, bit = matriz_departamental(deptos, anio, inds)
        if X.empty or X.shape[1] < 3:
            st.warning("No hay suficientes indicadores con datos cerca del año elegido.")
            return
        N = X.apply(minmax)
        if invertir:
            for c in N.columns:
                if polaridad(c) < 0:
                    N[c] = 1 - N[c]
        fig = go.Figure()
        theta = [etiqueta(c) for c in N.columns]
        prom = N.mean()
        fig.add_trace(go.Scatterpolar(r=list(prom.values) + [prom.values[0]], theta=theta + [theta[0]],
                                      name="Promedio departamental", line=dict(color=PIZARRA, dash="dash"),
                                      fill="none"))
        for k, d in enumerate(sel):
            if d not in N.index:
                continue
            v = N.loc[d]
            fig.add_trace(go.Scatterpolar(
                r=list(v.values) + [v.values[0]], theta=theta + [theta[0]], name=d, fill="toself", opacity=0.55,
                line=dict(color=PALETA[k % len(PALETA)], width=2),
                customdata=list(X.loc[d].values) + [X.loc[d].values[0]],
                hovertemplate="<b>" + d + "</b><br>%{theta}<br>Normalizado: %{r:.2f}<br>Valor: %{customdata:,.2f}<extra></extra>"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0, .25, .5, .75, 1], gridcolor="#E2E8F0"),
                                     angularaxis=dict(gridcolor="#E2E8F0")),
                          title=f"Perfil multidimensional normalizado (0–1) · {anio}", height=600,
                          legend=dict(orientation="h", y=-0.08))
        mostrar_fig(fig, key="radar")
        callout("Normalización min–max sobre todos los departamentos con dato en el año. "
                + ("Con la inversión activa, <b>1 representa el mejor desempeño relativo</b> en todas las dimensiones." if invertir
                   else "Sin inversión, 1 representa el valor más alto observado (no necesariamente el mejor)."))
        with st.expander("🧾 Bitácora de construcción de la matriz (pivot + imputación por media del año)"):
            mostrar_tabla(bit)


# =============================================================================
# MÓDULO 4 – EDA, CORRELACIONES Y PCA
# =============================================================================
def modulo_eda_pca(panel: pd.DataFrame) -> None:
    encabezado("🧪", "Análisis Exploratorio (EDA) y Reducción de Dimensionalidad",
               "Correlaciones de Pearson · dispersión con OLS · PCA en tiempo real · Índice Sintético Multidimensional (0–100)")
    deptos = solo_deptos(panel)
    cand = [i for i in indicadores_con_datos(deptos) if polaridad(i) != 0 or i in INDICADORES_CLAVE]
    c1, c2, c3 = st.columns([1, 3, 1])
    anios = sorted(deptos["anio"].unique())
    anio_def = anio_con_mejor_cobertura(deptos, INDICADORES_CLAVE)
    anio = c1.selectbox("Año de análisis", anios[::-1], index=anios[::-1].index(anio_def), key="pca_anio")
    inds = c2.multiselect("Variables de la matriz departamental", cand, format_func=etiqueta, key="pca_inds",
                          default=[i for i in INDICADORES_CLAVE if i in cand])
    tol = c3.number_input("Tolerancia de años", 0, 5, 2, key="pca_tol",
                          help="Si un indicador no existe en el año, se usa el año más cercano dentro de esta tolerancia.")
    if len(inds) < 3:
        st.info("Seleccione al menos 3 variables.")
        return
    X, bit = matriz_departamental(deptos, anio, inds, tolerancia=int(tol))
    if X.shape[1] < 3 or X.shape[0] < 6:
        st.warning("La matriz resultante es demasiado pequeña para el análisis (se requieren ≥3 variables y ≥6 departamentos).")
        mostrar_tabla(bit)
        return
    n_imp = int(bit["imputados (media del año)"].sum()) if not bit.empty else 0
    if n_imp:
        st.warning(f"Se imputaron {n_imp} celdas faltantes con la media del indicador en el año "
                   f"({fmt_pct(n_imp / X.size)} de la matriz). Revise la bitácora.")
    with st.expander(f"🧾 Matriz departamental {X.shape[0]} × {X.shape[1]} y bitácora de imputación"):
        mostrar_tabla(bit)
        mostrar_tabla(X.reset_index().rename(columns={"index": "departamento_canonico"}))

    t1, t2, t3 = st.tabs(["🔗 Correlaciones", "🧭 PCA y biplot", "🏅 Índice Sintético Multidimensional"])
    with t1:
        c1, c2 = st.columns([1.15, 1])
        with c1:
            corr = X.corr(method="pearson")
            lab = [etiqueta(c) for c in corr.columns]
            mostrar_p = st.toggle("Ocultar correlaciones no significativas (p ≥ 0,05)", key="corr_p")
            z = corr.values.copy()
            texto = np.vectorize(lambda v: f"{v:.2f}")(z)
            if mostrar_p and HAS_SCIPY:
                n = len(X)
                tstat = corr.values * np.sqrt((n - 2) / np.clip(1 - corr.values ** 2, 1e-12, None))
                pv = 2 * sps.t.sf(np.abs(tstat), n - 2)
                texto = np.where(pv < 0.05, texto, "")
            fig = go.Figure(go.Heatmap(z=z, x=lab, y=lab, colorscale="RdBu_r", zmin=-1, zmax=1, text=texto,
                                       texttemplate="%{text}", hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
            fig.update_layout(title=f"Matriz de correlación de Pearson · {anio} (n = {len(X)})", height=560)
            fig.update_xaxes(tickangle=-40)
            mostrar_fig(fig, key="corr_heat")
        with c2:
            cols = list(X.columns)
            xv = st.selectbox("Eje X", cols, index=0, format_func=etiqueta, key="sc_x")
            yv = st.selectbox("Eje Y", cols, index=min(len(cols) - 1, cols.index("IPM") if "IPM" in cols else 1),
                              format_func=etiqueta, key="sc_y")
            df = X.reset_index().rename(columns={"index": "departamento_canonico"})
            df["region"] = df["departamento_canonico"].map(DIVIPOLA).map(CODIGO_A_REGION).fillna("Otra")
            fig = px.scatter(df, x=xv, y=yv, color="region", hover_name="departamento_canonico", text="departamento_canonico",
                             color_discrete_sequence=PALETA)
            fig.update_traces(textposition="top center", textfont=dict(size=9), marker=dict(size=11, line=dict(width=1, color="white")))
            res = ols_simple(df[xv], df[yv])
            if res:
                xs = np.linspace(df[xv].min(), df[xv].max(), 50)
                fig.add_trace(go.Scatter(x=xs, y=res["b0"] + res["b1"] * xs, mode="lines", name="OLS",
                                         line=dict(color=ROJO, width=2.5, dash="dash")))
            fig.update_layout(title="Dispersión con línea de tendencia OLS", height=470,
                              xaxis_title=etiqueta(xv), yaxis_title=etiqueta(yv),
                              legend=dict(orientation="h", y=-0.2))
            mostrar_fig(fig, key="sc_ols")
            if res:
                callout(f"<b>ŷ = {res['b0']:.3f} + {res['b1']:.4f}·x</b> · R² = {res['r2']:.3f} · r = {res['r']:.3f}"
                        f" · p = {res['p']:.4f} · n = {res['n']}. Correlación ecológica: no implica causalidad.")

    # ---------------- PCA ----------------
    Z = StandardScaler().fit_transform(X.values)
    n_comp = min(X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp).fit(Z)
    scores = pca.transform(Z)
    evr = pca.explained_variance_ratio_
    eig = pca.explained_variance_
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(eig), index=X.columns,
                            columns=[f"PC{i + 1}" for i in range(n_comp)])
    with t2:
        fila_tarjetas([
            dict(titulo="Varianza PC1", valor=fmt_pct(evr[0]), icono="1️⃣"),
            dict(titulo="Varianza PC1+PC2", valor=fmt_pct(evr[:2].sum()), icono="2️⃣", acento="accent-gold"),
            dict(titulo="Componentes con λ > 1 (Kaiser)", valor=str(int((eig > 1).sum())), icono="📐",
                 acento="accent-emerald"),
            dict(titulo="Observaciones / variables", valor=f"{X.shape[0]} / {X.shape[1]}", icono="🧮",
                 acento="accent-slate"),
        ])
        st.write("")
        c1, c2 = st.columns([1, 1.45])
        with c1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            pcs = [f"PC{i + 1}" for i in range(n_comp)]
            fig.add_trace(go.Bar(x=pcs, y=evr, name="Varianza explicada", marker_color=AZUL,
                                 text=[f"{v:.0%}" for v in evr], textposition="outside"), secondary_y=False)
            fig.add_trace(go.Scatter(x=pcs, y=np.cumsum(evr), name="Acumulada", mode="lines+markers",
                                     line=dict(color=DORADO, width=3)), secondary_y=True)
            fig.add_hline(y=0.8, line_dash="dot", line_color=ESMERALDA, secondary_y=True,
                          annotation_text="80 %", annotation_position="bottom right")
            fig.update_yaxes(tickformat=".0%", secondary_y=False, title_text="Individual")
            fig.update_yaxes(tickformat=".0%", secondary_y=True, range=[0, 1.02], dtick=0.2, title_text="Acumulada")
            fig.update_layout(title="Gráfico de sedimentación y varianza acumulada", height=480,
                              legend=dict(orientation="h", y=-0.15))
            mostrar_fig(fig, key="scree")
        with c2:
            ca, cb = st.columns(2)
            pcx = ca.selectbox("Componente X", list(loadings.columns), index=0, key="bp_x")
            pcy = cb.selectbox("Componente Y", list(loadings.columns), index=min(1, n_comp - 1), key="bp_y")
            ix, iy = int(pcx[2:]) - 1, int(pcy[2:]) - 1
            df_s = pd.DataFrame({"departamento": X.index, "x": scores[:, ix], "y": scores[:, iy]})
            df_s["region"] = df_s["departamento"].map(DIVIPOLA).map(CODIGO_A_REGION).fillna("Otra")
            fig = px.scatter(df_s, x="x", y="y", color="region", hover_name="departamento", text="departamento",
                             color_discrete_sequence=PALETA)
            fig.update_traces(textposition="top center", textfont=dict(size=9), marker=dict(size=10, opacity=.85))
            esc = 0.85 * np.abs(scores[:, [ix, iy]]).max() / max(np.abs(loadings[[pcx, pcy]].values).max(), 1e-9)
            for var, row in loadings.iterrows():
                fig.add_annotation(x=row[pcx] * esc, y=row[pcy] * esc, ax=0, ay=0, xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2,
                                   arrowcolor=ROJO if polaridad(var) < 0 else ESMERALDA)
                fig.add_annotation(x=row[pcx] * esc * 1.12, y=row[pcy] * esc * 1.12, text=f"<b>{etiqueta(var)}</b>",
                                   showarrow=False, font=dict(size=10, color=AZUL_OSC),
                                   bgcolor="rgba(255,255,255,.75)")
            fig.add_hline(y=0, line_color="#CBD5E1"); fig.add_vline(x=0, line_color="#CBD5E1")
            fig.update_layout(title=f"Biplot {pcx} ({evr[ix]:.0%}) vs {pcy} ({evr[iy]:.0%})", height=560,
                              xaxis_title=pcx, yaxis_title=pcy, legend=dict(orientation="h", y=-0.12))
            mostrar_fig(fig, key="biplot")
        lt = loadings.copy()
        lt.index = [etiqueta(i) for i in lt.index]
        fig = px.imshow(lt.iloc[:, :min(5, n_comp)], color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        text_auto=".2f", aspect="auto")
        fig.update_layout(title="Cargas (loadings = correlación variable–componente)", height=max(320, 34 * len(lt) + 110))
        mostrar_fig(fig, key="loadings")
        callout("Flechas verdes: indicadores donde mayor es mejor · rojas: mayor es peor. Departamentos próximos en el "
                "biplot comparten perfiles socioeconómicos similares.")

    with t3:
        c1, c2 = st.columns([1.3, 1])
        metodo = c1.radio("Método de agregación", ["PC1 (primer componente)",
                                                    "Ponderado por varianza (componentes con λ > 1)"],
                          key="isd_met", horizontal=True)
        if metodo.startswith("PC1"):
            comp = scores[:, 0]
            pesos_txt = "PC1"
        else:
            k = max(1, int((eig > 1).sum()))
            # Se orienta cada componente antes de ponderar
            orient = np.array([_orientar(scores[:, j], X) for j in range(k)])
            comp = (scores[:, :k] * orient) @ (evr[:k] / evr[:k].sum())
            pesos_txt = f"{k} componentes ponderados por su varianza explicada"
        comp = comp * _orientar(comp, X)
        isd = 100 * (comp - comp.min()) / (comp.max() - comp.min()) if np.ptp(comp) > 0 else np.full_like(comp, 50)
        res = pd.DataFrame({"departamento_canonico": X.index, "ISM": isd}).sort_values("ISM", ascending=False)
        res["puesto"] = np.arange(1, len(res) + 1)
        res["categoría"] = pd.cut(res["ISM"], [-0.1, 25, 50, 75, 100.1],
                                  labels=["Rezago crítico", "Rezago medio", "Desarrollo medio-alto", "Desarrollo alto"])
        c2.markdown(f'<span class="pill">Base: {pesos_txt}</span><span class="pill">Año {anio}</span>'
                    f'<span class="pill">{X.shape[1]} variables</span>', unsafe_allow_html=True)
        cc1, cc2 = st.columns([1.4, 1])
        with cc1:
            fig = px.bar(res.iloc[::-1], x="ISM", y="departamento_canonico", orientation="h", color="ISM",
                         color_continuous_scale=[[0, ROJO], [0.5, DORADO], [1, ESMERALDA]],
                         text=res.iloc[::-1]["ISM"].map(lambda v: f"{v:.1f}"))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(title="Índice Sintético Multidimensional (0 = mayor rezago, 100 = mayor desarrollo)",
                              height=max(500, 22 * len(res) + 120), coloraxis_showscale=False, yaxis_title=None,
                              xaxis=dict(range=[0, 112]))
            mostrar_fig(fig, key="ism_bar")
        with cc2:
            cnt = res["categoría"].value_counts().reindex(["Desarrollo alto", "Desarrollo medio-alto",
                                                          "Rezago medio", "Rezago crítico"]).fillna(0)
            fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.6,
                                   marker=dict(colors=[ESMERALDA, "#6EE7B7", DORADO, ROJO])))
            fig.update_layout(title="Departamentos por categoría", height=340, showlegend=True,
                              legend=dict(orientation="h", y=-0.1))
            mostrar_fig(fig, key="ism_pie")
            cor = pd.Series({etiqueta(c): np.corrcoef(X[c], isd)[0, 1] for c in X.columns}).sort_values()
            fig = px.bar(cor, orientation="h", color=cor.values, color_continuous_scale="RdBu", range_color=[-1, 1])
            fig.update_layout(title="Correlación de cada variable con el índice", height=340,
                              coloraxis_showscale=False, showlegend=False, xaxis_title="r", yaxis_title=None)
            mostrar_fig(fig, key="ism_cor")
        mostrar_tabla(res.merge(X, left_on="departamento_canonico", right_index=True),
                      column_config={"ISM": st.column_config.ProgressColumn("ISM (0-100)", min_value=0, max_value=100,
                                                                            format="%.1f")})
        descargar_csv(res, f"indice_sintetico_multidimensional_{anio}.csv", key="dl_ism")
        callout("El índice se orienta automáticamente para que valores altos signifiquen <b>mayor desarrollo</b> "
                "(correlación positiva con la suma de variables estandarizadas multiplicadas por su polaridad). "
                "Es un índice exploratorio, relativo al año y a las variables seleccionadas; no sustituye índices oficiales.",
                "warn")


def _orientar(componente: np.ndarray, X: pd.DataFrame) -> int:
    """Devuelve +1/-1 para que el componente correlacione positivamente con 'desarrollo'."""
    Zs = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    pol = np.array([polaridad(c) if polaridad(c) != 0 else 1 for c in X.columns])
    ref = (Zs.values * pol).sum(axis=1)
    r = np.corrcoef(componente, ref)[0, 1] if np.std(componente) > 0 else 1
    return 1 if (np.isnan(r) or r >= 0) else -1


# =============================================================================
# MÓDULO 5 – ECONOMETRÍA ESPACIAL Y CARTOGRAFÍA INTERACTIVA
# =============================================================================
def _haversine(a, b) -> float:
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


@st.cache_resource(show_spinner=False)
def _geometrias(_gj: dict | None, firma: str) -> dict:
    if _gj is None or not HAS_SHAPELY:
        return {}
    geoms = {}
    for f in _gj["features"]:
        try:
            g = shp_shape(f["geometry"])
            if not g.is_valid:
                g = g.buffer(0)
            geoms[f["properties"]["codigo_divipola"]] = g
        except Exception:
            continue
    return geoms


def centroides(gj: dict | None, firma: str) -> dict:
    geoms = _geometrias(gj, firma)
    out = dict(CENTROIDES)
    for c, g in geoms.items():
        p = g.representative_point() if g.geom_type == "MultiPolygon" else g.centroid
        out[c] = (p.y, p.x)
    return out


@st.cache_data(show_spinner="Construyendo matriz de pesos espaciales…")
def construir_vecinos(codigos: tuple, tipo: str, k: int, _gj: dict | None, firma: str) -> tuple[dict, str]:
    """Devuelve dict código → lista de vecinos y una nota metodológica.
    tipo: 'queen' (contigüidad, con islas enlazadas a su vecino más cercano) o 'knn'."""
    cents = centroides(_gj, firma)
    codigos = [c for c in codigos if c in cents]
    nota = ""
    vec: dict[str, list[str]] = {c: [] for c in codigos}
    if tipo == "queen":
        geoms = _geometrias(_gj, firma)
        if not geoms or any(c not in geoms for c in codigos):
            tipo = "knn"
            nota = "Sin polígonos disponibles para todos los territorios: se usó KNN por centroides. "
        else:
            hecho = False
            if HAS_PYSAL and HAS_GPD:
                try:
                    gdf = gpd.GeoDataFrame({"codigo": codigos}, geometry=[geoms[c] for c in codigos], crs="EPSG:4326")
                    gdf.index = codigos
                    try:
                        wq = libpysal.weights.Queen.from_dataframe(gdf, use_index=True, silence_warnings=True)
                    except TypeError:
                        wq = libpysal.weights.Queen.from_dataframe(gdf, ids=codigos, silence_warnings=True)
                    vec = {c: [str(n) for n in wq.neighbors.get(c, [])] for c in codigos}
                    hecho = True
                    nota = "Contigüidad tipo reina (libpysal). "
                except Exception:
                    hecho = False
            if not hecho:
                buf = {c: geoms[c].buffer(0.005) for c in codigos}
                for i, a in enumerate(codigos):
                    for b in codigos[i + 1:]:
                        if buf[a].intersects(buf[b]):
                            vec[a].append(b); vec[b].append(a)
                nota = "Contigüidad tipo reina (shapely, tolerancia 0,005°). "
            islas = [c for c in codigos if not vec[c]]
            for c in islas:
                cercano = min([o for o in codigos if o != c], key=lambda o: _haversine(cents[c], cents[o]))
                vec[c].append(cercano)
                if c not in vec[cercano]:
                    vec[cercano].append(c)
            if islas:
                nota += (f"Islas enlazadas a su vecino más cercano: "
                         f"{', '.join(CODIGO_A_NOMBRE.get(c, c) for c in islas)}. ")
    if tipo == "knn":
        for c in codigos:
            d = sorted([o for o in codigos if o != c], key=lambda o: _haversine(cents[c], cents[o]))
            vec[c] = d[:k]
        nota += f"K vecinos más cercanos (k = {k}, distancia geodésica entre centroides). "
    return vec, nota + "Pesos estandarizados por fila."


def _matriz_w(codigos: list[str], vec: dict) -> np.ndarray:
    idx = {c: i for i, c in enumerate(codigos)}
    W = np.zeros((len(codigos), len(codigos)))
    for c in codigos:
        for n in vec.get(c, []):
            if n in idx:
                W[idx[c], idx[n]] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    return np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)


@st.cache_data(show_spinner="Calculando autocorrelación espacial…")
def moran_completo(y: tuple, codigos: tuple, vec_items: tuple, permutaciones: int = 999, semilla: int = 42) -> dict:
    """I de Moran global y local. Usa esda/libpysal si están instalados; si no, implementación propia."""
    y = np.asarray(y, float)
    codigos = list(codigos)
    vec = {c: list(n) for c, n in vec_items}
    vec = {c: [n for n in vec.get(c, []) if n in codigos] for c in codigos}
    W = _matriz_w(codigos, vec)
    z = y - y.mean()
    lag = W @ z
    n = len(y)
    if HAS_PYSAL:
        try:
            w = libpysal.weights.W({c: vec[c] for c in codigos}, id_order=codigos, silence_warnings=True)
            w.transform = "r"
            mi = Moran(y, w, permutations=permutaciones)
            ml = Moran_Local(y, w, permutations=permutaciones, seed=semilla)
            return dict(I=float(mi.I), EI=float(mi.EI), z_norm=float(mi.z_norm), p_norm=float(mi.p_norm),
                        p_sim=float(mi.p_sim), z_sim=float(mi.z_sim), sim=np.asarray(mi.sim).tolist(),
                        Is=np.asarray(ml.Is).tolist(), q=np.asarray(ml.q).tolist(),
                        p_local=np.asarray(ml.p_sim).tolist(), z=z.tolist(), lag=lag.tolist(), motor="esda / libpysal")
        except Exception:
            pass
    rng = np.random.default_rng(semilla)
    zz = float(z @ z)
    I = float(z @ W @ z / zz) if zz > 0 else np.nan
    EI = -1.0 / (n - 1)
    S0 = W.sum(); S1 = 0.5 * ((W + W.T) ** 2).sum(); S2 = ((W.sum(1) + W.sum(0)) ** 2).sum()
    VI = (n * n * S1 - n * S2 + 3 * S0 * S0) / ((n * n - 1) * S0 * S0) - EI ** 2
    z_norm = (I - EI) / np.sqrt(VI) if VI > 0 else np.nan
    p_norm = 2 * (1 - sps.norm.cdf(abs(z_norm))) if HAS_SCIPY and np.isfinite(z_norm) else np.nan
    sim = np.array([(zp @ W @ zp) / zz for zp in (rng.permutation(z) for _ in range(permutaciones))])
    mayor = (sim >= I).sum() if I >= EI else (sim <= I).sum()
    p_sim = (mayor + 1) / (permutaciones + 1)
    z_sim = (I - sim.mean()) / sim.std() if sim.std() > 0 else np.nan
    m2 = zz / (n - 1)  # misma convención que esda.Moran_Local
    Is = z * lag / m2
    q = np.where((z > 0) & (lag > 0), 1, np.where((z < 0) & (lag > 0), 2, np.where((z < 0) & (lag < 0), 3, 4)))
    p_local = np.ones(n)
    for i in range(n):
        otros = np.delete(z, i)
        wi = W[i, np.arange(n) != i]
        k_i = int((wi > 0).sum())
        if k_i == 0:
            continue
        pesos = wi[wi > 0]
        sims = np.array([z[i] * (pesos @ rng.choice(otros, k_i, replace=False)) / m2 for _ in range(permutaciones)])
        mayor = (sims >= Is[i]).sum() if Is[i] >= 0 else (sims <= Is[i]).sum()
        p_local[i] = (mayor + 1) / (permutaciones + 1)
    return dict(I=I, EI=EI, z_norm=float(z_norm), p_norm=float(p_norm), p_sim=float(p_sim), z_sim=float(z_sim),
                sim=sim.tolist(), Is=Is.tolist(), q=q.tolist(), p_local=p_local.tolist(), z=z.tolist(),
                lag=lag.tolist(), motor="implementación interna (numpy)")


def _mapa(df: pd.DataFrame, color: str, gj: dict | None, cents: dict, titulo: str, escala=None,
          categorico: dict | None = None, hover_extra: list | None = None, rango=None, altura: int = 640) -> go.Figure:
    hover_extra = hover_extra or []
    if gj is not None:
        if categorico:
            fig = px.choropleth(df, geojson=gj, locations="codigo_divipola", featureidkey="properties.codigo_divipola",
                                color=color, hover_name="departamento_canonico", color_discrete_map=categorico,
                                category_orders={color: list(categorico.keys())}, hover_data=hover_extra)
        else:
            fig = px.choropleth(df, geojson=gj, locations="codigo_divipola", featureidkey="properties.codigo_divipola",
                                color=color, hover_name="departamento_canonico", color_continuous_scale=escala,
                                hover_data=hover_extra, range_color=rango)
        fig.update_traces(marker_line_color="white", marker_line_width=0.8)
    else:
        d = df.copy()
        d["lat"] = d["codigo_divipola"].map(lambda c: cents.get(c, (np.nan, np.nan))[0])
        d["lon"] = d["codigo_divipola"].map(lambda c: cents.get(c, (np.nan, np.nan))[1])
        if categorico:
            fig = px.scatter_geo(d, lat="lat", lon="lon", color=color, hover_name="departamento_canonico",
                                 color_discrete_map=categorico, hover_data=hover_extra)
            fig.update_traces(marker=dict(size=18))
        else:
            fig = px.scatter_geo(d, lat="lat", lon="lon", color=color, size=(d[color] - d[color].min() + 1e-9),
                                 hover_name="departamento_canonico", color_continuous_scale=escala,
                                 hover_data=hover_extra, size_max=30)
    fig.update_geos(visible=False, showcountries=True, countrycolor="#CBD5E1", showland=True, landcolor="#F8FAFC",
                    lataxis_range=[-4.6, 13.6], lonaxis_range=[-82.2, -66.6], projection_type="mercator",
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(title=titulo, height=altura, margin=dict(l=0, r=0, t=60, b=0))
    return fig


def modulo_espacial(panel: pd.DataFrame) -> None:
    encabezado("🗺️", "Econometría Espacial y Cartografía Interactiva",
               "Mapa coroplético departamental · matriz de pesos espaciales · I de Moran global y LISA")
    deptos = solo_deptos(panel)
    gj, origen_geo, notas_geo = cargar_geojson()
    firma = origen_geo + str(len(gj["features"]) if gj else 0)
    cents = centroides(gj, firma)

    c1, c2, c3 = st.columns([2, 1.4, 1])
    ind = c1.selectbox("Indicador", indicadores_con_datos(deptos), format_func=etiqueta, key="sp_ind")
    desag = c2.selectbox("Desagregación", desags_de(deptos, ind), key="sp_desag")
    anios = anios_de(deptos, ind, desag)
    anio = c3.selectbox("Año", anios[::-1], key="sp_anio")
    sub = corte(deptos, ind, anio, desag)
    sub = sub[sub["codigo_divipola"].notna() & sub["codigo_divipola"].isin(cents.keys())]
    st.markdown(f'<span class="pill">Cartografía: {origen_geo}</span>'
                f'<span class="pill">Motor espacial: {"esda/libpysal" if HAS_PYSAL else "numpy (respaldo)"}</span>'
                f'<span class="pill">{len(sub)} departamentos con dato</span>', unsafe_allow_html=True)
    for n in notas_geo:
        callout(n, "warn")
    if len(sub) < 5:
        st.warning("Muy pocos departamentos con dato para análisis espacial.")
        return

    t1, t2, t3, t4 = st.tabs(["🗺️ Mapa coroplético", "🧮 Matriz de pesos espaciales",
                              "📐 I de Moran global y LISA", "⏱️ Dinámica de la autocorrelación"])
    with t1:
        c1, c2 = st.columns([2.2, 1])
        with c1:
            fig = _mapa(sub, "valor", gj, cents, f"{etiqueta(ind)} · {desag} · {anio}", escala=escala_color(ind),
                        hover_extra=["codigo_divipola"])
            fig.update_coloraxes(colorbar_title=unidad_de(panel, ind)[:18])
            mostrar_fig(fig, key="mapa_coro")
        with c2:
            sub2 = sub.copy()
            sub2["región"] = sub2["codigo_divipola"].map(CODIGO_A_REGION).fillna("Otra")
            reg = sub2.groupby("región")["valor"].agg(["mean", "min", "max", "count"]).reset_index()
            fig = px.bar(reg.sort_values("mean"), x="mean", y="región", orientation="h", color="mean",
                         color_continuous_scale=escala_color(ind), error_x=None,
                         hover_data={"min": ":.2f", "max": ":.2f", "count": True})
            fig.update_layout(title="Promedio por región natural", height=300, coloraxis_showscale=False,
                              yaxis_title=None, xaxis_title=None)
            mostrar_fig(fig, key="reg_bar")
            fig = px.strip(sub2, x="valor", y="región", color="región", hover_name="departamento_canonico",
                           color_discrete_sequence=PALETA)
            fig.update_layout(title="Dispersión intrarregional", height=320, showlegend=False, yaxis_title=None)
            mostrar_fig(fig, key="reg_strip")

    c1, c2, c3 = st.columns([1.4, 1, 1])
    tipo_w = c1.radio("Matriz de pesos", ["Contigüidad (reina)", "K vecinos más cercanos"], horizontal=True,
                      key="sp_w", index=0 if gj is not None else 1)
    k = c2.slider("k (sólo KNN)", 2, 8, 4, key="sp_k", disabled=tipo_w.startswith("Cont"))
    perms = c3.select_slider("Permutaciones", [99, 199, 499, 999, 1999], value=999, key="sp_perm")
    codigos = tuple(sub["codigo_divipola"].tolist())
    vec, nota_w = construir_vecinos(codigos, "queen" if tipo_w.startswith("Cont") else "knn", k, gj, firma)
    cods = [c for c in codigos if c in vec]
    y = sub.set_index("codigo_divipola").loc[cods, "valor"].astype(float).values
    nombres = [CODIGO_A_NOMBRE.get(c, c) for c in cods]

    with t2:
        W = _matriz_w(cods, vec)
        n_vec = np.array([len(vec[c]) for c in cods])
        fila_tarjetas([
            dict(titulo="Unidades espaciales", valor=str(len(cods)), icono="🧩"),
            dict(titulo="Vecinos promedio", valor=fmt_num(n_vec.mean(), 2), icono="🔗", acento="accent-gold",
                 sub=f"mín {n_vec.min()} · máx {n_vec.max()}"),
            dict(titulo="Densidad de la matriz", valor=fmt_pct((W > 0).mean()), icono="🕸️", acento="accent-emerald"),
            dict(titulo="Más conectado", valor=nombres[int(n_vec.argmax())], icono="⭐", acento="accent-slate",
                 sub=f"{n_vec.max()} vecinos"),
        ])
        callout(nota_w)
        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig = px.imshow(W, x=nombres, y=nombres, color_continuous_scale="Blues", aspect="auto",
                            labels=dict(color="w<sub>ij</sub>"))
            fig.update_layout(title="Matriz W estandarizada por fila", height=640)
            fig.update_xaxes(tickangle=-60, tickfont=dict(size=9)); fig.update_yaxes(tickfont=dict(size=9))
            mostrar_fig(fig, key="w_heat")
        with c2:
            fig = go.Figure()
            for c in cods:
                for nb in vec[c]:
                    if c < nb or c not in vec.get(nb, []):
                        fig.add_trace(go.Scattergeo(lat=[cents[c][0], cents[nb][0]], lon=[cents[c][1], cents[nb][1]],
                                                    mode="lines", line=dict(color="rgba(30,58,138,.45)", width=1.4),
                                                    showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scattergeo(lat=[cents[c][0] for c in cods], lon=[cents[c][1] for c in cods], text=nombres,
                                        mode="markers", marker=dict(size=8 + 2 * n_vec, color=DORADO,
                                                                    line=dict(color=AZUL, width=1)),
                                        customdata=n_vec, hovertemplate="<b>%{text}</b><br>%{customdata} vecinos<extra></extra>",
                                        showlegend=False))
            if gj is not None:
                fig.add_trace(go.Choropleth(geojson=gj, locations=list(cods), z=[1] * len(cods),
                                            featureidkey="properties.codigo_divipola", showscale=False,
                                            colorscale=[[0, "#EFF6FF"], [1, "#EFF6FF"]], marker_line_color="#94A3B8",
                                            hoverinfo="skip"))
                fig.data = fig.data[-1:] + fig.data[:-1]
            fig.update_geos(visible=False, lataxis_range=[-4.6, 13.6], lonaxis_range=[-82.2, -66.6],
                            projection_type="mercator")
            fig.update_layout(title="Grafo de vecindad", height=640, margin=dict(l=0, r=0, t=60, b=0))
            mostrar_fig(fig, key="w_graph")

    with t3:
        res = moran_completo(tuple(y), tuple(cods), tuple((c, tuple(vec[c])) for c in cods), int(perms))
        sig = res["p_sim"] < 0.05
        interpret = ("agrupamiento espacial significativo (valores similares tienden a ser vecinos)" if sig and res["I"] > res["EI"]
                     else "dispersión espacial significativa (patrón tipo tablero de ajedrez)" if sig
                     else "sin evidencia de autocorrelación espacial (patrón compatible con aleatoriedad)")
        fila_tarjetas([
            dict(titulo="I de Moran", valor=f"{res['I']:.4f}", icono="📐", sub=f"E[I] = {res['EI']:.4f}",
                 acento="accent-emerald" if sig else "accent-slate"),
            dict(titulo="z (normalidad)", valor=f"{res['z_norm']:.3f}", icono="📈", sub=f"p = {res['p_norm']:.4f}"),
            dict(titulo="p-valor por permutaciones", valor=f"{res['p_sim']:.4f}", icono="🎲",
                 sub=f"{perms} permutaciones · z_sim = {res['z_sim']:.2f}", acento="accent-gold"),
            dict(titulo="Conclusión (α = 5 %)", valor="Significativo" if sig else "No significativo", icono="✅" if sig else "➖",
                 sub=res["motor"], acento="accent-emerald" if sig else "accent-red"),
        ])
        callout(f"<b>Interpretación:</b> {etiqueta(ind)} ({anio}) presenta {interpret}.", "ok" if sig else "info")
        z = np.array(res["z"]); lag = np.array(res["lag"]); q = np.array(res["q"]); pl = np.array(res["p_local"])
        etiquetas_q = {1: "Alto-Alto (HH)", 2: "Bajo-Alto (LH)", 3: "Bajo-Bajo (LL)", 4: "Alto-Bajo (HL)"}
        alfa_l = st.select_slider("Significancia LISA", [0.01, 0.05, 0.10], value=0.05, key="lisa_alfa")
        cluster = [etiquetas_q[int(qq)] if p < alfa_l else "No significativo" for qq, p in zip(q, pl)]
        df_l = pd.DataFrame({"departamento_canonico": nombres, "codigo_divipola": cods, "valor": y, "z": z, "Wz": lag,
                             "I_local": res["Is"], "p_local": pl, "cluster": cluster})
        colores_l = {"Alto-Alto (HH)": "#B91C1C", "Bajo-Bajo (LL)": "#1D4ED8", "Alto-Bajo (HL)": "#F87171",
                     "Bajo-Alto (LH)": "#93C5FD", "No significativo": "#E2E8F0"}
        c1, c2 = st.columns([1, 1.15])
        with c1:
            zs = z / z.std() if z.std() > 0 else z
            ls = lag / z.std() if z.std() > 0 else lag
            fig = px.scatter(df_l.assign(zs=zs, ls=ls), x="zs", y="ls", color="cluster", hover_name="departamento_canonico",
                             color_discrete_map=colores_l, text="departamento_canonico")
            fig.update_traces(textposition="top center", textfont=dict(size=8), marker=dict(size=11, line=dict(width=1, color="#334155")))
            xs = np.linspace(zs.min(), zs.max(), 20)
            fig.add_trace(go.Scatter(x=xs, y=res["I"] * xs, mode="lines", name=f"Pendiente = I ({res['I']:.3f})",
                                     line=dict(color=ROJO, width=2.5)))
            fig.add_hline(y=0, line_color="#94A3B8", line_dash="dot"); fig.add_vline(x=0, line_color="#94A3B8", line_dash="dot")
            fig.update_layout(title="Diagrama de dispersión de Moran", height=560, xaxis_title="Valor estandarizado (z)",
                              yaxis_title="Rezago espacial (Wz)", legend=dict(orientation="h", y=-0.15))
            mostrar_fig(fig, key="moran_sc")
        with c2:
            fig = _mapa(df_l, "cluster", gj, cents, f"Mapa de clústeres LISA (α = {alfa_l})", categorico=colores_l,
                        hover_extra=["valor", "I_local", "p_local"], altura=560)
            mostrar_fig(fig, key="lisa_map")
        sim = np.array(res["sim"])
        fig = go.Figure(go.Histogram(x=sim, nbinsx=40, marker_color="rgba(30,58,138,.55)", name="I bajo H0"))
        fig.add_vline(x=res["I"], line_color=ROJO, line_width=3, annotation_text=f"I observado = {res['I']:.3f}")
        fig.add_vline(x=res["EI"], line_color=PIZARRA, line_dash="dash", annotation_text="E[I]", annotation_position="bottom")
        fig.update_layout(title="Distribución de referencia por permutaciones aleatorias", height=330,
                          xaxis_title="I de Moran", yaxis_title="Frecuencia", showlegend=False)
        mostrar_fig(fig, key="moran_hist")
        mostrar_tabla(df_l.sort_values("p_local"), column_config={
            "valor": st.column_config.NumberColumn(format="%.3f"), "z": st.column_config.NumberColumn(format="%.3f"),
            "Wz": st.column_config.NumberColumn(format="%.3f"), "I_local": st.column_config.NumberColumn(format="%.3f"),
            "p_local": st.column_config.NumberColumn(format="%.3f")})
        descargar_csv(df_l, f"lisa_{ind}_{anio}.csv", key="dl_lisa")

    with t4:
        filas = []
        for a in anios:
            s = corte(deptos, ind, a, desag)
            s = s[s["codigo_divipola"].isin(vec.keys())]
            if len(s) < 5:
                continue
            cs = tuple(s["codigo_divipola"])
            vv, _ = construir_vecinos(cs, "queen" if tipo_w.startswith("Cont") else "knn", k, gj, firma)
            cs = tuple(c for c in cs if c in vv)
            ya = s.set_index("codigo_divipola").loc[list(cs), "valor"].astype(float).values
            r = moran_completo(tuple(ya), cs, tuple((c, tuple(vv[c])) for c in cs), 499)
            filas.append({"año": a, "I de Moran": r["I"], "p_sim": r["p_sim"], "z_norm": r["z_norm"], "n": len(cs)})
        if not filas:
            st.info("No hay suficientes años con datos.")
        else:
            dm = pd.DataFrame(filas)
            dm["significativo"] = np.where(dm["p_sim"] < 0.05, "p < 0,05", "No significativo")
            fig = px.bar(dm, x="año", y="I de Moran", color="significativo",
                         color_discrete_map={"p < 0,05": ESMERALDA, "No significativo": "#CBD5E1"},
                         hover_data={"p_sim": ":.4f", "z_norm": ":.2f", "n": True})
            fig.add_hline(y=0, line_color=PIZARRA)
            fig.update_layout(title=f"Evolución del I de Moran global – {etiqueta(ind)} ({desag})", height=420,
                              xaxis=dict(dtick=1))
            mostrar_fig(fig, key="moran_ev")
            mostrar_tabla(dm)


# =============================================================================
# MÓDULO 6 – ECONOMETRÍA DE SERIES DE TIEMPO (VAR / VEC)
# =============================================================================
def _serie_unidad(panel: pd.DataFrame, modo: str, depto: str | None) -> pd.DataFrame:
    base = panel[(panel["desagregacion"] == DESAG_TOTAL) & panel["valor"].notna()]
    if modo == "Total nacional":
        sub = base[base["departamento_canonico"] == TOTAL_NACIONAL]
        return sub.pivot_table(index="anio", columns="indicador", values="valor")
    if modo == "Departamento":
        sub = base[base["departamento_canonico"] == depto]
        return sub.pivot_table(index="anio", columns="indicador", values="valor")
    sub = solo_deptos(base)
    # Promedio departamental: sólo años con cobertura ≥ 60 % del máximo de departamentos del indicador
    cnt = sub.groupby(["indicador", "anio"])["departamento_canonico"].nunique()
    mx = cnt.groupby(level=0).transform("max")
    validos = cnt[cnt >= 0.6 * mx].reset_index()[["indicador", "anio"]]
    sub = sub.merge(validos, on=["indicador", "anio"])
    return sub.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="mean")


def _bloque_contiguo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().sort_index()
    if df.empty:
        return df
    anios = df.index.to_numpy()
    cortes = np.where(np.diff(anios) != 1)[0] + 1
    bloques = np.split(np.arange(len(anios)), cortes)
    mejor = max(bloques, key=len)
    return df.iloc[mejor]


def _adf(s: pd.Series) -> tuple[float, float]:
    try:
        r = adfuller(s.dropna().values, maxlag=min(2, max(0, len(s) // 4 - 1)), autolag="AIC")
        return float(r[0]), float(r[1])
    except Exception:
        return np.nan, np.nan


def _transformar(df: pd.DataFrame, modo: str) -> tuple[pd.DataFrame, dict]:
    out, aplicada = {}, {}
    for c in df.columns:
        s = df[c].astype(float)
        es_log = transformacion_sugerida(c) == "log" and (s > 0).all()
        if modo == "Niveles":
            t = "nivel"
        elif modo == "Primera diferencia":
            t = "Δ"
        elif modo == "Log-diferencia (×100)":
            t = "Δlog×100" if (s > 0).all() else "Δ"
        else:
            _, p = _adf(s)
            t = "nivel" if (np.isfinite(p) and p < 0.10) else ("Δlog×100" if es_log else "Δ")
        if t == "nivel":
            out[c] = s
        elif t == "Δ":
            out[c] = s.diff()
        else:
            out[c] = np.log(s).diff() * 100
        aplicada[c] = t
    return pd.DataFrame(out).dropna(), aplicada


@st.cache_resource(show_spinner="Estimando modelo VAR…")
def ajustar_var(datos: pd.DataFrame, p: int, criterio: str):
    modelo = VAR(datos.reset_index(drop=True))
    if criterio == "Fijo":
        return modelo.fit(p, trend="c")
    res = modelo.fit(maxlags=p, ic=criterio, trend="c")
    if res.k_ar == 0:
        res = modelo.fit(1, trend="c")
    return res


@st.cache_resource(show_spinner="Estimando modelo VEC…")
def ajustar_vecm(datos: pd.DataFrame, k_ar_diff: int, rango: int, determinista: str):
    return VECM(datos.reset_index(drop=True), k_ar_diff=k_ar_diff, coint_rank=rango, deterministic=determinista).fit()


def _corto(ind: str, k: int) -> str:
    t = re.sub(r"\s*\(.*?\)", "", etiqueta(ind)).replace("Desempleo – ", "").replace("Ocupación – ", "")
    t = t.replace("Participación – ", "")
    lim = 30 if k <= 2 else (22 if k == 3 else 16)
    return t if len(t) <= lim else t[:lim - 1] + "…"


def _grafico_irf(irfs: np.ndarray, se: np.ndarray | None, nombres: list[str], titulo: str) -> go.Figure:
    k = len(nombres)
    fig = make_subplots(rows=k, cols=k, shared_xaxes=True, horizontal_spacing=0.05, vertical_spacing=0.07,
                        subplot_titles=[f"{_corto(nombres[j], k)} → {_corto(nombres[i], k)}"
                                        for i in range(k) for j in range(k)])
    h = np.arange(irfs.shape[0])
    for i in range(k):
        for j in range(k):
            y = irfs[:, i, j]
            if se is not None:
                up, lo = y + 1.96 * se[:, i, j], y - 1.96 * se[:, i, j]
                fig.add_trace(go.Scatter(x=np.r_[h, h[::-1]], y=np.r_[up, lo[::-1]], fill="toself", mode="lines",
                                         fillcolor="rgba(30,58,138,.12)", line=dict(width=0), hoverinfo="skip",
                                         showlegend=False), row=i + 1, col=j + 1)
            fig.add_trace(go.Scatter(x=h, y=y, mode="lines+markers", line=dict(color=AZUL, width=2.2),
                                     marker=dict(size=4), showlegend=False,
                                     hovertemplate="h=%{x}<br>respuesta=%{y:.4f}<extra></extra>"), row=i + 1, col=j + 1)
            fig.add_hline(y=0, line_color=ROJO, line_width=1, line_dash="dot", row=i + 1, col=j + 1)
    fig.update_annotations(font_size=10)
    fig.update_layout(title=titulo, height=230 * k + 90)
    return fig


def modulo_series_tiempo(panel: pd.DataFrame) -> None:
    encabezado("⏳", "Econometría de Series de Tiempo (VAR / VEC)",
               "Series agregadas o por departamento · raíces unitarias · VAR/VECM · causalidad de Granger · impulso-respuesta")
    if not HAS_SM:
        st.error("Este módulo requiere `statsmodels`. Instálelo con `pip install statsmodels`.")
        return
    deptos = solo_deptos(panel)
    c1, c2 = st.columns([1.6, 1.4])
    modo = c1.radio("Unidad de análisis", ["Total nacional", "Promedio departamental", "Departamento"],
                    horizontal=True, key="ts_modo")
    depto = None
    if modo == "Departamento":
        depto = c2.selectbox("Departamento", sorted(deptos["departamento_canonico"].unique()),
                             index=sorted(deptos["departamento_canonico"].unique()).index("Antioquia")
                             if "Antioquia" in deptos["departamento_canonico"].unique() else 0, key="ts_dep")
    serie = _serie_unidad(panel, modo, depto)
    if serie.empty:
        st.warning("No hay series para esta unidad (p. ej. el panel no contiene 'Total nacional'). "
                   "Pruebe 'Promedio departamental'.")
        return
    largas = [c for c in serie.columns if serie[c].notna().sum() >= 10]
    if len(largas) < 2:
        st.warning("Se requieren al menos dos indicadores con ≥ 10 años de información para esta unidad.")
        return
    defecto = [i for i in ["PIB_tasa_crecimiento", "Tasa de Desocupación (TD)", "Tasa de Ocupación (TO)"] if i in largas]
    if len(defecto) < 2:
        defecto = ordenar_indicadores(largas)[:2]
    c1, c2 = st.columns([2.4, 1])
    vars_ = c1.multiselect("Variables endógenas (2–4)", ordenar_indicadores(largas), default=defecto,
                           max_selections=4, format_func=etiqueta, key="ts_vars")
    modelo_tipo = c2.radio("Modelo", ["VAR", "VECM"], horizontal=True, key="ts_mod")
    if len(vars_) < 2:
        st.info("Seleccione al menos 2 variables.")
        return
    bruto = _bloque_contiguo(serie[vars_])
    T = len(bruto)
    k = len(vars_)
    if T < 8:
        st.warning(f"La ventana común completa sólo tiene {T} años: insuficiente para estimar un modelo multivariado. "
                   "Reduzca el número de variables o cambie la unidad.")
        return
    st.markdown(f'<span class="pill">Ventana: {int(bruto.index.min())}–{int(bruto.index.max())}</span>'
                f'<span class="pill">T = {T} observaciones anuales</span><span class="pill">k = {k} variables</span>',
                unsafe_allow_html=True)
    if T < 15:
        callout(f"<b>Muestra corta (T = {T}).</b> Las pruebas asintóticas (ADF, Johansen, Granger) tienen baja potencia; "
                "interprete los resultados como exploratorios y prefiera rezagos bajos.", "warn")

    t1, t2, t3, t4 = st.tabs(["🔍 Datos y raíces unitarias", "⚙️ Estimación", "🔀 Causalidad de Granger",
                              "📡 Impulso-respuesta y pronóstico"])
    with t1:
        modo_t = "Niveles" if modelo_tipo == "VECM" else st.radio(
            "Transformación (VAR)", ["Automática (ADF)", "Niveles", "Primera diferencia", "Log-diferencia (×100)"],
            horizontal=True, key="ts_trans")
        if modelo_tipo == "VECM":
            usar_log = st.toggle("Usar logaritmos para variables de nivel (PIB, población)", value=True, key="ts_log")
            datos = bruto.copy()
            aplic = {}
            for c in datos.columns:
                if usar_log and transformacion_sugerida(c) == "log" and (datos[c] > 0).all():
                    datos[c] = np.log(datos[c]); aplic[c] = "log(nivel)"
                else:
                    aplic[c] = "nivel"
            callout("El VECM se estima en <b>niveles</b>: la relación de largo plazo (cointegración) captura la "
                    "no estacionariedad común de las series.")
        else:
            datos, aplic = _transformar(bruto, modo_t)
        filas = []
        for c in bruto.columns:
            e0, p0 = _adf(bruto[c]); e1, p1 = _adf(bruto[c].diff())
            filas.append({"variable": etiqueta(c), "ADF nivel": e0, "p nivel": p0, "ADF Δ": e1, "p Δ": p1,
                          "orden de integración (α=10%)": "I(0)" if p0 < .1 else ("I(1)" if p1 < .1 else "I(2)?"),
                          "transformación usada": aplic.get(c, "—")})
        mostrar_tabla(pd.DataFrame(filas), column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                          for c in ["ADF nivel", "p nivel", "ADF Δ", "p Δ"]})
        zstd = (datos - datos.mean()) / datos.std().replace(0, 1)
        fig = go.Figure()
        for i, c in enumerate(datos.columns):
            fig.add_trace(go.Scatter(x=datos.index, y=zstd[c], name=f"{etiqueta(c)} [{aplic.get(c)}]",
                                     mode="lines+markers", line=dict(color=PALETA[i], width=2.4),
                                     customdata=datos[c], hovertemplate="%{x}: %{customdata:.3f}<extra></extra>"))
        fig.update_layout(title="Series del modelo (estandarizadas para comparación visual)", height=420,
                          hovermode="x unified", legend=dict(orientation="h", y=-0.15), yaxis_title="z")
        mostrar_fig(fig, key="ts_series")

    Tm = len(datos)
    with t2:
        if modelo_tipo == "VAR":
            p_max = max(1, min(4, (Tm - 2) // (k + 1)))
            c1, c2 = st.columns(2)
            criterio = c1.selectbox("Selección de rezagos", ["aic", "bic", "hqic", "fpe", "Fijo"], key="var_crit")
            p = c2.slider("Rezago máximo / fijo", 1, p_max, min(2, p_max), key="var_p")
            try:
                res = ajustar_var(datos, p, criterio)
            except Exception as e:
                st.error(f"No fue posible estimar el VAR: {e}")
                return
            raices = np.abs(res.roots) if len(res.roots) else np.array([np.inf])
            fila_tarjetas([
                dict(titulo="Rezagos (p)", valor=str(res.k_ar), icono="⏮️", sub=f"criterio: {criterio}"),
                dict(titulo="Observaciones efectivas", valor=str(res.nobs), icono="🧮", acento="accent-slate"),
                dict(titulo="AIC / BIC", valor=f"{res.aic:.2f}", sub=f"BIC {res.bic:.2f}", icono="📉", acento="accent-gold"),
                dict(titulo="Estabilidad", valor="Estable ✅" if res.is_stable() else "Inestable ⚠️", icono="⚖️",
                     sub=f"mín |raíz| = {raices.min():.3f}",
                     acento="accent-emerald" if res.is_stable() else "accent-red"),
            ])
            st.write("")
            pr = res.params.copy(); pv = res.pvalues.copy()
            tabla = pr.stack().reset_index()
            tabla.columns = ["regresor", "ecuación", "coeficiente"]
            tabla["p-valor"] = pv.stack().values
            tabla["sig."] = np.select([tabla["p-valor"] < .01, tabla["p-valor"] < .05, tabla["p-valor"] < .1],
                                      ["***", "**", "*"], "")
            c1, c2 = st.columns([1.3, 1])
            with c1:
                st.markdown("##### Coeficientes estimados")
                mostrar_tabla(tabla, column_config={"coeficiente": st.column_config.NumberColumn(format="%.4f"),
                                                    "p-valor": st.column_config.NumberColumn(format="%.4f")}, height=380)
            with c2:
                rc = pd.DataFrame(res.resid).corr()
                rc.index = rc.columns = [etiqueta(c)[:20] for c in rc.columns]
                fig = px.imshow(rc, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig.update_layout(title="Correlación de residuos", height=380)
                mostrar_fig(fig, key="var_rc")
            ang = np.angle(res.roots) if len(res.roots) else np.array([])
            inv = 1 / np.abs(res.roots) if len(res.roots) else np.array([])
            th = np.linspace(0, 2 * np.pi, 200)
            fig = go.Figure([go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color="#94A3B8"), name="Círculo unitario"),
                             go.Scatter(x=inv * np.cos(ang), y=inv * np.sin(ang), mode="markers", name="Raíces inversas",
                                        marker=dict(size=11, color=DORADO, line=dict(color=AZUL, width=1)))])
            fig.update_layout(title="Raíces inversas del polinomio característico", height=380,
                              yaxis=dict(scaleanchor="x", scaleratio=1))
            mostrar_fig(fig, key="var_roots")
            tipo_res = "VAR"
        else:
            c1, c2, c3 = st.columns(3)
            kd_max = max(1, min(3, (Tm - 4) // (k + 1)))
            kd = c1.slider("Rezagos en diferencias (k_ar_diff)", 1, kd_max, 1, key="vec_kd")
            det = c2.selectbox("Término determinístico", ["ci", "co", "li", "lo", "n"], key="vec_det",
                               help="ci: constante dentro de la relación de cointegración; co: constante fuera; "
                                    "li/lo: tendencia lineal dentro/fuera; n: ninguno.")
            det_order = {"n": -1, "ci": 0, "co": 0, "li": 1, "lo": 1}[det]
            try:
                jo = coint_johansen(datos.values, det_order, kd)
            except Exception as e:
                st.error(f"Prueba de Johansen no disponible para esta muestra: {e}")
                return
            filas = []
            sug = 0
            for r in range(k):
                traza, cv95 = jo.lr1[r], jo.cvt[r, 1]
                filas.append({"H0: rango ≤": r, "estadístico traza": traza, "valor crítico 95 %": cv95,
                              "λ-max": jo.lr2[r], "VC λ-max 95 %": jo.cvm[r, 1], "rechaza H0": traza > cv95})
            for f in filas:
                if f["rechaza H0"]:
                    sug += 1
                else:
                    break
            st.markdown("##### Prueba de cointegración de Johansen")
            mostrar_tabla(pd.DataFrame(filas), column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                              for c in ["estadístico traza", "valor crítico 95 %",
                                                                        "λ-max", "VC λ-max 95 %"]})
            if sug == 0:
                callout("Johansen no encuentra relaciones de cointegración al 5 %: un <b>VAR en diferencias</b> es "
                        "probablemente más adecuado. Se permite estimar el VECM con rango 1 con fines exploratorios.", "warn")
            elif sug == k:
                callout("Rango completo sugerido: las series podrían ser estacionarias en niveles (VAR en niveles).", "warn")
            rango = c3.slider("Rango de cointegración", 1, k - 1, int(min(max(sug, 1), k - 1)), key="vec_r")
            try:
                res = ajustar_vecm(datos, kd, rango, det)
            except Exception as e:
                st.error(f"No fue posible estimar el VECM: {e}")
                return
            fila_tarjetas([
                dict(titulo="Rango sugerido (traza 95 %)", valor=str(sug), icono="🔗"),
                dict(titulo="Rango usado", valor=str(rango), icono="🎯", acento="accent-gold"),
                dict(titulo="Rezagos en Δ", valor=str(kd), icono="⏮️", acento="accent-slate"),
                dict(titulo="Log-verosimilitud", valor=f"{res.llf:.2f}", icono="📈", acento="accent-emerald"),
            ])
            c1, c2 = st.columns(2)
            nombres = [etiqueta(c) for c in datos.columns]
            with c1:
                beta = pd.DataFrame(res.beta, index=nombres[:res.beta.shape[0]] if res.beta.shape[0] == k else
                                    nombres + [f"det{i}" for i in range(res.beta.shape[0] - k)],
                                    columns=[f"CE{i + 1}" for i in range(rango)])
                st.markdown("##### Vectores de cointegración (β, normalizados)")
                mostrar_tabla(beta.reset_index().rename(columns={"index": "variable"}))
            with c2:
                alpha = pd.DataFrame(res.alpha, index=nombres, columns=[f"α CE{i + 1}" for i in range(rango)])
                st.markdown("##### Coeficientes de ajuste (α, velocidad de corrección)")
                mostrar_tabla(alpha.reset_index().rename(columns={"index": "variable"}))
            callout("α negativo y significativo en la ecuación de una variable indica que ésta corrige los desequilibrios "
                    "respecto a la relación de largo plazo.")
            tipo_res = "VECM"

    nombres_c = list(datos.columns)
    with t3:
        pm = pd.DataFrame(np.nan, index=nombres_c, columns=nombres_c)
        for causado in nombres_c:
            for causante in nombres_c:
                if causado == causante:
                    continue
                try:
                    if tipo_res == "VAR":
                        pm.loc[causado, causante] = res.test_causality(causado, [causante], kind="f").pvalue
                    else:
                        pm.loc[causado, causante] = res.test_granger_causality(caused=causado, causing=causante).pvalue
                except Exception:
                    pm.loc[causado, causante] = np.nan
        lab = [etiqueta(c) for c in nombres_c]
        texto = np.where(np.isnan(pm.values), "—", np.vectorize(lambda v: f"{v:.3f}")(np.nan_to_num(pm.values)))
        fig = go.Figure(go.Heatmap(z=pm.values, x=lab, y=lab, colorscale=[[0, ESMERALDA], [0.05, "#A7F3D0"],
                                                                          [0.1, "#FEF3C7"], [1, "#F1F5F9"]],
                                   zmin=0, zmax=1, text=texto, texttemplate="%{text}",
                                   hovertemplate="¿%{x} causa (Granger) a %{y}?<br>p = %{z:.4f}<extra></extra>"))
        fig.update_layout(title="p-valores de causalidad de Granger (columna → fila)", height=440,
                          xaxis_title="Variable causante", yaxis_title="Variable causada")
        mostrar_fig(fig, key="granger")
        rel = [{"relación": f"{etiqueta(b)} → {etiqueta(a)}", "p-valor": pm.loc[a, b],
                "decisión (α=5 %)": "Causa en sentido de Granger" if pm.loc[a, b] < 0.05 else "No se rechaza H0"}
               for a in nombres_c for b in nombres_c if a != b and np.isfinite(pm.loc[a, b])]
        mostrar_tabla(pd.DataFrame(rel).sort_values("p-valor"),
                      column_config={"p-valor": st.column_config.NumberColumn(format="%.4f")})
        callout("H0: los rezagos de la variable causante no ayudan a predecir a la causada. "
                "La causalidad de Granger es <b>precedencia predictiva</b>, no causalidad estructural.")

    with t4:
        c1, c2, c3 = st.columns(3)
        h = c1.slider("Horizonte (años)", 3, 15, 8, key="irf_h")
        orto = c2.toggle("Impulsos ortogonalizados (Cholesky)", value=True, key="irf_orth")
        pasos = c3.slider("Años de pronóstico", 1, 8, 4, key="fc_h")
        try:
            irf = res.irf(h)
            irfs = irf.orth_irfs if orto else irf.irfs
            se = None
            try:
                se = irf.stderr(orth=orto)
            except Exception:
                se = None
            mostrar_fig(_grafico_irf(irfs, se, nombres_c,
                                     f"Funciones impulso-respuesta {'ortogonalizadas' if orto else 'simples'}"
                                     f"{' (bandas 95 %)' if se is not None else ''} · {tipo_res}"), key="irf")
            callout("Cada panel muestra la respuesta de la variable de la <b>fila</b> ante un choque de una desviación "
                    "estándar (o unitario) en la variable de la <b>columna</b>. El orden de Cholesky sigue el orden de selección.")
        except Exception as e:
            st.warning(f"No fue posible calcular las IRF: {e}")
        if tipo_res == "VAR":
            try:
                fe = res.fevd(h).decomp
                fig = make_subplots(rows=1, cols=k, subplot_titles=[etiqueta(c)[:28] for c in nombres_c])
                for i in range(k):
                    for j in range(k):
                        fig.add_trace(go.Bar(x=np.arange(1, h + 1), y=fe[i, :, j], name=etiqueta(nombres_c[j]),
                                             marker_color=PALETA[j], showlegend=(i == 0), legendgroup=str(j)),
                                      row=1, col=i + 1)
                fig.update_layout(barmode="stack", title="Descomposición de la varianza del error de pronóstico (FEVD)",
                                  height=380, legend=dict(orientation="h", y=-0.2))
                fig.update_yaxes(tickformat=".0%")
                mostrar_fig(fig, key="fevd")
            except Exception:
                pass
        try:
            if tipo_res == "VAR":
                media, lo, up = res.forecast_interval(datos.values[-res.k_ar:], steps=pasos, alpha=0.05)
            else:
                media, lo, up = res.predict(steps=pasos, alpha=0.05)
            futuros = np.arange(datos.index.max() + 1, datos.index.max() + 1 + pasos)
            fig = make_subplots(rows=1, cols=k, subplot_titles=[etiqueta(c)[:28] for c in nombres_c])
            for i, c in enumerate(nombres_c):
                fig.add_trace(go.Scatter(x=datos.index, y=datos[c], mode="lines+markers", line=dict(color=AZUL),
                                         name="Observado", showlegend=(i == 0)), row=1, col=i + 1)
                fig.add_trace(go.Scatter(x=np.r_[futuros, futuros[::-1]], y=np.r_[up[:, i], lo[::-1, i]], fill="toself", mode="lines",
                                         fillcolor="rgba(212,160,23,.2)", line=dict(width=0), name="IC 95 %",
                                         showlegend=(i == 0), hoverinfo="skip"), row=1, col=i + 1)
                fig.add_trace(go.Scatter(x=futuros, y=media[:, i], mode="lines+markers", line=dict(color=DORADO, dash="dash"),
                                         name="Pronóstico", showlegend=(i == 0)), row=1, col=i + 1)
            fig.update_layout(title="Pronóstico fuera de muestra (escala transformada del modelo)", height=380,
                              legend=dict(orientation="h", y=-0.2))
            mostrar_fig(fig, key="forecast")
        except Exception as e:
            st.info(f"Pronóstico no disponible: {e}")


# =============================================================================
# MÓDULO 7 – MACHINE LEARNING Y MINERÍA DE DATOS
# =============================================================================
def _log_cols(X, idx=()):
    """Transforma a logaritmo las columnas indicadas (para variables de nivel como el PIB)."""
    A = np.array(X, dtype=float, copy=True)
    if len(idx):
        A[:, list(idx)] = np.log(np.clip(A[:, list(idx)], 1e-9, None))
    return A


def _crear_pipeline(estimador, idx_log: tuple) -> Pipeline:
    return Pipeline([("log", FunctionTransformer(_log_cols, kw_args={"idx": idx_log})),
                     ("escalado", StandardScaler()), ("modelo", estimador)])


def config_ml_defecto(deptos: pd.DataFrame) -> dict:
    disponibles = deptos["indicador"].unique()
    objetivo = "IPM" if "IPM" in disponibles else ("incidencia_pobreza_monetaria" if "incidencia_pobreza_monetaria"
                                                    in disponibles else ordenar_indicadores(disponibles)[0])
    feats = [f for f in ["PIB_per_capita", "PIB_tasa_crecimiento", "IDC", "Tasa de Desocupación (TD)",
                         "Tasa de Ocupación (TO)", "Tasa Global de Participación (TGP)", "coeficiente_gini"]
             if f in disponibles and f != objetivo]
    return dict(objetivo=objetivo, features=feats, log=True, cv="GroupKFold (por departamento)", k=5,
                alpha_ridge=1.0, alpha_lasso=0.1, C=10.0, epsilon=0.5, max_imp=0.6)


def _excluir_por_fuga(objetivo: str) -> set:
    fugas = {"incidencia_pobreza_monetaria": {"personas_pobreza_monetaria"},
             "personas_pobreza_monetaria": {"incidencia_pobreza_monetaria"},
             "Tasa de Desocupación (TD)": {"Población desocupada", "Tasa Global de Participación (TGP)"},
             "Tasa de Ocupación (TO)": {"Población ocupada"},
             "PIB_volumen_encadenado": {"PIB_contribucion_crecimiento"}}
    return fugas.get(objetivo, set()) | {objetivo}


@st.cache_data(show_spinner="Construyendo dataset departamento-año…")
def construir_dataset_ml(deptos: pd.DataFrame, objetivo: str, features: tuple, desag_obj: str = DESAG_TOTAL):
    """Dataset panel (departamento, año) con imputación:
    1) media del año (transversal entre departamentos);
    2) si el indicador no existe en ese año: valor del año más cercano del mismo departamento;
    3) en última instancia, media global del indicador."""
    y_df = deptos[(deptos["indicador"] == objetivo) & (deptos["desagregacion"] == desag_obj) & deptos["valor"].notna()]
    base = y_df.groupby(["departamento_canonico", "anio"])["valor"].mean().rename("y").reset_index()
    F = panel_ancho_dept_anio(deptos, list(features))
    medias_anio = F.groupby(level="anio").mean()
    filas_imp = []
    for f in features:
        vals, met = [], []
        serie_f = F[f].dropna() if f in F.columns else pd.Series(dtype=float)
        global_mean = serie_f.mean() if len(serie_f) else np.nan
        for d, a in zip(base["departamento_canonico"], base["anio"]):
            v = serie_f.get((d, a), np.nan) if len(serie_f) else np.nan
            if pd.notna(v):
                vals.append(v); met.append("observado"); continue
            m = medias_anio[f].get(a, np.nan) if f in medias_anio.columns else np.nan
            if pd.notna(m):
                vals.append(m); met.append("media del año"); continue
            try:
                sd = serie_f.xs(d, level="departamento_canonico")
            except Exception:
                sd = pd.Series(dtype=float)
            if len(sd):
                a_cerc = min(sd.index, key=lambda x: (abs(x - a), -x))
                vals.append(sd.loc[a_cerc]); met.append("año más cercano (mismo depto.)"); continue
            vals.append(global_mean); met.append("media global")
        base[f] = vals
        cnt = pd.Series(met).value_counts()
        filas_imp.append({"variable": f, "observado": int(cnt.get("observado", 0)),
                          "media del año": int(cnt.get("media del año", 0)),
                          "año más cercano": int(cnt.get("año más cercano (mismo depto.)", 0)),
                          "media global": int(cnt.get("media global", 0))})
    imp = pd.DataFrame(filas_imp)
    if not imp.empty:
        imp["% imputado"] = 1 - imp["observado"] / len(base) if len(base) else np.nan
    base = base.dropna(subset=list(features)) if features else base
    return base.reset_index(drop=True), imp


@st.cache_resource(show_spinner="Entrenando modelos con validación cruzada…")
def entrenar_modelos(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, cfg_items: tuple) -> dict:
    cfg = dict(cfg_items)
    idx_log = tuple(i for i, c in enumerate(X.columns) if cfg["log"] and transformacion_sugerida(c) == "log")
    n_grupos = grupos.nunique()
    if cfg["cv"].startswith("Group") and n_grupos >= 3:
        cv = GroupKFold(n_splits=int(min(cfg["k"], n_grupos)))
    else:
        cv = KFold(n_splits=int(min(cfg["k"], len(y))), shuffle=True, random_state=42)
    estimadores = {
        "Ridge": Ridge(alpha=cfg["alpha_ridge"]),
        "LASSO": Lasso(alpha=cfg["alpha_lasso"], max_iter=50000),
        "SVR (RBF)": SVR(kernel="rbf", C=cfg["C"], epsilon=cfg["epsilon"], gamma="scale"),
    }
    salida = {}
    for nombre, est in estimadores.items():
        pipe = _crear_pipeline(est, idx_log)
        cvr = cross_validate(pipe, X, y, groups=grupos, cv=cv, return_train_score=True,
                             scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error",
                                      "mae": "neg_mean_absolute_error"})
        oof = cross_val_predict(pipe, X, y, groups=grupos, cv=cv)
        pipe.fit(X, y)
        salida[nombre] = dict(
            modelo=pipe, oof=oof,
            r2=cvr["test_r2"], rmse=-cvr["test_rmse"], mae=-cvr["test_mae"],
            r2_train=cvr["train_r2"], rmse_train=-cvr["train_rmse"],
        )
    salida["_cv"] = f"{type(cv).__name__} ({cv.get_n_splits()} pliegues)"
    return salida


def preparar_ml(deptos: pd.DataFrame, cfg: dict):
    datos, imp = construir_dataset_ml(deptos, cfg["objetivo"], tuple(cfg["features"]))
    feats = list(cfg["features"])
    excluidas = []
    if not imp.empty:
        excluidas = imp.loc[imp["% imputado"] > cfg["max_imp"], "variable"].tolist()
        feats = [f for f in feats if f not in excluidas]
    if len(feats) == 0 or len(datos) < 15:
        return None
    X = datos[feats].astype(float)
    y = datos["y"].astype(float)
    grupos = datos["departamento_canonico"]
    claves = ("objetivo", "log", "cv", "k", "alpha_ridge", "alpha_lasso", "C", "epsilon")
    cfg_items = tuple((k, cfg[k]) for k in claves)
    modelos = entrenar_modelos(X, y, grupos, cfg_items)
    return dict(datos=datos, imp=imp, X=X, y=y, grupos=grupos, feats=feats, excluidas=excluidas, modelos=modelos)


def modulo_ml(panel: pd.DataFrame) -> None:
    encabezado("🤖", "Machine Learning y Minería de Datos",
               "Ridge · LASSO · SVR con validación cruzada agrupada por departamento para predecir indicadores sociales")
    deptos = solo_deptos(panel)
    cfg = st.session_state.get("ml_cfg", config_ml_defecto(deptos))
    disponibles = indicadores_con_datos(deptos)
    with st.form("form_ml"):
        c1, c2 = st.columns([1, 2.4])
        obj_opts = [i for i in disponibles if polaridad(i) != 0] or disponibles
        objetivo = c1.selectbox("🎯 Variable objetivo", obj_opts,
                                index=obj_opts.index(cfg["objetivo"]) if cfg["objetivo"] in obj_opts else 0,
                                format_func=etiqueta)
        feat_opts = [i for i in disponibles if i not in _excluir_por_fuga(objetivo)]
        features = c2.multiselect("🧩 Variables predictoras", feat_opts, format_func=etiqueta,
                                  default=[f for f in cfg["features"] if f in feat_opts])
        c1, c2, c3, c4 = st.columns(4)
        cv = c1.radio("Esquema de validación", ["GroupKFold (por departamento)", "KFold aleatorio"],
                      index=0 if cfg["cv"].startswith("Group") else 1,
                      help="GroupKFold evita que un mismo departamento esté en entrenamiento y prueba (más honesto).")
        k = c2.slider("Pliegues (k)", 3, 10, int(cfg["k"]))
        log = c3.toggle("Log de variables de nivel (PIB, población)", value=cfg["log"])
        max_imp = c4.slider("Máx. % imputado por variable", 0.1, 1.0, float(cfg["max_imp"]), 0.05,
                            help="Variables con mayor proporción imputada se excluyen.")
        st.markdown("**Hiperparámetros**")
        c1, c2, c3, c4 = st.columns(4)
        grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        a_r = c1.select_slider("Ridge α", grid, value=cfg["alpha_ridge"] if cfg["alpha_ridge"] in grid else 1.0)
        a_l = c2.select_slider("LASSO α", grid, value=cfg["alpha_lasso"] if cfg["alpha_lasso"] in grid else 0.1)
        C = c3.select_slider("SVR C", [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
                             value=cfg["C"] if cfg["C"] in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0] else 10.0)
        eps = c4.select_slider("SVR ε", [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
                               value=cfg["epsilon"] if cfg["epsilon"] in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0] else 0.5)
        enviado = st.form_submit_button("🚀 Entrenar y evaluar modelos", type="primary")
    if enviado:
        cfg = dict(objetivo=objetivo, features=features, log=log, cv=cv, k=k, alpha_ridge=a_r, alpha_lasso=a_l,
                   C=C, epsilon=eps, max_imp=max_imp)
        st.session_state["ml_cfg"] = cfg
    if not cfg["features"]:
        st.info("Seleccione variables predictoras y pulse **Entrenar**.")
        return
    try:
        ml = preparar_ml(deptos, cfg)
    except Exception as e:
        st.error(f"No fue posible entrenar los modelos: {e}")
        return
    if ml is None:
        st.warning("Datos insuficientes: se requieren ≥ 15 observaciones departamento-año y al menos una variable "
                   "predictora con imputación por debajo del umbral.")
        return
    st.session_state["ml_cfg"] = cfg
    m = ml["modelos"]
    nombres = [n for n in m if not n.startswith("_")]
    resumen = pd.DataFrame([{
        "modelo": n, "R² CV": m[n]["r2"].mean(), "± R²": m[n]["r2"].std(), "RMSE CV": m[n]["rmse"].mean(),
        "± RMSE": m[n]["rmse"].std(), "MAE CV": m[n]["mae"].mean(), "± MAE": m[n]["mae"].std(),
        "R² entrenamiento": m[n]["r2_train"].mean(),
    } for n in nombres])
    mejor = resumen.sort_values("R² CV", ascending=False).iloc[0]["modelo"]
    st.markdown(f'<span class="pill">Objetivo: {etiqueta(cfg["objetivo"])}</span>'
                f'<span class="pill">n = {len(ml["y"])} obs. depto-año</span>'
                f'<span class="pill">{ml["grupos"].nunique()} departamentos</span>'
                f'<span class="pill">{len(ml["feats"])} predictores</span><span class="pill">{m["_cv"]}</span>',
                unsafe_allow_html=True)
    if ml["excluidas"]:
        callout(f"Variables excluidas por superar el umbral de imputación: <b>{', '.join(etiqueta(e) for e in ml['excluidas'])}</b>.", "warn")
    tarjetas = []
    for n, ac in zip(nombres, ["", "accent-gold", "accent-emerald"]):
        r = resumen[resumen["modelo"] == n].iloc[0]
        tarjetas.append(dict(titulo=f"{n}{' 🏆' if n == mejor else ''}", valor=f"R² {fmt_num(r['R² CV'], 3)}", icono="🧠",
                             sub=f"RMSE {fmt_num(r['RMSE CV'], 3)} · MAE {fmt_num(r['MAE CV'], 3)}", acento=ac))
    tarjetas.append(dict(titulo="Desv. estándar del objetivo", valor=fmt_num(ml["y"].std()), icono="📏",
                         sub="referencia para el RMSE", acento="accent-slate"))
    fila_tarjetas(tarjetas)
    st.write("")
    t1, t2, t3, t4 = st.tabs(["🏁 Comparación de modelos", "🎯 Observado vs. predicho", "🧬 Coeficientes y selección",
                              "🗃️ Dataset e imputación"])
    with t1:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = make_subplots(rows=1, cols=3, subplot_titles=["R² (↑)", "RMSE (↓)", "MAE (↓)"])
            for j, (met, sd) in enumerate([("R² CV", "± R²"), ("RMSE CV", "± RMSE"), ("MAE CV", "± MAE")], start=1):
                fig.add_trace(go.Bar(x=resumen["modelo"], y=resumen[met], error_y=dict(type="data", array=resumen[sd]),
                                     marker_color=[DORADO if n == mejor else AZUL for n in resumen["modelo"]],
                                     showlegend=False, text=resumen[met].map(lambda v: f"{v:.3f}"),
                                     textposition="outside"), row=1, col=j)
            fig.update_layout(title="Desempeño en validación cruzada (media ± desviación entre pliegues)", height=430)
            mostrar_fig(fig, key="ml_bars")
        with c2:
            folds = pd.concat([pd.DataFrame({"modelo": n, "R²": m[n]["r2"]}) for n in nombres])
            fig = px.box(folds, x="modelo", y="R²", points="all", color="modelo", color_discrete_sequence=PALETA)
            fig.update_layout(title="R² por pliegue", height=430, showlegend=False)
            mostrar_fig(fig, key="ml_box")
        mostrar_tabla(resumen, column_config={c: st.column_config.NumberColumn(format="%.4f")
                                              for c in resumen.columns if c != "modelo"})
        brecha = resumen.set_index("modelo")["R² entrenamiento"] - resumen.set_index("modelo")["R² CV"]
        if (brecha > 0.25).any():
            callout("Diferencia amplia entre R² de entrenamiento y de validación en "
                    f"<b>{', '.join(brecha[brecha > 0.25].index)}</b>: posible sobreajuste; aumente la regularización.", "warn")
    with t2:
        df_p = pd.concat([pd.DataFrame({"modelo": n, "observado": ml["y"].values, "predicho": m[n]["oof"],
                                        "departamento": ml["datos"]["departamento_canonico"].values,
                                        "año": ml["datos"]["anio"].values}) for n in nombres])
        fig = px.scatter(df_p, x="observado", y="predicho", color="modelo", facet_col="modelo",
                         hover_data=["departamento", "año"], color_discrete_sequence=PALETA, opacity=.75)
        lo, hi = float(df_p[["observado", "predicho"]].min().min()), float(df_p[["observado", "predicho"]].max().max())
        for j in range(1, len(nombres) + 1):
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=ROJO, dash="dash"),
                                     showlegend=False), row=1, col=j)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(title="Predicciones fuera de pliegue (out-of-fold) vs. valores observados", height=440,
                          showlegend=False)
        mostrar_fig(fig, key="ml_oof")
        sel = st.selectbox("Residuos del modelo", nombres, index=nombres.index(mejor), key="ml_res_sel")
        d = df_p[df_p["modelo"] == sel].assign(residuo=lambda x: x["observado"] - x["predicho"])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(d, x="predicho", y="residuo", hover_data=["departamento", "año"], color_discrete_sequence=[AZUL])
            fig.add_hline(y=0, line_color=ROJO, line_dash="dash")
            fig.update_layout(title="Residuos vs. predicción", height=380)
            mostrar_fig(fig, key="ml_resid")
        with c2:
            err = d.groupby("departamento")["residuo"].apply(lambda r: np.sqrt((r ** 2).mean())).sort_values().tail(12)
            fig = px.bar(err, orientation="h", color=err.values, color_continuous_scale="OrRd")
            fig.update_layout(title="Departamentos con mayor error (RMSE)", height=380, coloraxis_showscale=False,
                              showlegend=False, yaxis_title=None, xaxis_title="RMSE")
            mostrar_fig(fig, key="ml_err_dep")
    with t3:
        coefs = []
        for n in ["Ridge", "LASSO"]:
            mod = m[n]["modelo"].named_steps["modelo"]
            for f, c in zip(ml["feats"], mod.coef_):
                coefs.append({"modelo": n, "variable": etiqueta(f), "coeficiente estandarizado": c})
        dc = pd.DataFrame(coefs)
        fig = px.bar(dc, x="coeficiente estandarizado", y="variable", color="modelo", barmode="group", orientation="h",
                     color_discrete_sequence=[AZUL, DORADO])
        fig.add_vline(x=0, line_color=PIZARRA)
        fig.update_layout(title="Coeficientes estandarizados (efecto de +1 desviación estándar del predictor)",
                          height=max(380, 50 * len(ml["feats"]) + 120))
        mostrar_fig(fig, key="ml_coef")
        las = m["LASSO"]["modelo"].named_steps["modelo"].coef_
        sel_l = [etiqueta(f) for f, c in zip(ml["feats"], las) if abs(c) > 1e-8]
        callout(f"LASSO retiene <b>{len(sel_l)}</b> de {len(ml['feats'])} variables: {', '.join(sel_l) or 'ninguna'}. "
                "Aumente α para una selección más agresiva.")
        # Trayectoria de regularización LASSO
        alphas = np.logspace(-3, 1.5, 30)
        idx_log = tuple(i for i, c in enumerate(ml["X"].columns) if cfg["log"] and transformacion_sugerida(c) == "log")
        Xt = StandardScaler().fit_transform(_log_cols(ml["X"].values, idx_log))
        tray = np.array([Lasso(alpha=a, max_iter=20000).fit(Xt, ml["y"]).coef_ for a in alphas])
        fig = go.Figure([go.Scatter(x=alphas, y=tray[:, j], mode="lines", name=etiqueta(f),
                                    line=dict(color=PALETA[j % len(PALETA)], width=2.2))
                         for j, f in enumerate(ml["feats"])])
        fig.add_vline(x=cfg["alpha_lasso"], line_dash="dash", line_color=ROJO, annotation_text="α elegido")
        fig.update_layout(title="Trayectoria de regularización LASSO", xaxis_type="log", height=420,
                          xaxis_title="α", yaxis_title="coeficiente")
        mostrar_fig(fig, key="ml_path")
    with t4:
        st.markdown("##### Bitácora de imputación (media del año → año más cercano → media global)")
        mostrar_tabla(ml["imp"], column_config={"% imputado": st.column_config.ProgressColumn(format="percent",
                                                                                             min_value=0, max_value=1)})
        mostrar_tabla(ml["datos"], height=360)
        descargar_csv(ml["datos"], f"dataset_ml_{cfg['objetivo']}.csv", key="dl_ml")
    callout("Los modelos aprenden <b>asociaciones</b> departamentales (ecológicas) y no efectos causales. "
            "La validación agrupada por departamento estima la capacidad de generalizar a territorios no vistos.", "warn")


# =============================================================================
# MÓDULO 8 – SIMULADOR DE POLÍTICA PÚBLICA E INTERPRETABILIDAD (XAI)
# =============================================================================
def _predictor(modelo: Pipeline, columnas: list[str]):
    def f(A):
        return modelo.predict(pd.DataFrame(np.atleast_2d(A), columns=columnas))
    return f


def _shap_manual(f, Xb: np.ndarray, Xe: np.ndarray, M: int = 24, semilla: int = 0) -> tuple[np.ndarray, float]:
    """Valores de Shapley por muestreo de permutaciones (antitéticas) promediando sobre todo el fondo
    (enfoque intervencional). Respaldo cuando la librería shap no está instalada."""
    rng = np.random.default_rng(semilla)
    p, B = Xe.shape[1], len(Xb)
    base = float(np.mean(f(Xb)))
    phis = np.zeros_like(Xe, dtype=float)
    perms = []
    for _ in range(max(1, M // 2)):
        pm = rng.permutation(p)
        perms += [pm, pm[::-1]]
    for r, x in enumerate(Xe):
        bloques = []
        for perm in perms:
            Z = np.repeat(Xb[None, :, :], p + 1, axis=0).copy()  # (p+1, B, p)
            for paso, j in enumerate(perm, start=1):
                Z[paso:, :, j] = x[j]
            bloques.append(Z.reshape(-1, p))
        pred = f(np.vstack(bloques)).reshape(len(perms), p + 1, B).mean(axis=2)
        for k_, perm in enumerate(perms):
            phis[r, perm] += np.diff(pred[k_])
        phis[r] /= len(perms)
    return phis, base


def valores_shap(modelo: Pipeline, columnas: list[str], Xb: pd.DataFrame, Xe: pd.DataFrame) -> tuple[np.ndarray, float, str]:
    f = _predictor(modelo, columnas)
    fondo = Xb.sample(min(50, len(Xb)), random_state=0).values
    if HAS_SHAP:
        try:
            masker = shap.maskers.Independent(fondo, max_samples=len(fondo))
            algoritmo = "exact" if len(columnas) <= 10 else "permutation"
            expl = shap.Explainer(f, masker, algorithm=algoritmo, feature_names=columnas)
            sv = expl(Xe.values)
            base = float(np.atleast_1d(sv.base_values)[0])
            return np.asarray(sv.values), base, f"shap.Explainer ({algoritmo})"
        except Exception:
            pass
    phis, base = _shap_manual(f, fondo, Xe.values)
    return phis, base, "Shapley por permutaciones (implementación interna)"


@st.cache_data(show_spinner="Calculando importancia global SHAP…")
def shap_global(_modelo: Pipeline, columnas: tuple, X: pd.DataFrame, clave: str) -> tuple[np.ndarray, float, str, pd.DataFrame]:
    Xe = X.sample(min(80, len(X)), random_state=1)
    v, b, motor = valores_shap(_modelo, list(columnas), X, Xe)
    return v, b, motor, Xe


def _reset_sliders(feats: list[str]) -> None:
    for f in feats:
        st.session_state[f"sim_{f}"] = 0.0


def modulo_simulador(panel: pd.DataFrame) -> None:
    encabezado("🎛️", "Simulador de Política Pública e Interpretabilidad (XAI)",
               "Escenarios contrafactuales con sliders · explicación SHAP local y global · curvas ceteris paribus")
    deptos = solo_deptos(panel)
    cfg = st.session_state.get("ml_cfg", config_ml_defecto(deptos))
    try:
        ml = preparar_ml(deptos, cfg)
    except Exception as e:
        st.error(f"No fue posible preparar los modelos: {e}")
        return
    if ml is None:
        st.warning("No hay datos suficientes para entrenar el modelo del simulador. Ajuste la configuración en el módulo 🤖.")
        return
    m = ml["modelos"]
    nombres = [n for n in m if not n.startswith("_")]
    r2 = {n: m[n]["r2"].mean() for n in nombres}
    mejor = max(r2, key=r2.get)
    feats, X, datos = ml["feats"], ml["X"], ml["datos"]
    objetivo = cfg["objetivo"]
    pol_obj = polaridad(objetivo)
    callout(f"Modelos configurados en el módulo 🤖 · objetivo <b>{etiqueta(objetivo)}</b> · "
            f"{len(feats)} predictores. Cambie la configuración allí para simular otra variable.")
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    modelo_n = c1.selectbox("Modelo", nombres, index=nombres.index(mejor),
                            format_func=lambda n: f"{n} (R² CV {fmt_num(r2[n], 2)}){' 🏆' if n == mejor else ''}", key="sim_mod")
    deps = sorted(datos["departamento_canonico"].unique())
    dep = c2.selectbox("Departamento", deps, index=deps.index("Chocó") if "Chocó" in deps else 0, key="sim_dep")
    anios = sorted(datos.loc[datos["departamento_canonico"] == dep, "anio"].unique())
    anio = c3.selectbox("Año base", anios[::-1], key="sim_anio")
    modelo = m[modelo_n]["modelo"]
    fila = datos[(datos["departamento_canonico"] == dep) & (datos["anio"] == anio)].iloc[0]
    x0 = fila[feats].astype(float)
    y_obs = float(fila["y"])

    st.markdown("#### 🎚️ Palancas de política (variación respecto al año base)")
    st.button("↺ Restablecer escenario", on_click=_reset_sliders, args=(feats,), key="sim_reset")
    x1 = x0.copy()
    cols = st.columns(3)
    for i, f in enumerate(feats):
        sd = float(X[f].std()) if X[f].std() > 0 else 1.0
        with cols[i % 3]:
            if transformacion_sugerida(f) == "log":
                pct = st.slider(f"{etiqueta(f)} · variación %", -50.0, 100.0, 0.0, 5.0, key=f"sim_{f}",
                                help=f"Base: {fmt_num(x0[f])}")
                x1[f] = x0[f] * (1 + pct / 100)
            else:
                paso = float(10 ** np.floor(np.log10(sd / 10))) if sd > 0 else 0.1
                lim = float(np.ceil(2 * sd / paso) * paso)
                dlt = st.slider(f"{etiqueta(f)} · Δ unidades", -lim, lim, 0.0, paso, key=f"sim_{f}",
                                help=f"Base: {fmt_num(x0[f])} · σ = {fmt_num(sd)}")
                x1[f] = x0[f] + dlt
            st.caption(f"{fmt_num(x0[f])} → **{fmt_num(x1[f])}**")
    f_pred = _predictor(modelo, feats)
    p0 = float(f_pred(x0.values)[0]); p1 = float(f_pred(x1.values)[0])
    d = p1 - p0
    fila_tarjetas([
        dict(titulo=f"Observado {anio}", valor=fmt_num(y_obs), icono="📌", sub=etiqueta(objetivo), acento="accent-slate"),
        dict(titulo="Predicción línea base", valor=fmt_num(p0), icono="🧠", sub=f"error {fmt_num(p0 - y_obs)}"),
        dict(titulo="Predicción del escenario", valor=fmt_num(p1), icono="🎯", acento="accent-gold",
             delta=d, delta_txt=f"{fmt_num(d)} ({fmt_pct(d / p0, signo=True)})" if p0 else fmt_num(d), polar=pol_obj),
        dict(titulo="Impacto estimado", valor=("Mejora" if (d * pol_obj > 0) else ("Deterioro" if d * pol_obj < 0 else "Sin cambio")),
             icono="✅" if d * pol_obj > 0 else ("⚠️" if d * pol_obj < 0 else "➖"),
             sub="según polaridad del indicador", acento="accent-emerald" if d * pol_obj > 0 else "accent-red"),
    ])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["🔬 Explicación local (SHAP)", "🌐 Importancia global", "📉 Curvas ceteris paribus",
                              "🗺️ Escenario territorial"])
    with t1:
        Xe = pd.DataFrame([x0.values, x1.values], columns=feats)
        sv, base, motor = valores_shap(modelo, feats, X, Xe)
        c1, c2 = st.columns([1.25, 1])
        with c1:
            orden = np.argsort(np.abs(sv[1]))[::-1]
            fig = go.Figure(go.Waterfall(
                orientation="h", measure=["absolute"] + ["relative"] * len(feats) + ["total"],
                y=["E[f(X)] (base)"] + [f"{etiqueta(feats[j])} = {fmt_num(x1.iloc[j])}" for j in orden] + ["Predicción escenario"],
                x=[base] + [sv[1][j] for j in orden] + [0],
                text=[fmt_num(base)] + [f"{sv[1][j]:+.3f}" for j in orden] + [fmt_num(base + sv[1].sum())],
                increasing=dict(marker=dict(color=ROJO if pol_obj < 0 else ESMERALDA)),
                decreasing=dict(marker=dict(color=ESMERALDA if pol_obj < 0 else ROJO)),
                totals=dict(marker=dict(color=AZUL)), connector=dict(line=dict(color="#CBD5E1"))))
            fig.update_layout(title=f"Cascada SHAP del escenario – {dep} {anio}", height=max(420, 44 * len(feats) + 160),
                              yaxis=dict(autorange="reversed"))
            mostrar_fig(fig, key="shap_wf")
        with c2:
            imp_cf = pd.DataFrame({"variable": [etiqueta(f) for f in feats], "Δ contribución": sv[1] - sv[0]})
            imp_cf = imp_cf.reindex(imp_cf["Δ contribución"].abs().sort_values().index)
            fig = px.bar(imp_cf, x="Δ contribución", y="variable", orientation="h", color="Δ contribución",
                         color_continuous_scale="RdYlGn_r" if pol_obj < 0 else "RdYlGn", color_continuous_midpoint=0)
            fig.update_layout(title="Impacto contrafactual por palanca (SHAP escenario − base)", height=max(420, 44 * len(feats) + 160),
                              coloraxis_showscale=False, yaxis_title=None)
            mostrar_fig(fig, key="shap_cf")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=p1, delta=dict(reference=p0, increasing=dict(color=ROJO if pol_obj < 0 else ESMERALDA),
                                                               decreasing=dict(color=ESMERALDA if pol_obj < 0 else ROJO)),
                gauge=dict(axis=dict(range=[float(ml["y"].min()), float(ml["y"].max())]), bar=dict(color=AZUL),
                           steps=[dict(range=[float(ml["y"].min()), float(ml["y"].quantile(.33))], color="#D1FAE5" if pol_obj < 0 else "#FEE2E2"),
                                  dict(range=[float(ml["y"].quantile(.33)), float(ml["y"].quantile(.66))], color="#FEF3C7"),
                                  dict(range=[float(ml["y"].quantile(.66)), float(ml["y"].max())], color="#FEE2E2" if pol_obj < 0 else "#D1FAE5")],
                           threshold=dict(line=dict(color=DORADO, width=4), value=p0)),
                title=dict(text=f"{etiqueta(objetivo)}<br><span style='font-size:.8em;color:#64748B'>escenario vs. base</span>")))
            fig.update_layout(height=280, margin=dict(l=30, r=30, t=70, b=10))
            mostrar_fig(fig, key="gauge")
        st.caption(f"Motor de explicabilidad: {motor}. Fondo: muestra de {min(50, len(X))} observaciones departamento-año.")
    with t2:
        clave = f"{modelo_n}|{cfg['objetivo']}|{'|'.join(feats)}|{len(X)}"
        svg, bg, motor_g, Xg = shap_global(modelo, tuple(feats), X, clave)
        imp = pd.DataFrame({"variable": [etiqueta(f) for f in feats], "media |SHAP|": np.abs(svg).mean(axis=0)}).sort_values("media |SHAP|")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            fig = px.bar(imp, x="media |SHAP|", y="variable", orientation="h", color="media |SHAP|",
                         color_continuous_scale=[[0, "#BFDBFE"], [1, AZUL]])
            fig.update_layout(title="Importancia global (media |SHAP|)", height=max(380, 46 * len(feats) + 120),
                              coloraxis_showscale=False, yaxis_title=None)
            mostrar_fig(fig, key="shap_imp")
        with c2:
            filas = []
            for j, f in enumerate(feats):
                v = Xg[f].values
                vn = (v - v.min()) / (np.ptp(v) or 1)
                filas.append(pd.DataFrame({"variable": etiqueta(f), "SHAP": svg[:, j], "valor normalizado": vn, "valor": v}))
            bee = pd.concat(filas)
            orden_v = imp["variable"].tolist()
            bee["y"] = bee["variable"].map({v: i for i, v in enumerate(orden_v)}) + np.random.default_rng(0).uniform(-.28, .28, len(bee))
            fig = px.scatter(bee, x="SHAP", y="y", color="valor normalizado", color_continuous_scale="Bluered",
                             hover_data={"variable": True, "valor": ":.3f", "y": False})
            fig.update_yaxes(tickvals=list(range(len(orden_v))), ticktext=orden_v, title=None)
            fig.add_vline(x=0, line_color=PIZARRA)
            fig.update_layout(title="Resumen SHAP (beeswarm): color = valor del predictor (rojo alto, azul bajo)",
                              height=max(380, 46 * len(feats) + 120))
            mostrar_fig(fig, key="shap_bee")
        st.caption(f"Motor: {motor_g} · {len(Xg)} observaciones explicadas.")
    with t3:
        fsel = st.selectbox("Variable a perturbar", feats, format_func=etiqueta, key="cp_var")
        lo, hi = float(X[fsel].quantile(.02)), float(X[fsel].quantile(.98))
        grid = np.linspace(min(lo, x1[fsel]), max(hi, x1[fsel]), 60)
        filas = np.tile(x1.values, (len(grid), 1)); filas[:, feats.index(fsel)] = grid
        curvas = {n: _predictor(m[n]["modelo"], feats)(filas) for n in nombres}
        fig = go.Figure()
        for i, n in enumerate(nombres):
            fig.add_trace(go.Scatter(x=grid, y=curvas[n], mode="lines", name=n,
                                     line=dict(color=PALETA[i], width=3 if n == modelo_n else 1.6,
                                               dash="solid" if n == modelo_n else "dot")))
        fig.add_trace(go.Scatter(x=[x0[fsel]], y=[_predictor(modelo, feats)(np.where(np.arange(len(feats)) == feats.index(fsel), x0.values, x1.values))[0]],
                                 mode="markers", marker=dict(size=13, color=PIZARRA, symbol="circle-open", line=dict(width=3)),
                                 name="Valor base"))
        fig.add_trace(go.Scatter(x=[x1[fsel]], y=[p1], mode="markers", marker=dict(size=14, color=DORADO, symbol="star"),
                                 name="Escenario"))
        fig.update_layout(title=f"Curva ceteris paribus – {etiqueta(fsel)} ({dep}, demás variables en el escenario)",
                          height=440, xaxis_title=etiqueta(fsel), yaxis_title=etiqueta(objetivo))
        mostrar_fig(fig, key="cp_curve")
        callout("La curva muestra cómo cambia la predicción al mover sólo una variable. Pendientes distintas entre modelos "
                "revelan no linealidades (SVR) frente a efectos lineales (Ridge/LASSO).")
    with t4:
        ult = datos.sort_values("anio").groupby("departamento_canonico").tail(1)
        Xb = ult[feats].astype(float).copy()
        Xs = Xb.copy()
        for f in feats:
            if transformacion_sugerida(f) == "log":
                Xs[f] = Xb[f] * (x1[f] / x0[f] if x0[f] else 1)
            else:
                Xs[f] = Xb[f] + (x1[f] - x0[f])
        pb, ps = f_pred(Xb.values), f_pred(Xs.values)
        esc = pd.DataFrame({"departamento_canonico": ult["departamento_canonico"].values, "año": ult["anio"].values,
                            "predicción base": pb, "predicción escenario": ps, "Δ": ps - pb}).sort_values("Δ")
        esc["codigo_divipola"] = esc["departamento_canonico"].map(DIVIPOLA)
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(esc, x="Δ", y="departamento_canonico", orientation="h", color="Δ",
                         color_continuous_scale="RdYlGn_r" if pol_obj < 0 else "RdYlGn", color_continuous_midpoint=0)
            fig.update_layout(title="Efecto del mismo paquete de política en cada departamento (último año)",
                              height=max(460, 20 * len(esc) + 120), coloraxis_showscale=False, yaxis_title=None)
            mostrar_fig(fig, key="esc_bar")
        with c2:
            gj, _, _ = cargar_geojson()
            cents = centroides(gj, "sim")
            fig = _mapa(esc, "Δ", gj, cents, "Mapa del impacto simulado", escala="RdYlGn_r" if pol_obj < 0 else "RdYlGn",
                        hover_extra=["predicción base", "predicción escenario"], altura=max(460, 20 * len(esc) + 120))
            mostrar_fig(fig, key="esc_map")
        descargar_csv(esc, "escenario_politica_territorial.csv", key="dl_esc")
    callout("⚠️ Simulación basada en asociaciones estadísticas aprendidas de datos departamentales; los resultados son "
            "orientativos y no constituyen estimaciones causales del efecto de una política.", "warn")


# =============================================================================
# MÓDULO 9 – FICHA DEPARTAMENTAL Y DESCARGA DE DATOS
# =============================================================================
def _kpi_departamento(panel: pd.DataFrame, deptos: pd.DataFrame, dep: str, ind: str) -> dict | None:
    s = deptos[(deptos["departamento_canonico"] == dep) & (deptos["indicador"] == ind) &
               (deptos["desagregacion"] == DESAG_TOTAL) & deptos["valor"].notna()].sort_values("anio")
    if s.empty:
        return None
    ult = s.iloc[-1]
    a = int(ult["anio"])
    prev = s[s["anio"] < a]
    delta = float(ult["valor"] - prev.iloc[-1]["valor"]) if not prev.empty else None
    corte_a = corte(deptos, ind, a)
    pol = polaridad(ind)
    asc = pol < 0
    rk = corte_a["valor"].rank(ascending=asc, method="min")
    puesto = int(rk[corte_a["departamento_canonico"] == dep].iloc[0]) if (corte_a["departamento_canonico"] == dep).any() else None
    nac = corte(nacional(panel), ind, a)
    pct = float((corte_a["valor"] < ult["valor"]).mean() if pol >= 0 else (corte_a["valor"] > ult["valor"]).mean())
    return dict(indicador=ind, anio=a, valor=float(ult["valor"]), delta=delta, puesto=puesto, n=len(corte_a),
                nacional=float(nac["valor"].iloc[0]) if not nac.empty else np.nan,
                promedio=float(corte_a["valor"].mean()), percentil=pct * 100, marca=ult.get("marca_calidad"))


def modulo_ficha(panel: pd.DataFrame, catalogo: pd.DataFrame) -> None:
    encabezado("📋", "Ficha Departamental y Descarga de Datos",
               "KPIs del último año disponible · posición relativa · series históricas en facetas · exportación de microdatos")
    deptos = solo_deptos(panel)
    deps = sorted(deptos["departamento_canonico"].unique())
    c1, c2 = st.columns([1.3, 2])
    dep = c1.selectbox("Departamento", deps, index=deps.index("Antioquia") if "Antioquia" in deps else 0, key="fi_dep")
    cod = DIVIPOLA.get(dep, "—")
    sub_dep = deptos[(deptos["departamento_canonico"] == dep) & deptos["valor"].notna()]
    c2.markdown(
        f"""<div class="metric-card accent-gold" style="min-height:auto">
        <div class="mc-title">Ficha técnica ejecutiva</div>
        <div class="mc-value">{dep}</div>
        <div class="mc-sub">DIVIPOLA <b>{cod}</b> · Región <b>{CODIGO_A_REGION.get(cod, '—')}</b> ·
        {sub_dep['indicador'].nunique()} indicadores · {int(sub_dep['anio'].min()) if not sub_dep.empty else '—'}–{int(sub_dep['anio'].max()) if not sub_dep.empty else '—'}
        · {len(sub_dep):,} observaciones</div></div>""".replace(",", "."),
        unsafe_allow_html=True)
    st.write("")
    kpis = [k for k in (_kpi_departamento(panel, deptos, dep, i) for i in INDICADORES_CLAVE) if k]
    for bloque in [kpis[i:i + 4] for i in range(0, len(kpis), 4)]:
        fila_tarjetas([dict(
            titulo=f"{etiqueta(k['indicador'])} · {k['anio']}", valor=fmt_num(k["valor"]), icono="📊",
            sub=f"Puesto {k['puesto']}/{k['n']} · Nal. {fmt_num(k['nacional'])}", delta=k["delta"],
            delta_txt=f"{fmt_num(k['delta'])} vs. año previo" if k["delta"] is not None else None,
            polar=polaridad(k["indicador"]),
            acento="accent-emerald" if k["percentil"] >= 66 else ("accent-red" if k["percentil"] < 33 else "accent-gold"))
            for k in bloque], n_cols=4)
        st.write("")
    t1, t2, t3, t4 = st.tabs(["📈 Series históricas", "🧭 Posición relativa", "🧾 Tabla comparativa",
                              "📥 Descarga de microdatos"])
    with t1:
        opts = indicadores_con_datos(sub_dep)
        inds = st.multiselect("Indicadores", opts, default=[i for i in INDICADORES_CLAVE if i in opts],
                              format_func=etiqueta, key="fi_inds")
        ref = st.toggle("Comparar con Total nacional y promedio departamental", value=True, key="fi_ref")
        if inds:
            s = panel[(panel["departamento_canonico"] == dep) & panel["indicador"].isin(inds) &
                      (panel["desagregacion"] == DESAG_TOTAL) & panel["valor"].notna()].assign(territorio=dep)
            capas = [s]
            if ref:
                capas.append(panel[(panel["departamento_canonico"] == TOTAL_NACIONAL) & panel["indicador"].isin(inds) &
                                   (panel["desagregacion"] == DESAG_TOTAL) & panel["valor"].notna()].assign(territorio="Total nacional"))
                pm = (deptos[deptos["indicador"].isin(inds) & (deptos["desagregacion"] == DESAG_TOTAL)]
                      .groupby(["indicador", "anio"])["valor"].mean().reset_index().assign(territorio="Promedio departamental"))
                capas.append(pm)
            df = pd.concat(capas, ignore_index=True)
            df["serie"] = df["indicador"].map(etiqueta)
            fig = px.line(df, x="anio", y="valor", color="territorio", facet_col="serie", facet_col_wrap=3,
                          markers=True, facet_col_spacing=0.06, facet_row_spacing=0.09,
                          color_discrete_map={dep: AZUL, "Total nacional": DORADO, "Promedio departamental": PIZARRA_CL})
            fig.update_yaxes(matches=None, showticklabels=True, title=None)
            fig.update_xaxes(showticklabels=True, title=None)
            fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
            filas = int(np.ceil(len(inds) / 3))
            fig.update_layout(height=290 * filas + 110, title=f"Trayectorias históricas – {dep}",
                              legend=dict(orientation="h", y=1.04 + 0.03 / filas, x=0))
            mostrar_fig(fig, key="fi_facets")
    with t2:
        if kpis:
            pk = pd.DataFrame(kpis)
            pk["etiqueta"] = pk["indicador"].map(etiqueta)
            pk = pk.sort_values("percentil")
            fig = go.Figure(go.Bar(x=pk["percentil"], y=pk["etiqueta"], orientation="h",
                                   marker=dict(color=pk["percentil"], colorscale=[[0, ROJO], [.5, DORADO], [1, ESMERALDA]],
                                               cmin=0, cmax=100),
                                   text=[f"P{p:.0f} · puesto {r}/{n}" for p, r, n in zip(pk["percentil"], pk["puesto"], pk["n"])],
                                   textposition="outside", cliponaxis=False))
            fig.add_vline(x=50, line_dash="dash", line_color=PIZARRA, annotation_text="Mediana")
            fig.update_layout(title="Percentil de desempeño frente a los demás departamentos (100 = mejor)",
                              height=max(380, 44 * len(pk) + 120), xaxis=dict(range=[0, 118]), yaxis_title=None)
            mostrar_fig(fig, key="fi_pct")
        ipm = deptos[(deptos["departamento_canonico"] == dep) & (deptos["indicador"] == "IPM") & deptos["valor"].notna()]
        if not ipm.empty:
            pv = ipm.pivot_table(index="anio", columns="desagregacion", values="valor")
            fig = go.Figure()
            for i, c in enumerate([x for x in [DESAG_TOTAL, DESAG_CAB, DESAG_RURAL, "Hombre", "Mujer"] if x in pv.columns]):
                fig.add_trace(go.Scatter(x=pv.index, y=pv[c], name=c, mode="lines+markers", line=dict(color=PALETA[i], width=2.5)))
            fig.update_layout(title=f"IPM por desagregación – {dep}", height=380, yaxis_title="%", hovermode="x unified")
            mostrar_fig(fig, key="fi_ipm")
    with t3:
        if kpis:
            tk = pd.DataFrame(kpis)
            tk.insert(1, "nombre", tk["indicador"].map(etiqueta))
            tk["brecha vs. nacional"] = tk["valor"] - tk["nacional"]
            tk = tk.rename(columns={"anio": "año", "n": "deptos. comparados"})
            mostrar_tabla(tk, column_config={
                "valor": st.column_config.NumberColumn(format="%.3f"), "nacional": st.column_config.NumberColumn(format="%.3f"),
                "promedio": st.column_config.NumberColumn("promedio dptal.", format="%.3f"),
                "delta": st.column_config.NumberColumn("Δ año previo", format="%.3f"),
                "brecha vs. nacional": st.column_config.NumberColumn(format="%.3f"),
                "percentil": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
            descargar_csv(tk, f"ficha_{normalizar_texto(dep).replace(' ', '_').lower()}.csv", "⬇️ Descargar ficha (CSV)",
                          key="dl_ficha")
    with t4:
        st.markdown("##### Constructor de consultas sobre el panel")
        c1, c2 = st.columns(2)
        sel_deps = c1.multiselect("Territorios", [TOTAL_NACIONAL] + deps if TOTAL_NACIONAL in panel["departamento_canonico"].values else deps,
                                  default=[dep], key="dl_deps")
        sel_inds = c2.multiselect("Indicadores (vacío = todos)", ordenar_indicadores(panel["indicador"].unique()),
                                  format_func=etiqueta, key="dl_inds")
        c3, c4, c5 = st.columns([1, 1.6, 1])
        sel_des = c3.multiselect("Desagregaciones (vacío = todas)", sorted(panel["desagregacion"].dropna().unique()), key="dl_des")
        a_min, a_max = int(panel["anio"].min()), int(panel["anio"].max())
        rango = c4.slider("Años", a_min, a_max, (a_min, a_max), key="dl_rng")
        formato = c5.radio("Formato", ["Largo (microdatos)", "Ancho (pivot)"], key="dl_fmt")
        solo_obs = st.checkbox("Excluir registros sin valor", value=True, key="dl_obs")
        q = panel[panel["anio"].between(*rango)]
        if sel_deps:
            q = q[q["departamento_canonico"].isin(sel_deps)]
        if sel_inds:
            q = q[q["indicador"].isin(sel_inds)]
        if sel_des:
            q = q[q["desagregacion"].isin(sel_des)]
        if solo_obs:
            q = q[q["valor"].notna()]
        if formato.startswith("Ancho") and not q.empty:
            q = q.pivot_table(index=["departamento_canonico", "codigo_divipola", "anio"],
                              columns=["indicador", "desagregacion"], values="valor")
            q.columns = [f"{i} | {d}" for i, d in q.columns]
            q = q.reset_index()
        fila_tarjetas([dict(titulo="Filas seleccionadas", valor=fmt_num(len(q), 0), icono="🧾"),
                       dict(titulo="Columnas", valor=str(q.shape[1]), icono="📑", acento="accent-gold"),
                       dict(titulo="Tamaño estimado", valor=f"{q.to_csv(index=False).__len__() / 1024:,.0f} KB".replace(",", "."),
                            icono="💾", acento="accent-emerald")])
        st.write("")
        mostrar_tabla(q.head(500), height=360)
        if len(q) > 500:
            st.caption("Vista previa limitada a 500 filas; la descarga incluye todas.")
        cA, cB = st.columns(2)
        with cA:
            descargar_csv(q, f"observatorio_microdatos_{rango[0]}_{rango[1]}.csv", "⬇️ Descargar microdatos filtrados (CSV)",
                          key="dl_micro")
        with cB:
            descargar_csv(catalogo, "catalogo_indicadores.csv", "⬇️ Descargar catálogo de indicadores", key="dl_catalogo")


# =============================================================================
# BARRA LATERAL, PANTALLA SIN DATOS Y PROGRAMA PRINCIPAL
# =============================================================================
def barra_lateral(panel: pd.DataFrame, origen: str) -> str:
    with st.sidebar:
        st.markdown('<div class="sb-brand"><div class="flag">🇨🇴</div><div class="t1">Observatorio Socioeconómico</div>'
                    '<div class="t2">Colombia · Fase 4</div></div>', unsafe_allow_html=True)
        modulo = st.radio("Módulos", MODULOS, key="nav", label_visibility="collapsed")
        st.markdown("---")
        ok = lambda b: "✅" if b else "⚪"
        st.markdown(
            f"""<div class="sb-status"><b>Fuente de datos</b><br>{origen}<br>
            {len(panel):,} registros · {panel['indicador'].nunique()} indicadores<br>
            {int(panel['anio'].min())}–{int(panel['anio'].max())}<br><br>
            <b>Motores analíticos</b><br>
            {ok(HAS_SM)} statsmodels (VAR/VEC)<br>{ok(HAS_PYSAL)} libpysal + esda (Moran)<br>
            {ok(HAS_GPD)} geopandas · {ok(HAS_SHAPELY)} shapely<br>{ok(HAS_SHAP)} shap (XAI)<br>
            <span style="opacity:.75">⚪ = respaldo interno activo</span></div>""".replace(",", "."),
            unsafe_allow_html=True)
        st.caption("Fuentes: DANE (Cuentas nacionales, GEIH, ENCV) · Consejo Privado de Competitividad.")
    return modulo


def pantalla_sin_datos() -> None:
    st.markdown(
        """<div class="hero"><span class="badge">Fase 4 · Configuración</span>
        <h1>🇨🇴 Observatorio Integrado de Indicadores Socioeconómicos</h1>
        <p>No se encontraron los productos de la Fase 2 en la carpeta <code>data/</code>.</p></div>""",
        unsafe_allow_html=True)
    callout("<b>Opción 1.</b> Copie <code>panel_observatorio_fase2.parquet</code> y <code>catalogo_indicadores.csv</code> "
            "en una carpeta <code>data/</code> junto a <code>app.py</code> y recargue la página.<br>"
            "<b>Opción 2.</b> Cárguelos desde la barra lateral (📂 Cargar datos).", "info")
    st.code("proyecto/\n├── app.py\n└── data/\n    ├── panel_observatorio_fase2.parquet\n"
            "    ├── catalogo_indicadores.csv\n    └── colombia_departamentos.geojson   (opcional)", language="text")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    configurar_plotly()
    panel, catalogo, origen = obtener_datos()
    if panel is None or panel.empty:
        pantalla_sin_datos()
        st.stop()
    modulo = barra_lateral(panel, origen)
    vistas = {
        MODULOS[0]: lambda: modulo_bienvenida(panel, catalogo),
        MODULOS[1]: lambda: modulo_explorador(panel),
        MODULOS[2]: lambda: modulo_evolucion(panel),
        MODULOS[3]: lambda: modulo_eda_pca(panel),
        MODULOS[4]: lambda: modulo_espacial(panel),
        MODULOS[5]: lambda: modulo_series_tiempo(panel),
        MODULOS[6]: lambda: modulo_ml(panel),
        MODULOS[7]: lambda: modulo_simulador(panel),
        MODULOS[8]: lambda: modulo_ficha(panel, catalogo),
    }
    try:
        vistas[modulo]()
    except Exception as e:  # Manejo defensivo: el error de un módulo no tumba la aplicación
        st.error(f"Se produjo un error inesperado en el módulo «{modulo}»: {e}")
        with st.expander("Detalle técnico"):
            st.exception(e)
    st.markdown('<div class="foot">Observatorio Integrado de Indicadores Socioeconómicos de Colombia · Fase 4 · '
                'Datos: DANE y Consejo Privado de Competitividad · Resultados exploratorios con fines analíticos.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
