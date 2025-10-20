import io
import json
import os
import inspect
import datetime
import colorsys
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
try:
    from streamlit_tree_select import tree_select
except Exception:
    tree_select = None
DEFAULT_CONFIG = {'version': '1.0.0', 'ui': {'quick_examples_count': 9, 'column_examples_count': 12}, 'theme': {'mode': 'dark', 'canvas_bg': '#0B1220', 'panel_bg': '#121A2A', 'panel_border': '#1E2A44', 'primary_text': '#E8EEF9', 'secondary_text': '#A9B7D0', 'accent': '#4DA3FF', 'accent_alt': '#7AC8FF', 'kpi_good': '#34D399', 'kpi_neutral': '#FBBF24', 'kpi_bad': '#F87171', 'rev_color': '#4DA3FF', 'ebitda_color': '#7AC8FF', 'margin_color': '#34D399'}, 'fonts': {'title_size': 24, 'section_size': 18, 'label_size': 13}, 'startup': {'default_main_category': 'Aerospace'}, 'metrics': {'revenue': {'fy23': 'Financial_Revenue_FY23', 'fy24': 'Financial_ReVENUE_FY24' if False else 'Financial_Revenue_FY24', 'fy25': 'Financial_Revenue_FY25'}, 'ebitda': {'fy23': 'Financial_EBITDA_FY23', 'fy24': 'Financial_EBITDA_FY24', 'fy25': 'Financial_EBITDA_FY25'}, 'player_revenue': {'fy25': 'Player_Revenue_FY25'}, 'player_ebitda': {'fy25': 'Player_EBITDA_FY25'}}, 'hierarchy': {'levels': ['Hierarchy - Main Category', 'Hierarchy - Sector', 'Hierarchy - Subsector', 'Hierarchy - Sub-Sub-Sector']}, 'comments_sources': {'financial_comment_col': 'Financial Data - Financial Commentary', 'financial_source_col': 'Financial Data - Financial Sources', 'player_comment_col': 'Player data - Player Commentary', 'player_source_col': 'Player data - Player Sources'}, 'player_data': {'names': ['Player data - Name', 'Player data - Name_1', 'Player data - Name_2', 'Player data - Name_3', 'Player data - Name_4'], 'countries': ['Player data - Country', 'Player data - Country_1', 'Player data - Country_2', 'Player data - Country_3', 'Player data - Country_4'], 'types': ['Player data - Type', 'Player data - Type_1', 'Player data - Type_2', 'Player data - Type_3', 'Player data - Type_4']}, 'data_sources': {'revenue': 'Company filings, Market research', 'ebitda': 'Company filings, Market research', 'ebitda_margin': 'Calculated from EBITDA/Revenue'}}
CONFIG_PATH = 'config.json'

# helper to render Plotly figures without Streamlit's deprecated kwargs  
import plotly.io as pio  
import streamlit as st  
from streamlit.components.v1 import html as st_html  
  
import plotly.io as pio
from streamlit.components.v1 import html as st_html
from typing import Optional

def show_plotly(fig, height: Optional[int] = None, config: Optional[dict] = None):
    """
    Render a Plotly figure via HTML to avoid deprecated Streamlit kwargs.
    - fig: plotly.graph_objects.Figure or plotly figure-like object
    - height: int (px) or None
    - config: dict passed to plotly.io.to_html
    """
    # default config (user-provided config overrides)
    cfg = {"displayModeBar": False, "responsive": True}
    if isinstance(config, dict):
        cfg.update(config)

    # try to infer height from figure layout if not provided
    if height is None:
        try:
            h = getattr(fig.layout, "height", None)
            if h is not None:
                try:
                    height = int(h)
                except Exception:
                    height = None
        except Exception:
            height = None

    # fallback height
    h = height or 600

    # render HTML and inject
    html_str = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config=cfg)
    st_html(html_str, height=h, scrolling=True)

def load_config() -> Dict:
    """Load user config.json and merge into DEFAULT_CONFIG safely."""
    cfg = DEFAULT_CONFIG.copy()
    if not os.path.exists(CONFIG_PATH):
        return cfg
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as fh:
            user_cfg = json.load(fh)
        if not isinstance(user_cfg, dict):
            return cfg

        def merge(a: Dict, b: Dict):
            for k, v in b.items():
                if isinstance(v, dict) and k in a and isinstance(a[k], dict):
                    merge(a[k], v)
                else:
                    a[k] = v
            return a

        merge(cfg, user_cfg)
        return cfg
    except Exception:
        # If config is malformed, return defaults rather than crash
        return cfg

CFG = load_config()
st.set_page_config(page_title='Aerospace & Defence Dashboard', layout='wide')

def theme_vars(cfg_theme: Dict) -> Dict[str, str]:
    if cfg_theme.get('mode', 'dark') == 'light':
        return {'CANVAS_BG': '#FAFAFB', 'PANEL_BG': '#FFFF', 'PANEL_BORDER': '#E6E9EE', 'PRIMARY': '#0B1220', 'SECONDARY': '#505766', 'ACCENT': cfg_theme.get('accent', '#2563EB')}
    else:
        return {'CANVAS_BG': cfg_theme.get('canvas_bg', '#0B1220'), 'PANEL_BG': cfg_theme.get('panel_bg', '#121A2A'), 'PANEL_BORDER': cfg_theme.get('panel_border', '#1E2A44'), 'PRIMARY': cfg_theme.get('primary_text', '#E8EEF9'), 'SECONDARY': cfg_theme.get('secondary_text', '#A9B7D0'), 'ACCENT': cfg_theme.get('accent', '#4DA3FF')}

TV = theme_vars(CFG['theme'])
CANVAS_BG = TV['CANVAS_BG']
PANEL_BG = TV['PANEL_BG']
PANEL_BORDER = TV['PANEL_BORDER']
PRIMARY = TV['PRIMARY']
SECONDARY = TV['SECONDARY']
ACCENT = TV['ACCENT']
REV_COLOR = CFG['theme'].get('rev_color', '#4DA3FF')
EBITDA_COLOR = CFG['theme'].get('ebitda_color', '#7AC8FF')
MARGIN_COLOR = CFG['theme'].get('margin_color', '#34D399')

st.markdown(f"""
    <style>
    .stApp {{ background-color: {CANVAS_BG}; color: {PRIMARY}; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
    .panel {{ background: {PANEL_BG}; border: 1px solid {PANEL_BORDER}; border-radius: 10px; padding: 16px 16px 8px 16px; }}
    .section-title {{ font-size: {CFG['fonts']['section_size']}px; font-weight: 600; color: {PRIMARY}; margin-bottom: 8px; }}
    .kpi {{ background: {PANEL_BG}; border: 1px solid {PANEL_BORDER}; border-radius: 10px; padding: 14px; }}
    .kpi .label {{ font-size: 12px; color: {SECONDARY}; margin-bottom: 6px; }}
    .kpi .value {{ font-size: 20px; color: {PRIMARY}; font-weight: 700; }}

    /* Tabs: increase visibility */
    div[role="tablist"] > button {{
    border: 1px solid {ACCENT};
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    padding: 8px 12px;
    margin-right: 6px;
    color: {PRIMARY};
    }}
    /* Selected tab */
    div[role="tablist"] > button[aria-selected="true"] {{
    background: {ACCENT};
    color: #ffff;
    box-shadow: 0 6px 14px rgba(0,0,0,0.25);
    }}

    /* Make plotted text not clipped */
    .js-plotly-plot .plot-container .svg-container {{
    overflow: visible !important;
    }}

    /* Buttons look slightly heavier for clickable examples */
    .stButton>button {{
    border-radius: 6px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---- hide Streamlit header/menu/footer (OPTION 2) ----
st.markdown(
    """
    <style>
    /* Hide Streamlit top-left menu and header (and footer) */
    #MainMenu { visibility: hidden !important; }
    header { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }
    footer { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }

    /* keep a little top padding so content doesn't stick to browser top */
    .block-container {
        padding-top: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_data_file(path: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if not path or not os.path.exists(path):
        return df
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        df = pd.DataFrame()
    return df

def get_active_df() -> pd.DataFrame:
    """Return uploaded dataframe if present, otherwise baseline."""
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    return st.session_state.uploaded_df if st.session_state.uploaded_df is not None else globals().get('df_baseline', pd.DataFrame())

LOCAL_DATA_DEFAULT = 'Aero_and_Defence_Tidy_Data_AIReady.normalized.xlsx'
ALTERNATIVE_DATA = 'Aero_and_Defence_Tidy_Data_AIReady.xlsx'
DEFAULT_DATA_PATH = None
if os.path.exists(LOCAL_DATA_DEFAULT):
    DEFAULT_DATA_PATH = LOCAL_DATA_DEFAULT
elif os.path.exists(ALTERNATIVE_DATA):
    DEFAULT_DATA_PATH = ALTERNATIVE_DATA
try:
    df_baseline = load_data_file(DEFAULT_DATA_PATH) if DEFAULT_DATA_PATH else pd.DataFrame()
except Exception:
    df_baseline = pd.DataFrame()

# session defaults
st.session_state.setdefault('uploaded_df', None)
st.session_state.setdefault('currency_unit', 'Billions')
st.session_state.setdefault('sel_path', {})
st.session_state.setdefault('_rerun_trigger', 0)
st.session_state.setdefault('rollup_mode', 'strict')
st.session_state.setdefault('target_focus', None)
st.session_state.setdefault('show_inspector_hint', False)

HIER_LEVELS: List[str] = CFG['hierarchy']['levels']
QUICK_EXAMPLES_COUNT = int(CFG.get('ui', {}).get('quick_examples_count', 9))
COLUMN_EXAMPLES_COUNT = int(CFG.get('ui', {}).get('column_examples_count', 12))
BASE_COLORS = {'aerospace': '#1f77b4', 'aero': '#1f77b4', 'defence': '#2ca02c', 'defense': '#2ca02c'}

def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    if not hex_color:
        hex_color = '888888'
    h = (hex_color or '').lstrip('#').strip()
    if len(h) == 3:
        h = ''.join([c*2 for c in h])
    if len(h) != 6:
        h = '888888'
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return (136/255.0, 136/255.0, 136/255.0)

def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    try:
        r = int(max(0, min(1, rgb[0])) * 255)
        g = int(max(0, min(1, rgb[1])) * 255)
        b = int(max(0, min(1, rgb[2])) * 255)
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    except Exception:
        return '#888888'

def generate_shades(base_hex: str, n: int, light_min: float=0.35, light_max: float=0.85) -> List[str]:
    """Return n hex shades derived from base_hex by varying lightness (HLS)."""
    if n <= 0:
        return []
    r, g, b = hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    shades = []
    for i in range(n):
        if n == 1:
            li = (light_min + light_max) / 2
        else:
            li = light_min + (light_max - light_min) * (i / (n - 1))
        rr, gg, bb = colorsys.hls_to_rgb(h, li, s)
        shades.append(rgb_to_hex((rr, gg, bb)))
    return shades

def get_base_color_for_top(val: str) -> str:
    if not val:
        return '#888888'
    key = str(val).strip().lower()
    return BASE_COLORS.get(key, None) or BASE_COLORS.get('aero')

def build_horizontal_color_map(df: pd.DataFrame, parent_col: str, child_col: str, top_level_values: Optional[List[str]]=None) -> Dict[Tuple[str, str], str]:
    color_map: Dict[Tuple[str, str], str] = {}
    if parent_col not in df.columns or child_col not in df.columns:
        return color_map
    if top_level_values is None:
        parents = list(df[parent_col].dropna().astype(str).str.strip().unique())
    else:
        parents = top_level_values
    top_col = None
    if len(CFG.get('hierarchy', {}).get('levels', [])) > 0:
        top_col = CFG['hierarchy']['levels'][0]
    for parent in parents:
        if parent is None:
            continue
        mask_parent = df[parent_col].astype(str).str.strip().str.lower() == str(parent).strip().lower()
        children = list(df.loc[mask_parent, child_col].dropna().astype(str).str.strip().unique())
        base = get_base_color_for_top(parent)
        if (not base or base == BASE_COLORS.get('aero')) and top_col and (top_col in df.columns):
            try:
                top_vals = df.loc[mask_parent, top_col].dropna().astype(str).str.strip().unique().tolist()
                if top_vals:
                    base_candidate = get_base_color_for_top(top_vals[0])
                    if base_candidate:
                        base = base_candidate
            except Exception:
                pass
        if not base:
            base = BASE_COLORS.get('aero')
        shades = generate_shades(base, max(1, len(children)))
        for child, shade in zip(children, shades):
            color_map[str(parent), str(child)] = shade
    return color_map

def build_top_level_color_map(df: pd.DataFrame, top_col: str) -> Dict[str, str]:
    col_map: Dict[str, str] = {}
    if top_col not in df.columns:
        return col_map
    vals = list(df[top_col].dropna().astype(str).str.strip().unique())
    fallback = px.colors.qualitative.Plotly
    fi = 0
    for v in vals:
        base = get_base_color_for_top(v)
        if not base:
            base = fallback[fi % len(fallback)]
            fi += 1
        col_map[str(v)] = base
    return col_map

def get_contrast_text(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#000' if lum > 0.6 else '#fff'

def get_unique_in_family_order(series: pd.Series) -> List[str]:
    out = []
    seen = set()
    try:
        for x in series.dropna().astype(str):
            s = x.strip()
            if s == '' or s.lower() == 'nan':
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
    except Exception:
        pass
    return out

def filter_by_path_progressive(df: pd.DataFrame, path: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df
    for level in HIER_LEVELS:
        if level not in out.columns:
            continue
        sel = (path or {}).get(level)
        if sel and sel != 'All':
            out = out[out[level].astype(str).str.strip().str.lower() == str(sel).strip().lower()]
    return out

def is_blank_or_nan(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.isna() | (series.astype(str).str.strip() == '')
    return series.isna()

def apply_selection_strict_rollup(df: pd.DataFrame, path: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df
    for level in HIER_LEVELS:
        if level not in out.columns:
            continue
        sel = (path or {}).get(level)
        if sel and sel != 'All':
            out = out[out[level].astype(str).str.strip().str.lower() == str(sel).strip().lower()]
        else:
            out = out[is_blank_or_nan(out[level])]
    return out

def best_effort_selection(df: pd.DataFrame, path: Dict[str, Optional[str]]) -> pd.DataFrame:
    if st.session_state.rollup_mode == 'strict':
        out = apply_selection_strict_rollup(df, path)
        if len(out) > 0:
            return out
        return filter_by_path_progressive(df, path)
    else:
        out = filter_by_path_progressive(df, path)
        if len(out) > 0:
            return out
        return apply_selection_strict_rollup(df, path)

def convert_currency(value: Optional[float], from_unit='Billions', to_unit='Billions') -> Optional[float]:
    if value is None:
        return None
    if from_unit == to_unit:
        return value
    if from_unit == 'Billions' and to_unit == 'Millions':
        return value * 1000.0
    if from_unit == 'Millions' and to_unit == 'Billions':
        return value / 1000.0
    return value

def get_metric_values(df: pd.DataFrame, label_map: Dict[str, str]) -> Dict[str, Optional[float]]:
    out = {k: None for k in label_map.keys()}
    cols_lower = [c.lower() for c in df.columns]
    if 'fiscal metric flattened' not in cols_lower or 'value' not in cols_lower:
        return out
    metric_col = [c for c in df.columns if c.strip().lower() == 'fiscal metric flattened'][0]
    value_col = [c for c in df.columns if c.strip().lower() == 'value'][0]
    fiscal_metric_col = None
    for c in df.columns:
        if c.strip().lower() == 'fiscal metric':
            fiscal_metric_col = c
            break
    for key, label in label_map.items():
        subset = df[df[metric_col].astype(str).str.strip().str.lower() == str(label).strip().lower()]
        if subset.empty:
            continue
        chosen = None
        if fiscal_metric_col:
            dollar_rows = subset[subset[fiscal_metric_col].astype(str).str.contains('\\$B', na=False)]
            if not dollar_rows.empty:
                vals = pd.to_numeric(dollar_rows[value_col], errors='coerce').dropna()
                if not vals.empty:
                    chosen = float(vals.iloc[0])
        if chosen is None:
            vals = pd.to_numeric(subset[value_col], errors='coerce').dropna()
            if not vals.empty:
                chosen = float(vals.iloc[0])
        out[key] = chosen
    return out

def calculate_ebitda_margin(revenue: Optional[float], ebitda: Optional[float]) -> Optional[float]:
    if revenue and ebitda and (revenue > 0):
        return ebitda / revenue * 100.0
    return None

def kpi_card(label: str, value: Optional[float], suffix: str='', color: Optional[str]=None):
    """Render a KPI card (HTML)."""
    color_style = f'color:{color};' if color else ''
    if value is None:
        st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value" style="{color_style}">—</div></div>', unsafe_allow_html=True)
    else:
        disp = f'{value:,.2f}{suffix}' if isinstance(value, (int, float)) else f'{value}{suffix}'
        st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value" style="{color_style}">{disp}</div></div>', unsafe_allow_html=True)

def plot_bars_with_labels(title: str, series_map: Dict[str, Optional[float]], color: str, is_percent: bool=False, source: str='') -> go.Figure:
    x = ['FY23', 'FY24', 'FY25']
    y = [series_map.get('fy23'), series_map.get('fy24'), series_map.get('fy25')]
    if not is_percent:
        y = [convert_currency(v, 'Billions', st.session_state.currency_unit) if v is not None else None for v in y]
    text = []
    for val in y:
        if val is None:
            text.append('')
        else:
            text.append(f'{val:.1f}%' if is_percent else f'{val:,.1f}{st.session_state.currency_unit[0]}')
    fig = go.Figure()
    fig.add_bar(x=x, y=y, text=text, textposition='outside', marker_color=color, hovertemplate='%{x}: %{y}<extra></extra>', cliponaxis=False)
    fig.update_layout(title=title + (f"<br><sub style='font-size:9px;color:{SECONDARY};'>Source: {source}</sub>" if source else ''), paper_bgcolor=CANVAS_BG, plot_bgcolor=PANEL_BG, font_color=PRIMARY, margin=dict(l=10, r=10, t=100, b=140), yaxis=dict(gridcolor=PANEL_BORDER, rangemode='tozero'), xaxis=dict(showgrid=False, tickfont=dict(color=SECONDARY), tickangle=-45, automargin=True, tickvals=x, ticktext=x), height=360, uniformtext_mode='hide', showlegend=False)
    return fig

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='sheet1')
    return output.getvalue()

tabs = st.tabs(['Home', 'Main Dashboard', 'Players', 'Comments & Sources', 'Taxonomy Explorer', 'Heatmap'])
tab_home, tab_main, tab_players, tab_comments, tab_taxonomy, tab_heatmap = tabs

with tab_home:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{PRIMARY}; margin-bottom:0.25rem;'>Home</h2>", unsafe_allow_html=True)
    st.markdown('### What this tool is')
    st.write('This dashboard lets you explore the Aerospace & Defence taxonomy and associated KPIs. Use the tabs to navigate (Main Dashboard, Players, Inspector, Taxonomy Explorer).')
    df_active = get_active_df()
    st.markdown('### Dataset summary')
    if df_active is None or df_active.empty:
        st.info('No data loaded. Upload a dataset in Settings or provide the baseline file.')
    else:
        try:
            st.write(f'- Rows: **{len(df_active):,}**')
            st.write(f'- Columns: **{len(df_active.columns):,}**')
            for lvl in HIER_LEVELS:
                if lvl in df_active.columns:
                    n = df_active[lvl].dropna().astype(str).str.strip().nunique()
                    st.write(f"- {lvl.split(' - ')[-1]} unique values: **{n:,}**")
        except Exception:
            st.write('_Dataset summary failed to compute._')
    st.markdown('### Data available')
    hierarchy_cols = ['Hierarchy - Main Category', 'Hierarchy - Sector', 'Hierarchy - Subsector', 'Hierarchy - Sub-Sub-Sector']
    tops = get_unique_in_family_order(df_active[hierarchy_cols[0]]) if hierarchy_cols[0] in df_active.columns else []
    row_index = 0
    top_color_map = build_top_level_color_map(df_active, hierarchy_cols[0]) if not df_active.empty and hierarchy_cols[0] in df_active.columns else {}
    cmap_01 = build_horizontal_color_map(df_active, hierarchy_cols[0], hierarchy_cols[1]) if len(hierarchy_cols) > 1 else {}
    cmap_12 = build_horizontal_color_map(df_active, hierarchy_cols[1], hierarchy_cols[2]) if len(hierarchy_cols) > 2 else {}
    cmap_23 = build_horizontal_color_map(df_active, hierarchy_cols[2], hierarchy_cols[3]) if len(hierarchy_cols) > 3 else {}
    prev_top = prev_sector = prev_subsector = None
    row_index = 0
    for top in tops:
        if not top:
            continue
        df_top = df_active[df_active[hierarchy_cols[0]].astype(str).str.strip().str.lower() == str(top).strip().lower()]
        top_color = top_color_map.get(top, '#888888')
        sectors = get_unique_in_family_order(df_top[hierarchy_cols[1]]) if len(hierarchy_cols) > 1 and hierarchy_cols[1] in df_top.columns else ['']
        if not sectors:
            sectors = ['']
        for sector in sectors:
            df_sector = df_top[df_top[hierarchy_cols[1]].astype(str).str.strip().str.lower() == str(sector).strip().lower()] if sector and hierarchy_cols[1] in df_top.columns else df_top
            sector_color = cmap_01.get((top, sector), top_color)
            subsectors = get_unique_in_family_order(df_sector[hierarchy_cols[2]]) if len(hierarchy_cols) > 2 and hierarchy_cols[2] in df_sector.columns else ['']
            if not subsectors:
                subsectors = ['']
            for subsector in subsectors:
                df_subsector = df_sector[df_sector[hierarchy_cols[2]].astype(str).str.strip().str.lower() == str(subsector).strip().lower()] if subsector and hierarchy_cols[2] in df_sector.columns else df_sector
                subsector_color = cmap_12.get((sector, subsector), sector_color)
                subsubs = get_unique_in_family_order(df_subsector[hierarchy_cols[3]]) if len(hierarchy_cols) > 3 and hierarchy_cols[3] in df_subsector.columns else ['']
                if not subsubs:
                    subsubs = ['']
                for subsub in subsubs:
                    nodes = [top or '', sector or '', subsector or '', subsub or '']
                    cols = st.columns(len(hierarchy_cols))
                    display_top = top if prev_top != top else ''
                    display_sector = sector if prev_sector != sector or display_top else ''
                    display_subsector = subsector if prev_subsector != subsector or display_sector else ''
                    for i, node in enumerate(nodes):
                        color = '#888888'
                        if i == 0:
                            color = top_color
                        elif i == 1:
                            color = sector_color
                        elif i == 2:
                            color = subsector_color
                        elif len(hierarchy_cols) > 3:
                            parent = nodes[i - 1]
                            color = cmap_23.get((parent, node), '#888888')
                        with cols[i]:
                            if i == 0:
                                disp = display_top
                            elif i == 1:
                                disp = display_sector
                            elif i == 2:
                                disp = display_subsector
                            else:
                                disp = node
                            if disp:
                                inner = st.columns([0.08, 0.92])
                                with inner[0]:
                                    st.markdown(f"<div style='width:18px;height:18px;background:{color};border-radius:4px;border:1px solid {PANEL_BORDER};margin-top:4px;'></div>", unsafe_allow_html=True)
                                with inner[1]:
                                    if st.button(disp, key=f'home_tax_{i}_{row_index}_{disp}'):
                                        st.session_state.sel_path[hierarchy_cols[i]] = disp
                                        for deeper in HIER_LEVELS[i + 1:]:
                                            st.session_state.sel_path[deeper] = 'All'
                                        st.session_state._rerun_trigger = st.session_state.get('_rerun_trigger', 0) + 1
                            else:
                                st.markdown(f"<div style='height:34px;background:transparent;border-radius:3px;'></div>", unsafe_allow_html=True)
                    prev_top = top
                    prev_sector = sector
                    prev_subsector = subsector
                    row_index += 1
    st.markdown('</div>', unsafe_allow_html=True)

with tab_main:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Financial Overview</div>', unsafe_allow_html=True)
    df_active = get_active_df()

    def build_tree_ui(df: pd.DataFrame, key_prefix: str='') -> Dict[str, Optional[str]]:
        st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
        path: Dict[str, Optional[str]] = {}
        l1_vals = ['All'] + (list(df[HIER_LEVELS[0]].dropna().astype(str).str.strip().unique()) if HIER_LEVELS[0] in df.columns else [])
        default_l1 = CFG['startup'].get('default_main_category', 'All')
        if default_l1 not in l1_vals:
            default_l1 = 'All'
        l1 = st.selectbox(HIER_LEVELS[0], l1_vals, index=l1_vals.index(default_l1), key=f'{key_prefix}l1')
        path[HIER_LEVELS[0]] = l1
        df_l2 = filter_by_path_progressive(df, {HIER_LEVELS[0]: l1})
        l2_vals = ['All'] + (list(df_l2[HIER_LEVELS[1]].dropna().astype(str).str.strip().unique()) if HIER_LEVELS[1] in df_l2.columns else [])
        l2 = st.selectbox(HIER_LEVELS[1], l2_vals, index=0, key=f'{key_prefix}l2')
        path[HIER_LEVELS[1]] = l2
        df_l3 = filter_by_path_progressive(df, {HIER_LEVELS[0]: l1, HIER_LEVELS[1]: l2})
        l3_vals = ['All'] + (list(df_l3[HIER_LEVELS[2]].dropna().astype(str).str.strip().unique()) if HIER_LEVELS[2] in df_l3.columns else [])
        l3 = st.selectbox(HIER_LEVELS[2], l3_vals, index=0, key=f'{key_prefix}l3')
        path[HIER_LEVELS[2]] = l3
        df_l4 = filter_by_path_progressive(df, {HIER_LEVELS[0]: l1, HIER_LEVELS[1]: l2, HIER_LEVELS[2]: l3})
        l4_vals = ['All'] + (list(df_l4[HIER_LEVELS[3]].dropna().astype(str).str.strip().unique()) if HIER_LEVELS[3] in df_l4.columns else [])
        l4 = st.selectbox(HIER_LEVELS[3], l4_vals, index=0, key=f'{key_prefix}l4')
        path[HIER_LEVELS[3]] = l4
        return path

    st.session_state.sel_path = build_tree_ui(df_active)
    breadcrumb_parts = []
    for lvl in HIER_LEVELS:
        val = st.session_state.sel_path.get(lvl, 'All')
        if val and val != 'All':
            breadcrumb_parts.append(val)
    breadcrumb = ' / '.join(breadcrumb_parts) if breadcrumb_parts else 'Top level (all)'
    st.markdown(f'**Breadcrumb:** {breadcrumb}')
    df_node = best_effort_selection(df_active, st.session_state.sel_path)
    if len(df_node) == 0:
        df_node = filter_by_path_progressive(df_active, st.session_state.sel_path)
    MET = CFG['metrics']
    rev = get_metric_values(df_node, MET.get('revenue', {}))
    ebt = get_metric_values(df_node, MET.get('ebitda', {}))
    player_rev = get_metric_values(df_node, MET.get('player_revenue', {}))
    player_ebt = get_metric_values(df_node, MET.get('player_ebitda', {}))
    mgn = {'fy23': calculate_ebitda_margin(rev.get('fy23'), ebt.get('fy23')), 'fy24': calculate_ebitda_margin(rev.get('fy24'), ebt.get('fy24')), 'fy25': calculate_ebitda_margin(rev.get('fy25'), ebt.get('fy25'))}
    player_mgn = {'fy25': calculate_ebitda_margin(player_rev.get('fy25'), player_ebt.get('fy25'))}
    kcols = st.columns([1, 1, 1, 1, 1, 1])
    with kcols[0]:
        v = convert_currency(rev.get('fy23'), 'Billions', st.session_state.currency_unit)
        suffix = f' {st.session_state.currency_unit[0]}' if v is not None else ''
        kpi_card('Revenue FY23', v, suffix)
        if st.button('Go to Players (revenue)', key='k_rev_players'):
            st.session_state.target_focus = 'players'
            st.success("Players tab selected as target — click the 'Players' tab to view the list.")
    with kcols[1]:
        v = convert_currency(ebt.get('fy23'), 'Billions', st.session_state.currency_unit)
        suffix = f' {st.session_state.currency_unit[0]}' if v is not None else ''
        kpi_card('EBITDA FY23', v, suffix)
        if st.button('Inspect rows (EBITDA)', key='k_ebt_inspect'):
            st.session_state.target_focus = 'inspector'
            st.success("Inspector tab selected as target — click the 'Inspector' tab to inspect supporting rows.")
    with kcols[2]:
        kpi_card('EBITDA Margin FY23', mgn.get('fy23'), '%', color=CFG['theme']['kpi_good'])
    with kcols[3]:
        v = convert_currency(player_rev.get('fy25'), 'Billions', st.session_state.currency_unit)
        suffix = f' {st.session_state.currency_unit[0]}' if v is not None else ''
        kpi_card('Player Revenue FY25', v, suffix)
        if st.button('Show Players (player rev)', key='k_plr_rev'):
            st.session_state.target_focus = 'players'
            st.success("Players tab selected as target — click the 'Players' tab to view the list.")
    with kcols[4]:
        v = convert_currency(player_ebt.get('fy25'), 'Billions', st.session_state.currency_unit)
        suffix = f' {st.session_state.currency_unit[0]}' if v is not None else ''
        kpi_card('Player EBITDA FY25', v, suffix)
    with kcols[5]:
        kpi_card('Player EBITDA Margin FY25', player_mgn.get('fy25'), '%', color=CFG['theme']['kpi_good'])
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        fig1 = plot_bars_with_labels('Revenue', {'fy23': rev.get('fy23'), 'fy24': rev.get('fy24'), 'fy25': rev.get('fy25')}, REV_COLOR, is_percent=False, source=CFG['data_sources']['revenue'])
        show_plotly(fig1)
    with c2:
        fig2 = plot_bars_with_labels('EBITDA', {'fy23': ebt.get('fy23'), 'fy24': ebt.get('fy24'), 'fy25': ebt.get('fy25')}, EBITDA_COLOR, is_percent=False, source=CFG['data_sources']['ebitda'])
        show_plotly(fig2)
    with c3:
        fig3 = plot_bars_with_labels('EBITDA Margin', {'fy23': mgn.get('fy23'), 'fy24': mgn.get('fy24'), 'fy25': mgn.get('fy25')}, MARGIN_COLOR, is_percent=True, source=CFG['data_sources']['ebitda_margin'])
        show_plotly(fig3)
    st.markdown('### Export current slice')
    try:
        if len(df_node) > 0:
            csv_bytes = df_node.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', csv_bytes, file_name='slice.csv', mime='text/csv')
            excel_bytes = df_to_excel_bytes(df_node)
            st.download_button('Download Excel', excel_bytes, file_name='slice.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        else:
            st.info('No rows available to export for the current selection.')
    except Exception as e:
        st.error('Export failed: ' + str(e))
    comment_col = CFG['comments_sources']['financial_comment_col']
    source_col = CFG['comments_sources']['financial_source_col']
    comments = [str(x) for x in df_node.get(comment_col, pd.Series(dtype=object)).dropna().unique() if str(x).strip() != '']
    sources = [str(x) for x in df_node.get(source_col, pd.Series(dtype=object)).dropna().unique() if str(x).strip() != '']
    if comments or sources:
        st.markdown('### Financial Commentary & Sources')
        col1, col2 = st.columns(2)
        with col1:
            if comments:
                st.markdown('**Financial Commentary**')
                for c in comments:
                    bullet_text = c.replace('\n', '<br>')
                    st.markdown(f'- {bullet_text}', unsafe_allow_html=True)
            else:
                st.markdown('_No financial commentary available._')
        with col2:
            if sources:
                st.markdown('**Financial Sources**')
                for s in sources:
                    bullet_text = s.replace('\n', '<br>')
                    st.markdown(f'- {bullet_text}', unsafe_allow_html=True)
            else:
                st.markdown('_No financial sources available._')
    current_path = st.session_state.sel_path.copy()
    levels = CFG['hierarchy']['levels']
    top_color_map = build_top_level_color_map(df_active, levels[0]) if not df_active.empty and levels[0] in df_active.columns else {}
    level_color_maps = []
    for i in range(1, len(levels)):
        parent_col = levels[i - 1]
        child_col = levels[i]
        level_color_maps.append(build_horizontal_color_map(df_active, parent_col, child_col))

    def get_deepest_selected_level(path, levels):
        for lvl in reversed(levels):
            if lvl in path and path[lvl] != 'All':
                return lvl
        return None

    deepest_level = get_deepest_selected_level(current_path, levels)
    if deepest_level is None:
        next_level = levels[0]
    else:
        idx = levels.index(deepest_level)
        next_level = levels[idx + 1] if idx + 1 < len(levels) else None
    if next_level:
        filter_path = {lvl: current_path.get(lvl, 'All') for lvl in levels[:levels.index(next_level)]}
        df_filtered = filter_by_path_progressive(df_active, filter_path)
        child_sectors = list(df_filtered[next_level].dropna().astype(str).str.strip().unique()) if next_level in df_filtered.columns else []
        if child_sectors:
            st.markdown(f"### Drill down: Select {next_level.split(' - ')[-1]}")
            cols_d = st.columns(min(len(child_sectors), 5))
            for i, sector in enumerate(child_sectors):
                color = '#888888'
                try:
                    if deepest_level is None:
                        parent_val = None
                    else:
                        parent_val = current_path.get(deepest_level, None)
                    if parent_val:
                        cmap_index = levels.index(next_level) - 1
                        cmap = level_color_maps[cmap_index] if cmap_index < len(level_color_maps) else {}
                        color = cmap.get((str(parent_val), str(sector)), top_color_map.get(parent_val, '#888888'))
                    elif levels.index(next_level) > 0:
                        parent_col = levels[levels.index(next_level) - 1]
                        subset = df_filtered[df_filtered[next_level].astype(str).str.strip().str.lower() == str(sector).strip().lower()]
                        if parent_col in subset.columns and len(subset[parent_col].dropna()) > 0:
                            parent_val = str(subset[parent_col].dropna().unique()[0])
                            cmap_index = levels.index(next_level) - 1
                            cmap = level_color_maps[cmap_index] if cmap_index < len(level_color_maps) else {}
                            color = cmap.get((parent_val, str(sector)), top_color_map.get(parent_val, '#888888'))
                except Exception:
                    color = '#888888'
                sw = cols_d[i % 5].columns([0.08, 0.92])
                with sw[0]:
                    st.markdown(f"<div style='width:18px;height:18px;background:{color};border-radius:4px;border:1px solid {PANEL_BORDER};'></div>", unsafe_allow_html=True)
                with sw[1]:
                    if st.button(sector, key=f'drill_{i}_{sector}'):
                        st.session_state.sel_path[next_level] = sector
                        for lvl in levels[levels.index(next_level) + 1:]:
                            st.session_state.sel_path[lvl] = 'All'
                        st.session_state._rerun_trigger += 1
    if all((v is None for v in [rev.get('fy23'), rev.get('fy24'), rev.get('fy25')])):
        st.info("No metrics found for current selection. To view top-level totals, set deeper levels to 'All'.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_players:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Players</div>', unsafe_allow_html=True)
    df_active = get_active_df()
    sel_path = st.session_state.sel_path
    df_sel = best_effort_selection(df_active, sel_path)
    if len(df_sel) == 0:
        df_sel = filter_by_path_progressive(df_active, sel_path)
    player_cols = list(zip(CFG['player_data']['names'], CFG['player_data']['countries'], CFG['player_data']['types']))
    players = []
    for _, r in df_sel.iterrows():
        for name_col, country_col, type_col in player_cols:
            name = str(r.get(name_col, '')).strip()
            if name and name.lower() != 'nan':
                players.append({'Name': name, 'Country': str(r.get(country_col, '')).strip(), 'Type': str(r.get(type_col, '')).strip()})
    if players:
        df_players = pd.DataFrame(players).drop_duplicates().reset_index(drop=True)
        cols_lower = [c.lower() for c in df_sel.columns]
        if 'fiscal metric flattened' in cols_lower and 'value' in cols_lower:
            metric_col = [c for c in df_sel.columns if c.strip().lower() == 'fiscal metric flattened'][0]
            value_col = [c for c in df_sel.columns if c.strip().lower() == 'value'][0]
        else:
            metric_col = None
            value_col = None
        name_cols = [c for c in CFG['player_data']['names'] if c in df_sel.columns]

        def player_metric_aggregate(player_name: str, metric_label: str) -> Optional[float]:
            if metric_col is None or value_col is None:
                return None
            mask = False
            for nc in name_cols:
                mask = mask | (df_sel[nc].astype(str).str.strip().str.lower() == str(player_name).strip().lower())
            subset = df_sel[mask]
            if subset.empty:
                return None
            metric_rows = subset[subset[metric_col].astype(str).str.strip().str.lower() == str(metric_label).strip().lower()]
            vals = pd.to_numeric(metric_rows[value_col], errors='coerce').dropna()
            if vals.empty:
                return None
            return float(vals.sum())

        metrics_rows = []
        player_revenue_label = CFG['metrics'].get('player_revenue', {}).get('fy25')
        player_ebitda_label = CFG['metrics'].get('player_ebitda', {}).get('fy25')
        for _, prow in df_players.iterrows():
            name = prow['Name']
            rev25 = player_metric_aggregate(name, player_revenue_label) if player_revenue_label else None
            ebt25 = player_metric_aggregate(name, player_ebitda_label) if player_ebitda_label else None
            mgn25 = calculate_ebitda_margin(rev25, ebt25) if rev25 is not None and ebt25 is not None else None
            metrics_rows.append({'Name': name, 'Revenue_FY25': rev25, 'EBITDA_FY25': ebt25, 'EBITDA_Margin_FY25': mgn25})
        df_player_metrics = pd.DataFrame(metrics_rows).set_index('Name')
        df_player_metrics['Revenue_FY25_display'] = df_player_metrics['Revenue_FY25'].apply(lambda v: convert_currency(v, 'Billions', st.session_state.currency_unit) if v is not None else None)
        df_player_metrics['EBITDA_FY25_display'] = df_player_metrics['EBITDA_FY25'].apply(lambda v: convert_currency(v, 'Billions', st.session_state.currency_unit) if v is not None else None)
        top_n = st.slider('Number of players to show in charts', 5, 50, 15)
        if df_player_metrics['Revenue_FY25_display'].dropna().any():
            df_rev_plot = df_player_metrics.dropna(subset=['Revenue_FY25_display']).sort_values('Revenue_FY25_display', ascending=True).tail(top_n)
            fig_rev = px.bar(df_rev_plot, x='Revenue_FY25_display', y=df_rev_plot.index, orientation='h', labels={'Revenue_FY25_display': f'Revenue FY25 ({st.session_state.currency_unit})', 'index': 'Player'}, color_discrete_sequence=[REV_COLOR])
            fig_rev.update_layout(paper_bgcolor=CANVAS_BG, plot_bgcolor=PANEL_BG, font_color=PRIMARY, margin=dict(l=140, r=10, t=40, b=40), height=420)
            st.markdown('#### Revenue FY25 (top players)')
            show_plotly(fig_rev)
        else:
            st.info('No Player Revenue FY25 values found for charting.')
        if df_player_metrics['EBITDA_FY25_display'].dropna().any():
            df_ebt_plot = df_player_metrics.dropna(subset=['EBITDA_FY25_display']).sort_values('EBITDA_FY25_display', ascending=True).tail(top_n)
            fig_ebt = px.bar(df_ebt_plot, x='EBITDA_FY25_display', y=df_ebt_plot.index, orientation='h', labels={'EBITDA_FY25_display': f'EBITDA FY25 ({st.session_state.currency_unit})', 'index': 'Player'}, color_discrete_sequence=[EBITDA_COLOR])
            fig_ebt.update_layout(paper_bgcolor=CANVAS_BG, plot_bgcolor=PANEL_BG, font_color=PRIMARY, margin=dict(l=140, r=10, t=40, b=40), height=420)
            st.markdown('#### EBITDA FY25 (top players)')
            show_plotly(fig_ebt)
        else:
            st.info('No Player EBITDA FY25 values found for charting.')
        if df_player_metrics['EBITDA_Margin_FY25'].dropna().any():
            df_mgn_plot = df_player_metrics.dropna(subset=['EBITDA_Margin_FY25']).sort_values('EBITDA_Margin_FY25', ascending=True).tail(top_n)
            fig_mgn = px.bar(df_mgn_plot, x='EBITDA_Margin_FY25', y=df_mgn_plot.index, orientation='h', labels={'EBITDA_Margin_FY25': 'EBITDA Margin FY25 (%)', 'index': 'Player'}, color_discrete_sequence=[MARGIN_COLOR])
            fig_mgn.update_layout(paper_bgcolor=CANVAS_BG, plot_bgcolor=PANEL_BG, font_color=PRIMARY, margin=dict(l=140, r=10, t=40, b=40), height=420)
            st.markdown('#### EBITDA Margin FY25 (top players)')
            show_plotly(fig_mgn)
        else:
            st.info('No Player EBITDA Margin FY25 values found for charting.')
        df_players_display = df_players.set_index('Name').join(df_player_metrics[['Revenue_FY25_display', 'EBITDA_FY25_display', 'EBITDA_Margin_FY25']], how='left').reset_index()
        df_players_display = df_players_display.rename(columns={'Revenue_FY25_display': f'Revenue FY25 ({st.session_state.currency_unit})', 'EBITDA_FY25_display': f'EBITDA FY25 ({st.session_state.currency_unit})', 'EBITDA_Margin_FY25': 'EBITDA Margin FY25 (%)'})
        st.dataframe(df_players_display, width='stretch', hide_index=True)
        csv_bytes = df_players_display.to_csv(index=False).encode('utf-8')
        st.download_button('Download Players CSV', csv_bytes, file_name='players.csv', mime='text/csv')
    else:
        st.info('No players found for current selection.')
    st.markdown('</div>', unsafe_allow_html=True)

with tab_comments:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comments & Sources</div>', unsafe_allow_html=True)
    df_active = get_active_df()
    df_node = best_effort_selection(df_active, st.session_state.sel_path)
    comment_col = CFG['comments_sources']['financial_comment_col']
    source_col = CFG['comments_sources']['financial_source_col']
    comments = [str(x) for x in df_node.get(comment_col, pd.Series(dtype=object)).dropna().unique() if str(x).strip() != '']
    sources = [str(x) for x in df_node.get(source_col, pd.Series(dtype=object)).dropna().unique() if str(x).strip() != '']
    st.markdown('#### Financial commentary')
    if comments:
        for c in comments:
            st.markdown(f"- {c.replace(chr(10), '<br>')}", unsafe_allow_html=True)
    else:
        st.markdown('_No financial commentary available for this slice._')
    st.markdown('#### Financial sources')
    if sources:
        for s in sources:
            st.markdown(f"- {s.replace(chr(10), '<br>')}", unsafe_allow_html=True)
    else:
        st.markdown('_No financial sources available for this slice._')
    st.markdown('</div>', unsafe_allow_html=True)

with tab_taxonomy:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Taxonomy Explorer</div>', unsafe_allow_html=True)

    def build_tree_nodes(df: pd.DataFrame, levels: List[str]):
        df_copy = df.copy().fillna('').astype(str)

        def recurse(sub_df: pd.DataFrame, level_idx: int, prefix: List[str]):
            if level_idx >= len(levels):
                return []
            level = levels[level_idx]
            vals = [v for v in sub_df[level].astype(str).str.strip().dropna().unique() if v != '']
            nodes = []
            for val in vals:
                path = prefix + [val]
                mask = sub_df[level].astype(str).str.strip().str.lower() == val.strip().lower()
                child_subset = sub_df[mask]
                children = recurse(child_subset, level_idx + 1, path)
                nodes.append({'label': val, 'value': '|||'.join(path), 'children': children})
            return nodes
        return recurse(df_copy, 0, [])

    df_active = get_active_df()
    if df_active.empty:
        st.info('No data available to build taxonomy. Upload a dataset in Settings or load the baseline file.')
    else:
        tree_data = build_tree_nodes(df_active, CFG['hierarchy']['levels'])
        st.markdown('Use the tree below to explore the taxonomy. Click a node to select it — the Main Dashboard will drill into that node.')

        def safe_tree_select(tree_data, key='taxonomy_tree', height=520, multi=False):
            if tree_select is None:
                st.warning('Tree widget not installed. Install `streamlit-tree-select` to enable interactive taxonomy tree.')
                return None
            try:
                sig = inspect.signature(tree_select)
                params = sig.parameters
                call_kwargs = {}
                call_kwargs['key'] = key
                if 'height' in params:
                    call_kwargs['height'] = height
                if multi:
                    if 'multi_select' in params:
                        call_kwargs['multi_select'] = True
                    elif 'multiple' in params:
                        call_kwargs['multiple'] = True
                    elif 'allow_multi_select' in params:
                        call_kwargs['allow_multi_select'] = True
                return tree_select(tree_data, **call_kwargs)
            except Exception:
                try:
                    return tree_select(tree_data)
                except Exception as err:
                    st.error('Taxonomy tree widget failed to load: ' + str(err))
                    return None

        selected = safe_tree_select(tree_data, key='taxonomy_tree', height=520, multi=False)
        sel_value = None
        if selected:
            if isinstance(selected, (list, tuple)) and len(selected) > 0:
                sel_value = selected[0]
            elif isinstance(selected, str):
                sel_value = selected
            elif isinstance(selected, dict) and 'value' in selected:
                sel_value = selected['value']
        if sel_value:
            parts = sel_value.split('|||')
            for i, lvl in enumerate(CFG['hierarchy']['levels']):
                if i < len(parts):
                    st.session_state.sel_path[lvl] = parts[i]
                else:
                    st.session_state.sel_path[lvl] = 'All'
            st.session_state._rerun_trigger = st.session_state.get('_rerun_trigger', 0) + 1

        if tree_select is None:
            try:
                top_color_map = build_top_level_color_map(df_active, CFG['hierarchy']['levels'][0]) if not df_active.empty else {}
                level_color_maps = []
                for i in range(1, len(CFG['hierarchy']['levels'])):
                    parent_col = CFG['hierarchy']['levels'][i - 1]
                    child_col = CFG['hierarchy']['levels'][i]
                    cmap = build_horizontal_color_map(df_active, parent_col, child_col)
                    level_color_maps.append(cmap)
                try:
                    from pyvis.network import Network
                    import tempfile
                    import os as _os
                    df_tax = df_active.copy()
                    for level in CFG['hierarchy']['levels']:
                        if level not in df_tax.columns:
                            df_tax[level] = 'Unknown'
                    df_paths = df_tax[CFG['hierarchy']['levels']].drop_duplicates()
                    nodes_meta = {}
                    edges = set()
                    levels_list = CFG['hierarchy']['levels']
                    for _, row in df_paths.iterrows():
                        path_vals = [str(row[l]).strip() if str(row[l]).strip() != '' else None for l in levels_list]
                        last_id = None
                        for i_lvl, val in enumerate(path_vals):
                            if not val:
                                break
                            node_id = f'{i_lvl}|||{val}'
                            if node_id not in nodes_meta:
                                if i_lvl == 0:
                                    color = top_color_map.get(val, '#888888')
                                else:
                                    parent = path_vals[i_lvl - 1]
                                    cmap = level_color_maps[i_lvl - 1] if i_lvl - 1 < len(level_color_maps) else {}
                                    color = cmap.get((parent, val), top_color_map.get(parent, '#888888') if parent else '#888888')
                                title = ' / '.join([p for p in path_vals[:i_lvl + 1] if p])
                                nodes_meta[node_id] = {'label': val, 'level': i_lvl, 'color': color, 'title': title}
                            if last_id is not None:
                                edges.add((last_id, node_id))
                            last_id = node_id
                    net = Network(height='650px', width='100%', bgcolor=CANVAS_BG, font_color=PRIMARY)
                    net.barnes_hut()
                    for nid, meta in nodes_meta.items():
                        lvl = meta['level']
                        size = max(12, 28 - lvl * 4)
                        net.add_node(nid, label=meta['label'], title=meta['title'], color=meta['color'], size=size)
                    for a, b in edges:
                        net.add_edge(a, b)
                    net.set_options('var options = { "nodes": {"font": {"size": 14}}, "physics": {"stabilization": {"iterations": 100}, "barnesHut": {"gravitationalConstant": -8000}} }')
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                    net.save_graph(tmp.name)
                    html = open(tmp.name, 'r', encoding='utf-8').read()
                    components.html(html, height=720, scrolling=True)
                    try:
                        _os.unlink(tmp.name)
                    except Exception:
                        pass
                except Exception:
                    df_tax = df_active.copy()
                    for level in CFG['hierarchy']['levels']:
                        if level not in df_tax.columns:
                            df_tax[level] = 'Unknown'
                    df_tax_unique = df_tax[CFG['hierarchy']['levels']].drop_duplicates()
                    fig = px.treemap(df_tax_unique, path=CFG['hierarchy']['levels'], color=CFG['hierarchy']['levels'][0], color_discrete_sequence=px.colors.qualitative.Pastel, title='Aerospace & Defence Taxonomy (Treemap fallback)')
                    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0), paper_bgcolor=CANVAS_BG, plot_bgcolor=CANVAS_BG, font_color=PRIMARY)
                    show_plotly(fig)
            except Exception as e:
                st.info('Could not build taxonomy visualization: ' + str(e))

with tab_heatmap:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Taxonomy × Metrics Heatmap</div>', unsafe_allow_html=True)
    st.write('Select taxonomy rows (one checkbox per node). Select metrics (columns). Click Generate to build a column-normalised heatmap.')
    df_active = get_active_df()
    levels = CFG['hierarchy']['levels']

    def build_node_paths(df: pd.DataFrame, levels: List[str]) -> List[str]:
        if df is None or df.empty:
            return []
        out = []
        seen = set()
        for _, r in df[levels].iterrows():
            parts = []
            for lvl in levels:
                val = str(r.get(lvl, '')).strip()
                if val != '' and val.lower() != 'nan':
                    parts.append(val)
                else:
                    break
            if not parts:
                continue
            path = ' / '.join(parts)
            if path not in seen:
                seen.add(path)
                out.append(path)
        return out

    node_paths = build_node_paths(df_active, levels)
    if not node_paths:
        st.info('No taxonomy nodes available (dataset empty or hierarchy columns missing).')
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        MAX_CHECKBOXES = 120
        if len(node_paths) > MAX_CHECKBOXES:
            st.warning(f'Found {len(node_paths)} taxonomy nodes. For performance, a compact multi-select is shown instead of many checkboxes.')
            compact_multi = True
        else:
            compact_multi = False
        st.markdown('#### Select columns (metrics)')
        metric_cols = []
        col1, col2, col3 = st.columns(3)
        with col1:
            rev23 = st.checkbox('Revenue FY23', key='hm_rev23', value=False)
            rev24 = st.checkbox('Revenue FY24', key='hm_rev24', value=False)
            rev25 = st.checkbox('Revenue FY25', key='hm_rev25', value=False)
        with col2:
            ebt23 = st.checkbox('EBITDA FY23', key='hm_ebt23', value=False)
            ebt24 = st.checkbox('EBITDA FY24', key='hm_ebt24', value=False)
            ebt25 = st.checkbox('EBITDA FY25', key='hm_ebt25', value=False)
        with col3:
            mgn23 = st.checkbox('EBITDA Margin FY23', key='hm_mgn23', value=False)
            mgn24 = st.checkbox('EBITDA Margin FY24', key='hm_mgn24', value=False)
            mgn25 = st.checkbox('EBITDA Margin FY25', key='hm_mgn25', value=False)
        if rev23:
            metric_cols.append(('Revenue FY23', ('revenue', 'fy23')))
        if rev24:
            metric_cols.append(('Revenue FY24', ('revenue', 'fy24')))
        if rev25:
            metric_cols.append(('Revenue FY25', ('revenue', 'fy25')))
        if ebt23:
            metric_cols.append(('EBITDA FY23', ('ebitda', 'fy23')))
        if ebt24:
            metric_cols.append(('EBITDA FY24', ('ebitda', 'fy24')))
        if ebt25:
            metric_cols.append(('EBITDA FY25', ('ebitda', 'fy25')))
        if mgn23:
            metric_cols.append(('EBITDA Margin FY23', ('margin', 'fy23')))
        if mgn24:
            metric_cols.append(('EBITDA Margin FY24', ('margin', 'fy24')))
        if mgn25:
            metric_cols.append(('EBITDA Margin FY25', ('margin', 'fy25')))
        if not metric_cols:
            st.info('No metrics selected. Choose one or more metric checkboxes to build the heatmap.')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            result_holder = {}
            with st.form(key='heatmap_form'):
                st.markdown('#### Select taxonomy rows to include')
                if compact_multi:
                    selected_nodes = st.multiselect('Select taxonomy rows (multi-select)', options=node_paths, default=[])
                else:
                    cols = st.columns(3)
                    checks = {}
                    for i, p in enumerate(node_paths):
                        with cols[i % 3]:
                            checks[p] = st.checkbox(p, key=f'hm_chk_{i}', value=False)
                    selected_nodes = [p for p, v in checks.items() if v]
                submitted = st.form_submit_button('Generate heatmap')
            if submitted:
                if not selected_nodes:
                    st.warning('No taxonomy rows selected — the heatmap will be empty. Select at least one row.')
                else:
                    MET = CFG.get('metrics', {})

                    def subset_for_path(df: pd.DataFrame, path_str: str, levels: List[str]) -> pd.DataFrame:
                        parts = [p.strip() for p in path_str.split(' / ') if p.strip() != '']
                        if not parts:
                            return pd.DataFrame()
                        mask = pd.Series([True] * len(df), index=df.index)
                        for i, p in enumerate(parts):
                            lvl = levels[i]
                            if lvl in df.columns:
                                mask = mask & (df[lvl].astype(str).str.strip().str.lower() == str(p).strip().lower())
                            else:
                                return pd.DataFrame()
                        return df[mask]
                    rows = []
                    for node in selected_nodes:
                        df_node = subset_for_path(df_active, node, levels)
                        row = {'__node': node}
                        rev_map = MET.get('revenue', {})
                        ebt_map = MET.get('ebitda', {})
                        rev_vals = get_metric_values(df_node, rev_map) if rev_map else {'fy23': None, 'fy24': None, 'fy25': None}
                        ebt_vals = get_metric_values(df_node, ebt_map) if ebt_map else {'fy23': None, 'fy24': None, 'fy25': None}
                        for fy in ('fy23', 'fy24', 'fy25'):
                            row[f'Revenue {fy.upper()}'] = rev_vals.get(fy)
                            row[f'EBITDA {fy.upper()}'] = ebt_vals.get(fy)
                            rv = rev_vals.get(fy)
                            eb = ebt_vals.get(fy)
                            row[f'EBITDA Margin {fy.upper()}'] = calculate_ebitda_margin(rv, eb) if rv is not None and eb is not None else None
                        rows.append(row)
                    df_heat_raw = pd.DataFrame(rows).set_index('__node')
                    final_cols = []
                    for label, key in metric_cols:
                        if key[0] == 'revenue':
                            final_cols.append(f'Revenue {key[1].upper()}')
                        elif key[0] == 'ebitda':
                            final_cols.append(f'EBITDA {key[1].upper()}')
                        elif key[0] == 'margin':
                            final_cols.append(f'EBITDA Margin {key[1].upper()}')
                    df_final = df_heat_raw.reindex(columns=final_cols)
                    result_holder['df_final'] = df_final
            if result_holder.get('df_final') is not None:
                df_final = result_holder['df_final']
                if df_final.isna().all(axis=None):
                    st.warning('No numeric data found for the selected rows/metrics. Check dataset or metric mappings.')
                    st.dataframe(df_final.fillna('—'), width='stretch')
                else:
                    z = df_final.copy().astype(float)
                    z_norm = z.copy()
                    for c in z.columns:
                        col = z[c]
                        if col.dropna().empty:
                            z_norm[c] = None
                        else:
                            mn = float(col.dropna().min())
                            mx = float(col.dropna().max())
                            if mn == mx:
                                z_norm[c] = col.apply(lambda v: 0.5 if pd.notna(v) else None)
                            else:
                                z_norm[c] = col.apply(lambda v: (float(v) - mn) / (mx - mn) if pd.notna(v) else None)

                    def fmt_val(v):
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return ''
                        if isinstance(v, float):
                            return f'{v:,.2f}'
                        return str(v)
                    annotations = z.apply(lambda col: col.map(fmt_val)).values.tolist()
                    z_plot = z_norm.values.tolist()
                    colorscale = 'Blues'
                    fig = go.Figure(data=go.Heatmap(z=z_plot, x=z.columns.tolist(), y=z.index.tolist(), text=annotations, hoverinfo='text', colorscale=colorscale, zmin=0, zmax=1, colorbar=dict(title='Relative (col)'), showscale=True))
                    text_annotations = []
                    for yi, row_idx in enumerate(z.index):
                        for xi, colname in enumerate(z.columns):
                            val_norm = z_norm.iloc[yi, xi]
                            raw_text = annotations[yi][xi]
                            if raw_text == '' or val_norm is None:
                                continue
                            text_color = 'white' if float(val_norm) > 0.55 else 'black'
                            text_annotations.append(dict(x=z.columns[xi], y=z.index.tolist()[yi], text=raw_text, showarrow=False, font=dict(color=text_color, size=11), xref='x', yref='y', xanchor='center', yanchor='middle'))
                    fig.update_layout(paper_bgcolor=CANVAS_BG, plot_bgcolor=PANEL_BG, font_color=PRIMARY, margin=dict(l=180 if len(z.index) > 15 else 120, r=20, t=60, b=80), height=max(400, 32 * len(z.index)), xaxis=dict(tickangle=-45))
                    fig.update_xaxes(tickmode='array', tickvals=z.columns.tolist(), ticktext=z.columns.tolist())
                    fig.update_yaxes(tickmode='array', tickvals=z.index.tolist(), ticktext=z.index.tolist(), autorange='reversed')
                    for a in text_annotations:
                        fig.add_annotation(a)
                    st.markdown('#### Heatmap (column-normalised shading)')
                    show_plotly(fig)
                    csv_bytes = df_final.reset_index().to_csv(index=False).encode('utf-8')
                    st.download_button('Download heatmap table (CSV)', csv_bytes, file_name='heatmap_table.csv', mime='text/csv')
                    st.markdown('**Notes:**\n- Colors are normalised per column (min→light, max→dark) so shading shows relative strength within each metric column.\n- Empty cells indicate missing data for that node/metric.\n- Values shown on cells are the raw aggregated values (formatted).')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    pass

if __name__ == '__main__':
    main()