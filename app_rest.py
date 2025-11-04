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
import plotly.io as pio
import re
try:
    from streamlit_tree_select import tree_select
except Exception:
    tree_select = None

DEFAULT_CONFIG = {
    'version': '1.0.0',
    'ui': {'quick_examples_count': 9, 'column_examples_count': 12},
    'theme': {
        'mode': 'dark',
        'canvas_bg': '#0B1220',
        'panel_bg': '#121A2A',
        'panel_border': '#1E2A44',
        'primary_text': '#E8EEF9',
        'secondary_text': '#A9B7D0',
        'accent': '#4DA3FF',
        'accent_alt': '#7AC8FF',
        'kpi_good': '#34D399',
        'kpi_neutral': '#FBBF24',
        'kpi_bad': '#F87171',
        'rev_color': '#4DA3FF',
        'ebitda_color': '#7AC8FF',
        'margin_color': '#34D399'
    },
    'fonts': {'title_size': 24, 'section_size': 18, 'label_size': 13},
    'startup': {'default_main_category': 'Aerospace'},
    'metrics': {
        'revenue': {'fy23': 'Financial_Revenue_FY23', 'fy24': 'Financial_Revenue_FY24', 'fy25': 'Financial_Revenue_FY25'},
        'ebitda': {'fy23': 'Financial_EBITDA_FY23', 'fy24': 'Financial_EBITDA_FY24', 'fy25': 'Financial_EBITDA_FY25'},
        'player_revenue': {'fy25': 'Player_Revenue_FY25'},
        'player_ebitda': {'fy25': 'Player_EBITDA_FY25'}
    },
    'hierarchy': {'levels': ['Hierarchy - Main Category', 'Hierarchy - Sector', 'Hierarchy - Subsector', 'Hierarchy - Sub-Sub-Sector']},
    'comments_sources': {
        'financial_comment_col': 'Financial Data - Financial Commentary',
        'financial_source_col': 'Financial Data - Financial Sources',
        'player_comment_col': 'Player data - Player Commentary',
        'player_source_col': 'Player data - Player Sources'
    },
    'player_data': {
        'names': ['Player data - Name', 'Player data - Name_1', 'Player data - Name_2', 'Player data - Name_3', 'Player data - Name_4'],
        'countries': ['Player data - Country', 'Player data - Country_1', 'Player data - Country_2', 'Player data - Country_3', 'Player data - Country_4'],
        'types': ['Player data - Type', 'Player data - Type_1', 'Player data - Type_2', 'Player data - Type_3', 'Player data - Type_4']
    },
    'data_sources': {'revenue': 'Company filings, Market research', 'ebitda': 'Company filings, Market research', 'ebitda_margin': 'Calculated from EBITDA/Revenue'}
}

CONFIG_PATH = 'config.json'


def show_plotly(fig, height: Optional[int] = None, config: Optional[dict] = None):
    """
    Render a Plotly figure via HTML to avoid deprecated Streamlit kwargs.
    - fig: plotly.graph_objects.Figure or similar
    - height: px or None
    - config: additional plotly config dict
    """
    cfg = {"displayModeBar": False, "responsive": True}
    if isinstance(config, dict):
        cfg.update(config)

    # Ensure transparent backgrounds so no white box shows up behind the chart
    try:
        # This will harmlessly noop if fig has no layout
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        # Make modebar background transparent on plotly.js where supported
        if "modebar" not in cfg:
            cfg.setdefault("displayModeBar", False)
    except Exception:
        pass

    # Prefer Streamlit's native Plotly renderer (allows width stretching via use_container_width)
    try:
        # Pass height through if provided — st.plotly_chart accepts height and will render responsively for width.
        if height:
            st.plotly_chart(fig, use_container_width=True, height=height, config=cfg)
        else:
            st.plotly_chart(fig, use_container_width=True, config=cfg)
        return
    except Exception:
        # Fallback to embedding HTML when native renderer fails in some hosting environments
        pass

    inferred_h = None
    try:
        h = getattr(fig.layout, "height", None)
        if h is not None:
            inferred_h = int(h)
    except Exception:
        inferred_h = None

    h = height or inferred_h or 600
    try:
        # Use to_html but force transparent styling and wrap in a transparent container.
        html_str = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config=cfg, default_height=h)
        wrapper = f"""
            <div style="background: transparent;">
                <style>
                    /* Force Plotly internals to be transparent where possible */
                    .js-plotly-plot .plot-container, .js-plotly-plot .svg-container, .plotly-graph-div {{
                        background: transparent !important;
                    }}
                    .modebar {{ background: rgba(0,0,0,0) !important; }}
                </style>
                {html_str}
            </div>
        """
        st.components.v1.html(wrapper, height=h, scrolling=True)
    except Exception:
        # Last-resort: write the figure object so the app doesn't hard-fail.
        try:
            st.write(fig)
        except Exception:
            # Swallow — we don't want the whole app to crash
            pass


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
        return {
            'CANVAS_BG': '#FAFAFB',
            'PANEL_BG': '#FFFF',
            'PANEL_BORDER': '#E6E9EE',
            'PRIMARY': '#0B1220',
            'SECONDARY': '#505766',
            'ACCENT': cfg_theme.get('accent', '#2563EB')
        }
    else:
        return {
            'CANVAS_BG': cfg_theme.get('canvas_bg', '#0B1220'),
            'PANEL_BG': cfg_theme.get('panel_bg', '#121A2A'),
            'PANEL_BORDER': cfg_theme.get('panel_border', '#1E2A44'),
            'PRIMARY': cfg_theme.get('primary_text', '#E8EEF9'),
            'SECONDARY': cfg_theme.get('secondary_text', '#A9B7D0'),
            'ACCENT': cfg_theme.get('accent', '#4DA3FF')
        }

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
    .panel p, .panel li, .panel span, .panel div.body-text {{ color: {SECONDARY}; font-size: 14px; line-height: 1.45; }}
    .section-title {{ font-size: {CFG['fonts']['section_size']}px; font-weight: 600; color: {PRIMARY}; margin-bottom: 8px; }}
    /* KPI bubble styling: full-width in column, vertically centred and consistent height */
    .kpi {{ 
        background: {PANEL_BG}; 
        border: 1px solid {PANEL_BORDER}; 
        border-radius: 10px; 
        padding: 14px; 
        box-sizing: border-box;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;   /* vertically center label/value within the KPI block */
        align-items: flex-start;   /* align text to the left to match column content */
        min-height: 72px;         /* consistent height so KPI blocks line up */
        margin-bottom: 8px;
    }}
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

# Hide Streamlit header/menu/footer
st.markdown(
    """
    <style>
    #MainMenu { visibility: hidden !important; }
    header { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }
    footer { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }
    .block-container { padding-top: 12px !important; }
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
st.session_state.setdefault('_last_nav_checked', [])

HIER_LEVELS: List[str] = CFG['hierarchy']['levels']
QUICK_EXAMPLES_COUNT = int(CFG.get('ui', {}).get('quick_examples_count', 9))
COLUMN_EXAMPLES_COUNT = int(CFG.get('ui', {}).get('column_examples_count', 12))
BASE_COLORS = {'aerospace': '#1f77b4', 'aero': '#1f77b4', 'defence': '#2ca02c', 'defense': '#2ca02c'}


def _build_xlsx_bytes(sheets: dict) -> bytes:
    """
    Build an in-memory XLSX workbook from a dict of sheet_name -> DataFrame.
    Only writes non-empty DataFrames. Returns bytes.
    """
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for name, df in sheets.items():
                if df is None:
                    continue
                safe_name = str(name)[:31]
                try:
                    df.to_excel(writer, sheet_name=safe_name, index=False)
                except Exception:
                    pd.DataFrame({"info": [f"Failed to write sheet {safe_name}"]}).to_excel(
                        writer, sheet_name=safe_name, index=False
                    )
        output.seek(0)
        return output.read()
    finally:
        try:
            output.close()
        except Exception:
            pass


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    if not hex_color:
        h = '8888'
    else:
        h = str(hex_color).lstrip('#').strip()
    if len(h) == 3:
        h = ''.join([c * 2 for c in h])
    if len(h) != 6:
        h = '8888'
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return (136 / 255.0, 136 / 255.0, 136 / 255.0)


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    try:
        r = int(max(0, min(1, rgb[0])) * 255)
        g = int(max(0, min(1, rgb[1])) * 255)
        b = int(max(0, min(1, rgb[2])) * 255)
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    except Exception:
        return '#8888'


def get_contrast_text(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#0000' if lum > 0.6 else '#ffff'


def generate_shades(base_hex: str, n: int, light_min: float = 0.35, light_max: float = 0.85) -> List[str]:
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
        return '#8888'
    key = str(val).strip().lower()
    return BASE_COLORS.get(key, None) or BASE_COLORS.get('aero')


def build_horizontal_color_map(df: pd.DataFrame, parent_col: str, child_col: str, top_level_values: Optional[List[str]] = None) -> Dict[Tuple[str, str], str]:
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


def safe_rerun():
    """
    Flexible helper to request a script rerun across different Streamlit versions.
    """
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass

    try:
        from streamlit.runtime.scriptrunner import RerunException
        raise RerunException()
    except Exception:
        pass

    try:
        import streamlit as _st
        if hasattr(_st, 'rerun'):
            _st.rerun()
        else:
            pass
    except Exception:
        pass

    st.session_state._rerun_trigger = st.session_state.get('_rerun_trigger', 0) + 1
    return


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


def rows_for_selected_node_exact(df: pd.DataFrame, sel_path: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Return rows that exactly correspond to the selected node.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    deepest = get_deepest_selected_level(sel_path, HIER_LEVELS)
    mask = pd.Series(True, index=df.index)

    for i, lvl in enumerate(HIER_LEVELS):
        if lvl not in df.columns:
            continue
        sel_val = sel_path.get(lvl, 'All')
        if sel_val and sel_val != 'All':
            mask = mask & (df[lvl].astype(str).str.strip().str.lower() == str(sel_val).strip().lower())
        else:
            if deepest:
                try:
                    if HIER_LEVELS.index(lvl) > HIER_LEVELS.index(deepest):
                        mask = mask & is_blank_or_nan(df[lvl])
                except Exception:
                    pass
    return df[mask].copy()


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


def kpi_card(label: str, value: Optional[float], suffix: str = '', color: Optional[str] = None):
    """Render a KPI card (HTML)."""
    color_style = f'color:{color};' if color else ''
    if value is None:
        st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value" style="{color_style}">—</div></div>', unsafe_allow_html=True)
    else:
        disp = f'{value:,.2f}{suffix}' if isinstance(value, (int, float)) else f'{value}{suffix}'
        st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value" style="{color_style}">{disp}</div></div>', unsafe_allow_html=True)


def plot_bars_with_labels(title: str, series_map: Dict[str, Optional[float]], color: str, is_percent: bool = False, source: str = '') -> go.Figure:
    x = ['FY23', 'FY24', 'FY25']
    y = [series_map.get('fy23'), series_map.get('fy24'), series_map.get('fy25')]
    if not is_percent:
        y = [convert_currency(v, 'Billions', st.session_state.get('currency_unit', 'Billions')) if v is not None else None for v in y]

    text = []
    unit_letter = (st.session_state.get('currency_unit', 'Billions')[0] if st.session_state.get('currency_unit') else 'B')
    for val in y:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            text.append('')
        else:
            if is_percent:
                # For margins/percents show one decimal place to match player table formatting (e.g. "9.3%")
                text.append(f'{val:.1f}%')
            else:
                text.append(f'{val:,.2f} {unit_letter}')

    fig = go.Figure()
    hovertemplate = '%{x}: %{text}<extra></extra>'
    fig.add_bar(x=x, y=y, text=text, textposition='outside', marker_color=color, hovertemplate=hovertemplate, cliponaxis=False)

    fig.update_layout(
        title=title + (f"<br><sub style='font-size:9px;color:{SECONDARY};'>Source: {source}</sub>" if source else ''),
        paper_bgcolor=CANVAS_BG,
        plot_bgcolor=PANEL_BG,
        font_color=PRIMARY,
        margin=dict(l=10, r=10, t=100, b=140),
        # Hide y-axis labels/tick marks to reduce visual clutter on the financial charts
        yaxis=dict(gridcolor=PANEL_BORDER, rangemode='tozero', showticklabels=False, ticks=''),
        xaxis=dict(showgrid=False, tickfont=dict(color=SECONDARY), tickangle=-45, automargin=True, tickvals=x, ticktext=x),
        height=360,
        uniformtext_mode='hide',
        showlegend=False
    )
    return fig


def render_horizontal_player_bars(title: str, df_plot: pd.DataFrame, x_col: str, y_col: str = 'Name', color: Optional[str] = None, x_label: Optional[str] = None, height: int = 360, category_order: Optional[List[str]] = None):
    """
    Build a horizontal bar chart for player-level metrics and render via show_plotly.
    """
    if df_plot is None or df_plot.empty:
        st.info("No data available to plot.")
        return

    if x_col not in df_plot.columns:
        st.error(f"Column '{x_col}' not found in dataframe. Available columns: {list(df_plot.columns)}")
        return

    chosen_y = y_col

    if chosen_y not in df_plot.columns:
        for alt in ('Name', 'Player'):
            if alt in df_plot.columns:
                chosen_y = alt
                break

    if chosen_y not in df_plot.columns and y_col:
        cols_lower = {c.lower(): c for c in df_plot.columns}
        if y_col.lower() in cols_lower:
            chosen_y = cols_lower[y_col.lower()]

    if chosen_y not in df_plot.columns:
        str_cols = [c for c in df_plot.columns if df_plot[c].dtype == object or df_plot[c].dtype.name == 'string']
        if str_cols:
            chosen_y = str_cols[0]
        else:
            st.error(f"Could not find a suitable y-column for plotting. Dataframe columns: {list(df_plot.columns)}")
            return

    try:
        # Work on a copy so we don't mutate caller's dataframe
        dfp = df_plot.copy()

        # Format labels to show to the right of each bar.
        # Decide whether to format as percent (margins) or numeric/currency.
        def _fmt_label(v):
            try:
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ''
                label_hint = (x_label or '').lower()
                # Determine percent vs absolute by checking label or column name
                if '%' in (x_label or '') or 'margin' in label_hint or 'margin' in str(x_col).lower():
                    # keep percent formatting (e.g. "9.32%")
                    return f'{float(v):.2f}%'
                else:
                    # numeric/currency — format with thousands separator and 2 decimals
                    # caller converts units already (e.g., Billions)
                    return f'{float(v):,.2f}'
            except Exception:
                return str(v)

        dfp['_label'] = dfp[x_col].apply(_fmt_label)

        labels = {x_col: (x_label or x_col), chosen_y: 'Player'}
        color_seq = [color] if color else None

        # Build the bar chart using the prepared label column
        fig = px.bar(
            dfp,
            x=x_col,
            y=chosen_y,
            orientation='h',
            labels=labels,
            color_discrete_sequence=color_seq,
            text='_label'
        )

        # show value text to the right of bars and allow it to overflow (not clipped)
        fig.update_traces(textposition='outside', cliponaxis=False)

        # compute left margin based on longest player name to reduce wasted space
        try:
            max_name_len = int(dfp[chosen_y].astype(str).map(len).max() or 0)
            # scale factor (pixels per char) — tuned to give reasonable left gutter
            left_margin = max(80, min(220, 8 + max_name_len * 7))
        except Exception:
            left_margin = 120

        # If caller supplied a desired category order (list of player names), reindex the dataframe
        # so all charts can share the same vertical ordering. Do this before building the y_order.
        try:
            if category_order:
                # reindex by the chosen_y column (player names). Missing players will become NaN rows
                # and will be placed at the end (na_position='last' equivalent behavior via reindex).
                dfp = dfp.set_index(chosen_y).reindex(category_order).reset_index()
        except Exception:
            # Fall back to original ordering if anything goes wrong
            pass

        try:
            y_order = dfp[chosen_y].astype(str).tolist()
        except Exception:
            y_order = []

        fig.update_layout(
            title=title,
            paper_bgcolor=CANVAS_BG,
            plot_bgcolor=PANEL_BG,
            font_color=PRIMARY,
            margin=dict(l=left_margin, r=64, t=40, b=110),
            height=height,
            showlegend=False
        )

        # enforce category order and show largest at top by reversing autorange
        if y_order:
            fig.update_yaxes(categoryorder='array', categoryarray=y_order, autorange='reversed', automargin=True)
        else:
            fig.update_yaxes(title='', automargin=True)

        fig.update_xaxes(title=(x_label or x_col), automargin=True)

        show_plotly(fig, height=height)
    except Exception as e:
        st.error("Failed to build player bar chart:")
        st.exception(e)


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='sheet1')
    return output.getvalue()


def get_deepest_selected_level(path: Dict[str, Optional[str]], levels: List[str]) -> Optional[str]:
    """Return the deepest selected hierarchy level (i.e. last non-'All')."""
    for lvl in reversed(levels):
        if lvl in path and path[lvl] != 'All':
            return lvl
    return None


def _build_first_child_path(df: pd.DataFrame, sel_path: Dict[str, Optional[str]]) -> Optional[Dict[str, str]]:
    """
    Return a selection path dict that corresponds to the FIRST child value one level below the current selection.
    Example: sel_path = {Level0: 'A', Level1: 'All', Level2: 'All'}. If Level1 has children under A,
    return a path where Level1 == first_child and deeper levels == 'All'.
    Returns None if no child level/value exists.
    """
    if df is None or df.empty or not sel_path:
        return None
    try:
        deepest = get_deepest_selected_level(sel_path, HIER_LEVELS)
        if deepest is None:
            # nothing selected; cannot find child
            return None
        # index of deepest selected
        try:
            idx = HIER_LEVELS.index(deepest)
        except Exception:
            return None
        child_idx = idx + 1
        if child_idx >= len(HIER_LEVELS):
            # no lower level exists
            return None
        child_level = HIER_LEVELS[child_idx]

        # parent value (the deepest selected)
        parent_val = sel_path.get(deepest)
        if not parent_val:
            return None

        # build mask for rows belonging to the selected parent (case-insensitive)
        if deepest not in df.columns or child_level not in df.columns:
            return None
        try:
            mask = df[deepest].astype(str).str.strip().str.lower() == str(parent_val).strip().lower()
            child_vals = df.loc[mask, child_level].dropna().astype(str).str.strip().unique().tolist()
        except Exception:
            child_vals = []

        if not child_vals:
            return None

        first_child = child_vals[0]
        # build a new path where levels up-to child_idx are set (parent levels preserved, child set, deeper = 'All')
        new_path: Dict[str, str] = {}
        for i, lvl in enumerate(HIER_LEVELS):
            if i < child_idx:
                # preserve existing selection (or default to 'All' if missing)
                new_path[lvl] = sel_path.get(lvl, 'All')
            elif i == child_idx:
                new_path[lvl] = first_child
            else:
                new_path[lvl] = 'All'
        return new_path
    except Exception:
        return None


# Heatmap - all (helpers + UI) -- improved and robust version
import numpy as _np
import pandas as _pd
import plotly.graph_objects as _go
import streamlit as _st

def _collect_hierarchy_cols(df):
    """
    Prefer the configured hierarchy order (CFG['hierarchy']['levels']) when present in df.
    Fall back to any columns that start with 'Hierarchy -' preserving dataframe order.
    """
    if df is None or df.empty:
        return []
    # prefer configured levels (keeps expected left->right ordering)
    cfg_levels = [lvl for lvl in CFG.get('hierarchy', {}).get('levels', []) if lvl in df.columns]
    if cfg_levels:
        return cfg_levels
    # fallback: any column that starts with "Hierarchy -", in df column order
    return [c for c in df.columns if isinstance(c, str) and c.startswith("Hierarchy -")]

def _find_metric_and_value_cols(df):
    """Return (metric_col, value_col) found case-insensitively in df, or (None, None)."""
    if df is None or df.empty:
        return None, None
    cols = list(df.columns)
    metric_col = next((c for c in cols if isinstance(c, str) and c.strip().lower() == 'fiscal metric flattened'), None)
    value_col = next((c for c in cols if isinstance(c, str) and c.strip().lower() == 'value'), None)
    return metric_col, value_col

def build_node_metric_map(df, metric_name, value_col=None):
    """
    Build a mapping from hierarchy-path (string) to numeric metric value for metric_name.
    More robust detection of metric/value columns and hierarchy order.
    Returns: (metric_map: dict path -> value, hierarchy_cols: list, matched_count: int)
    """
    if df is None or df.empty:
        return {}, [], 0

    metric_col, detected_value_col = _find_metric_and_value_cols(df)
    if metric_col is None:
        # cannot operate without metric column
        return {}, _collect_hierarchy_cols(df), 0
    value_col = value_col or detected_value_col

    # Try to match metric rows robustly:
    # 1) prefer exact (case-insensitive) equality of the metric string,
    # 2) fallback to substring contains if no exact matches were found.
    mname = '' if metric_name is None else str(metric_name).strip()
    metric_col_series = df[metric_col].astype(str)
    mask_exact = metric_col_series.str.strip().str.lower() == mname.lower()
    if mask_exact.any():
        df_metric = df.loc[mask_exact].copy()
        used_fallback = False
    else:
        # fallback: substring match (case-insensitive) — kept for backwards compatibility
        mask_sub = metric_col_series.str.contains(mname, case=False, na=False)
        df_metric = df.loc[mask_sub].copy()
        used_fallback = True if not mask_exact.any() and mask_sub.any() else False

    if df_metric.empty:
        return {}, _collect_hierarchy_cols(df), 0

    # Ensure numeric values (if value column missing, try to detect a numeric-like column)
    if value_col is None:
        # try to locate first numeric column apart from hierarchy/metric cols
        candidates = [c for c in df_metric.columns if c not in _collect_hierarchy_cols(df) and c != metric_col and df_metric[c].dtype.kind in 'fi']
        value_col = candidates[0] if candidates else None
    if value_col is None:
        return {}, _collect_hierarchy_cols(df), int(len(df_metric))

    df_metric[value_col] = _pd.to_numeric(df_metric[value_col], errors="coerce")

    hierarchy_cols = _collect_hierarchy_cols(df)
    records = []
    for _, row in df_metric.iterrows():
        path_parts = []
        for hc in hierarchy_cols:
            val = row.get(hc)
            if val is not None and str(val).strip() not in ["", "nan", "none"]:
                path_parts.append(str(val).strip())
            else:
                break
        if not path_parts:
            # skip rows that do not map to any hierarchy node
            continue
        node_path = " / ".join(path_parts)
        records.append({"node_path": node_path, "value": row[value_col]})

    nodes_df = _pd.DataFrame.from_records(records)
    if nodes_df.empty:
        return {}, hierarchy_cols, int(len(df_metric))

    # Aggregate duplicates (multiple rows mapping to same node) — use mean and drop NaNs
    agg = nodes_df.dropna(subset=["value"]).groupby("node_path", as_index=False).value.mean()
    metric_map = dict(zip(agg.node_path, agg.value))
    return metric_map, hierarchy_cols, int(len(df_metric))

def build_heatmap_matrix_from_paths(df, metric_name="EBITDA margin 2025", value_col=None):
    """
    Build matrix where rows are full-paths and columns are hierarchy levels.
    Entry [i,j] is the metric value for the ancestor at level j of row i's path (NaN if missing).
    """
    metric_map, hierarchy_cols, matched_count = build_node_metric_map(df, metric_name, value_col=value_col)

    if not hierarchy_cols:
        return _np.empty((0,0)), [], [], matched_count

    def get_full_path(row):
        parts = []
        for hc in hierarchy_cols:
            v = row.get(hc)
            if v is None or str(v).strip().lower() in ["", "nan", "none"]:
                break
            parts.append(str(v).strip())
        return " / ".join(parts)

    full_paths_series = df.apply(get_full_path, axis=1)
    full_paths = [_p for _p in pd.Series(full_paths_series).unique() if _p and _p.strip() != ""]
    if not full_paths:
        return _np.empty((0, len(hierarchy_cols))), [], hierarchy_cols, matched_count

    rows = []
    for path in full_paths:
        parts = path.split(" / ")
        row_vals = []
        for level_idx in range(len(hierarchy_cols)):
            if level_idx < len(parts):
                ancestor_path = " / ".join(parts[: level_idx + 1])
                row_vals.append(metric_map.get(ancestor_path, _np.nan))
            else:
                row_vals.append(_np.nan)
        rows.append(row_vals)

    matrix = _np.array(rows, dtype=float)
    return matrix, full_paths, hierarchy_cols, matched_count

def render_heatmap_figure(matrix, row_labels, col_labels, metric_name="Metric", colorscale="RdYlGn"):
    """Render heatmap; uses global min/max across matrix so shading is comparable across levels."""
    fig = _go.Figure()
    if matrix.size == 0 or not _np.isfinite(matrix).any():
        fig.update_layout(title=f"No numeric data available for metric: {metric_name}")
        return fig

    z = matrix
    finite = _np.isfinite(z)
    zmin = float(_np.nanmin(_np.where(finite, z, _np.nan)))
    zmax = float(_np.nanmax(_np.where(finite, z, _np.nan)))

    hover_text = []
    for r_idx, rlabel in enumerate(row_labels):
        row_hover = []
        for c_idx, clabel in enumerate(col_labels):
            val = z[r_idx, c_idx]
            if _np.isfinite(val):
                txt = f"{rlabel} — {clabel}: {val:,.2f}"
            else:
                txt = f"{rlabel} — {clabel}: n/a"
            row_hover.append(txt)
        hover_text.append(row_hover)

    fig.add_trace(_go.Heatmap(
        z=z,
        x=col_labels,
        y=row_labels,
        hoverinfo="text",
        text=hover_text,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=metric_name),
    ))

    fig.update_layout(
        autosize=True,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, r=40, t=60, b=80),
        title=f"Heatmap - all (metric: {metric_name})",
    )
    return fig

def build_matrix_from_metric_map(metric_map, df):
    """
    Build a matrix identical in shape to build_heatmap_matrix_from_paths but using
    a precomputed metric_map: { "A / B / C": value }.
    Returns: matrix, full_paths, hierarchy_cols, matched_count
    """
    if df is None or df.empty:
        return _np.empty((0,0)), [], [], 0

    hierarchy_cols = _collect_hierarchy_cols(df)
    if not hierarchy_cols:
        return _np.empty((0,0)), [], hierarchy_cols, 0

    def get_full_path(row):
        parts = []
        for hc in hierarchy_cols:
            v = row.get(hc)
            if v is None or str(v).strip().lower() in ["", "nan", "none"]:
                break
            parts.append(str(v).strip())
        return " / ".join(parts)

    full_paths_series = df.apply(get_full_path, axis=1)
    full_paths = [_p for _p in _pd.Series(full_paths_series).unique() if _p and _p.strip() != ""]
    if not full_paths:
        return _np.empty((0, len(hierarchy_cols))), [], hierarchy_cols, int(len(metric_map))

    rows = []
    for path in full_paths:
        parts = path.split(" / ")
        row_vals = []
        for level_idx in range(len(hierarchy_cols)):
            if level_idx < len(parts):
                ancestor_path = " / ".join(parts[: level_idx + 1])
                row_vals.append(metric_map.get(ancestor_path, _np.nan))
            else:
                row_vals.append(_np.nan)
        rows.append(row_vals)

    matrix = _np.array(rows, dtype=float)
    return matrix, full_paths, hierarchy_cols, int(len(metric_map))

def heatmap_all_tab_ui(df, default_metric="EBITDA Margin FY25", value_col=None):
    """
    Streamlit UI for the 'Heatmap - all' tab. Metric choices are read from the dataset to avoid
    mismatches between hard-coded names and what's actually present in the 'Fiscal Metric Flattened' column.
    """
    st = _st
    st.header("Heatmap - all")

    metric_col, detected_value_col = _find_metric_and_value_cols(df)
    if metric_col is None:
        st.warning("Dataset does not contain a 'Fiscal Metric Flattened' column (case-insensitive). Heatmap cannot be built.")
        return

    # Build list of metric options from the dataframe values (non-empty, deduped)
    metric_options_df = [str(x).strip() for x in pd.Series(df[metric_col].astype(str)).unique()
                         if str(x).strip() and str(x).strip().lower() not in ['nan', 'none']]
    if not metric_options_df:
        st.warning("No metric values found in 'Fiscal Metric Flattened' column.")
        return

    # Present a fixed set of user-facing choices (per request).
    desired_choices = [
        "EBITDA Margin FY23",
        "EBITDA Margin FY24",
        "EBITDA Margin FY25",
        "EBITDA CAGR 23-25",
        "Revenue CAGR 23-25",
    ]

    # Normalization helper for cast-iron matching:
    # Lowercase, remove punctuation, collapse whitespace, normalise 'fy' tokens.
    def _normalize_label(s: Optional[str]) -> str:
        if s is None:
            return ''
        s2 = str(s).lower()
        # Replace 'fy 25' or 'fy-25' -> 'fy25'
        s2 = re.sub(r'fy[\s\-]*([0-9]{2,4})', r'fy\1', s2)
        # Remove any non-alphanumeric characters except spaces
        s2 = re.sub(r'[^a-z0-9\s]', ' ', s2)
        # Collapse whitespace
        s2 = re.sub(r'\s+', ' ', s2).strip()
        return s2

    # Build map normalized_option -> original_option (first occurrence wins)
    norm_to_option = {}
    for opt in metric_options_df:
        n = _normalize_label(opt)
        if n and n not in norm_to_option:
            norm_to_option[n] = opt

    # Looser fallback matching (token presence) retained but used only if normalized exact fails
    def _loose_match(desired_label: str, options: List[str]) -> Optional[str]:
        toks = [t.strip() for t in re.split(r'[\s\-/]+', desired_label) if t.strip()]
        toks = [t for t in toks if t and len(t) > 0]
        for opt in options:
            low = opt.lower()
            ok = True
            for t in toks:
                if t.isdigit():
                    if f"fy{t}" not in low and t not in low:
                        ok = False
                        break
                else:
                    if t.lower() not in low:
                        ok = False
                        break
            if ok:
                return opt
        # final fallback: any option that contains any token
        for opt in options:
            low = opt.lower()
            if any((t.isdigit() and (f"fy{t}" in low or t in low)) or (not t.isdigit() and t.lower() in low) for t in toks):
                return opt
        return None

    # Present only the desired labels as choices (user UI requirement)
    default_choice = default_metric if default_metric in desired_choices else "EBITDA Margin FY25"
    if default_choice not in desired_choices:
        default_choice = desired_choices[0]
    metric_choice = st.selectbox("Determining factor (metric)", options=desired_choices,
                                 index=desired_choices.index(default_choice),
                                 help="Choose metric used to color the heatmap.")

    # CAST-IRON matching: normalized exact match first, then loose fallback
    norm_desired = _normalize_label(metric_choice)
    chosen_dataset_metric = None
    if norm_desired and norm_desired in norm_to_option:
        chosen_dataset_metric = norm_to_option[norm_desired]
        matched_by = "exact (normalized)"
    else:
        # try looser matching
        loose = _loose_match(metric_choice, metric_options_df)
        if loose:
            chosen_dataset_metric = loose
            matched_by = "loose token match"
        else:
            chosen_dataset_metric = None
            matched_by = None

    # Show user what dataset metric was matched (helpful for debugging)
    if chosen_dataset_metric:
        st.caption(f"Using dataset metric: '{chosen_dataset_metric}' (matched by: {matched_by})")
    else:
        st.warning(f"No dataset metric could be matched for the chosen label '{metric_choice}'. The heatmap will likely be empty.")

    with st.spinner("Building matrix..."):
        # If user requested a margin metric, compute margins exactly as the Dashboard:
        # use CFG['metrics'] to find revenue & ebitda rows for the requested FY and compute margin.
        if 'margin' in (metric_choice or '').lower():
            # detect FY token from the chosen label (default to fy25)
            m = re.search(r'fy[\s\-]*([0-9]{2,4})', metric_choice, flags=re.I)
            fy_token = f'fy{m.group(1)}' if m else 'fy25'

            hierarchy_cols = _collect_hierarchy_cols(df)

            def _get_full_path(row):
                parts = []
                for hc in hierarchy_cols:
                    v = row.get(hc)
                    if v is None or str(v).strip().lower() in ["", "nan", "none"]:
                        break
                    parts.append(str(v).strip())
                return " / ".join(parts)

            full_paths_series = df.apply(_get_full_path, axis=1)
            full_paths = [_p for _p in _pd.Series(full_paths_series).unique() if _p and _p.strip() != ""]

            metric_map = {}
            # For each path, subset the dataframe and compute margin from config-driven revenue/ebitda lookups
            for path in full_paths:
                parts = [p.strip() for p in path.split(' / ') if p.strip() != '']
                if not parts:
                    continue
                # build mask to select rows that match this path (match provided parts only)
                mask = _pd.Series(True, index=df.index)
                for i, part in enumerate(parts):
                    lvl = hierarchy_cols[i] if i < len(hierarchy_cols) else None
                    if lvl and lvl in df.columns:
                        mask = mask & (df[lvl].astype(str).str.strip().str.lower() == str(part).strip().lower())
                df_node = df.loc[mask].copy() if not mask.empty else _pd.DataFrame()

                # lookup revenue & ebitda using the same CFG-driven labels used by Dashboard
                rev_map_cfg = CFG.get('metrics', {}).get('revenue', {})
                ebt_map_cfg = CFG.get('metrics', {}).get('ebitda', {})
                rev_vals = get_metric_values(df_node, rev_map_cfg) if rev_map_cfg else {'fy23': None, 'fy24': None, 'fy25': None}
                ebt_vals = get_metric_values(df_node, ebt_map_cfg) if ebt_map_cfg else {'fy23': None, 'fy24': None, 'fy25': None}

                rev_v = rev_vals.get(fy_token)
                ebt_v = ebt_vals.get(fy_token)
                try:
                    val = calculate_ebitda_margin(rev_v, ebt_v) if (rev_v is not None and ebt_v is not None) else None
                except Exception:
                    val = None
                metric_map[path] = val

            # Build matrix from computed metric_map
            matrix, row_labels, col_labels, matched_count = build_matrix_from_metric_map(metric_map, df)
        else:
            matrix, row_labels, col_labels, matched_count = build_heatmap_matrix_from_paths(
                df,
                metric_name=chosen_dataset_metric if chosen_dataset_metric is not None else metric_choice,
                value_col=value_col or detected_value_col
            )

    # Show how many dataset rows matched the metric search (exact or fallback)
    try:
        st.caption(f"Dataset rows matched for metric search: {matched_count}")
    except Exception:
        pass

    if matrix.size == 0 or not _np.isfinite(matrix).any():
        st.warning(f"No numeric data found for metric '{metric_choice}'. Verify mappings and that a numeric 'Value' column exists (matched rows: {matched_count}).")
        return

    colorscale = st.selectbox("Color scale", options=["RdYlGn", "Viridis", "Blues", "RdBu"], index=0)
    fig = render_heatmap_figure(matrix, row_labels, col_labels, metric_name=chosen_dataset_metric or metric_choice, colorscale=colorscale)

    try:
        # Sensible default so heatmap isn't enormous by default.
        default_height = 600
        if row_labels:
            try:
                # Compact scaling: 20px per row, clamped to [400, 900]
                scaled = int(20 * len(row_labels))
                default_height = min(900, max(400, scaled))
            except Exception:
                default_height = 600
    except Exception:
        default_height = 600

    resize_enabled = st.checkbox(
        "Enable vertical resize for heatmap area (click & drag)",
        value=True,
        help="Allow adjusting the visible height of the heatmap by dragging the bottom edge."
    )

    if resize_enabled:
        # Let the user pick the initial height (px) quickly, then they can drag to fine-tune.
        initial_height = st.slider("Initial heatmap height (px)", min_value=300, max_value=1200, value=default_height, step=50)
        try:
            # Ensure the figure has transparent background for embedding (avoid white card in iframe)
            try:
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            except Exception:
                pass
            plot_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False, "responsive": True})
            # Constrain vertical resize: min-height and max-height (90vh) to avoid absurd tall sizes.
            # Make wrapper background transparent so the embedded plot doesn't get a white backdrop.
            wrapper_html = f"""
            <div style="resize: vertical; overflow: auto; border:1px solid {PANEL_BORDER}; border-radius:6px; padding:6px;
            height:{initial_height}px; min-height:200px; max-height:90vh; background: transparent;">
            {plot_html}
            </div>
            """
            # Set the components iframe a bit taller than the initial inner height so the handle is reachable.
            outer_iframe_h = min(1400, initial_height + 120)
            components.html(wrapper_html, height=outer_iframe_h, scrolling=True)
        except Exception:
            # Fallback: Streamlit's native renderer
            try:
                # Also try to render with transparent layout when falling back
                try:
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # Last resort: write the figure object so the app doesn't hard-fail.
                try:
                    st.write(fig)
                except Exception:
                    pass
    else:
        st.plotly_chart(fig, use_container_width=True)

    # Provide CSV download of the matrix (with labels)
    mat_df = _pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    csv_bytes = mat_df.to_csv(index=True).encode('utf-8')
    st.download_button("Download matrix CSV", data=csv_bytes, file_name="heatmap_all_matrix.csv", mime="text/csv")

# ---- UI: tabs ----
tabs = st.tabs(['Home', 'Dashboard', 'Heatmap', 'Heatmap - all'])
tab_home, tab_dashboard, tab_heatmap, tab_heatmap_all = tabs

# ---- Home tab (refreshed layout & styling) ----
with tab_home:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{PRIMARY}; margin-bottom:0.25rem;'>Home</h2>", unsafe_allow_html=True)

    # --- dataset + KPI prep (robust to empty uploads) ---
    df_active = get_active_df()
    rows_count = len(df_active) if (df_active is not None and not df_active.empty) else 0
    cols_count = len(df_active.columns) if (df_active is not None and not df_active.empty) else 0

    # unique taxonomy nodes (concatenate present hierarchy columns per row)
    hier_cols_present = [c for c in HIER_LEVELS if (df_active is not None and c in df_active.columns)]
    unique_nodes = 0
    try:
        if df_active is not None and not df_active.empty and hier_cols_present:
            node_series = df_active[hier_cols_present].fillna('').astype(str).apply(
                lambda r: ' / '.join([p.strip() for p in r.tolist() if p and str(p).strip().lower() != 'nan']), axis=1
            )
            unique_nodes = int(pd.Series(node_series[node_series.str.strip() != ''].unique()).shape[0])
    except Exception:
        unique_nodes = 0

    # unique players
    unique_players = 0
    try:
        name_cols = [c for c in CFG.get('player_data', {}).get('names', []) if (df_active is not None and c in df_active.columns)]
        if df_active is not None and not df_active.empty and name_cols:
            all_players = []
            for c in name_cols:
                vals = df_active[c].dropna().astype(str).str.strip().tolist()
                all_players.extend([v for v in vals if v and v.lower() != 'nan'])
            unique_players = len(set(all_players))
    except Exception:
        unique_players = 0

    # KPI row — use the same column width ratios as the three main columns below
    # so KPI cards align directly above their corresponding content columns.
    k1, k2, k3 = st.columns([1, 1.25, 1], gap="large")
    with k1:
        kpi_card('Rows', rows_count, suffix='', color=PRIMARY)
    with k2:
        kpi_card('Unique taxonomy nodes', unique_nodes, suffix='', color=ACCENT)
    with k3:
        kpi_card('Unique players', unique_players, suffix='', color=SECONDARY)

    # --- Three-column Home content: Overview, About dataset, How to use ---
    col1, col2, col3 = st.columns([1, 1.25, 1], gap="large")

    # Column 1: Overview & insights (clean typography; avoid harsh colours)
    with col1:
        st.markdown("<div class='section-title'>Introduction & approach</div>", unsafe_allow_html=True)
        st.markdown("<div class='body-text'>"
                    "Structured, taxonomy-driven analysis using a multi-tier hierarchy (Total → Main Category → Sector → Subsector → Sub-Sub-Sector). "
                    "Nodes are mutually exclusive slices of commercial recurring revenue (industry contract revenue only). "
                    "Internal government budgets, asset values and one‑off program valuations are excluded to provide a transparent 'cash to industry' view."
                    "</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Data foundation & methods (summary)</div>", unsafe_allow_html=True)
        st.markdown("<div class='body-text'>"
                    "- Source mix: public company filings, contract award databases and multi‑year market research; cross‑checked top‑down vs bottom‑up.<br>"
                    "- Mapping: segments are cross‑walked to the taxonomy and validated against peer sums and contract flows.<br>"
                    "- Objective: anchor estimates to recurring contract revenue rather than fleet or asset valuations."
                    "</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Key financial findings (high level)</div>", unsafe_allow_html=True)
        st.markdown("<div class='body-text'>"
                    "- Aerospace: Platforms (large share), Propulsion, Avionics and Aftermarket remain material; avionics show elevated margins where proprietary software exists.<br>"
                    "- Defence: Missiles, weapons and naval platforms are sizeable with differentiated margin profiles; C4ISR is large and growing with a strong software share.<br>"
                    "- Support & Services: Training, Logistics & Base and managed services show attractive recurring revenue and margin expansion potential."
                    "</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Strategic takeaways</div>", unsafe_allow_html=True)
        st.markdown("<div class='body-text'>"
                    "- Prioritise recurring contract revenue to reveal sustainable cash flows and investment attractiveness.<br>"
                    "- C4ISR, LVC, analytics and managed services are the best growth/margin opportunities.<br>"
                    "- Use taxonomy-driven comparisons to prioritise capital allocation, M&A and product strategy."
                    "</div>", unsafe_allow_html=True)

        st.markdown("<div class='body-text'><em>Note: full dataset column mappings and debugging details are shown in the middle column's 'Technical details' expander.</em></div>", unsafe_allow_html=True)

    # Column 2: Short dataset summary + collapsible technical details (keeps Home visually clean)
    with col2:
        st.markdown("<div class='section-title'>Dataset summary</div>", unsafe_allow_html=True)
        if df_active is None or df_active.empty:
            st.info('No data loaded. Upload a dataset in Settings or provide the baseline file.')
        else:
            st.markdown(f"<div class='body-text'>Rows: <strong>{rows_count:,}</strong> &middot; Columns: <strong>{cols_count:,}</strong></div>", unsafe_allow_html=True)
            baseline_name = os.path.basename(DEFAULT_DATA_PATH) if DEFAULT_DATA_PATH else 'No baseline file found'
            st.markdown(f"<div class='body-text'>Baseline file: <strong>{baseline_name}</strong></div>", unsafe_allow_html=True)

        st.markdown("<div class='body-text'>Primary node-level metrics shown in the app: Revenue (FY23–FY25), EBITDA (FY23–FY25) and EBITDA margin per year.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Per-node financial metrics:</div>", unsafe_allow_html=True)
        st.markdown("<div class='body-text'>"
                    "- Revenue: FY23, FY24, FY25 (absolute values) and Revenue CAGR 23→25<br>"
                    "- EBITDA: FY23, FY24, FY25 (absolute values) and EBITDA CAGR 23→25<br>"
                    "- EBITDA Margin: FY23, FY24, FY25 (percent). Margin = EBITDA / Revenue × 100"
                    "</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Per-node player data (top 5):</div>", unsafe_allow_html=True)
        st.markdown("<div class='body-text'>"
                    "Name, Geography/Country, Type (Prime/Tier/etc.), Revenue & EBITDA (node-specific, FY25) and EBITDA margin (calculated or provided)."
                    "</div>", unsafe_allow_html=True)

        # Keep verbose, developer/debug info inside an expander so normal users don't get overwhelmed
        with st.expander("Technical details: columns, metric & player mappings (expand if you need exact config)", expanded=False):
            st.markdown("<div class='body-text'>This section lists the exact config-driven column mappings and other technical diagnostics helpful for mapping and debugging.</div>", unsafe_allow_html=True)
            try:
                metrics_cfg = CFG.get('metrics', {})
                player_cfg = CFG.get('player_data', {})
                if metrics_cfg:
                    st.markdown("**Metrics config (from config.json):**", unsafe_allow_html=True)
                    for k, v in metrics_cfg.items():
                        st.write(f"- {k}: {v}")
                if player_cfg:
                    st.markdown("**Player data slots (from config.json):**", unsafe_allow_html=True)
                    st.write("- Player name columns:")
                    for c in player_cfg.get('names', []):
                        st.write(f"  - {c}")
                    st.write("- Player country columns:")
                    for c in player_cfg.get('countries', []):
                        st.write(f"  - {c}")
                    st.write("- Player type columns:")
                    for c in player_cfg.get('types', []):
                        st.write(f"  - {c}")
            except Exception:
                st.markdown("<div class='body-text'>Technical diagnostics unavailable.</div>", unsafe_allow_html=True)

    # Column 3: How to use / navigation (kept concise)
    with col3:
        st.markdown("<div class='section-title'>How to use this app</div>", unsafe_allow_html=True)
        how_md = (
            "<div class='body-text'>"
            "<ul>"
            "<li><strong>Home:</strong> overview, dataset KPIs and a clickable taxonomy map. Click a node to set the Dashboard selection.</li>"
            "<li><strong>Dashboard:</strong> main charts (Revenue, EBITDA, Margin), commentary & sources, player charts and download options.</li>"
            "<li><strong>Heatmap:</strong> column-normalised Taxonomy × Metrics views for multi-node comparison.</li>"
            "</ul>"
            "<p><em>Tips:</em> click nodes on Home or use the left navigation in Dashboard to focus analysis. Use CSV/Download buttons to export tables.</p>"
            "</div>"
        )
        st.markdown(how_md, unsafe_allow_html=True)

    # --- Taxonomy map (interactive) placed below the three columns ---
    st.markdown('<hr />', unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Taxonomy map (click a node to select)</div>", unsafe_allow_html=True)

    hierarchy_cols = ['Hierarchy - Main Category', 'Hierarchy - Sector', 'Hierarchy - Subsector', 'Hierarchy - Sub-Sub-Sector']
    hierarchy_cols = [c for c in hierarchy_cols if (df_active is not None and c in df_active.columns)]
    if not hierarchy_cols:
        st.info('No taxonomy columns found in dataset.')
    else:
        tops = get_unique_in_family_order(df_active[hierarchy_cols[0]]) if hierarchy_cols[0] in df_active.columns else []
        top_color_map = build_top_level_color_map(df_active, hierarchy_cols[0]) if not df_active.empty and hierarchy_cols[0] in df_active.columns else {}
        cmap_01 = build_horizontal_color_map(df_active, hierarchy_cols[0], hierarchy_cols[1]) if len(hierarchy_cols) > 1 and hierarchy_cols[1] in df_active.columns else {}
        cmap_12 = build_horizontal_color_map(df_active, hierarchy_cols[1], hierarchy_cols[2]) if len(hierarchy_cols) > 2 and hierarchy_cols[2] in df_active.columns else {}
        cmap_23 = build_horizontal_color_map(df_active, hierarchy_cols[2], hierarchy_cols[3]) if len(hierarchy_cols) > 3 and hierarchy_cols[3] in df_active.columns else {}
        prev_top = prev_sector = prev_subsector = None
        row_index = 0

        for top in tops:
            if not top:
                continue
            df_top = df_active[df_active[hierarchy_cols[0]].astype(str).str.strip().str.lower() == str(top).strip().lower()]
            top_color = top_color_map.get(top, '#8888')
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
                            color_blk = '#8888'
                            if i == 0:
                                color_blk = top_color
                            elif i == 1:
                                color_blk = sector_color
                            elif i == 2:
                                color_blk = subsector_color
                            elif len(hierarchy_cols) > 3:
                                parent = nodes[i - 1]
                                color_blk = cmap_23.get((parent, node), '#8888')
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
                                        st.markdown(
                                            f"<div style='width:18px;height:18px;background:{color_blk};border-radius:4px;border:1px solid {PANEL_BORDER};margin-top:4px;'></div>",
                                            unsafe_allow_html=True
                                        )
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
      

with tab_dashboard:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Financial Overview</div>', unsafe_allow_html=True)
    df_active = get_active_df()

    # Left navigation builder (simple single-select; no tree, no forced reruns)
    def build_left_nav(df: pd.DataFrame, key_prefix: str = '') -> Dict[str, str]:
        st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)

        # Build flat list of full-path node labels from the dataframe in row order.
        present_levels = [lvl for lvl in HIER_LEVELS if isinstance(df, pd.DataFrame) and lvl in df.columns]
        node_paths: List[str] = []
        seen = set()
        if df is not None and not df.empty and present_levels:
            try:
                for _, row in df[present_levels].iterrows():
                    parts = []
                    for lvl in present_levels:
                        val = str(row.get(lvl, '')).strip()
                        if val and val.lower() not in ('nan', 'none'):
                            parts.append(val)
                        else:
                            break
                    if not parts:
                        continue
                    path = ' / '.join(parts)
                    if path not in seen:
                        seen.add(path)
                        node_paths.append(path)
            except Exception:
                node_paths = []

        default_top = CFG.get('startup', {}).get('default_main_category', 'Aerospace')

        # If no full paths, fall back to a top-level select on the first hierarchy column.
        if not node_paths:
            top_col = HIER_LEVELS[0] if len(HIER_LEVELS) > 0 else None
            top_vals: List[str] = []
            if top_col and isinstance(df, pd.DataFrame) and top_col in df.columns and not df[top_col].dropna().empty:
                top_vals = list(df[top_col].dropna().astype(str).str.strip().unique())
            if default_top in top_vals:
                top_vals = [default_top] + [v for v in top_vals if v != default_top]
            if not top_vals:
                top_vals = [default_top]
            sel = st.selectbox(top_col if top_col else 'Category', top_vals, index=0, key=f'{key_prefix}l1_nav')
            # build sel_path (top level selected, deeper levels = 'All')
            path: Dict[str, str] = {}
            for i, lvl in enumerate(HIER_LEVELS):
                path[lvl] = sel if i == 0 else 'All'
            # Persist selection only (no manual rerun triggers)
            if path != st.session_state.get('sel_path', {}):
                st.session_state.sel_path = path
            return path

        # Prefer nodes whose top-level equals default_top in the select index (UX nicety)
        default_index = 0
        try:
            for i, p in enumerate(node_paths):
                if p.split(' / ')[0].strip().lower() == str(default_top).strip().lower():
                    default_index = i
                    break
        except Exception:
            default_index = 0

        sel_key = f'{key_prefix}left_nav_select'
        selected = st.selectbox("Select taxonomy node", node_paths, index=default_index if default_index < len(node_paths) else 0, key=sel_key)

        # Convert chosen full-path into the nested sel_path dict the app uses
        parts = [p.strip() for p in selected.split(' / ') if p.strip() != '']
        path: Dict[str, str] = {}
        for i, lvl in enumerate(HIER_LEVELS):
            path[lvl] = parts[i] if i < len(parts) and parts[i] != '' else 'All'

        # Persist selection only; Streamlit will rerun automatically on widget change
        if path != st.session_state.get('sel_path', {}):
            st.session_state.sel_path = path
        return path

    # --- MAIN DASHBOARD content ---
    st.subheader("Main Dashboard")
    df_active = get_active_df()
    col_left, col_right = st.columns([1, 3], gap="large")

    with col_left:
        try:
            selected_path = build_left_nav(df_active)
        except Exception as e:
            st.error("Failed to load left navigation.")
            st.exception(e)
            selected_path = st.session_state.get('sel_path', {})
        if not selected_path:
            selected_path = {lvl: ('Aerospace' if i == 0 else 'All') for i, lvl in enumerate(HIER_LEVELS)}
        st.session_state.sel_path = selected_path
        try:
            crumb = " / ".join([selected_path.get(l, '') for l in HIER_LEVELS if selected_path.get(l, '') and selected_path.get(l, '') != 'All'])
            if crumb:
                st.markdown(f"**Selection:** {crumb}")
        except Exception:
            pass

        # Download all data as XLSX
        try:
            sheets = {}
            # Grab commonly used dataframes if they exist in the module globals
            g = globals()
            if isinstance(g.get("players_df"), pd.DataFrame) and not g.get("players_df").empty:
                sheets["Players"] = g.get("players_df")
            if isinstance(g.get("selected_node_rows_df"), pd.DataFrame) and not g.get("selected_node_rows_df").empty:
                sheets["Selected Node"] = g.get("selected_node_rows_df")
            if isinstance(g.get("financials_df"), pd.DataFrame) and not g.get("financials_df").empty:
                sheets["Financials"] = g.get("financials_df")
            if isinstance(g.get("heatmap_df"), pd.DataFrame) and not g.get("heatmap_df").empty:
                sheets["Heatmap"] = g.get("heatmap_df")
            # fallback: raw dataframe
            if isinstance(g.get("raw_df"), pd.DataFrame) and not g.get("raw_df").empty:
                sheets["Raw Data"] = g.get("raw_df")
            # best-effort: common variable names
            if not sheets:
                df_try = g.get("df") or g.get("adata") or g.get("full_df")
                if isinstance(df_try, pd.DataFrame):
                    sheets["Raw Data"] = df_try

            if sheets:
                xlsx_bytes = _build_xlsx_bytes(sheets)
                st.download_button(
                    label="Download all (XLSX)",
                    data=xlsx_bytes,
                    file_name="export_all.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as _e:
            # non-fatal: show small message in sidebar
            try:
                st.text("Download unavailable")
            except Exception:
                pass

    with col_right:
        try:
            df_node = best_effort_selection(df_active, selected_path) if df_active is not None else pd.DataFrame()
        except Exception:
            df_node = pd.DataFrame()

        try:  
            # --- Fetch fiscal metrics for the selected node (Revenue / EBITDA FY23-FY25) ---  
            rev = get_metric_values(df_node, CFG.get('metrics', {}).get('revenue', {})) if df_node is not None else {}  
            ebt = get_metric_values(df_node, CFG.get('metrics', {}).get('ebitda', {})) if df_node is not None else {}  
    
            # Compute CAGR helper  
            def _compute_cagr(v_start, v_end, years=2):  
                try:  
                    if v_start is None or v_end is None:  
                        return None  
                    vs = float(v_start)  
                    ve = float(v_end)  
                    if vs <= 0:  
                        return None  
                    return (ve / vs) ** (1.0 / years) - 1.0  
                except Exception:  
                    return None  
    
            cagr_rev = _compute_cagr(rev.get('fy23'), rev.get('fy25'))  
            cagr_ebt = _compute_cagr(ebt.get('fy23'), ebt.get('fy25'))  
    
            # Average EBITDA margin across FY23-FY25 (where calculable)  
            margin_list = []  
            for fy in ('fy23', 'fy24', 'fy25'):  
                r = rev.get(fy)  
                e = ebt.get(fy)  
                try:  
                    if r is not None and e is not None and float(r) != 0:  
                        m = calculate_ebitda_margin(float(r), float(e))  
                        if m is not None:  
                            margin_list.append(m)  
                except Exception:  
                    pass  
            avg_margin = (sum(margin_list) / len(margin_list)) if margin_list else None  
        except Exception:  
            # non-fatal: keep dashboard working even if KPI calc fails  
            cagr_rev = cagr_ebt = avg_margin = None  
    
        try:  
            # Prepare series for plotting  
            rev_vals = {k: rev.get(k) for k in ('fy23', 'fy24', 'fy25')} if isinstance(rev, dict) else {'fy23': None, 'fy24': None, 'fy25': None}  
            ebt_vals = {k: ebt.get(k) for k in ('fy23', 'fy24', 'fy25')} if isinstance(ebt, dict) else {'fy23': None, 'fy24': None, 'fy25': None}  
    
            # Title above the financial charts with current selected node  
            try:  
                crumb_right = " / ".join([selected_path.get(l, '') for l in HIER_LEVELS if selected_path.get(l, '') and selected_path.get(l, '') != 'All'])  
                if not crumb_right:  
                    crumb_right = "All"  
                st.markdown(f"<div class='section-title'>Segment overal financials: {crumb_right}</div>", unsafe_allow_html=True)  
            except Exception:  
                pass  
    
            c1, c2, c3 = st.columns(3, gap="large")  
    
            # Column 1: Revenue KPI + chart  
            with c1:  
                # KPI bubble: Revenue CAGR 23-25 (percent)  
                try:  
                    kpi_card('Revenue CAGR 23-25', (cagr_rev * 100.0) if cagr_rev is not None else None, suffix='%', color=REV_COLOR)  
                except Exception:  
                    kpi_card('Revenue CAGR 23-25', None, suffix='%', color=REV_COLOR)  
    
                st.markdown("**Revenue**")  
                try:  
                    if any(v is not None for v in rev_vals.values()):  
                        fig_rev = plot_bars_with_labels(  
                            "Revenue (FY23–FY25)",  
                            rev_vals,  
                            color=REV_COLOR,  
                            is_percent=False,  
                            source=CFG.get('data_sources', {}).get('revenue', '')  
                        )  
                        show_plotly(fig_rev, height=360)  
                    else:  
                        st.info("No Revenue data available for this selection.")  
                except Exception as e:  
                    st.warning("Revenue chart failed: " + str(e))  
    
            # Column 2: EBITDA KPI + chart  
            with c2:  
                try:  
                    kpi_card('EBITDA CAGR 23-25', (cagr_ebt * 100.0) if cagr_ebt is not None else None, suffix='%', color=EBITDA_COLOR)  
                except Exception:  
                    kpi_card('EBITDA CAGR 23-25', None, suffix='%', color=EBITDA_COLOR)  
    
                st.markdown("**EBITDA**")  
                try:  
                    if any(v is not None for v in ebt_vals.values()):  
                        fig_ebt = plot_bars_with_labels(  
                            "EBITDA (FY23–FY25)",  
                            ebt_vals,  
                            color=EBITDA_COLOR,  
                            is_percent=False,  
                            source=CFG.get('data_sources', {}).get('ebitda', '')  
                        )  
                        show_plotly(fig_ebt, height=360)  
                    else:  
                        st.info("No EBITDA data available for this selection.")  
                except Exception as e:  
                    st.warning("EBITDA chart failed: " + str(e))  
    
            # Column 3: Avg Margin KPI + chart  
            with c3:  
                try:  
                    # calculate_ebitda_margin() returns a percent value (e.g. 12.34),  
                    # so do NOT multiply by 100 again — show the percent as-is.  
                    kpi_card('Avg EBITDA Margin 23-25', (avg_margin) if avg_margin is not None else None, suffix='%', color=MARGIN_COLOR)  
                except Exception:  
                    kpi_card('Avg EBITDA Margin 23-25', None, suffix='%', color=MARGIN_COLOR)  
    
                st.markdown("**EBITDA Margin**")  
                try:  
                    margin_map = {}  
                    for fy in ('fy23', 'fy24', 'fy25'):  
                        r = rev_vals.get(fy)  
                        e = ebt_vals.get(fy)  
                        margin_map[fy] = calculate_ebitda_margin(r, e) if (r is not None and e is not None) else None  
                    if any(v is not None for v in margin_map.values()):  
                        fig_mgn = plot_bars_with_labels(  
                            "EBITDA Margin (FY23–FY25)",  
                            margin_map,  
                            color=MARGIN_COLOR,  
                            is_percent=True,  
                            source=CFG.get('data_sources', {}).get('ebitda_margin', '')  
                        )  
                        show_plotly(fig_mgn, height=360)  
                    else:  
                        st.info("No EBITDA margin data available for this selection.")  
                except Exception as e:  
                    st.warning("Margin chart failed: " + str(e))  

        except Exception as e:
            st.warning("Dashboard charts could not be built: " + str(e))

        # --- Financial commentary & sources (hierarchical lookup, nearest ancestor if needed) ---
        try:
            fin_comment_col = CFG.get('comments_sources', {}).get('financial_comment_col', 'Financial Data - Financial Commentary')
            fin_source_col = CFG.get('comments_sources', {}).get('financial_source_col', 'Financial Data - Financial Sources')

            def find_node_comment_and_sources(df: pd.DataFrame, sel_path: Dict[str, str],
                                              comment_col: str, source_col: str,
                                              levels: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
                """
                Return (comment_text, source_text, provenance_label)
                provenance_label is None for an exact-node match, or a short label like
                'Parent commentary (for: Sector / Subsector)' when falling back to ancestors.
                """
                if df is None or df.empty or not sel_path:
                    return None, None, None

                def extract_texts(df_slice: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
                    com = None
                    src = None
                    try:
                        if comment_col in df_slice.columns:
                            comments = (
                                df_slice[comment_col].astype(str).fillna('')
                                .map(lambda s: s.strip()).replace({'': None}).dropna().unique().tolist()
                            )
                            if comments:
                                com = "\n\n".join(comments)
                    except Exception:
                        com = None
                    try:
                        if source_col in df_slice.columns:
                            sources = (
                                df_slice[source_col].astype(str).fillna('')
                                .map(lambda s: s.strip()).replace({'': None}).dropna().unique().tolist()
                            )
                            if sources:
                                src = "\n\n".join(sources)
                    except Exception:
                        src = None
                    return com, src

                # 1) exact node
                exact = rows_for_selected_node_exact(df, sel_path)
                if exact is not None and not exact.empty:
                    c, s = extract_texts(exact)
                    if c or s:
                        return c, s, None

                # 2) walk ancestors: closest parent first
                deepest = get_deepest_selected_level(sel_path, levels)
                if deepest is None:
                    return None, None, None
                try:
                    deep_idx = levels.index(deepest)
                except Exception:
                    return None, None, None

                for parent_idx in range(deep_idx - 1, -1, -1):
                    trial_path: Dict[str, str] = {}
                    for i, lvl in enumerate(levels):
                        if i <= parent_idx:
                            trial_path[lvl] = sel_path.get(lvl, 'All')
                        else:
                            trial_path[lvl] = 'All'
                    trial_rows = rows_for_selected_node_exact(df, trial_path)
                    if trial_rows is not None and not trial_rows.empty:
                        c, s = extract_texts(trial_rows)
                        if c or s:
                            prov_parts = [trial_path[levels[i]] for i in range(parent_idx + 1) if trial_path.get(levels[i]) and trial_path.get(levels[i]) != 'All']
                            prov_label = " / ".join(prov_parts) if prov_parts else None
                            return c, s, f"Parent commentary (for: {prov_label})" if prov_label else ("", "", "Parent commentary")

                # nothing found
                return None, None, None

            sel_path_local = st.session_state.get('sel_path', {}) or {}
            # Pass the full active dataframe so ancestor lookup can find parent-level rows
            fin_comment_text, fin_source_text, fin_provenance = find_node_comment_and_sources(
                df_active, sel_path_local, fin_comment_col, fin_source_col, HIER_LEVELS
            )

        except Exception:
            fin_comment_text = None
            fin_source_text = None
            fin_provenance = None

        # Render commentary (left) and sources (right) and show provenance when falling back
        try:
            # Use 2:1 layout so commentary takes ~2/3 and sources ~1/3
            com_col_left, com_col_right = st.columns([2, 1], gap="large")
            with com_col_left:
                st.markdown("#### Financial commentary")
                if fin_provenance:
                    try:
                        st.markdown(f"_{fin_provenance}_")
                    except Exception:
                        pass
                if fin_comment_text:
                    for para in str(fin_comment_text).split("\n\n"):
                        st.markdown(para)
                else:
                    st.markdown("_No financial commentary found for this selection._")
            with com_col_right:
                st.markdown("#### Financial sources")
                if fin_source_text:
                    for s in str(fin_source_text).split("\n\n"):
                        st.markdown(f"- {s}")
                else:
                    st.markdown("_No financial sources found for this selection._")
        except Exception:
            # do not block dashboard if rendering fails
            pass

        # --- Merge: Players content included here inside Dashboard ---
        st.markdown('<div class="section-title">Players</div>', unsafe_allow_html=True)
        df_active = get_active_df()
        sel_path = st.session_state.get('sel_path', {}) or {}
        # IMPORTANT: use the exact same node dataframe that the financial charts used.
        # Financial metrics above use `df_node = best_effort_selection(df_active, selected_path)`.
        # Reuse `df_node` when available so commentary/sources and players align exactly with the fiscal metrics.
        try:
            if 'df_node' in locals() and (df_node is not None) and (not df_node.empty):
                df_sel = df_node.copy()
            else:
                df_sel = best_effort_selection(df_active, sel_path) if not df_active.empty else pd.DataFrame()
        except Exception:
            # Fallback robust path to avoid breaking the UI
            try:
                df_sel = best_effort_selection(df_active, sel_path) if not df_active.empty else pd.DataFrame()
            except Exception:
                df_sel = pd.DataFrame()

        player_cols = list(zip(CFG['player_data']['names'], CFG['player_data']['countries'], CFG['player_data']['types']))
        players = []
        for _, r in df_sel.iterrows():
            for name_col, country_col, type_col in player_cols:
                name = str(r.get(name_col, '')).strip()
                if name and name.lower() != 'nan':
                    players.append({
                        'Name': name,
                        'Country': str(r.get(country_col, '')).strip(),
                        'Type': str(r.get(type_col, '')).strip()
                    })
        if not players:
            st.info('No players found for current selection.')
        else:
            df_players = pd.DataFrame(players).drop_duplicates().reset_index(drop=True)

            cols_lower = [c.strip().lower() for c in df_sel.columns]
            metric_col = None
            value_col = None
            if 'fiscal metric flattened' in cols_lower and 'value' in cols_lower:
                metric_col = [c for c in df_sel.columns if c.strip().lower() == 'fiscal metric flattened'][0]
                value_col = [c for c in df_sel.columns if c.strip().lower() == 'value'][0]
            else:
                for c in df_sel.columns:
                    lc = c.strip().lower()
                    if metric_col is None and 'fiscal' in lc and 'metric' in lc:
                        metric_col = c
                    if value_col is None and lc == 'value':
                        value_col = c

            fiscal_metric_col = next((c for c in df_sel.columns if c.strip().lower() == 'fiscal metric'), None)
            name_cols = [c for c in CFG['player_data']['names'] if c in df_sel.columns]

            def player_metric_aggregate(player_name: str, metric_label: str, df_lookup: pd.DataFrame) -> Optional[float]:
                if metric_label is None or metric_label == '' or df_lookup is None or df_lookup.empty:
                    return None
                pname = str(player_name).strip()
                if not pname:
                    return None

                def parse_slot_from_texts(metric_text: str, fiscal_text: Optional[str]) -> int:
                    def try_parse(t):
                        if not t:
                            return None
                        m = re.search(r'\$B[_\s]*([0-9]+)\b', str(t), flags=re.I)
                        if m:
                            try:
                                return int(m.group(1))
                            except Exception:
                                return None
                        m2 = re.search(r'_(\d+)\b', str(t))
                        if m2:
                            try:
                                return int(m2.group(1))
                            except Exception:
                                return None
                        return None

                    s = try_parse(metric_text)
                    if s is not None:
                        return s
                    s = try_parse(fiscal_text)
                    if s is not None:
                        return s
                    return 0

                name_cols_local = [c for c in CFG['player_data']['names'] if c in df_lookup.columns]
                if not name_cols_local:
                    return None

                metric_col_local = metric_col
                value_col_local = value_col
                fiscal_metric_col_local = fiscal_metric_col

                candidates = pd.DataFrame()
                try:
                    if metric_col_local in df_lookup.columns:
                        lc = df_lookup[metric_col_local].astype(str).str.lower()
                        lbl = str(metric_label or '').lower()
                        metric_tokens = []
                        if 'revenue' in lbl:
                            metric_tokens.append('revenue')
                        elif 'ebitda' in lbl:
                            metric_tokens.append('ebitda')
                        elif 'margin' in lbl or 'marg' in lbl:
                            metric_tokens.append('margin')
                        required_mask = lc.str.contains('player', na=False) & lc.str.contains('25', na=False)
                        for tok in metric_tokens:
                            required_mask = required_mask & lc.str.contains(tok, na=False)
                        candidates = df_lookup[required_mask].copy()
                except Exception:
                    candidates = pd.DataFrame()

                if candidates.empty and metric_col_local in df_lookup.columns:
                    try:
                        lc = df_lookup[metric_col_local].astype(str).str.lower()
                        candidates = df_lookup[lc.str.contains('player', na=False) & lc.str.contains('25', na=False)].copy()
                    except Exception:
                        candidates = pd.DataFrame()

                if candidates.empty:
                    return None

                rows = []
                for idx, r in candidates.iterrows():
                    fm = r.get(metric_col_local, '') if metric_col_local in r.index else ''
                    fiscal_text = r.get(fiscal_metric_col_local, '') if (fiscal_metric_col_local and fiscal_metric_col_local in r.index) else ''
                    slot = parse_slot_from_texts(fm, fiscal_text)

                    if slot == 0:
                        pname_col = name_cols_local[0]
                    else:
                        match_col = next((c for c in name_cols_local if re.search(rf'_{slot}$', c)), None)
                        if match_col:
                            pname_col = match_col
                        else:
                            try:
                                pname_col = name_cols_local[slot]
                            except Exception:
                                pname_col = name_cols_local[0]

                    player_in_mapped_col = str(r.get(pname_col, '')).strip() if pname_col in r.index else ''
                    player_names_in_row = [str(r.get(c, '')).strip() if c in r.index else '' for c in name_cols_local]

                    is_dollar = False
                    try:
                        if fiscal_metric_col_local and fiscal_metric_col_local in r.index:
                            if re.search(r'\$B', str(r.get(fiscal_metric_col_local, '')), flags=re.I):
                                is_dollar = True
                        if not is_dollar and re.search(r'\$B', str(fm), flags=re.I):
                            is_dollar = True
                    except Exception:
                        is_dollar = False

                    try:
                        val = float(pd.to_numeric(r.get(value_col_local), errors='coerce'))
                    except Exception:
                        val = None

                    rows.append({
                        'idx': idx,
                        'slot': slot,
                        'pname_col': pname_col,
                        'player_in_mapped_col': player_in_mapped_col,
                        'player_names_in_row': player_names_in_row,
                        'is_dollar': is_dollar,
                        'value': val,
                        'fm': fm
                    })

                exact_mapped = [r for r in rows if r['player_in_mapped_col'] and r['player_in_mapped_col'].strip().lower() == pname.lower() and r['value'] is not None]
                if exact_mapped:
                    chosen = sorted(exact_mapped, key=lambda x: int(x.get('is_dollar', False)), reverse=True)[0]
                    return chosen['value']

                exact_any = []
                for r in rows:
                    for pn in r.get('player_names_in_row', []):
                        if pn and pn.strip().lower() == pname.lower() and r['value'] is not None:
                            exact_any.append(r)
                            break
                if exact_any:
                    chosen = sorted(exact_any, key=lambda x: int(x.get('is_dollar', False)), reverse=True)[0]
                    return chosen['value']

                contains_mapped = [r for r in rows if r['player_in_mapped_col'] and re.search(rf'\b{re.escape(pname)}\b', r['player_in_mapped_col'], flags=re.I) and r['value'] is not None]
                if contains_mapped:
                    chosen = sorted(contains_mapped, key=lambda x: int(x.get('is_dollar', False)), reverse=True)[0]
                    return chosen['value']

                contains_any = []
                for r in rows:
                    for pn in r.get('player_names_in_row', []):
                        if pn and re.search(rf'\b{re.escape(pname)}\b', pn, flags=re.I) and r['value'] is not None:
                            contains_any.append(r)
                            break
                if contains_any:
                    chosen = sorted(contains_any, key=lambda x: int(x.get('is_dollar', False)), reverse=True)[0]
                    return chosen['value']

                return None

            metrics_rows = []
            player_revenue_label = CFG['metrics'].get('player_revenue', {}).get('fy25')
            player_ebitda_label = CFG['metrics'].get('player_ebitda', {}).get('fy25')
            player_margin_label = 'EBITDA Margin FY25 (%)'

            def player_margin_lookup(player_name: str, metric_label: str, df_lookup: pd.DataFrame) -> Optional[float]:
                if not metric_label or df_lookup is None or df_lookup.empty or value_col is None:
                    return None
                try:
                    metric_mask = df_lookup[metric_col].astype(str).str.strip().str.lower() == str(metric_label).strip().lower() if metric_col in df_lookup.columns else pd.Series(False, index=df_lookup.index)
                except Exception:
                    metric_mask = pd.Series(False, index=df_lookup.index)
                rows = df_lookup[metric_mask].copy()
                if rows.empty:
                    return None
                pname = str(player_name).strip()
                if not pname:
                    return None
                for nc in name_cols:
                    if nc in rows.columns:
                        exact = rows[rows[nc].astype(str).str.strip().str.lower() == pname.lower()]
                        if not exact.empty:
                            vals = pd.to_numeric(exact[value_col], errors='coerce').dropna()
                            if not vals.empty:
                                return float(vals.iloc[0])
                pat = rf'\b{re.escape(pname)}\b'
                for nc in name_cols:
                    if nc in rows.columns:
                        try:
                            cont = rows[rows[nc].astype(str).str.contains(pat, case=False, na=False, regex=True)]
                        except Exception:
                            cont = rows[rows[nc].astype(str).str.contains(re.escape(pname), case=False, na=False)]
                        if not cont.empty:
                            vals = pd.to_numeric(cont[value_col], errors='coerce').dropna()
                            if not vals.empty:
                                return float(vals.iloc[0])
                return None

            for _, prow in df_players.iterrows():
                name = prow['Name']
                rev25 = player_metric_aggregate(name, player_revenue_label, df_sel) if player_revenue_label else None
                ebt25 = player_metric_aggregate(name, player_ebitda_label, df_sel) if player_ebitda_label else None
                mgn25 = None
                try:
                    mgn25 = player_margin_lookup(name, player_margin_label, df_sel)
                except Exception:
                    mgn25 = None
                if (mgn25 is None) and (rev25 is not None) and (ebt25 is not None):
                    try:
                        mgn25 = calculate_ebitda_margin(rev25, ebt25)
                    except Exception:
                        mgn25 = None
                metrics_rows.append({
                    'Name': name,
                    'Revenue_FY25': rev25,
                    'EBITDA_FY25': ebt25,
                    'EBITDA_Margin_FY25': mgn25
                })

            df_player_metrics = pd.DataFrame(metrics_rows).set_index('Name') if metrics_rows else pd.DataFrame(columns=['Revenue_FY25', 'EBITDA_FY25', 'EBITDA_Margin_FY25']).set_index(pd.Index([]))
            if not df_player_metrics.empty:
                df_player_metrics.index = df_player_metrics.index.astype(str).str.strip()
                df_player_metrics = df_player_metrics[~df_player_metrics.index.duplicated(keep='first')]

            expected_metrics = ['Revenue_FY25', 'EBITDA_FY25', 'EBITDA_Margin_FY25']
            for c in expected_metrics:
                if c not in df_player_metrics.columns:
                    df_player_metrics[c] = None

            df_player_metrics['Revenue_FY25_display'] = df_player_metrics['Revenue_FY25'].apply(lambda v: convert_currency(v, 'Billions', st.session_state.currency_unit) if pd.notna(v) else None)
            df_player_metrics['EBITDA_FY25_display'] = df_player_metrics['EBITDA_FY25'].apply(lambda v: convert_currency(v, 'Billions', st.session_state.currency_unit) if pd.notna(v) else None)
            if 'EBITDA_Margin_FY25' not in df_player_metrics.columns:
                df_player_metrics['EBITDA_Margin_FY25'] = None

            top_n = 5
            c1, c2, c3 = st.columns(3, gap="large")

            # Select top players by Revenue (largest first), then build ascending list (small -> large)
            # because the renderer sets autorange='reversed' so largest appears at the top visually.
            try:
                if 'Revenue_FY25_display' in df_player_metrics.columns:
                    rev_series = df_player_metrics['Revenue_FY25_display'].dropna()
                else:
                    rev_series = pd.Series(dtype=float)

                if not rev_series.empty:
                    # top_players_desc: largest -> smallest
                    top_players_desc = rev_series.sort_values(ascending=False).head(top_n).index.tolist()
                    # top_players_asc kept for compatibility if needed, but we'll pass the descending order to the renderer
                    top_players_asc = list(reversed(top_players_desc))
                else:
                    top_players_desc = []
                    top_players_asc = []
            except Exception:
                top_players_desc = []
                top_players_asc = []

            with c1:
                st.markdown('#### Revenue FY25 (top players)')
                if top_players_desc:
                    # Reindex to the descending order (largest -> smallest) so the renderer will display largest at the top
                    df_rev_plot = df_player_metrics.reindex(top_players_desc).reset_index().rename(columns={'index': 'Player'})
                    render_horizontal_player_bars(
                        "Revenue FY25 (top players)",
                        df_rev_plot,
                        'Revenue_FY25_display',
                        'Player',
                        REV_COLOR,
                        f'Revenue FY25 ({st.session_state.currency_unit})',
                        height=360,
                        category_order=top_players_desc,
                    )
                else:
                    st.info('No Player Revenue FY25 values found for charting.')

            with c2:
                st.markdown('#### EBITDA FY25 (top players)')
                if top_players_desc:
                    # Reuse same descending order (largest -> smallest) so vertical positions align across charts
                    df_ebt_plot = df_player_metrics.reindex(top_players_desc).reset_index().rename(columns={'index': 'Player'})
                    render_horizontal_player_bars(
                        "EBITDA FY25 (top players)",
                        df_ebt_plot,
                        'EBITDA_FY25_display',
                        'Player',
                        EBITDA_COLOR,
                        f'EBITDA FY25 ({st.session_state.currency_unit})',
                        height=360,
                        category_order=top_players_desc,
                    )
                else:
                    st.info('No Player EBITDA FY25 values found for charting.')

            with c3:
                st.markdown('#### EBITDA Margin FY25 (top players)')
                if top_players_desc:
                    df_mgn_plot = df_player_metrics.reindex(top_players_desc).reset_index().rename(columns={'index': 'Player'})
                    render_horizontal_player_bars(
                        "EBITDA Margin FY25 (top players)",
                        df_mgn_plot,
                        'EBITDA_Margin_FY25',
                        'Player',
                        MARGIN_COLOR,
                        'EBITDA Margin FY25 (%)',
                        height=360,
                        category_order=top_players_desc,
                    )
                else:
                    st.info('No Player EBITDA Margin FY25 values found for charting.')

            # --- Player data table preparation (player commentary & sources restored) ---
            df_players_display = df_players.set_index('Name').join(
                df_player_metrics[['Revenue_FY25_display', 'EBITDA_FY25_display', 'EBITDA_Margin_FY25']].rename(columns={}),
                how='left'
            ).reset_index()

            df_players_display = df_players_display.rename(columns={
                'Revenue_FY25_display': f'Revenue FY25 ({st.session_state.currency_unit})',
                'EBITDA_FY25_display': f'EBITDA FY25 ({st.session_state.currency_unit})',
                'EBITDA_Margin_FY25': 'EBITDA Margin FY25 (%)'
            })

            # Format EBITDA margin column for display with 1 decimal place (e.g. "9.3%")
            if 'EBITDA Margin FY25 (%)' in df_players_display.columns:
                try:
                    df_players_display['EBITDA Margin FY25 (%)'] = (
                        df_players_display['EBITDA Margin FY25 (%)']
                        .apply(lambda v: f'{v:.1f}%' if pd.notna(v) else '')
                    )
                except Exception:
                    # Fail silently and fall back to raw values to avoid breaking the UI
                    pass

            # --- Player commentary & sources (restored for vertical layout) ---
            top_player_name = top_players_desc[0] if top_players_desc else None
            p_comment_text = None
            p_source_text = None
            try:
                if top_player_name and (df_sel is not None and not df_sel.empty):
                    p_comment_col = CFG.get('comments_sources', {}).get('player_comment_col', 'Player data - Player Commentary')
                    p_source_col = CFG.get('comments_sources', {}).get('player_source_col', 'Player data - Player Sources')

                    # Build mask for rows that reference the top player in any player name slot
                    name_cols_present = [c for c in CFG.get('player_data', {}).get('names', []) if c in df_sel.columns]
                    if name_cols_present:
                        mask = pd.Series(False, index=df_sel.index)
                        pat = rf'\b{re.escape(str(top_player_name).strip())}\b'
                        for nc in name_cols_present:
                            try:
                                mask = mask | df_sel[nc].astype(str).str.contains(pat, case=False, na=False, regex=True)
                            except Exception:
                                mask = mask | df_sel[nc].astype(str).str.strip().str.lower().eq(str(top_player_name).strip().lower())
                        candidates = df_sel[mask].copy() if not mask.empty else pd.DataFrame()
                    else:
                        candidates = pd.DataFrame()

                    # Extract commentary & sources (deduplicated)
                    if not candidates.empty:
                        try:
                            if p_comment_col in candidates.columns:
                                comments = (
                                    candidates[p_comment_col]
                                    .astype(str)
                                    .fillna('')
                                    .map(lambda s: s.strip())
                                    .replace({'': None})
                                    .dropna()
                                    .unique()
                                    .tolist()
                                )
                                if comments:
                                    p_comment_text = "\n\n".join(comments)
                        except Exception:
                            p_comment_text = None
                        try:
                            if p_source_col in candidates.columns:
                                sources = (
                                    candidates[p_source_col]
                                    .astype(str)
                                    .fillna('')
                                    .map(lambda s: s.strip())
                                    .replace({'': None})
                                    .dropna()
                                    .unique()
                                    .tolist()
                                )
                                if sources:
                                    p_source_text = "\n\n".join(sources)
                        except Exception:
                            p_source_text = None
            except Exception:
                p_comment_text = None
                p_source_text = None

            # --- Render players commentary and sources side-by-side (2/3 commentary, 1/3 sources) ---
            try:
                p_col_left, p_col_right = st.columns([2, 1], gap="large")
                with p_col_left:
                    st.markdown('#### Player commentary')
                    if top_player_name:
                        st.markdown(f"**Top player (by Revenue FY25):** {top_player_name}")
                    if p_comment_text:
                        for para in str(p_comment_text).split("\n\n"):
                            st.markdown(para)
                    else:
                        if top_player_name:
                            st.markdown(f"_No player commentary found for {top_player_name} in this node._")
                        else:
                            st.markdown("_No player commentary available._")

                with p_col_right:
                    st.markdown('#### Player sources')
                    if p_source_text:
                        for s in str(p_source_text).split("\n\n"):
                            st.markdown(f"- {s}")
                    else:
                        st.markdown("_No player sources found._")

            except Exception:
                # If something goes wrong rendering commentary, continue to table rendering
                pass

            # Now render the players table and CSV download below the commentary
            try:
                st.dataframe(df_players_display, width='stretch', hide_index=True)
                csv_bytes = df_players_display.to_csv(index=False).encode('utf-8')
                st.download_button('Download Players CSV', csv_bytes, file_name='players.csv', mime='text/csv')
            except Exception:
                # final fallback: attempt to at least render basic table output
                try:
                    st.dataframe(df_players_display, width='stretch', hide_index=True)
                    csv_bytes = df_players_display.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Players CSV', csv_bytes, file_name='players.csv', mime='text/csv')
                except Exception:
                    # if even that fails, silently continue so dashboard does not crash
                    pass

        st.markdown('</div>', unsafe_allow_html=True)


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
            # Add Revenue CAGR option (lookup-only; do NOT recalculate)
            rev_cagr = st.checkbox('Revenue CAGR 23-25', key='hm_rev_cagr', value=False)
        with col2:
            ebt23 = st.checkbox('EBITDA FY23', key='hm_ebt23', value=False)
            ebt24 = st.checkbox('EBITDA FY24', key='hm_ebt24', value=False)
            ebt25 = st.checkbox('EBITDA FY25', key='hm_ebt25', value=False)
            # Add EBITDA CAGR option (lookup-only; do NOT recalculate)
            ebt_cagr = st.checkbox('EBITDA CAGR 23-25', key='hm_ebt_cagr', value=False)
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
        if rev_cagr:
            metric_cols.append(('Revenue CAGR 23-25', ('revenue_cagr', '')))
        if ebt23:
            metric_cols.append(('EBITDA FY23', ('ebitda', 'fy23')))
        if ebt24:
            metric_cols.append(('EBITDA FY24', ('ebitda', 'fy24')))
        if ebt25:
            metric_cols.append(('EBITDA FY25', ('ebitda', 'fy25')))
        if ebt_cagr:
            metric_cols.append(('EBITDA CAGR 23-25', ('ebitda_cagr', '')))
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
                    
                    # Helper: try to lookup an existing metric row in the node DF by keyword tokens.
                    def _lookup_metric_by_keywords(df_node: pd.DataFrame, metric_col_name: Optional[str], value_col_name: Optional[str], keywords: List[str]):
                        if df_node is None or df_node.empty:
                            return None
                        # If metric_col_name/value_col_name not provided, fall back to detection
                        if not metric_col_name or metric_col_name not in df_node.columns:
                            metric_col_name, detected_value_col = _find_metric_and_value_cols(df_node)
                        else:
                            detected_value_col = value_col_name if (value_col_name and value_col_name in df_node.columns) else None
                        if not metric_col_name:
                            return None
                        try:
                            lc = df_node[metric_col_name].astype(str).str.lower()
                            mask = pd.Series(True, index=df_node.index)
                            for kw in keywords:
                                mask = mask & lc.str.contains(str(kw).strip().lower(), na=False)
                            matches = df_node.loc[mask]
                            if matches is None or matches.empty:
                                return None
                            # Choose first numeric value in the detected value column if present; else try to coerce any numeric-like column
                            value_col = detected_value_col
                            if not value_col:
                                # try to find a column named 'Value' case-insensitively
                                cols_lower = {c.strip().lower(): c for c in df_node.columns}
                                value_col = cols_lower.get('value')
                            if not value_col:
                                # try to find first numeric column
                                numeric_cols = [c for c in df_node.columns if df_node[c].dtype.kind in 'fi']
                                value_col = numeric_cols[0] if numeric_cols else None
                            if not value_col:
                                return None
                            # coerce numeric and take the first non-null
                            vals = pd.to_numeric(matches[value_col], errors='coerce').dropna()
                            if vals.empty:
                                return None
                            return float(vals.iloc[0])
                        except Exception:
                            return None

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
                        
                        # Lookup existing CAGR metrics in the node rows (do NOT recalculate)
                        # Detect metric/value column names once per node
                        metric_col_name, detected_value_col = _find_metric_and_value_cols(df_node)
                        # Revenue CAGR lookup: look for rows whose metric cell contains both 'cagr' and 'revenue'
                        rev_cagr_val = _lookup_metric_by_keywords(df_node, metric_col_name, detected_value_col, ['cagr', 'revenue'])
                        ebt_cagr_val = _lookup_metric_by_keywords(df_node, metric_col_name, detected_value_col, ['cagr', 'ebitda'])
                        # Keep the same labels as the UI checkboxes
                        row['Revenue CAGR 23-25'] = rev_cagr_val
                        row['EBITDA CAGR 23-25'] = ebt_cagr_val
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
                        elif key[0] == 'revenue_cagr':
                            # label used above is 'Revenue CAGR 23-25'
                            final_cols.append('Revenue CAGR 23-25')
                        elif key[0] == 'ebitda_cagr':
                            final_cols.append('EBITDA CAGR 23-25')
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
    
with tab_heatmap_all:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Heatmap - all</div>", unsafe_allow_html=True)
    try:
        # Use the same active dataframe as the rest of the app
        df_active = get_active_df()
        # Default to EBITDA Margin FY25 (user-facing label). The UI remains selectable.
        heatmap_all_tab_ui(df_active, default_metric="EBITDA Margin FY25", value_col="Value")
    except Exception as e:
        st.error("Failed to render 'Heatmap - all' tab: " + str(e))
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    pass


if __name__ == '__main__':
    main()



