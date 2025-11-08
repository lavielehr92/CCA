
import math
import json
import time
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(page_title="Education Desert — Tract Dashboard", layout="wide")
st.title("Education Desert — Census Tract Dashboard")
st.caption("ACS-based market scan you can extend. Uses your Census API key from Streamlit Secrets.")

# ------------------------------
# Helpers
# ------------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return default

def retry_get(url, params, tries=3, timeout=30):
    last = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(0.7)
    raise last if last else RuntimeError("GET failed")

def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or math.isclose(lo, hi):
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd

# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.header("Inputs")
    api_key = st.text_input("Census API key", value=get_secret("CENSUS_API_KEY"), type="password")
    if not api_key:
        st.warning("Add CENSUS_API_KEY to Streamlit Secrets or paste it here to query the API.")

    st.markdown("**Select Location** (State/County FIPS)")
    # Minimal list; default to Philadelphia County, PA (42-101)
    state = st.text_input("State FIPS (e.g., 42=PA)", value="42")
    county = st.text_input("County FIPS (e.g., 101=Philadelphia)", value="101")

    st.divider()
    st.markdown("**Scoring Weights**")
    w_no_hs = st.slider("% Adults w/o HS", 0.0, 1.0, 0.5, 0.05)
    w_kids  = st.slider("% Under 18", 0.0, 1.0, 0.3, 0.05)
    w_inc   = st.slider("Inverse Median Income", 0.0, 1.0, 0.2, 0.05)
    norm_choice = st.selectbox("Normalization", ["z-score", "min-max"])

# ------------------------------
# Fetch ACS 5-year (tract) for county
# Variables:
#   B15003: educational attainment 25+ (001=total; 002..016 less than HS)
#   B01001: age/sex (001 total; <18 approx indices)
#   B19013_001E: median household income
# ------------------------------
if not api_key:
    st.stop()

acs_base = "https://api.census.gov/data/2022/acs/acs5"
edu_vars = [f"B15003_{i:03d}E" for i in range(1, 26)]
age_vars = [f"B01001_{i:03d}E" for i in range(1, 50)]
inc_var  = ["B19013_001E"]
vars_all = ",".join(edu_vars + age_vars + inc_var)

params = {
    "get": vars_all + ",state,county,tract",
    "for": "tract:*",
    "in": f"state:{state} county:{county}",
    "key": api_key
}

with st.spinner("Querying ACS by tract..."):
    r = retry_get(acs_base, params)
    data = r.json()
    cols, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=cols)

# Cast numeric
num_cols = [c for c in df.columns if c.startswith(("B15003_", "B01001_", "B19013_"))]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df["GEOID"] = df["state"] + df["county"] + df["tract"]

# % without HS (sum 002..016 over 001)
less_hs = df[[f"B15003_{i:03d}E" for i in range(2,17)]].sum(axis=1)
total25 = df["B15003_001E"]
df["pct_no_hs"] = (less_hs / total25 * 100).where(total25 > 0)

# % under 18 from B01001 (approx bins)
under18_idx_m = list(range(3,10))
under18_idx_f = list(range(27,34))
df["total_pop"] = df["B01001_001E"]
df["under18"] = df[[f"B01001_{i:03d}E" for i in under18_idx_m + under18_idx_f]].sum(axis=1)
df["pct_under18"] = (df["under18"] / df["total_pop"] * 100).where(df["total_pop"] > 0)

df["median_income"] = df["B19013_001E"]

# Scoring
work = df[["GEOID","pct_no_hs","pct_under18","median_income","total_pop"]].copy()
if norm_choice == "min-max":
    work["no_hs_n"] = minmax(work["pct_no_hs"])
    work["kids_n"]  = minmax(work["pct_under18"])
    work["inc_n"]   = 1 - minmax(work["median_income"])
else:
    work["no_hs_n"] = zscore(work["pct_no_hs"])
    work["kids_n"]  = zscore(work["pct_under18"])
    work["inc_n"]   = -zscore(work["median_income"])

work["desert_score"] = w_no_hs*work["no_hs_n"] + w_kids*work["kids_n"] + w_inc*work["inc_n"]
work = work.sort_values("desert_score", ascending=False).reset_index(drop=True)

st.subheader("Tract Ranking (Education Desert Score)")
st.dataframe(work[["GEOID","pct_no_hs","pct_under18","median_income","total_pop","desert_score"]].round(2),
             use_container_width=True)

# ------------------------------
# Quick charts
# ------------------------------
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    top10 = work.head(10).iloc[::-1]
    ax.barh(top10["GEOID"], top10["desert_score"])
    ax.set_title("Top 10 Tracts by Desert Score")
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(work["median_income"], work["pct_no_hs"])
    ax2.set_xlabel("Median HH Income")
    ax2.set_ylabel("% Adults w/o HS")
    ax2.set_title("Income vs. Educational Attainment")
    st.pyplot(fig2)

# ------------------------------
# TIGERweb GeoJSON for tracts (choropleth)
# Layer 8 is Census Tracts (2023) in Tracts_Blocks service.
# ------------------------------
st.subheader("Choropleth Map (by Tract)")

tiger_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/8/query"
where = f"STATE='{state.zfill(2)}' AND COUNTY='{county.zfill(3)}'"
gparams = {
    "where": where,
    "outFields": "GEOID",
    "outSR": "4326",
    "f": "geojson"
}

with st.spinner("Downloading tract boundaries..."):
    gj = retry_get(tiger_url, gparams).json()

# Merge score onto GeoJSON features
score_map = work.set_index("GEOID")["desert_score"].to_dict()

# Build folium map
m = folium.Map(location=[39.9526, -75.1652], zoom_start=11, tiles="cartodbpositron")

folium.Choropleth(
    geo_data=gj,
    data=work,
    columns=["GEOID", "desert_score"],
    key_on="feature.properties.GEOID",
    fill_opacity=0.7,
    line_opacity=0.3,
    legend_name="Education Desert Score",
).add_to(m)

# Add hover popups with details
folium.features.GeoJson(
    gj,
    name="tracts",
    style_function=lambda x: {"fillOpacity":0, "color":"#00000000", "weight":0},
    tooltip=folium.features.GeoJsonTooltip(
        fields=["GEOID"],
        aliases=["Tract GEOID:"],
        sticky=False
    ),
    highlight_function=lambda x: {"weight":1, "color":"#666"}
).add_to(m)

st_folium(m, height=600, width=None)

with st.expander("Notes"):
    st.markdown("""
- **Data**: ACS 2022 5-year. Change the endpoint year if needed.
- **Variables**: % adults without a HS diploma (B15003), % under 18 (B01001), and median household income (B19013_001E).
- **Score**: Weighted combination of normalized indicators; tweak weights at left.
- **Map**: Tract boundaries from TIGERweb (Census). You can replace with local shapefiles if needed.
""")
