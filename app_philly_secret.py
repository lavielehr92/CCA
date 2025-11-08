
import math, time, requests, pandas as pd, streamlit as st, folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(page_title="Philadelphia Education Desert — Tracts", layout="wide")
st.title("Philadelphia Education Desert — Census Tracts")
st.caption("ACS 2022 5-year • Philadelphia County (FIPS: state 42, county 101)")

# --- Require API key from Streamlit Secrets ONLY ---
def get_api_key():
    try:
        return st.secrets["CENSUS_API_KEY"]
    except Exception:
        st.error(
            "Missing `CENSUS_API_KEY` secret.\n\n"
            "In Streamlit → **Manage app** → **Settings** → **Secrets**, add:\n"
            "```toml\nCENSUS_API_KEY = \"your_key_here\"\n```"
        )
        st.stop()

API_KEY = get_api_key()
STATE, COUNTY = "42", "101"  # Philadelphia, PA

# --- Helpers ---
def retry_get(url, params, tries=4, timeout=40, backoff=0.9):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                # Census throttling — backoff then retry
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(backoff + i * 0.7)
    raise last if last else RuntimeError("GET failed")

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - mu) / sd

# --- Fetch ACS data (tracts) ---
with st.status("Querying ACS for Philadelphia tracts…", expanded=False) as status:
    base = "https://api.census.gov/data/2022/acs/acs5"
    edu_vars = [f"B15003_{i:03d}E" for i in range(1, 26)]
    age_vars = [f"B01001_{i:03d}E" for i in range(1, 50)]
    inc_var  = ["B19013_001E"]
    vars_all = ",".join(edu_vars + age_vars + inc_var)
    params = {
        "get": vars_all + ",state,county,tract",
        "for": "tract:*",
        "in": f"state:{STATE} county:{COUNTY}",
        "key": API_KEY,
    }
    try:
        r = retry_get(base, params)
    except Exception as e:
        st.error(f"Census API error: {e}")
        st.stop()
    data = r.json()
    cols, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=cols)
    status.update(label=f"Fetched {len(df)} tracts.", state="complete")

# --- Compute indicators ---
num_cols = [c for c in df.columns if c.startswith(("B15003_", "B01001_", "B19013_"))]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df["GEOID"] = df["state"] + df["county"] + df["tract"]

less_hs = df[[f"B15003_{i:03d}E" for i in range(2,17)]].sum(axis=1)
total25 = df["B15003_001E"]
df["pct_no_hs"] = (less_hs / total25 * 100).where(total25 > 0)

under18_idx_m = list(range(3,10))
under18_idx_f = list(range(27,34))
df["total_pop"] = df["B01001_001E"]
df["under18"] = df[[f"B01001_{i:03d}E" for i in under18_idx_m + under18_idx_f]].sum(axis=1)
df["pct_under18"] = (df["under18"] / df["total_pop"] * 100).where(df["total_pop"] > 0)

df["median_income"] = df["B19013_001E"]

# --- Score (fixed weights) ---
w_no_hs, w_kids, w_inc = 0.5, 0.3, 0.2
work = df[["GEOID","pct_no_hs","pct_under18","median_income","total_pop"]].copy()
work["no_hs_n"] = zscore(work["pct_no_hs"])
work["kids_n"]  = zscore(work["pct_under18"])
work["inc_n"]   = -zscore(work["median_income"])
work["desert_score"] = w_no_hs*work["no_hs_n"] + w_kids*work["kids_n"] + w_inc*work["inc_n"]
work = work.sort_values("desert_score", ascending=False).reset_index(drop=True)

st.subheader("Tract Ranking")
st.dataframe(work[["GEOID","pct_no_hs","pct_under18","median_income","total_pop","desert_score"]].round(2),
             use_container_width=True)

# --- Charts ---
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

# --- Tract choropleth ---
st.subheader("Choropleth Map (Tracts)")
tiger_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/8/query"
where = f"STATE='{STATE}' AND COUNTY='{COUNTY}'"
gparams = {"where": where, "outFields": "GEOID", "outSR": "4326", "f": "geojson"}

try:
    gj = retry_get(tiger_url, gparams).json()
except Exception as e:
    st.error(f"TIGERweb boundary download failed: {e}")
    st.stop()

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

folium.features.GeoJson(
    gj,
    name="tracts",
    style_function=lambda x: {"fillOpacity":0, "color":"#00000000", "weight":0},
    tooltip=folium.features.GeoJsonTooltip(fields=["GEOID"], aliases=["Tract GEOID:"], sticky=False),
    highlight_function=lambda x: {"weight":1, "color":"#666"}
).add_to(m)

st_folium(m, height=600, width=None)

with st.expander("Notes"):
    st.write("Key is read from Streamlit Secrets only; never shown in the UI. ACS 2022 5-year; Philadelphia tracts.")
