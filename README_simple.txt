# Education Desert — Tract Dashboard (Streamlit)

A minimal Streamlit app that queries ACS by **tract** for a selected county, computes an
Education Desert score, and renders a tract-level choropleth. No addresses required.

## Run locally
```bash
pip install -r requirements_simple.txt
streamlit run app_simple.py
```

## Deploy (Streamlit Community Cloud)
1. Push `app_simple.py` and `requirements_simple.txt` to GitHub.
2. Deploy with main file = `app_simple.py`.
3. In *Settings → Secrets*, add:
```
CENSUS_API_KEY = "your_key_here"
```

## Customize
- Change default FIPS (sidebar) or hardcode your target county.
- Add more indicators (e.g., poverty, unemployment) and extend the score.
- Swap TIGERweb for local shapefiles if you need offline mapping.