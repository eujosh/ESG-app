# Nigeria AI-Powered ESG Rating System — Streamlit App

## What this app does
Interactive dashboard presenting the results of the thesis:
"Developing an AI-Powered ESG Rating System for Nigeria"

## Tabs
1. **ESG Rankings** — Company league table with sector breakdown
2. **Pillar Deep-Dive** — E, S, G sub-scores, disclosure rates, heatmap
3. **Validation** — H1 Spearman correlation vs Risk Insights external ratings
4. **Trends Over Time** — ESG score evolution 2020–2024
5. **ESG vs Financial Performance** — H2 ANOVA + H3 scatter plots
6. **Why Better** — Comparison table vs Risk Insights / Sustainalytics / MSCI

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## Deploy FREE on Streamlit Cloud (recommended)
1. Create a free account at https://streamlit.io/cloud
2. Push this folder to a GitHub repository
3. Click "New app" → select your repo → set app.py as the main file
4. Click Deploy — your app gets a public URL instantly

## File structure
```
esg_app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── data/
    └── ESG_dataset_v2.csv   ← Your 250-row dataset
```

## Notes
- To add your actual financial data (ROA, ROE), populate the
  roaa_pct and roae_pct columns in ESG_dataset_v2.csv
- The app automatically re-scores whenever the CSV is updated
- For thesis defence: run locally and present from your laptop
