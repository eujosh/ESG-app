"""
AI-Powered ESG Rating System — Nigeria
Master's Thesis | HSE University St. Petersburg
Supervisor: Prof. Maxim Storchevoy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigeria AI-Powered ESG Rating",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern Dark Professional CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0A0F1C; }
    
    .hero {
        background: linear-gradient(135deg, #0A1F3D 0%, #0B4A6B 50%, #0A6E5E 100%);
        border-radius: 20px;
        padding: 42px 48px;
        color: white;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px rgba(10, 111, 94, 0.18);
    }
    .hero h1 { font-size: 2.15rem; font-weight: 700; margin: 0 0 14px 0; }
    .hero p { font-size: 1.03rem; opacity: 0.92; line-height: 1.65; }

    .stMetric { 
        background: #1A2338; 
        border-radius: 16px; 
        padding: 20px 18px; 
        border: 1px solid #2A3550;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: #1A2338;
        border-radius: 14px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 26px;
        font-weight: 600;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background: #0A6E5E;
        color: white;
        box-shadow: 0 4px 15px rgba(10, 110, 94, 0.35);
    }

    .insight {
        background: #1A3C2E;
        border-left: 5px solid #10B981;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 18px 0;
        color: #D1FAE5;
    }
    .warning {
        background: #3F2A1E;
        border-left: 5px solid #F59E0B;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 18px 0;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #E2E8F0;
        margin: 28px 0 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Color Palettes ─────────────────────────────────────────────────────────────
SECTOR_COLORS = {
    "Banking & Finance": "#0EA5E9",
    "Oil & Gas": "#F97316",
    "Consumer Goods": "#22C55E",
    "Telecoms & Technology": "#8B5CF6",
    "Industrial Goods": "#EC4899",
}

GRADE_COLORS = {
    "A": "#10B981", "B": "#34D399", "C": "#FBBF24", 
    "D": "#FB923C", "F": "#EF4444"
}

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/esg_final_dataset.csv")
    df["year_std"] = pd.to_numeric(df["year_std"], errors="coerce").astype(int)
    df["Sector"] = df["Sector"].astype(str)
    
    numeric = [
        "ESG_Score_Composite","Env_Score_Raw","Social_Score_Raw","Governance_Score_Raw",
        "ROA","ROE","Log_Assets","RI_Numeric","Sust_Numeric","Female_Board_Pct",
        "Independent_Dir_Pct","Board_Meetings_pa","CEO_Chair_Separation",
        "renewable_energy_flag","env_policy_flag","carbon_audit_flag",
        "gri_302_flag","gri_305_flag","financial_inclusion_flag","gender_award_flag",
        "ungc_signatory","nsbp_signatory","human_rights_policy_flag","board_eval_flag",
        "sustainability_assurance","code_of_conduct_flag","anti_corruption_flag",
        "data_privacy_flag","iso_certified","Anti_Corruption_Policy",
        "GRI_Reporting","IFRS_S1_S2_Aligned"
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    ri_map = {"Poor":1,"Below Average":2,"Fair":3,"Good":4,"Excellent":5}
    if "RI_Rating" in df.columns:
        df["RI_Numeric"] = df["RI_Rating"].map(ri_map)
    return df

df = load_data()
companies = sorted(df["company_std"].dropna().unique())
sectors   = sorted(df["Sector"].dropna().unique())
years     = sorted(df["year_std"].unique())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/leaf.png", width=68)
    st.markdown("### 🌿 Nigeria ESG Rating System")
    st.caption("**AI-Powered** • 50 NGX Companies • 2020–2024")
    
    st.divider()
    st.markdown("**Filters**")
    sel_sectors = st.multiselect("Sectors", sectors, default=sectors)
    sel_years   = st.multiselect("Years", years, default=years)
    
    st.divider()
    st.markdown("**Thesis Information**")
    st.caption("""
    **Title:** Developing an AI-Powered ESG Rating System: A Case Study of Nigeria  
    **University:** HSE University St. Petersburg  
    **Supervisor:** Prof. Maxim Storchevoy  
    **Programme:** MSc Data Analytics for Business and Economics
    """)
    st.caption("© 2026 Group Thesis Research")

# ── Filtered Data ──────────────────────────────────────────────────────────────
dff = df[df["Sector"].isin(sel_sectors) & df["year_std"].isin(sel_years)].copy()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌿 AI-Powered ESG Rating System — Nigeria</h1>
  <p>First systematic, transparent and externally validated ESG rating framework for Nigerian
  Exchange-listed companies. Built using NLP and machine learning on 241 firm-year observations
  across 50 companies (2020–2024). Validated against Risk Insights (Spearman ρ = 0.466, p = 0.014).</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Companies", int(dff["company_std"].nunique()), "NGX-listed")
with k2: st.metric("Observations", len(dff), "firm-year")
with k3:
    m = dff["ESG_Score_Composite"].mean()
    s = dff["ESG_Score_Composite"].std()
    st.metric("Avg ESG Score", f"{m:.1f}/100", f"±{s:.1f}")
with k4:
    n_val = int(dff[dff["RI_Rating"].notna()]["company_std"].nunique()) if "RI_Rating" in dff.columns else 0
    st.metric("Externally Validated", f"{n_val} firms", "Risk Insights")
with k5:
    top = dff.groupby("company_std")["ESG_Score_Composite"].mean().idxmax()
    st.metric("Top Scorer", top[:22] + "..." if len(top) > 22 else top)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 ESG Rankings",
    "📋 Descriptive Stats",
    "🔬 Pillar Analysis",
    "✅ H1 — Convergent Validity",
    "🏭 H2 — Sector Variation",
    "💰 H3 — Financial Performance",
    "📈 Trends & Disclosure"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ESG Rankings
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Company ESG Rankings — Average 2020–2024</p>', unsafe_allow_html=True)
    
    avg = (dff.groupby(["company_std","Sector"])
           .agg(ESG=("ESG_Score_Composite","mean"),
                E=("Env_Score_Raw","mean"),
                S=("Social_Score_Raw","mean"),
                G=("Governance_Score_Raw","mean"),
                Grade=("ESG_Rating_Grade", lambda x: x.mode()[0] if len(x)>0 else ""))
           .reset_index().round(1).sort_values("ESG", ascending=False))

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig = px.bar(avg, x="ESG", y="company_std", color="Sector",
                     color_discrete_map=SECTOR_COLORS, orientation="h", height=720,
                     title="Average ESG Score by Company")
        mean_val = avg["ESG"].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#64748B",
                      annotation_text=f"Mean: {mean_val:.1f}")
        fig.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0",
                          xaxis=dict(range=[0,100], gridcolor="#334155"),
                          yaxis=dict(categoryorder="total ascending"),
                          legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, width='stretch')

    with col_r:
        st.markdown("#### ESG Grade Distribution")
        grade_counts = dff["ESG_Rating_Grade"].value_counts().reindex(["A","B","C","D","F"]).fillna(0)
        fig_g = px.bar(x=grade_counts.index, y=grade_counts.values,
                       color=grade_counts.index, color_discrete_map=GRADE_COLORS,
                       text=grade_counts.values.astype(int), height=260,
                       title="Grade Distribution")
        fig_g.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0", showlegend=False)
        st.plotly_chart(fig_g, width='stretch')

        st.markdown("#### Full Company Table")
        display = avg[["company_std","Sector","ESG","E","S","G","Grade"]].copy()
        display.columns = ["Company","Sector","ESG","E","S","G","Grade"]
        st.dataframe(display, hide_index=True, width='stretch',
                     column_config={"ESG": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f")})

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Descriptive Stats
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Descriptive Statistics</p>', unsafe_allow_html=True)

    score_cols = ["ESG_Score_Composite","Env_Score_Raw","Social_Score_Raw","Governance_Score_Raw"]
    labels = ["Composite ESG","Environmental","Social","Governance"]
    desc = df[score_cols].describe().T
    desc.index = labels
    desc = desc[["count","mean","std","min","25%","50%","75%","max"]].round(2)
    desc.columns = ["N","Mean","Std Dev","Min","Q1","Median","Q3","Max"]
    st.dataframe(desc, width='stretch')

    st.markdown("""
    <div class="insight">
    📌 <strong>Key finding:</strong> Governance pillar leads (mean = 53.02), followed by Social (34.78) 
    and Environmental (25.25). This reflects stronger regulatory maturity in governance disclosures in Nigeria.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(dff, x="ESG_Score_Composite", nbins=20, color="Sector",
                                color_discrete_map=SECTOR_COLORS, height=380,
                                title="Distribution of ESG Scores")
        fig_hist.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0")
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        fig_box = px.box(dff, x="ESG_Rating_Grade", y="ESG_Score_Composite",
                         color="ESG_Rating_Grade", color_discrete_map=GRADE_COLORS,
                         category_orders={"ESG_Rating_Grade": ["A", "B", "C", "D", "F"]},
                         height=380, title="Score Distribution by Grade")
        fig_box.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0", showlegend=False)
        st.plotly_chart(fig_box, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Pillar Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">ESG Pillar Analysis — Environmental · Social · Governance</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        sec_pil = (dff.groupby("Sector")[["Env_Score_Raw","Social_Score_Raw","Governance_Score_Raw"]]
                   .mean().reset_index().round(1))
        cats = ["Environmental","Social","Governance"]
        fig_rad = go.Figure()
        for _, row in sec_pil.iterrows():
            vals = [row["Env_Score_Raw"], row["Social_Score_Raw"], row["Governance_Score_Raw"]]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=row["Sector"],
                line_color=SECTOR_COLORS.get(row["Sector"], "#666"), opacity=0.75))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            title="Average Pillar Scores by Sector",
            height=440, plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_rad, width='stretch')

    with col_b:
        pillar_sector = (dff.groupby("Sector")[["Env_Score_Raw","Social_Score_Raw","Governance_Score_Raw"]]
                         .mean().round(1))
        fig_heat = px.imshow(pillar_sector, color_continuous_scale="RdYlGn",
                             zmin=0, zmax=80, text_auto=".1f", height=440,
                             title="Pillar Heatmap by Sector")
        fig_heat.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0")
        st.plotly_chart(fig_heat, width='stretch')

    # Disclosure Rates
    st.markdown("#### Indicator Disclosure Rates")
    indicator_map = {
        "renewable_energy_flag":   ("Renewable Energy Policy",  "Environmental"),
        "env_policy_flag":         ("Environmental Policy",      "Environmental"),
        "carbon_audit_flag":       ("Carbon Audit",              "Environmental"),
        "gri_302_flag":            ("GRI 302 (Energy)",          "Environmental"),
        "gri_305_flag":            ("GRI 305 (Emissions)",       "Environmental"),
        "financial_inclusion_flag":("Financial Inclusion",       "Social"),
        "gender_award_flag":       ("Gender Award",              "Social"),
        "ungc_signatory":          ("UN Global Compact",         "Social"),
        "nsbp_signatory":          ("NSBP Signatory",            "Social"),
        "human_rights_policy_flag":("Human Rights Policy",       "Social"),
        "board_eval_flag":         ("Board Evaluation",          "Governance"),
        "sustainability_assurance":("Sustainability Assurance",  "Governance"),
        "GRI_Reporting":           ("GRI Reporting",             "Governance"),
        "code_of_conduct_flag":    ("Code of Conduct",           "Governance"),
        "anti_corruption_flag":    ("Anti-Corruption Policy",    "Governance"),
        "data_privacy_flag":       ("Data Privacy Policy",       "Governance"),
        "iso_certified":           ("ISO Certification",         "Governance"),
        "CEO_Chair_Separation":    ("CEO-Chair Separation",      "Governance"),
    }

    rows, seen = [], set()
    for col, (label, pillar) in indicator_map.items():
        if col in dff.columns and label not in seen:
            rate = pd.to_numeric(dff[col], errors="coerce").mean() * 100
            if not np.isnan(rate):
                rows.append({"Indicator": label, "Pillar": pillar, "Disclosure Rate (%)": round(rate, 1)})
                seen.add(label)

    disc_df = pd.DataFrame(rows).sort_values("Disclosure Rate (%)", ascending=False)
    
    fig_disc = px.bar(disc_df, x="Disclosure Rate (%)", y="Indicator",
                      color="Pillar", orientation="h", height=520,
                      color_discrete_map={"Environmental":"#22C55E", "Social":"#0EA5E9", "Governance":"#8B5CF6"},
                      title="% of Firm-Year Observations Disclosing Each Indicator")
    fig_disc.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0",
                           yaxis={"categoryorder": "total ascending"},
                           xaxis=dict(range=[0, 110], gridcolor="#334155"))
    fig_disc.add_vline(x=50, line_dash="dash", line_color="#64748B", annotation_text="50%")
    st.plotly_chart(fig_disc, width='stretch')

    st.markdown("""
    <div class="warning">
    ⚠️ <strong>Key finding:</strong> High disclosure in governance (Code of Conduct, Board Evaluation) due to SEC requirements, 
    but critically low in environmental metrics (Carbon Audit ~1%, GRI 305 ~8%).
    </div>
    """, unsafe_allow_html=True)
    # ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — H1 Convergent Validity
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">H1 — Convergent Validity: AI Scores vs Risk Insights</p>', unsafe_allow_html=True)

    val_df = (df[df["RI_Rating"].notna()]
              .groupby("company_std")
              .agg(AI_Score=("ESG_Score_Composite","mean"),
                   RI_Numeric=("RI_Numeric","first"),
                   RI_Rating=("RI_Rating","first"),
                   Sector=("Sector","first"))
              .reset_index()
              .dropna(subset=["AI_Score","RI_Numeric"])
              .round(2))

    rho, pval = stats.spearmanr(val_df["AI_Score"], val_df["RI_Numeric"])

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Spearman ρ", f"{rho:.3f}", 
                       "✅ ≥ 0.40 threshold met" if rho >= 0.40 else "⚠️ Below threshold")
    with m2: st.metric("p-value", f"{pval:.4f}", 
                       "✅ Significant (p<0.05)" if pval < 0.05 else "Not significant")
    with m3: st.metric("N Companies", len(val_df), "matched")
    with m4: st.metric("Directional Consistency", "74%", "20 of 27 companies")

    verdict = "✅ H1 SUPPORTED" if rho >= 0.40 and pval < 0.05 else "⚠️ H1 NOT SUPPORTED"
    box_class = "insight" if rho >= 0.40 and pval < 0.05 else "warning"
    st.markdown(f"""
    <div class="{box_class}">
    <strong>{verdict}:</strong> Spearman ρ = {rho:.3f} (p = {pval:.4f}, N = {len(val_df)}). 
    The AI-generated ESG scores show statistically significant convergent validity with Risk Insights external ratings.
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        x_range = np.linspace(val_df["RI_Numeric"].min(), val_df["RI_Numeric"].max(), 50)
        m_fit, b_fit = np.polyfit(val_df["RI_Numeric"], val_df["AI_Score"], 1)
        
        fig_sc = px.scatter(val_df, x="RI_Numeric", y="AI_Score",
                            color="Sector", color_discrete_map=SECTOR_COLORS,
                            text="company_std", height=500,
                            title=f"AI Score vs Risk Insights Rating (ρ = {rho:.3f})")
        fig_sc.add_trace(go.Scatter(x=x_range, y=m_fit*x_range + b_fit,
                                    mode="lines", line=dict(dash="dash", color="#94A3B8"), name="Trend"))
        fig_sc.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0",
                             xaxis=dict(tickvals=[1,2,3,4,5],
                                        ticktext=["Poor","Below Avg","Fair","Good","Excellent"]))
        st.plotly_chart(fig_sc, width='stretch')

    with col_r:
        st.markdown("#### Validation Companies")
        tbl = val_df[["company_std","AI_Score","RI_Rating"]].sort_values("AI_Score", ascending=False)
        tbl.columns = ["Company", "AI Score", "RI Rating"]
        st.dataframe(tbl, hide_index=True, width='stretch',
                     column_config={"AI Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f")})

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — H2 Sector Variation
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">H2 — Sectoral Variation in ESG Scores</p>', unsafe_allow_html=True)

    groups = [dff[dff["Sector"] == s]["ESG_Score_Composite"].dropna().values
              for s in dff["Sector"].unique()
              if len(dff[dff["Sector"] == s]["ESG_Score_Composite"].dropna()) >= 3]
    
    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        a1, a2, a3 = st.columns(3)
        with a1: st.metric("ANOVA F-statistic", f"{f_stat:.3f}")
        with a2: st.metric("p-value", f"{p_anova:.5f}", 
                           "✅ H2 Supported" if p_anova < 0.05 else "Not significant")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig_box = px.box(dff, x="Sector", y="ESG_Score_Composite",
                         color="Sector", color_discrete_map=SECTOR_COLORS,
                         points="all", height=420, title="ESG Score Distribution by Sector")
        fig_box.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0", showlegend=False)
        st.plotly_chart(fig_box, width='stretch')

    with col_r:
        sec_stats = (dff.groupby("Sector")
                     .agg(ESG_Mean=("ESG_Score_Composite","mean"),
                          E=("Env_Score_Raw","mean"),
                          S=("Social_Score_Raw","mean"),
                          G=("Governance_Score_Raw","mean"),
                          N=("ESG_Score_Composite","count"))
                     .round(1).sort_values("ESG_Mean", ascending=False).reset_index())
        sec_stats.columns = ["Sector", "ESG Mean", "E", "S", "G", "N obs"]
        st.dataframe(sec_stats, hide_index=True, width='stretch',
                     column_config={"ESG Mean": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f")})

    st.markdown(f"""
    <div class="insight">
    ✅ <strong>H2 Supported:</strong> Significant differences exist across sectors (ANOVA p < 0.001). 
    Telecoms & Technology and Banking & Finance lead, while Insurance and Industrial Goods lag.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — H3 Financial Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-header">H3 — ESG Score and Financial Performance</p>', unsafe_allow_html=True)

    h3_roe = df[["ESG_Score_Composite","ROE","Sector"]].dropna()
    h3_roa = df[["ESG_Score_Composite","ROA","Sector"]].dropna()
    rho_roe, p_roe = stats.spearmanr(h3_roe["ESG_Score_Composite"], h3_roe["ROE"])
    rho_roa, p_roa = stats.spearmanr(h3_roa["ESG_Score_Composite"], h3_roa["ROA"])

    b1, b2, b3, b4 = st.columns(4)
    with b1: st.metric("ESG vs ROE (ρ)", f"{rho_roe:.3f}")
    with b2: st.metric("ROE p-value", f"{p_roe:.4f}", f"N = {len(h3_roe)}")
    with b3: st.metric("ESG vs ROA (ρ)", f"{rho_roa:.3f}")
    with b4: st.metric("ROA p-value", f"{p_roa:.4f}", f"N = {len(h3_roa)}")

    col_l, col_r = st.columns(2)
    with col_l:
        fig_roe = px.scatter(h3_roe, x="ESG_Score_Composite", y="ROE",
                             color="Sector", color_discrete_map=SECTOR_COLORS,
                             trendline="ols", height=400,
                             title=f"ESG Score vs ROE (ρ={rho_roe:.3f})")
        fig_roe.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0")
        st.plotly_chart(fig_roe, width='stretch')

    with col_r:
        fig_roa = px.scatter(h3_roa, x="ESG_Score_Composite", y="ROA",
                             color="Sector", color_discrete_map=SECTOR_COLORS,
                             trendline="ols", height=400,
                             title=f"ESG Score vs ROA (ρ={rho_roa:.3f})")
        fig_roa.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0")
        st.plotly_chart(fig_roa, width='stretch')

    # Pillar correlations with ROE
    st.markdown("#### Pillar Sub-score Correlations with ROE")
    pillar_rows = []
    for col,nm in [("Env_Score_Raw","Environmental"),("Social_Score_Raw","Social"),
                    ("Governance_Score_Raw","Governance"),("ESG_Score_Composite","Composite ESG")]:
        tmp = df[[col,"ROE"]].dropna()
        r,p = stats.spearmanr(tmp[col],tmp["ROE"])
        stars = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        pillar_rows.append({"Pillar/Score":nm,"Spearman ρ":round(r,3),
                             "p-value":round(p,4),"Sig":stars,"N":len(tmp)})
    st.dataframe(pd.DataFrame(pillar_rows), hide_index=True, width='stretch')

    st.markdown(f"""
    <div class="warning">
    ⚠️ <strong>H3 — Partially Supported:</strong> Significant positive correlation with ROE 
    (ρ = {rho_roe:.3f}, p = {p_roe:.4f}), but not with ROA. Larger firms tend to have higher ESG scores.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Trends & Disclosure
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<p class="section-header">Temporal Trends & Disclosure Patterns</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        sel_cos = st.multiselect("Track specific companies", companies,
                                 default=["MTN Nigeria", "Zenith Bank", "Dangote Cement", "Seplat Energy"])
    with col_r:
        pillar_choice = st.radio("Pillar to display", 
                                 ["ESG_Score_Composite", "Env_Score_Raw", "Social_Score_Raw", "Governance_Score_Raw"],
                                 horizontal=True,
                                 format_func=lambda x: {"ESG_Score_Composite":"Overall ESG",
                                                        "Env_Score_Raw":"Environmental",
                                                        "Social_Score_Raw":"Social",
                                                        "Governance_Score_Raw":"Governance"}[x])

    if sel_cos:
        trend_df = (df[df["company_std"].isin(sel_cos)]
                    .groupby(["company_std", "year_std"])[pillar_choice]
                    .mean().reset_index())
        fig_line = px.line(trend_df, x="year_std", y=pillar_choice, color="company_std",
                           markers=True, height=400, title="Selected Companies ESG Trend")
        fig_line.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0")
        st.plotly_chart(fig_line, width='stretch')

    st.markdown("#### Sector Average ESG Score Trends")
    sec_trend = (dff.groupby(["Sector", "year_std"])["ESG_Score_Composite"]
                 .mean().reset_index().round(1))
    fig_sec = px.line(sec_trend, x="year_std", y="ESG_Score_Composite", color="Sector",
                      color_discrete_map=SECTOR_COLORS, markers=True, height=400,
                      title="Sector ESG Score Trends 2020–2024")
    fig_sec.update_layout(plot_bgcolor="#1A2338", paper_bgcolor="#1A2338", font_color="#E2E8F0")
    st.plotly_chart(fig_sec, width='stretch')

    # Year summary
    year_trends = (df.groupby("year_std")[["ESG_Score_Composite","Env_Score_Raw",
                                            "Social_Score_Raw","Governance_Score_Raw"]]
                   .mean().round(2).reset_index())
    year_trends.columns = ["Year","Composite ESG","Environmental","Social","Governance"]
    st.dataframe(year_trends, hide_index=True, width='stretch')

    net = year_trends["Composite ESG"].iloc[-1] - year_trends["Composite ESG"].iloc[0]
    st.markdown(f"""
    <div class="insight">
    📈 <strong>Key Trend:</strong> Mean ESG score increased from 37.39 (2020) to 42.42 (2024), 
    a net gain of +{net:.2f} points. Governance continues to be the strongest pillar.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption("Built as part of MSc Thesis Research • HSE University St. Petersburg • 2026")
