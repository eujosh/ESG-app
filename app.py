import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigeria AI-Powered ESG Rating System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS - Modern Dark Professional Theme ───────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { 
        background: #0A0F1C; 
        color: #E0E7FF;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: #121A2E;
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #94A3B8;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1E2937;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0F766E 100%);
        border-radius: 20px;
        padding: 48px 40px;
        margin-bottom: 32px;
        color: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.6);
    }
    
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 12px 0;
        letter-spacing: -0.02em;
    }
    
    .insight-box {
        background: #052E16;
        border-left: 5px solid #10B981;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0;
    }
    
    .warning-box {
        background: #431407;
        border-left: 5px solid #F59E0B;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0;
    }
    
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Color Palettes ────────────────────────────────────────────────────────────
SECTOR_COLORS = {
    "Banking & Finance": "#3B82F6",
    "Oil & Gas": "#F97316",
    "Consumer Goods (FMCG)": "#22C55E",
    "Telecoms & Technology": "#8B5CF6",
    "Industrial Goods": "#EC4899",
    "Other": "#64748B",
}

RATING_COLORS = {
    "Excellent": "#10B981",
    "Good": "#34D399",
    "Fair": "#FBBF24",
    "Below Average": "#FB923C",
    "Poor": "#EF4444",
}

# ── Data Loading & Processing ─────────────────────────────────────────────────
@st.cache_data
def load_and_score():
    df = pd.read_csv("data/ESG_dataset_v2.csv")

    # Clean npl_ratio_pct
    df['npl_ratio_pct'] = pd.to_numeric(
        df['npl_ratio_pct'].astype(str).str.replace(r'^\.(\d)', r'0.\1', regex=True),
        errors='coerce')

    # Clean company names
    df['company_name'] = (df['company_name']
        .str.replace('SR$', '', regex=True)
        .str.replace('S$', '', regex=True)
        .str.strip())

    name_map = {
        'Access Bank':'Access Bank','AccessBank':'Access Bank',
        'DangoteSugar':'Dangote Sugar','DangoteCementPlc':'Dangote Cement',
        'AirtelAfrica':'Airtel Africa','FCMBBank':'FCMB',
        'FidelityBank':'Fidelity Bank','FirstBank':'First Bank',
        'GuinnessNigeriaplc':'Guinness Nigeria','JuliusBerger':'Julius Berger',
        'NigerianBreweriesplc':'Nigerian Breweries','StanbicIBTCBank':'Stanbic IBTC',
        'SterlingBank':'Sterling Bank','TotalEnergies':'TotalEnergies',
        'UnionBank':'Union Bank','ZenithBank':'Zenith Bank',
        'Nestleplc':'Nestle Nigeria','Unileverplc':'Unilever Nigeria',
        'Lafargeplc':'Lafarge Africa','NasconAllied':'Nascon Allied',
        'FMNplc':'Flour Mills Nigeria','Honeywellplc':'Honeywell Flour',
        'Cadburyplc':'Cadbury Nigeria','BUACement':'BUA Cement',
        'BUAfoodsplc':'BUA Foods','Geregupowerplc':'Geregu Power',
        'EternaPlc':'Eterna','Oandoplc':'Oando','Conoilplc':'Conoil',
        'Ardovaplc':'Ardova','Aradelplc':'Aradel','MTNNG':'MTN Nigeria',
        'CWGplc':'CWG','Chamsholdco':'Chams','CHAMPIONBREWPLC':'Champion Breweries',
        'Intlbrew':'International Breweries','NNFMplc':'NNFM',
        'BETAGLASSPLC':'Beta Glass','WemaBank':'Wema Bank','JaizBank':'Jaiz Bank',
        'AXAMansard':'AXA Mansard','AIICO':'AIICO Insurance',
        'Custodianplc':'Custodian Investment','MBENEFIT':'Mutual Benefits',
        'VFDGroup':'VFD Group','PZCN':'PZ Cussons','MRS':'MRS Oil',
        'Ecobank':'Ecobank','Seplat':'Seplat Energy','UBA':'UBA',
        'MR':'MRS Oil',
    }
    df['company_name'] = df['company_name'].map(lambda x: name_map.get(x, x))

    sector_map = {
        'Access Bank':'Banking & Finance','FCMB':'Banking & Finance',
        'Fidelity Bank':'Banking & Finance','First Bank':'Banking & Finance',
        'Stanbic IBTC':'Banking & Finance','Sterling Bank':'Banking & Finance',
        'Union Bank':'Banking & Finance','Wema Bank':'Banking & Finance',
        'Zenith Bank':'Banking & Finance','Jaiz Bank':'Banking & Finance',
        'AXA Mansard':'Banking & Finance','AIICO Insurance':'Banking & Finance',
        'Custodian Investment':'Banking & Finance','Mutual Benefits':'Banking & Finance',
        'VFD Group':'Banking & Finance','Ecobank':'Banking & Finance','UBA':'Banking & Finance',
        'Seplat Energy':'Oil & Gas','TotalEnergies':'Oil & Gas','Ardova':'Oil & Gas',
        'MRS Oil':'Oil & Gas','Conoil':'Oil & Gas','Oando':'Oil & Gas',
        'Eterna':'Oil & Gas','Aradel':'Oil & Gas','Geregu Power':'Oil & Gas',
        'Nestle Nigeria':'Consumer Goods (FMCG)','Unilever Nigeria':'Consumer Goods (FMCG)',
        'Guinness Nigeria':'Consumer Goods (FMCG)','Dangote Sugar':'Consumer Goods (FMCG)',
        'BUA Foods':'Consumer Goods (FMCG)','Cadbury Nigeria':'Consumer Goods (FMCG)',
        'Flour Mills Nigeria':'Consumer Goods (FMCG)','Honeywell Flour':'Consumer Goods (FMCG)',
        'PZ Cussons':'Consumer Goods (FMCG)','Champion Breweries':'Consumer Goods (FMCG)',
        'International Breweries':'Consumer Goods (FMCG)','Nascon Allied':'Consumer Goods (FMCG)',
        'NNFM':'Consumer Goods (FMCG)','Nigerian Breweries':'Consumer Goods (FMCG)',
        'MTN Nigeria':'Telecoms & Technology','Airtel Africa':'Telecoms & Technology',
        'Chams':'Telecoms & Technology','CWG':'Telecoms & Technology',
        'Dangote Cement':'Industrial Goods','BUA Cement':'Industrial Goods',
        'Lafarge Africa':'Industrial Goods','Julius Berger':'Industrial Goods',
        'Beta Glass':'Industrial Goods',
    }
    df['Sector'] = df['company_name'].map(sector_map).fillna('Other')

    # ESG Scoring Functions
    def minmax(s):
        mn, mx = s.min(), s.max()
        if mx == mn: return s * 0
        return (s - mn) / (mx - mn)

    def minmax_inv(s): return 1 - minmax(s)

    # Environmental
    df['e_electricity'] = minmax_inv(df['electricity_kwh'])
    df['e_diesel']      = minmax_inv(df['diesel_litres'])
    df['e_ghg']         = minmax_inv(df['ghg_emissions_mtco2e'])
    df['e_ghgchg']      = minmax_inv(df['ghg_change_pct'].clip(-50, 50))
    e_cols = ['renewable_energy_flag','env_policy_flag','carbon_audit_flag',
              'gri_302_flag','gri_305_flag','e_electricity','e_diesel','e_ghg','e_ghgchg']
    df['E_score'] = df[e_cols].mean(axis=1)

    # Social
    df['s_femp']   = minmax(df['female_employee_pct'])
    df['s_train']  = minmax(df['employees_trained_hr'])
    df['s_pret']   = minmax(df['parental_leave_return_pct'])
    df['s_csr']    = minmax(df['csr_pct_pat'])
    s_cols = ['financial_inclusion_flag','gender_award_flag','ungc_signatory',
              'nsbp_signatory','human_rights_policy_flag',
              's_femp','s_train','s_pret','s_csr']
    df['S_score'] = df[s_cols].mean(axis=1)

    # Governance
    df['g_bfem']   = minmax(df['board_female_pct'])
    df['g_bsize']  = minmax(df['board_total'].clip(5, 20))
    df['g_bigfour']= df['external_auditor'].str.contains(
        'KPMG|PwC|Deloitte|EY|PricewaterHouse|Ernst', case=False, na=False).astype(float)
    g_cols = ['board_eval_flag','sustainability_assurance','gri_reporting',
              'code_of_conduct_flag','anti_corruption_flag','data_privacy_flag',
              'iso_certified','g_bfem','g_bsize','g_bigfour']
    df['G_score'] = df[g_cols].mean(axis=1)

    # Composite ESG Score (30% E, 30% S, 40% G)
    df['ESG_score'] = (df['E_score']*0.30 + df['S_score']*0.30 + df['G_score']*0.40) * 100
    df['ESG_score'] = df['ESG_score'].round(2)
    df['E_score']   = (df['E_score'] * 100).round(2)
    df['S_score']   = (df['S_score'] * 100).round(2)
    df['G_score']   = (df['G_score'] * 100).round(2)

    # Risk Insights validation
    ri_map = {
        'Zenith Bank':'Fair','Fidelity Bank':'Below Average','Wema Bank':'Fair',
        'FCMB':'Below Average','Stanbic IBTC':'Good','Union Bank':'Good',
        'Jaiz Bank':'Fair','First Bank':'Good','MRS Oil':'Below Average',
        'Eterna':'Below Average','Geregu Power':'Below Average',
        'Flour Mills Nigeria':'Good','Dangote Cement':'Good',
        'Dangote Sugar':'Good','Guinness Nigeria':'Good',
        'International Breweries':'Good','Nascon Allied':'Good',
        'PZ Cussons':'Good','MTN Nigeria':'Good','Chams':'Poor',
        'BUA Cement':'Poor','Lafarge Africa':'Below Average',
        'Julius Berger':'Below Average','Sterling Bank':'Good',
        'AXA Mansard':'Below Average','Custodian Investment':'Below Average',
        'CWG':'Below Average',
    }
    ri_num = {'Poor':1,'Below Average':2,'Fair':3,'Good':4,'Excellent':5}
    df['RI_rating'] = df['company_name'].map(ri_map)
    df['RI_numeric'] = df['RI_rating'].map(ri_num)

    return df

df = load_and_score()
companies = sorted(df['company_name'].unique())
sectors   = sorted(df['Sector'].unique())
years     = sorted(df['report_year'].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/leaf.png", width=64)
    st.markdown("### 🇳🇬 Nigeria ESG Rating System")
    st.markdown("**AI-Powered · NGX-Listed Companies · 2020–2024**")
    st.divider()

    st.subheader("Filter Dashboard")
    sel_sectors = st.multiselect("Sectors", sectors, default=sectors)
    sel_years   = st.multiselect("Years", years, default=years)
    
    st.divider()
    st.subheader("Thesis Info")
    st.markdown("""
    **Title:** Developing an AI-Powered ESG Rating System for Nigeria  
    **University:** HSE University St. Petersburg  
    **Supervisor:** Prof. M. Storchevoy  
    **Data:** 50 NGX-listed companies, 2020–2024
    """)
    st.divider()
    st.caption("© 2025 Master's Thesis Research")

# ── Filtered Data ─────────────────────────────────────────────────────────────
dff = df[df['Sector'].isin(sel_sectors) & df['report_year'].isin(sel_years)].copy()

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 Nigeria AI-Powered ESG Rating System</h1>
    <p style="font-size:1.05rem; opacity:0.9;">
        First systematic ESG rating framework for Nigerian Exchange-listed companies, 
        built using NLP and machine learning on 250 annual reports across 50 companies (2020–2024). 
        Compare AI-generated scores against Risk Insights external ratings and explore the ESG–financial performance relationship.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics Row ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Companies Rated", f"{dff['company_name'].nunique()}", "NGX-listed")
with col2:
    st.metric("Firm-Year Observations", f"{len(dff)}", "2020–2024")
with col3:
    st.metric("Avg ESG Score", f"{dff['ESG_score'].mean():.1f}", f"Std ±{dff['ESG_score'].std():.1f}")
with col4:
    validated = dff[dff['RI_rating'].notna()]['company_name'].nunique()
    st.metric("Externally Validated", f"{validated}", "Risk Insights matches")
with col5:
    top = dff.groupby('company_name')['ESG_score'].mean().idxmax()
    st.metric("Top Scorer", top[:20] + "..." if len(top) > 20 else top)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 ESG Rankings",
    "🔬 Pillar Deep-Dive",
    "✅ Validation vs External Ratings",
    "📈 Trends Over Time",
    "🏦 ESG vs Financial Performance",
    "🆚 Why Better Than Current System"
])

# TAB 1 — ESG Rankings
with tab1:
    st.subheader("AI-Generated ESG Rankings — All Companies")
    
    avg = (dff.groupby(['company_name','Sector'])
           .agg(ESG_avg=('ESG_score','mean'), E_avg=('E_score','mean'),
                S_avg=('S_score','mean'), G_avg=('G_score','mean'))
           .reset_index()
           .sort_values('ESG_avg', ascending=False)
           .round(1))

    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        fig = px.bar(
            avg, x='ESG_avg', y='company_name', color='Sector',
            color_discrete_map=SECTOR_COLORS,
            orientation='h', height=700,
            labels={'ESG_avg':'Average ESG Score (0–100)','company_name':''},
            title='Average ESG Score by Company (2020–2024)'
        )
        fig.update_layout(
            plot_bgcolor='#0F172A', paper_bgcolor='#0F172A',
            font_family='Inter', font_color="#E0E7FF",
            yaxis={'categoryorder':'total ascending'},
            xaxis=dict(range=[0,100]),
            legend=dict(orientation='h', y=-0.12)
        )
        fig.add_vline(x=avg['ESG_avg'].mean(), line_dash='dash', 
                      line_color='#94A3B8', annotation_text='Average')
        st.plotly_chart(fig, width='stretch')

    with col_r:
        st.markdown("#### Score Distribution")
        fig2 = px.histogram(
            dff, x='ESG_score', nbins=20, color='Sector',
            color_discrete_map=SECTOR_COLORS, height=280,
            title='Distribution of All ESG Scores'
        )
        fig2.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
        st.plotly_chart(fig2, width='stretch')

        st.markdown("#### Sector Averages")
        sec_avg = (dff.groupby('Sector')['ESG_score']
                   .agg(['mean','std','count'])
                   .round(1).reset_index())
        sec_avg.columns = ['Sector','Avg Score','Std Dev','N obs']
        st.dataframe(sec_avg, hide_index=True, width='stretch')

# TAB 2 — Pillar Deep-Dive
with tab2:
    st.subheader("ESG Pillar Analysis — Environmental · Social · Governance")

    col_a, col_b = st.columns(2)

    with col_a:
        sec_pillars = (dff.groupby('Sector')[['E_score','S_score','G_score']]
                       .mean().reset_index().round(1))
        fig_rad = go.Figure()
        categories = ['Environmental','Social','Governance']
        for _, row in sec_pillars.iterrows():
            vals = [row['E_score'], row['S_score'], row['G_score']]
            vals_closed = vals + [vals[0]]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=categories + [categories[0]],
                fill='toself', name=row['Sector'],
                line_color=SECTOR_COLORS.get(row['Sector'], '#666')
            ))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            title='Average Pillar Scores by Sector',
            plot_bgcolor='#0F172A', paper_bgcolor='#0F172A',
            height=420
        )
        st.plotly_chart(fig_rad, width='stretch')

    with col_b:
        avg_pillars = (dff.groupby('company_name')[['E_score','S_score','G_score']]
                       .mean().round(1).reset_index()
                       .sort_values('G_score', ascending=False).head(20))
        fig_heat = px.imshow(
            avg_pillars.set_index('company_name')[['E_score','S_score','G_score']].T,
            color_continuous_scale='RdYlGn',
            zmin=0, zmax=100,
            title='Pillar Heatmap — Top 20 Companies',
            height=420
        )
        fig_heat.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
        st.plotly_chart(fig_heat, width='stretch')

    # Disclosure Rate
    st.markdown("#### Disclosure Rate by Indicator")
    flag_cols = {
        'renewable_energy_flag':'Renewable Energy',
        'env_policy_flag':'Environmental Policy',
        'carbon_audit_flag':'Carbon Audit',
        'gri_302_flag':'GRI 302 (Energy)',
        'gri_305_flag':'GRI 305 (Emissions)',
        'financial_inclusion_flag':'Financial Inclusion',
        'gender_award_flag':'Gender Award',
        'ungc_signatory':'UN Global Compact',
        'nsbp_signatory':'NSBP Signatory',
        'human_rights_policy_flag':'Human Rights Policy',
        'board_eval_flag':'Board Evaluation',
        'sustainability_assurance':'Sust. Assurance',
        'gri_reporting':'GRI Reporting',
        'code_of_conduct_flag':'Code of Conduct',
        'anti_corruption_flag':'Anti-Corruption Policy',
        'data_privacy_flag':'Data Privacy Policy',
        'iso_certified':'ISO Certified',
    }
    rates = {label: dff[col].mean()*100 for col, label in flag_cols.items() if col in dff.columns}
    rates_df = pd.DataFrame({'Indicator': list(rates.keys()), 'Disclosure Rate (%)': list(rates.values())})
    rates_df = rates_df.sort_values('Disclosure Rate (%)', ascending=False)
    rates_df['Pillar'] = rates_df['Indicator'].map(lambda x: 
        'Environmental' if any(k in x for k in ['Renewable','Environmental','Carbon','GRI 302','GRI 305']) else
        'Governance' if any(k in x for k in ['Board','Sust.','GRI Reporting','Code','Anti-Corruption','Data Privacy','ISO']) else 'Social')

    fig_bar = px.bar(
        rates_df, x='Disclosure Rate (%)', y='Indicator',
        color='Pillar', orientation='h', height=480,
        color_discrete_map={'Environmental':'#22C55E', 'Social':'#3B82F6', 'Governance':'#8B5CF6'},
        title='% of Company-Year Reports Disclosing Each Indicator'
    )
    fig_bar.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
    st.plotly_chart(fig_bar, width='stretch')

# TAB 3 — Validation vs External Ratings
with tab3:
    st.subheader("H1 Validation — AI Scores vs Risk Insights External Ratings")

    val_df = (df[df['RI_rating'].notna()]
              .groupby('company_name')
              .agg(AI_score=('ESG_score','mean'),
                   RI_numeric=('RI_numeric','first'),
                   RI_rating=('RI_rating','first'),
                   Sector=('Sector','first'))
              .reset_index().round(2))

    rho, pval = stats.spearmanr(val_df['AI_score'], val_df['RI_numeric'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Spearman ρ", f"{rho:.3f}", "✅ Acceptable (≥0.40)" if rho >= 0.4 else "⚠️ Below threshold")
    with col2:
        st.metric("p-value", f"{pval:.4f}", "✅ Significant (p<0.05)" if pval < 0.05 else "Not significant")
    with col3:
        st.metric("Validation Companies", len(val_df))

    if rho >= 0.4 and pval < 0.05:
        st.markdown(f"""<div class="insight-box">
        ✅ <strong>H1 Supported:</strong> The AI-generated ESG scores demonstrate statistically significant 
        convergent validity with Risk Insights external ratings (Spearman ρ = {rho:.3f}, p = {pval:.4f}).
        </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig_scat = px.scatter(
            val_df, x='RI_numeric', y='AI_score',
            color='Sector', color_discrete_map=SECTOR_COLORS,
            height=480,
            labels={'RI_numeric':'Risk Insights Rating (1=Poor → 5=Excellent)', 'AI_score':'AI ESG Score (0–100)'},
            title=f'AI Score vs Risk Insights Rating (ρ = {rho:.3f})'
        )
        # Trend line
        x_range = np.linspace(val_df['RI_numeric'].min(), val_df['RI_numeric'].max(), 50)
        m, b = np.polyfit(val_df['RI_numeric'], val_df['AI_score'], 1)
        fig_scat.add_trace(go.Scatter(x=x_range, y=m*x_range+b, mode='lines', 
                                      line=dict(dash='dash', color='#94A3B8'), name='Trend'))
        fig_scat.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
        st.plotly_chart(fig_scat, width='stretch')

    with col_r:
        st.markdown("#### AI Score vs External Rating Table")
        tbl = val_df[['company_name','Sector','AI_score','RI_rating','RI_numeric']].copy()
        tbl.columns = ['Company','Sector','AI Score','RI Rating','RI Numeric']
        tbl = tbl.sort_values('AI Score', ascending=False)
        st.dataframe(tbl, hide_index=True, width='stretch')

# TAB 4 — Trends Over Time
with tab4:
    st.subheader("ESG Score Trends 2020–2024")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        selected_cos = st.multiselect(
            "Select companies to track",
            companies,
            default=['Zenith Bank', 'MTN Nigeria', 'Dangote Cement', 'Seplat Energy', 'Nestle Nigeria']
        )
    with col_r:
        pillar_choice = st.radio("Show pillar", ['ESG_score','E_score','S_score','G_score'],
                                 horizontal=True,
                                 format_func=lambda x: x.replace('_score','').upper())

    if selected_cos:
        trend_df = (df[df['company_name'].isin(selected_cos)]
                    .groupby(['company_name','report_year'])[pillar_choice]
                    .mean().reset_index())
        fig_line = px.line(
            trend_df, x='report_year', y=pillar_choice,
            color='company_name', markers=True, height=420,
            title=f'{pillar_choice.replace("_score","").upper()} Score Trend by Company'
        )
        fig_line.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
        st.plotly_chart(fig_line, width='stretch')

    st.markdown("#### Sector Average ESG Score Over Time")
    sec_trend = (dff.groupby(['Sector','report_year'])['ESG_score']
                 .mean().reset_index().round(1))
    fig_sec = px.line(
        sec_trend, x='report_year', y='ESG_score',
        color='Sector', markers=True, height=380,
        color_discrete_map=SECTOR_COLORS,
        title='Sector ESG Score Trends 2020–2024'
    )
    fig_sec.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
    st.plotly_chart(fig_sec, width='stretch')

# TAB 5 — ESG vs Financial Performance
with tab5:
    st.subheader("H3 — ESG Score vs Financial Performance")

    st.info("Financial data availability is limited in this dataset. The charts below show available data points.")

    fin_df = df[df['total_assets_ngn_mn'].notna()].copy()
    fin_df['Log_Assets'] = np.log(fin_df['total_assets_ngn_mn'].clip(1))

    col_l, col_r = st.columns(2)
    with col_l:
        fig_assets = px.scatter(
            fin_df, x='ESG_score', y='Log_Assets',
            color='Sector', color_discrete_map=SECTOR_COLORS,
            trendline='ols', height=380,
            title='ESG Score vs Firm Size (Log Total Assets)'
        )
        fig_assets.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
        st.plotly_chart(fig_assets, width='stretch')

    with col_r:
        roae_df = df[df['roae_pct'].notna()].copy()
        roae_df['roae_pct'] = pd.to_numeric(roae_df['roae_pct'], errors='coerce')
        roae_df = roae_df[roae_df['roae_pct'].notna() & (roae_df['roae_pct'].abs() < 200)]
        if len(roae_df) > 3:
            fig_roe = px.scatter(
                roae_df, x='ESG_score', y='roae_pct',
                color='Sector', color_discrete_map=SECTOR_COLORS,
                trendline='ols', height=380,
                title='ESG Score vs ROE (%)'
            )
            fig_roe.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
            st.plotly_chart(fig_roe, width='stretch')

    st.markdown("#### H2 — Sector ESG Score Comparison (ANOVA)")
    fig_box = px.box(
        dff, x='Sector', y='ESG_score', color='Sector',
        color_discrete_map=SECTOR_COLORS,
        points='all', height=400,
        title='ESG Score Distribution by Sector'
    )
    fig_box.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
    st.plotly_chart(fig_box, width='stretch')

# TAB 6 — Why Better Than Current System
with tab6:
    st.subheader("Why This System is Better Than What Currently Exists in Nigeria")

    st.markdown("""
    Nigeria currently has no official, systematic ESG rating system for all NGX-listed companies. 
    Below is a direct comparison with the closest alternatives.
    """)

    comparison = {
        'Dimension': [
            'Nigerian Company Coverage', 'Methodology Transparency', 'Data Source',
            'Cost to Access', 'Years of Coverage', 'ESG Pillar Detail',
            'Contextual Relevance (Nigeria)', 'Academic Validation',
            'Reproducibility', 'Updatability'
        ],
        'This AI System (Thesis)': [
            '50 NGX-listed firms ✅', 'Fully documented (42 indicators) ✅',
            'Annual reports + news + governance coding ✅', 'Free / open-source ✅',
            '2020–2024 (5 years) ✅', 'E/S/G sub-scores + 42 indicators ✅',
            'Adapted to Nigerian regulatory context ✅', 'Validated against Risk Insights ✅',
            'Full Python codebase available ✅', 'Re-run annually ✅'
        ],
        'Risk Insights ESG GPS': [
            '91 firms (paid) ⚠️', 'Proprietary / not published ❌', 'Undisclosed ❌',
            'Paid subscription ❌', 'Annual (limited history) ⚠️', 'Single rating only ⚠️',
            'African-focused ✅', 'No peer-reviewed validation ❌', 'Black-box ❌',
            'Commercial update cycle ⚠️'
        ],
        'Sustainalytics / MSCI': [
            '5 / 0 firms ❌', 'Partially documented ⚠️', 'Multiple sources ✅',
            'Expensive institutional subscription ❌', 'Annual (for covered firms) ⚠️',
            'Detailed (for covered firms) ✅', 'Global framework, not Nigeria-specific ❌',
            'Peer-reviewed globally ✅', 'Not reproducible ❌', 'Commercial update cycle ⚠️'
        ],
    }
    comp_df = pd.DataFrame(comparison)
    st.dataframe(comp_df, hide_index=True, width='stretch')

    st.markdown("#### Coverage Gap — The Problem This Research Solves")
    coverage = pd.DataFrame({
        'System': ['This AI System', 'Risk Insights ESG GPS', 'Sustainalytics', 'MSCI'],
        'Nigerian Companies Rated': [50, 91, 5, 0]
    })
    fig_cov = px.bar(
        coverage, x='System', y='Nigerian Companies Rated',
        color='System', text='Nigerian Companies Rated',
        color_discrete_sequence=['#0F766E', '#3B82F6', '#8B5CF6', '#EC4899'],
        title='Number of Nigerian Companies Rated per System'
    )
    fig_cov.update_layout(plot_bgcolor='#0F172A', paper_bgcolor='#0F172A')
    st.plotly_chart(fig_cov, width='stretch')

    st.markdown("""
    <div class="insight-box">
    <strong>Key Contribution:</strong> This research fills a critical gap by providing a transparent, 
    reproducible, Nigeria-specific AI ESG rating framework that covers 50 NGX-listed companies 
    with full 2020–2024 panel data — freely accessible and academically validated.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("Built as part of a Master's Thesis at HSE University St. Petersburg • 2025")
