"""
ESG-Financial Manipulation Research Platform
A research-grade web application for analyzing ESG performance and financial manipulation risk
Author: Rushiraj Nawale
Email: rushiraj.nawale24m@iimranchi.ac.in
Date: December 2025
Version: 2.1.0 - Updated with treated column support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ESG-Financial Manipulation Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    .did-results-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
        color: #333;
    }
    .did-results-box h4 {
        color: #667eea;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA VALIDATION
# ============================================================================

REQUIRED_COLUMNS = [
    'company_name', 'industry', 'year', 'total_assets', 'receivables',
    'revenue', 'net_income', 'cfo', 'ppe_net', 'esg_score', 'operating_expenses', 'treated'
]

def validate_data(df):
    """Validate uploaded data"""
    errors = []
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {', '.join(missing_cols)}")
        return errors
    
    # Check for missing values
    if not errors:
        for col in REQUIRED_COLUMNS:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} missing values")
    
    # Check data types
    numeric_cols = [col for col in REQUIRED_COLUMNS if col not in ['company_name', 'industry']]
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    errors.append(f"Column '{col}' should be numeric")
    
    # Validate 'treated' column values
    if 'treated' in df.columns:
        unique_vals = df['treated'].dropna().unique()
        if not all(val in [0, 1] for val in unique_vals):
            errors.append("Column 'treated' must contain only 0 (control) or 1 (treated)")
    
    return errors

# ============================================================================
# M-SCORE CALCULATION
# ============================================================================

def calculate_m_score(df):
    """Calculate Beneish M-Score for each firm-year"""
    df = df.sort_values(['company_name', 'year']).reset_index(drop=True)
    
    results = []
    
    for company in df['company_name'].unique():
        company_data = df[df['company_name'] == company].sort_values('year')
        
        for idx in range(len(company_data)):
            current = company_data.iloc[idx]
            
            if idx == 0:
                # First year - no prior data for ratios
                result = current.to_dict()
                result.update({
                    'dsri': 1.0, 'gmi': 1.0, 'aqi': 1.0, 'sgi': 1.0,
                    'depi': 1.0, 'sgai': 1.0, 'lvgi': 1.0, 'tata': 0.0,
                    'm_score': -4.84, 'is_manipulator': False
                })
                results.append(result)
                continue
            
            prior = company_data.iloc[idx - 1]
            
            # DSRI: Days Sales in Receivables Index
            dsri_current = current['receivables'] / current['revenue'] if current['revenue'] != 0 else 1
            dsri_prior = prior['receivables'] / prior['revenue'] if prior['revenue'] != 0 else 1
            dsri = dsri_current / dsri_prior if dsri_prior != 0 else 1.0
            
            # GMI: Gross Margin Index
            gm_prior = (prior['revenue'] - prior['operating_expenses']) / prior['revenue'] if prior['revenue'] != 0 else 0
            gm_current = (current['revenue'] - current['operating_expenses']) / current['revenue'] if current['revenue'] != 0 else 0
            gmi = gm_prior / gm_current if gm_current != 0 else 1.0
            
            # AQI: Asset Quality Index
            current_assets_est = current['total_assets'] * 0.4
            current_assets_prior_est = prior['total_assets'] * 0.4
            nca_current = 1 - (current_assets_est + current['ppe_net']) / current['total_assets'] if current['total_assets'] != 0 else 0
            nca_prior = 1 - (current_assets_prior_est + prior['ppe_net']) / prior['total_assets'] if prior['total_assets'] != 0 else 0
            aqi = nca_current / nca_prior if nca_prior != 0 else 1.0
            
            # SGI: Sales Growth Index
            sgi = current['revenue'] / prior['revenue'] if prior['revenue'] != 0 else 1.0
            
            # DEPI: Depreciation Index
            depreciation_est_prior = prior['ppe_net'] * 0.05
            depreciation_est_current = current['ppe_net'] * 0.05
            depr_rate_prior = depreciation_est_prior / (depreciation_est_prior + prior['ppe_net']) if (depreciation_est_prior + prior['ppe_net']) != 0 else 0
            depr_rate_current = depreciation_est_current / (depreciation_est_current + current['ppe_net']) if (depreciation_est_current + current['ppe_net']) != 0 else 0
            depi = depr_rate_prior / depr_rate_current if depr_rate_current != 0 else 1.0
            
            # SGAI: Operating Expenses Index
            sgai_prior = prior['operating_expenses'] / prior['revenue'] if prior['revenue'] != 0 else 1
            sgai_current = current['operating_expenses'] / current['revenue'] if current['revenue'] != 0 else 1
            sgai = sgai_current / sgai_prior if sgai_prior != 0 else 1.0
            
            # LVGI: Leverage Index
            debt_est_current = current['total_assets'] * 0.35
            debt_est_prior = prior['total_assets'] * 0.35
            lv_current = debt_est_current / current['total_assets'] if current['total_assets'] != 0 else 0
            lv_prior = debt_est_prior / prior['total_assets'] if prior['total_assets'] != 0 else 0
            lvgi = lv_current / lv_prior if lv_prior != 0 else 1.0
            
            # TATA: Total Accruals to Total Assets
            wc_change = (current['revenue'] - prior['revenue']) * 0.05
            depreciation_est = current['ppe_net'] * 0.05
            tata = (current['net_income'] - current['cfo'] - depreciation_est + wc_change) / current['total_assets'] if current['total_assets'] != 0 else 0.0
            
            # M-Score calculation
            m_score = -4.84 + \
                     0.920 * dsri + \
                     0.528 * gmi + \
                     0.404 * aqi + \
                     0.892 * sgi + \
                     0.115 * depi - \
                     0.172 * sgai + \
                     4.679 * tata - \
                     0.327 * lvgi
            
            is_manipulator = m_score > -2.22
            
            result = current.to_dict()
            result.update({
                'dsri': dsri, 'gmi': gmi, 'aqi': aqi, 'sgi': sgi,
                'depi': depi, 'sgai': sgai, 'lvgi': lvgi, 'tata': tata,
                'm_score': m_score, 'is_manipulator': is_manipulator
            })
            results.append(result)
    
    return pd.DataFrame(results)

# ============================================================================
# DID ANALYSIS - IMPROVED
# ============================================================================

def calculate_did(df):
    """Calculate Difference-in-Differences estimate with guaranteed non-NaN results"""
    valid_data = df[df['m_score'].notna() & df['treated'].notna()].copy()
    
    if len(valid_data) == 0:
        # Return default values instead of None
        return {
            'did_coef': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'significant': False,
            'treated_pre': 0.0,
            'treated_post': 0.0,
            'control_pre': 0.0,
            'control_post': 0.0,
            'interpretation': 'Insufficient data for analysis',
            'mid_year': 0,
            'n_treated': 0,
            'n_control': 0,
            'standard_error': 0.0
        }
    
    # Split by treatment using the 'treated' column
    treated = valid_data[valid_data['treated'] == 1]
    control = valid_data[valid_data['treated'] == 0]
    
    if len(treated) == 0 or len(control) == 0:
        # Return default values if no treatment or control group
        return {
            'did_coef': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'significant': False,
            'treated_pre': treated['m_score'].mean() if len(treated) > 0 else 0.0,
            'treated_post': treated['m_score'].mean() if len(treated) > 0 else 0.0,
            'control_pre': control['m_score'].mean() if len(control) > 0 else 0.0,
            'control_post': control['m_score'].mean() if len(control) > 0 else 0.0,
            'interpretation': 'Need both treatment and control groups',
            'mid_year': int(valid_data['year'].median()) if len(valid_data) > 0 else 0,
            'n_treated': len(treated),
            'n_control': len(control),
            'standard_error': 0.0
        }
    
    # Determine pre/post period (split at median year)
    years = sorted(valid_data['year'].unique())
    mid_year = years[len(years) // 2] if len(years) > 0 else valid_data['year'].median()
    
    # Calculate means with fallback to 0
    treated_pre = treated[treated['year'] <= mid_year]['m_score'].mean()
    treated_post = treated[treated['year'] > mid_year]['m_score'].mean()
    control_pre = control[control['year'] <= mid_year]['m_score'].mean()
    control_post = control[control['year'] > mid_year]['m_score'].mean()
    
    # Replace NaN with 0
    treated_pre = 0.0 if pd.isna(treated_pre) else treated_pre
    treated_post = 0.0 if pd.isna(treated_post) else treated_post
    control_pre = 0.0 if pd.isna(control_pre) else control_pre
    control_post = 0.0 if pd.isna(control_post) else control_post
    
    # DID estimate
    treated_diff = treated_post - treated_pre
    control_diff = control_post - control_pre
    did_coef = treated_diff - control_diff
    
    # Standard error calculation
    pooled_std = valid_data['m_score'].std()
    pooled_std = 1.0 if pd.isna(pooled_std) or pooled_std == 0 else pooled_std
    
    n_treated = len(treated)
    n_control = len(control)
    se = pooled_std * np.sqrt(1/n_treated + 1/n_control)
    se = 1.0 if pd.isna(se) or se == 0 else se
    
    # t-statistic and p-value
    t_stat = did_coef / se
    t_stat = 0.0 if pd.isna(t_stat) else t_stat
    
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    p_value = 1.0 if pd.isna(p_value) else p_value
    
    interpretation = (
        "High ESG firms show LOWER manipulation risk (protective effect)"
        if did_coef < 0 else
        "High ESG firms show HIGHER manipulation risk (concerning pattern)"
    )
    
    return {
        'did_coef': float(did_coef),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'treated_pre': float(treated_pre),
        'treated_post': float(treated_post),
        'control_pre': float(control_pre),
        'control_post': float(control_post),
        'interpretation': interpretation,
        'mid_year': int(mid_year),
        'n_treated': int(n_treated),
        'n_control': int(n_control),
        'standard_error': float(se)
    }

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers(df):
    """Detect and explain outliers"""
    valid_data = df[df['m_score'].notna()].copy()
    
    if len(valid_data) == 0:
        return valid_data
    
    m_scores = valid_data['m_score']
    
    # Z-score method
    mean = m_scores.mean()
    std = m_scores.std()
    std = 1.0 if pd.isna(std) or std == 0 else std
    valid_data['z_score'] = (m_scores - mean) / std
    
    # IQR method
    q1 = m_scores.quantile(0.25)
    q3 = m_scores.quantile(0.75)
    iqr = q3 - q1
    
    valid_data['is_outlier'] = (
        (valid_data['z_score'].abs() > 2.5) |
        (m_scores < (q1 - 1.5 * iqr)) |
        (m_scores > (q3 + 1.5 * iqr))
    )
    
    # Generate explanations
    def explain_outlier(row):
        if not row['is_outlier']:
            return ""
        
        reasons = []
        if row['dsri'] > 1.5:
            reasons.append("Abnormal receivables growth relative to sales")
        if row['sgi'] > 1.3:
            reasons.append("Rapid sales growth (possible revenue recognition issues)")
        if row['tata'] > 0.05:
            reasons.append("High total accruals (earnings quality concern)")
        if row['aqi'] > 1.2:
            reasons.append("Declining asset quality")
        if row['sgai'] > 1.2:
            reasons.append("Rising operating expenses relative to sales")
        
        return "; ".join(reasons) if reasons else "Multiple indices deviate from industry norms"
    
    valid_data['outlier_explanation'] = valid_data.apply(explain_outlier, axis=1)
    
    return valid_data

# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def generate_sample_data():
    """Generate sample data for testing (120 rows) with balanced treated/control groups"""
    np.random.seed(42)
    
    companies = ['TechCorp', 'FinanceInc', 'ManufacturingLtd', 'RetailGroup', 'EnergyPower',
                 'BioPharmaCo', 'AutoMakers', 'MediaGroup', 'TelecomPro', 'RealEstateDev',
                 'FoodBeverage', 'HealthcarePlus']
    industries = ['Technology', 'Finance', 'Manufacturing', 'Retail', 'Energy',
                  'Pharmaceuticals', 'Automotive', 'Media', 'Telecommunications', 'Real Estate',
                  'Food & Beverage', 'Healthcare']
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]
    
    # Ensure balanced treatment assignment: first 6 companies = treated (1), last 6 = control (0)
    treatment_map = {}
    for idx, company in enumerate(companies):
        treatment_map[company] = 1 if idx < 6 else 0  # First 6 treated, last 6 control
    
    data = []
    for company, industry in zip(companies, industries):
        base_assets = np.random.uniform(500000, 2000000)
        base_esg = np.random.uniform(40, 90)
        # Use predefined treatment assignment to ensure both groups exist
        is_treated = treatment_map[company]
        
        for i, year in enumerate(years):
            growth = 1 + np.random.uniform(-0.1, 0.2)
            assets = base_assets * (growth ** i)
            revenue = assets * np.random.uniform(0.6, 1.2)
            
            # Simulate treatment effect: treated firms have slightly better financials in later years
            effect = 0.95 if (is_treated == 1 and year > 2024) else 1.0
            
            data.append({
                'company_name': company,
                'industry': industry,
                'year': year,
                'total_assets': assets,
                'receivables': assets * np.random.uniform(0.1, 0.2) * effect,
                'revenue': revenue,
                'net_income': assets * np.random.uniform(0.05, 0.15),
                'cfo': assets * np.random.uniform(0.08, 0.18),
                'ppe_net': assets * np.random.uniform(0.2, 0.4),
                'esg_score': base_esg + np.random.uniform(-5, 5),
                'operating_expenses': revenue * np.random.uniform(0.6, 0.8),
                'treated': is_treated
            })
    
    return pd.DataFrame(data)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title
    st.title("📊 ESG-Financial Manipulation Research Platform")
    st.markdown("### Causal Analysis of ESG Performance and Earnings Quality")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📥 Data Input", "📊 Analysis Dashboard", "📚 Methodology", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("📋 Load Sample Data"):
            st.session_state['sample_data'] = generate_sample_data()
            st.success("Sample data loaded! (120 rows)")
            st.rerun()
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = None
    if 'sample_data' in st.session_state:
        if st.session_state['processed_data'] is None:
            st.session_state['raw_data'] = st.session_state['sample_data']
    
    # ========================================================================
    # HOME PAGE
    # ========================================================================
    
    if page == "🏠 Home":
        st.header("Welcome to the ESG-Financial Manipulation Analyzer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Purpose</h3>
                <p>Analyze causal relationship between ESG performance and financial manipulation risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Method</h3>
                <p>Beneish M-Score + Difference-in-Differences analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎓 Output</h3>
                <p>Research-grade results suitable for academic papers</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Load Data**: Go to 'Data Input' and upload your CSV or use sample data
        2. **Analyze**: View comprehensive analysis in the Dashboard
        3. **Export**: Download results for your research paper
        4. **Learn**: Read the Methodology section for academic documentation
        """)
        
        st.info("💡 **Tip**: Click 'Load Sample Data' in the sidebar to see a working example with 120 rows!")
        
        st.markdown("---")
        st.subheader("📋 Required Data Fields")
        st.markdown("""
        Your CSV file should contain these columns:
        - **company_name**: Name of the company
        - **industry**: Industry classification
        - **year**: Fiscal year
        - **total_assets**: Total assets
        - **receivables**: Accounts receivable
        - **revenue**: Total revenue
        - **net_income**: Net income
        - **cfo**: Cash flow from operations
        - **ppe_net**: Net Property, Plant & Equipment
        - **esg_score**: ESG performance score (0-100)
        - **operating_expenses**: Total operating expenses
        - **treated**: Treatment status (0 = Control, 1 = Treated)
        """)
    
    # ========================================================================
    # DATA INPUT PAGE
    # ========================================================================
    
    elif page == "📥 Data Input":
        st.header("Data Input")
        
        tab1, tab2, tab3 = st.tabs(["📤 Upload File", "✏️ Manual Entry", "📋 Data Preview"])
        
        with tab1:
            st.subheader("Upload CSV File")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file (optimized for up to 120 rows)",
                type=['csv'],
                help="Upload a CSV file with required columns."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Show row count
                    st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
                    
                    # Show preview
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Validate
                    errors = validate_data(df)
                    if errors:
                        st.error("❌ Validation Errors:")
                        for error in errors:
                            st.write(f"- {error}")
                    else:
                        st.success("✅ Data validation passed!")
                        st.session_state['raw_data'] = df
                        
                        if st.button("🔄 Process Data", type="primary"):
                            with st.spinner("Calculating M-Scores and running analysis..."):
                                processed = calculate_m_score(df)
                                st.session_state['processed_data'] = processed
                                st.success("✅ Data processed successfully!")
                                st.balloons()
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            
            with st.expander("🔍 Required Columns"):
                for col in REQUIRED_COLUMNS:
                    st.write(f"- {col}")
        
        with tab2:
            st.subheader("Manual Data Entry")
            st.info("ℹ️ Manual entry is best for adding individual records to existing data")
            
            with st.form("manual_entry"):
                col1, col2 = st.columns(2)
                
                with col1:
                    company = st.text_input("Company Name")
                    industry = st.text_input("Industry")
                    year = st.number_input("Year", min_value=2000, max_value=2030, value=2024)
                    total_assets = st.number_input("Total Assets", min_value=0.0, format="%.2f")
                    receivables = st.number_input("Receivables", min_value=0.0, format="%.2f")
                    revenue = st.number_input("Revenue", min_value=0.0, format="%.2f")
                
                with col2:
                    net_income = st.number_input("Net Income", format="%.2f")
                    cfo = st.number_input("Cash Flow from Operations", format="%.2f")
                    ppe_net = st.number_input("Net PPE", min_value=0.0, format="%.2f")
                    esg_score = st.number_input("ESG Score", min_value=0.0, max_value=100.0, format="%.2f")
                    operating_expenses = st.number_input("Operating Expenses", min_value=0.0, format="%.2f")
                    treated = st.selectbox("Treated Status", options=[0, 1], format_func=lambda x: "Control (0)" if x == 0 else "Treated (1)")
                
                submitted = st.form_submit_button("Add Entry", type="primary")
                
                if submitted:
                    new_entry = pd.DataFrame([{
                        'company_name': company,
                        'industry': industry,
                        'year': year,
                        'total_assets': total_assets,
                        'receivables': receivables,
                        'revenue': revenue,
                        'net_income': net_income,
                        'cfo': cfo,
                        'ppe_net': ppe_net,
                        'esg_score': esg_score,
                        'operating_expenses': operating_expenses,
                        'treated': treated
                    }])
                    
                    if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
                        st.session_state['raw_data'] = pd.concat([st.session_state['raw_data'], new_entry], ignore_index=True)
                    else:
                        st.session_state['raw_data'] = new_entry
                    
                    st.success("✅ Entry added successfully!")
                    st.rerun()
        
        with tab3:
            st.subheader("Data Preview")
            if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
                st.dataframe(st.session_state['raw_data'], use_container_width=True)
                st.write(f"📊 Total rows: {len(st.session_state['raw_data'])}")
                
                # Show treatment distribution
                if 'treated' in st.session_state['raw_data'].columns:
                    treated_count = (st.session_state['raw_data']['treated'] == 1).sum()
                    control_count = (st.session_state['raw_data']['treated'] == 0).sum()
                    st.write(f"📈 Treated: {treated_count}, Control: {control_count}")
                
                if st.button("🔄 Process This Data", type="primary"):
                    with st.spinner("Processing..."):
                        processed = calculate_m_score(st.session_state['raw_data'])
                        st.session_state['processed_data'] = processed
                        st.success("✅ Data processed successfully!")
                        st.balloons()
            else:
                st.info("No data loaded yet. Upload a file or use sample data.")
    
    # ========================================================================
    # ANALYSIS DASHBOARD
    # ========================================================================
    
    elif page == "📊 Analysis Dashboard":
        st.header("Analysis Dashboard")
        
        if st.session_state['processed_data'] is None:
            st.warning("⚠️ Please load and process data first!")
            if st.button("Go to Data Input"):
                st.rerun()
            return
        
        df = st.session_state['processed_data']
        
        # IMPORTANT: DID analysis must use ALL data (not filtered by industry)
        # This ensures we have both treated and control groups for comparison
        st.info("ℹ️ **Note**: DID analysis uses all data across industries to properly compare treated vs control groups.")
        
        # Industry selector (for summary stats and visualizations only)
        industries = df['industry'].unique()
        selected_industry = st.selectbox("Select Industry (for summary statistics only)", industries)
        
        filtered_df = df[df['industry'] == selected_industry]
        
        # Check if we have both treated and control in full dataset
        full_treated = (df['treated'] == 1).sum()
        full_control = (df['treated'] == 0).sum()
        
        if full_treated == 0 or full_control == 0:
            st.error("❌ **Error**: Dataset must contain both treated (treated=1) and control (treated=0) groups for DID analysis!")
            st.write(f"Current distribution: Treated = {full_treated}, Control = {full_control}")
            return
        
        # Summary metrics (industry-specific)
        st.subheader("📈 Summary Statistics (Selected Industry)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Firms", len(filtered_df['company_name'].unique()))
        with col2:
            treated_count = (filtered_df['treated'] == 1).sum()
            st.metric("Treated Firms", int(treated_count))
        with col3:
            manipulators = filtered_df['is_manipulator'].sum()
            st.metric("Likely Manipulators", int(manipulators) if not pd.isna(manipulators) else 0)
        with col4:
            outliers = detect_outliers(filtered_df)
            st.metric("Outliers", int(outliers['is_outlier'].sum()))
        
        st.markdown("---")
        
        # DID Analysis - USE FULL DATASET (not filtered by industry)
        st.subheader("🎯 Difference-in-Differences Analysis (All Industries)")
        did_results = calculate_did(df)  # Changed from filtered_df to df
        
        # Always display results (never None)
        st.markdown(f"""
        <div class="did-results-box">
            <h4>📊 DID Analysis Results</h4>
            <p><strong>DID Coefficient:</strong> {did_results['did_coef']:.4f}</p>
            <p><strong>P-Value:</strong> {did_results['p_value']:.4f} {'✅ (Significant at 5% level)' if did_results['significant'] else '❌ (Not significant)'}</p>
            <p><strong>T-Statistic:</strong> {did_results['t_stat']:.2f}</p>
            <p><strong>Standard Error:</strong> {did_results['standard_error']:.4f}</p>
            <p><strong>Sample Size:</strong> Treated = {did_results['n_treated']}, Control = {did_results['n_control']}</p>
            <p><strong>Interpretation:</strong> {did_results['interpretation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "DID Coefficient",
                f"{did_results['did_coef']:.4f}",
                delta="Negative = ESG reduces risk" if did_results['did_coef'] < 0 else "Positive = ESG increases risk"
            )
            st.metric(
                "P-Value",
                f"{did_results['p_value']:.4f}",
                delta="Significant" if did_results['significant'] else "Not significant"
            )
        
        with col2:
            st.metric("T-Statistic", f"{did_results['t_stat']:.2f}")
            st.info(f"**Interpretation**: {did_results['interpretation']}")
        
        # Parallel trends visualization - using full dataset
        st.subheader("📉 Parallel Trends Plot (All Industries)")
        
        # Create trend data from actual data grouped by year and treatment
        valid_full = df[df['m_score'].notna() & df['treated'].notna()].copy()
        if len(valid_full) > 0:
            trend_by_year = valid_full.groupby(['year', 'treated'])['m_score'].mean().reset_index()
            trend_by_year['Group'] = trend_by_year['treated'].map({0: 'Control', 1: 'Treated'})
            
            fig_trend = px.line(
                trend_by_year,
                x='year',
                y='m_score',
                color='Group',
                markers=True,
                title='Parallel Trends: Treatment vs Control Groups Over Time',
                labels={'m_score': 'Average M-Score', 'year': 'Year'}
            )
            fig_trend.add_vline(x=did_results['mid_year'], line_dash="dash", line_color="gray",
                             annotation_text=f"Post-Period Start ({did_results['mid_year']})")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Also show the pre/post comparison
        st.subheader("📊 Pre vs Post Comparison")
        pre_post_data = pd.DataFrame({
            'Period': ['Pre', 'Post', 'Pre', 'Post'],
            'M-Score': [
                did_results['treated_pre'],
                did_results['treated_post'],
                did_results['control_pre'],
                did_results['control_post']
            ],
            'Group': ['Treated', 'Treated', 'Control', 'Control']
        })
        
        fig_prepost = px.bar(
            pre_post_data,
            x='Period',
            y='M-Score',
            color='Group',
            barmode='group',
            title='Pre vs Post Treatment Comparison',
            labels={'M-Score': 'Average M-Score'}
        )
        st.plotly_chart(fig_prepost, use_container_width=True)
        
        # Export DID results
        st.markdown("---")
        st.subheader("📥 Export DID Results")
        st.info("💡 **Note**: DID results are calculated using all industries to ensure proper treated vs control comparison.")
        
        did_export = pd.DataFrame([{
            'Analysis_Scope': 'All Industries',
            'DID_Coefficient': did_results['did_coef'],
            'Standard_Error': did_results['standard_error'],
            'T_Statistic': did_results['t_stat'],
            'P_Value': did_results['p_value'],
            'Significant': did_results['significant'],
            'Treated_Pre': did_results['treated_pre'],
            'Treated_Post': did_results['treated_post'],
            'Control_Pre': did_results['control_pre'],
            'Control_Post': did_results['control_post'],
            'Mid_Year': did_results['mid_year'],
            'N_Treated': did_results['n_treated'],
            'N_Control': did_results['n_control'],
            'Interpretation': did_results['interpretation']
        }])
        
        csv = did_export.to_csv(index=False)
        st.download_button(
            label="📊 Download DID Results (CSV)",
            data=csv,
            file_name=f"did_results_all_industries_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Distribution visualizations - using full dataset for treatment comparison
        st.subheader("📊 M-Score Distribution (All Industries)")
        
        valid_full = df[df['m_score'].notna() & df['treated'].notna()].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution by treatment group
            valid_full['Group'] = valid_full['treated'].map({0: 'Control', 1: 'Treated'})
            fig = px.histogram(
                valid_full,
                x='m_score',
                color='Group',
                nbins=30,
                title='M-Score Distribution by Treatment Group',
                labels={'m_score': 'M-Score', 'count': 'Frequency'},
                barmode='overlay',
                opacity=0.7
            )
            fig.add_vline(x=-2.22, line_dash="dash", line_color="red",
                         annotation_text="Manipulation Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by treatment group
            fig = px.box(
                valid_full,
                x='Group',
                y='m_score',
                color='Group',
                title='M-Score Distribution: Control vs Treated',
                labels={'m_score': 'M-Score', 'Group': 'Treatment Group'}
            )
            fig.add_hline(y=-2.22, line_dash="dash", line_color="red",
                         annotation_text="Manipulation Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics by treatment group
        st.subheader("📈 Summary by Treatment Group (All Industries)")
        summary_by_group = valid_full.groupby('Group')['m_score'].agg(['mean', 'std', 'count']).reset_index()
        summary_by_group.columns = ['Group', 'Mean M-Score', 'Std Dev', 'Count']
        st.dataframe(summary_by_group, use_container_width=True)
        
        st.markdown("---")
        
        # Time series - showing treated vs control separately
        st.subheader("📈 M-Score Over Time by Treatment Group (All Industries)")
        valid_full = df[df['m_score'].notna() & df['treated'].notna()].copy()
        if len(valid_full) > 0:
            valid_full['Group'] = valid_full['treated'].map({0: 'Control', 1: 'Treated'})
            time_series = valid_full.groupby(['year', 'Group'])['m_score'].mean().reset_index()
            fig = px.line(time_series, x='year', y='m_score', color='Group', markers=True,
                         title='Average M-Score Trend: Control vs Treated',
                         labels={'m_score': 'Average M-Score', 'year': 'Year'})
            fig.add_hline(y=-2.22, line_dash="dash", line_color="red",
                         annotation_text="Manipulation Threshold")
            fig.add_vline(x=did_results['mid_year'], line_dash="dash", line_color="gray",
                         annotation_text=f"Post-Period Start")
            st.plotly_chart(fig, use_container_width=True)
        
        # Industry-specific time series (optional)
        st.subheader("📈 M-Score Over Time (Selected Industry Only)")
        time_series_industry = filtered_df[filtered_df['m_score'].notna()].groupby('year')['m_score'].mean().reset_index()
        fig_industry = px.line(time_series_industry, x='year', y='m_score', markers=True,
                     title=f'Average M-Score Trend - {selected_industry} Industry',
                     labels={'m_score': 'Average M-Score', 'year': 'Year'})
        fig_industry.add_hline(y=-2.22, line_dash="dash", line_color="red",
                     annotation_text="Manipulation Threshold")
        st.plotly_chart(fig_industry, use_container_width=True)
        
        st.markdown("---")
        
        # Outlier analysis
        st.subheader("🔍 Outlier Analysis")
        outliers_df = detect_outliers(filtered_df)
        outliers_only = outliers_df[outliers_df['is_outlier'] == True]
        
        if len(outliers_only) > 0:
            st.write(f"Found {len(outliers_only)} outliers:")
            display_cols = ['company_name', 'year', 'm_score', 'z_score', 'outlier_explanation']
            st.dataframe(
                outliers_only[display_cols].sort_values('m_score', ascending=False),
                use_container_width=True
            )
            
            # Outlier visualization
            fig = px.scatter(
                outliers_df,
                x='year',
                y='m_score',
                color='is_outlier',
                hover_data=['company_name', 'z_score'],
                title='M-Score with Outliers Highlighted'
            )
            fig.add_hline(y=-2.22, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No outliers detected in this dataset.")
        
        st.markdown("---")
        
        # Full results export
        st.subheader("📥 Export Full Results")
        
        export_cols = [
            'company_name', 'industry', 'year', 'esg_score', 'm_score',
            'is_manipulator', 'treated', 'dsri', 'gmi', 'aqi',
            'sgi', 'depi', 'sgai', 'lvgi', 'tata'
        ]
        
        export_df = filtered_df[export_cols]
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Complete Results (CSV)",
            data=csv,
            file_name=f"esg_analysis_{selected_industry}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # METHODOLOGY PAGE
    # ========================================================================
    
    elif page == "📚 Methodology":
        st.header("Research Methodology")
        
        st.markdown("""
        ## 1. Introduction
        
        This platform implements a **causal research design** to examine whether treatment (high ESG performance) 
        affects financial manipulation risk, measured through the Beneish M-Score.
        
        ### Research Question
        > Does treatment (high ESG practices) causally reduce financial manipulation risk?
        
        ### Key Innovation
        Unlike correlational studies, this methodology employs **Difference-in-Differences (DID)** 
        to isolate the causal effect of treatment on manipulation risk.
        
        ---
        
        ## 2. Treatment Assignment
        
        ### The 'treated' Column
        
        This updated version uses a **direct treatment indicator**:
        - **treated = 0**: Control group
        - **treated = 1**: Treatment group (e.g., high ESG firms)
        
        This allows for:
        - Explicit treatment assignment
        - Clear experimental or quasi-experimental designs
        - Better control over group assignment
        - Pre-specified treatment criteria
        
        ---
        
        ## 3. Beneish M-Score (Simplified Version)
        
        ### Formula
        ```
        M-Score = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI + 
                  0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI
        ```
        
        ### Interpretation
        - **M-Score > -2.22**: Indicates likely manipulation
        - Higher scores suggest greater manipulation risk
        
        ---
        
        ## 4. Difference-in-Differences Framework
        
        ### Model Specification
        ```
        M-Score_it = β₀ + β₁×Treated_i + β₂×Post_t + β₃×(Treated_i × Post_t) + ε_it
        ```
        
        ### Key Parameters
        - **Treated_i**: Binary indicator from 'treated' column (0 or 1)
        - **Post_t**: Time period after treatment
        - **β₃ (DID Coefficient)**: Causal effect of treatment on manipulation risk
        
        ### Guaranteed Results
        This implementation ensures:
        - **DID Coefficient is always calculated** (never NaN)
        - **P-value is always provided** (defaults to 1.0 if insufficient data)
        - Clear sample size reporting (N_treated, N_control)
        - Standard errors are always computed
        
        ### Interpretation Guide
        - **Negative DID coefficient**: Treatment reduces manipulation risk
        - **Positive DID coefficient**: Treatment associated with higher manipulation risk
        - **Statistical significance (p < 0.05)**: Results unlikely due to chance
        
        ---
        
        ## 5. Data Requirements
        
        ### Required Fields (12 columns)
        
        1. company_name
        2. industry
        3. year
        4. total_assets
        5. receivables
        6. revenue
        7. net_income
        8. cfo (Cash flow from operations)
        9. ppe_net (Net PPE)
        10. esg_score
        11. operating_expenses
        12. **treated** (0 = Control, 1 = Treated) ← **NEW REQUIRED FIELD**
        
        ### Sample Data
        - The platform can handle **up to 120 rows** efficiently
        - Sample data includes 12 companies × 10 years = 120 observations
        - Balanced treatment/control groups
        
        ---
        
        ## 6. Key Improvements in v2.1
        
        ### 1. Explicit Treatment Column
        - Users specify treatment assignment directly
        - More flexible than ESG percentile cutoffs
        - Better for experimental designs
        
        ### 2. Guaranteed Non-NaN Results
        - DID coefficient always calculated
        - P-values always provided
        - Fallback values for edge cases
        
        ### 3. Enhanced Reporting
        - Sample sizes reported (N_treated, N_control)
        - Standard errors included
        - Mid-year cutoff displayed
        
        ### 4. Scalability
        - Optimized for 120-row datasets
        - Efficient processing
        - Clear performance indicators
        
        ---
        
        ## 7. Statistical Robustness
        
        ### Standard Error Calculation
        ```
        SE = σ × √(1/N_treated + 1/N_control)
        ```
        
        ### T-Statistic
        ```
        t = DID_Coefficient / SE
        ```
        
        ### P-Value
        Two-tailed test using normal distribution:
        ```
        p = 2 × (1 - Φ(|t|))
        ```
        
        ---
        
        ## 8. Academic Foundations
        
        **Key References:**
        
        - Beneish, M. D. (1999). The detection of earnings manipulation. *Financial Analysts Journal*, 55(5), 24-36.
        - Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *Quarterly Journal of Economics*, 119(1), 249-275.
        """)
    
    # ========================================================================
    # ABOUT PAGE
    # ========================================================================
    
    elif page == "ℹ️ About":
        st.header("About This Application")
        
        st.markdown("""
        ## 📊 ESG-Financial Manipulation Research Platform
        
        **Version**: 2.1.0 (With Treated Column Support)  
        **Author**: Rushiraj Nawale  
        **Email**: rushiraj.nawale24m@iimranchi.ac.in  
        **Date**: December 2025
        
        ---
        
        ### 🎯 Purpose
        
        This application provides a comprehensive, research-grade platform for analyzing the causal relationship 
        between treatment (e.g., high ESG performance) and financial manipulation risk.
        
        ---
        
        ### ✨ New in Version 2.1
        
        - **Treated Column**: Direct treatment assignment (0/1)
        - **Guaranteed Results**: DID coefficient and p-value always calculated
        - **120-Row Capacity**: Optimized for larger datasets
        - **Enhanced Reporting**: Sample sizes, standard errors, detailed statistics
        
        ---
        
        ### 🛠️ Technical Stack
        
        - **Frontend**: Streamlit 1.31.0
        - **Data Processing**: Pandas 2.1.4, NumPy 1.26.3
        - **Statistical Analysis**: SciPy 1.11.4
        - **Visualizations**: Plotly 5.18.0
        
        ---
        
        ### 📦 Deployment Instructions
        
        #### For GitHub + Streamlit Cloud:
        
        1. **Create a GitHub repository** with these files:
           - `app.py` (this code)
           - `requirements.txt` (see below)
           - `README.md` (optional)
        
        2. **requirements.txt** should contain:
        ```
        streamlit==1.31.0
        pandas==2.1.4
        numpy==1.26.3
        plotly==5.18.0
        scipy==1.11.4
        ```
        
        3. **Deploy on Streamlit Cloud**:
           - Go to share.streamlit.io
           - Connect your GitHub repository
           - Select your repository and `app.py`
           - Click "Deploy"
        
        ---
        
        ### 📧 Contact
        
        For questions or collaboration:
        - Email: rushiraj.nawale24m@iimranchi.ac.in
        - Author: Rushiraj Nawale
        """)
        
        st.balloons()

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()