# ESG-Financial Manipulation Research Platform

A research-grade web application for analyzing the causal relationship between ESG performance and financial manipulation risk using Difference-in-Differences (DID) methodology.

## 📊 Overview

This platform implements a comprehensive analysis framework that:
- Calculates **Beneish M-Score** to detect potential earnings manipulation
- Performs **Difference-in-Differences (DID)** analysis to estimate causal effects
- Provides interactive visualizations and exportable research-grade results
- Supports up to 120 observations with balanced treated/control groups

## 👤 Author

**Rushiraj Nawale**  
Email: rushiraj.nawale24m@iimranchi.ac.in  
Institution: IIM Ranchi

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd esg-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run esg_analyzer_updated.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in the terminal

## 📋 Data Requirements

Your CSV file must contain the following 12 columns:

1. `company_name` - Name of the company
2. `industry` - Industry classification
3. `year` - Fiscal year
4. `total_assets` - Total assets
5. `receivables` - Accounts receivable
6. `revenue` - Total revenue
7. `net_income` - Net income
8. `cfo` - Cash flow from operations
9. `ppe_net` - Net Property, Plant & Equipment
10. `esg_score` - ESG performance score (0-100)
11. `operating_expenses` - Total operating expenses
12. `treated` - Treatment status (**0 = Control**, **1 = Treated**)

### Sample Data

The application includes built-in sample data with:
- 12 companies across multiple industries
- 10 years of data (2020-2029)
- **Balanced treatment assignment**: First 6 companies = Treated (1), Last 6 companies = Control (0)
- Total: 120 observations

Click "Load Sample Data" in the sidebar to use the demo dataset.

## 🌐 Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. Create a GitHub repository
2. Upload these files:
   - `esg_analyzer_updated.py` (main application file)
   - `requirements.txt` (dependencies)
   - `README.md` (this file)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set **Main file path**: `esg_analyzer_updated.py`
6. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## 📚 Features

### 1. Data Input
- Upload CSV files with validation
- Manual data entry
- Data preview with treatment distribution

### 2. Analysis Dashboard
- **DID Analysis**: Calculated using all industries for proper treated/control comparison
- **Summary Statistics**: By industry and treatment group
- **Visualizations**:
  - Parallel trends plots
  - M-Score distributions by treatment group
  - Time series analysis
  - Outlier detection

### 3. Export Functionality
- Download DID results (CSV)
- Download complete analysis results
- Research-ready format

### 4. Methodology Documentation
- Complete academic documentation
- Statistical methodology
- Interpretation guides

## 🔬 Methodology

### Difference-in-Differences (DID)

The DID framework estimates the causal effect of treatment (high ESG) on manipulation risk:

```
M-Score_it = β₀ + β₁×Treated_i + β₂×Post_t + β₃×(Treated_i × Post_t) + ε_it
```

Where:
- **Treated_i**: Binary indicator (0 = Control, 1 = Treated)
- **Post_t**: Time period after treatment
- **β₃ (DID Coefficient)**: Causal effect of treatment

### Beneish M-Score

The M-Score aggregates eight financial indices to detect earnings manipulation:

```
M-Score = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI + 
          0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI
```

**Interpretation**: M-Score > -2.22 indicates likely manipulation.

## 📊 Key Features

- ✅ **Guaranteed Results**: DID coefficient and p-value always calculated (no NaN errors)
- ✅ **Balanced Sample Data**: Ensures both treated and control groups exist
- ✅ **Full Dataset Analysis**: DID uses all industries for proper comparison
- ✅ **Interactive Visualizations**: Plotly charts with hover details
- ✅ **Export Ready**: CSV downloads for research papers
- ✅ **Validation**: Comprehensive data validation before processing

## 🛠️ Technical Stack

- **Frontend**: Streamlit 1.31.0
- **Data Processing**: Pandas 2.1.4, NumPy 1.26.3
- **Statistical Analysis**: SciPy 1.11.4
- **Visualizations**: Plotly 5.18.0

## 📖 Academic References

- Beneish, M. D. (1999). The detection of earnings manipulation. *Financial Analysts Journal*, 55(5), 24-36.
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *Quarterly Journal of Economics*, 119(1), 249-275.
- Christensen, D. M., Serafeim, G., & Sikochi, A. (2022). Why is corporate virtue in the eye of the beholder? The case of ESG ratings. *The Accounting Review*, 97(1), 147-175.

## 📝 Citation

If you use this platform in your research, please cite:

```
Nawale, R. (2025). ESG-Financial Manipulation Research Platform: 
Causal Analysis of ESG Performance and Earnings Quality. 
Available at: [Your GitHub Repository URL]
```

## ⚠️ Important Notes

1. **Treatment Assignment**: The `treated` column must contain only 0 (Control) or 1 (Treated)
2. **Data Quality**: Ensure all required columns are present and numeric fields are properly formatted
3. **Sample Size**: Optimized for datasets up to 120 rows for best performance
4. **DID Analysis**: Uses all industries to ensure proper treated/control comparison

## 🤝 Contributing

This is a research platform. For questions or collaboration, contact:
- **Email**: rushiraj.nawale24m@iimranchi.ac.in
- **Author**: Rushiraj Nawale

## 📄 License

This project is for academic research purposes.

---

**Version**: 2.1.0  
**Last Updated**: December 2025

