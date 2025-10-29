# Drug Script Analysis Tool

A comprehensive Python tool for analyzing prescription data using two different methodologies side-by-side:
- **WoW Method** (Week-over-Week Percentage Change)
- **Z-Score Method** (Statistical Analysis with rolling baseline)

## 🚀 Deployment Options

- **📊 Web App**: [Deploy with Streamlit](#web-application)
- **💻 Command Line**: [Run locally](#quick-start)
- **📓 Jupyter Notebook**: [Use in Colab or locally](#jupyter-notebook)

## Features

✨ **Dual Analysis Methods**
- Week-over-Week percentage change classification
- Z-Score statistical analysis with holiday-aware baseline

📊 **Interactive Visualization**
- 4-panel Plotly comparison chart
- Script trend analysis with color-coded classifications
- Timeline view of classification changes
- Holiday markers and indicators

🎯 **Smart Drug Maturity Classification**
- Automatically adjusts thresholds based on drug lifecycle stage
- Brand New Release, Emerging/Expanding, or Fully Mature

🎄 **Holiday Detection**
- Flags major US holidays automatically
- Z-Score method excludes holidays from baseline calculations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Your Method

#### Web Application (Recommended) 🌐

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`
- Upload CSV or Excel files via web interface
- Interactive visualizations
- Easy to share and deploy

📖 See `docs/DEPLOYMENT_GUIDE.md` for deployment to Streamlit Cloud

#### Command Line 💻

```bash
python run_analysis.py
```

Or in a Python script:

```python
from scripts import main

# Run complete analysis
fig, df_wow, df_zscore, differences = main(filepath='data/neffy_scripts_google_cloud.csv')

# Display interactive chart
fig.show()
```

#### Jupyter Notebook 📓

Open `neffy_comparison_analysis_COLAB.ipynb` in:
- **Google Colab** - Upload via browser
- **Local Jupyter** - Run `jupyter notebook` and open the file

## Output

### Console Output
- Data loading confirmation (50 weeks)
- Holiday detection results (4 holidays flagged)
- Drug maturity classification (Emerging/Expanding)
- Method-specific thresholds
- Detailed disagreement analysis (37 weeks differ)
- Summary statistics showing:
  - WoW: 18 In-Line, 13 Meaningfully Above, 13 Slightly Above, etc.
  - Z-Score: 22 Slightly Above, 17 Meaningfully Above, 7 In-Line, etc.
  - Agreement: 26% (13/50 weeks)

### Visualization
Interactive 4-panel chart with:
1. **WoW Script Trend** - Colored by classification
2. **Z-Score Script Trend** - Colored by classification  
3. **WoW Timeline** - Classification over time
4. **Z-Score Timeline** - Classification over time

### Key Insights

**Agreement Rate:** 26% (13/50 weeks)

**Most Common Disagreement:** WoW says "In-Line" while Z-Score says "Slightly Above" (16 occurrences)

**Why They Disagree:**
- **WoW Method** only compares to previous week → More volatile, reacts to every fluctuation
- **Z-Score Method** compares to 8-week rolling average → More stable, filters noise, excludes holidays

**Example Disagreement (Week 13 - Thanksgiving):**
- Scripts: 1,409
- WoW: "Meaningfully Below" (-38.4% vs previous week)
- Z-Score: "In-Line" (+0.54σ vs 8-week baseline excluding holidays)

## Classification Thresholds

### For Neffy (Emerging/Expanding Drug)

**WoW Method:**
- In-Line: ±10%
- Slight Change: ±10-25%
- Meaningful Change: >±25%

**Z-Score Method:**
- In-Line: |z| ≤ 1.0
- Slight Change: 1.0 < |z| ≤ 2.0
- Meaningful Change: |z| > 2.0

## 📁 Project Structure

```
ishara_demo/
├── scripts.py                    # Core analysis module
├── streamlit_app.py              # Web application
├── run_analysis.py               # CLI tool
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
│
├── data/                         # Data files
│   ├── neffy_scripts_google_cloud.csv
│   └── neffy scripts -- sent to Anya 8.xlsx
│
├── docs/                         # Documentation
│   ├── PROJECT_SUMMARY.md
│   ├── VERIFICATION_CHECKLIST.md
│   ├── CUSTOMIZATION_GUIDE.md
│   └── DEPLOYMENT_GUIDE.md
│
├── notebooks/                    # Jupyter notebooks
│   ├── neffy_comparison_analysis.ipynb
│   └── working_visualization_z_scores.ipynb
│
├── archive/                      # Old files
└── neffy_comparison_analysis_COLAB.ipynb
```

## Data Format

### CSV Format

Place your data files in the `data/` folder.

```csv
neffy,EUTRX
2024-09-06,3
2024-09-13,7
...
```

### Excel Format (Multi-Drug Support)

Supports Excel files with multiple drugs and metrics:
- **Week column**: Date column (e.g., "Week")
- **Value columns**: Any numeric columns (e.g., "VOQUEZNA NRx", "XDEMVY EUTRX")

The Streamlit app automatically extracts all numeric columns as separate datasets.

### Required Columns:
- First column: Date (YYYY-MM-DD)
- Second column: Script counts (numeric)

## Advanced Usage

### Access Individual DataFrames

```python
from scripts import main

fig, df_wow, df_zscore, differences = main()

# WoW results
print(df_wow.head())
# Columns: week_number, date, scripts, is_holiday_week, holiday_name, 
#          classification, wow_pct, previous_week_scripts

# Z-Score results  
print(df_zscore.head())
# Columns: week_number, date, scripts, is_holiday_week, holiday_name,
#          classification, z_score, baseline_mean, baseline_std, 
#          baseline_weeks_used, holidays_excluded

# Disagreements
print(differences)
# Shows all weeks where methods disagree
```

### Export Results

```python
# Save visualization
fig.write_html("neffy_comparison.html")

# Export to CSV
df_wow.to_csv("wow_results.csv", index=False)
df_zscore.to_csv("zscore_results.csv", index=False)
differences.to_csv("disagreements.csv", index=False)
```

### Use Individual Functions

```python
from scripts import (
    load_data,
    flag_holiday_weeks,
    classify_drug_maturity,
    classify_wow_method,
    classify_zscore_method
)

# Load and prepare data
df = load_data("neffy_scripts_google_cloud.csv")
df = flag_holiday_weeks(df)

# Get drug maturity info
maturity = classify_drug_maturity(df)
print(f"Stage: {maturity['stage']}")
print(f"Baseline window: {maturity['baseline_window']} weeks")
```

## Color Coding

- 🔴 **Meaningfully Below:** #FF4444 (red)
- 🟠 **Slightly Below:** #F18F01 (orange)
- 🔵 **In-Line:** #2E86AB (blue)
- 🟣 **Slightly Above:** #A23B72 (purple)
- 🟢 **Meaningfully Above:** #06FFA5 (green)
- ⚪ **Baseline Building:** #CCCCCC (gray)

## Requirements

```python
pandas
plotly
numpy
```

## Testing Checklist

- ✅ Loads CSV correctly with proper column names
- ✅ Detects 4 holiday weeks in neffy data
- ✅ Classifies neffy as "Emerging/Expanding" (50 weeks)
- ✅ WoW uses ±10%/±25% thresholds for neffy
- ✅ Z-Score excludes holidays from baseline
- ✅ Both methods run without errors
- ✅ Visualization renders with 4 panels
- ✅ Holiday markers appear as red diamonds
- ✅ Difference report shows specific disagreements (37 weeks)
- ✅ Summary statistics print correctly
- ✅ Agreement rate: 26% (13/50 weeks)

## Key Findings

1. **Z-Score is more conservative** - Shows 22 "Slightly Above" vs WoW's 18 "In-Line"
2. **WoW is more reactive** - 13 "Meaningfully Above" vs Z-Score's 17
3. **Most common pattern:** WoW says stable ("In-Line") while Z-Score detects growth ("Slightly Above")
4. **Holiday handling matters:** Week 13 (Thanksgiving) shows dramatic difference due to Z-Score's holiday exclusion

## 📚 Additional Documentation

- **`docs/DEPLOYMENT_GUIDE.md`** - How to deploy the web app
- **`docs/CUSTOMIZATION_GUIDE.md`** - How to customize the Streamlit app
- **`docs/PROJECT_SUMMARY.md`** - Detailed project overview
- **`docs/VERIFICATION_CHECKLIST.md`** - Testing checklist
- **`CLEANUP_SUMMARY.md`** - Folder organization details

## 🎨 Customization

Want to customize the app? See `docs/CUSTOMIZATION_GUIDE.md` for:
- Visual customization (colors, themes, logos)
- Adding new features (authentication, export)
- Changing layout and components
- Integrating with databases

## 🚀 Deploy to Production

See `docs/DEPLOYMENT_GUIDE.md` for:
- Streamlit Cloud deployment (free, 5 min setup)
- Docker deployment
- AWS/Google Cloud/Azure deployment
- Authentication and security

## License

This tool is provided as-is for pharmaceutical prescription analysis.

