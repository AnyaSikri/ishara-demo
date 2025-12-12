"""
Streamlit Web App for Drug Script Analysis
Converts the notebook analysis into an interactive web interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta
import sys

# Import your existing analysis functions
sys.path.append('.')
from scripts import main, load_data, flag_holiday_weeks, classify_drug_maturity, classify_wow_method, classify_zscore_method

# ============================================================================
# EXISTING HELPER FUNCTIONS (DO NOT MODIFY)
# ============================================================================

def create_wow_only_chart(df_wow):
    """Create 2-panel WoW-only visualization"""
    color_map = {
        'Baseline Building': '#CCCCCC',
        'In-Line': '#2E86AB',
        'Slightly Above': '#A23B72',
        'Slightly Below': '#F18F01',
        'Meaningfully Above': '#06FFA5',
        'Meaningfully Below': '#FF4444'
    }
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('WoW Method - Script Trend', 'WoW Classification Timeline'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    all_classifications = set(df_wow['classification'].unique())
    legend_added = set()
    
    # Panel 1: WoW Script Trend
    for classification in all_classifications:
        df_subset = df_wow[df_wow['classification'] == classification]
        for _, row in df_subset.iterrows():
            symbol = 'diamond' if row['is_holiday_week'] else 'circle'
            size = 12 if row['is_holiday_week'] else 8
            showlegend = classification not in legend_added
            if showlegend:
                legend_added.add(classification)
            
            fig.add_trace(
                go.Scatter(x=[row['date']], y=[row['scripts']],
                    mode='markers', name=classification,
                    marker=dict(color=color_map[classification], size=size, symbol=symbol,
                    line=dict(color='red' if row['is_holiday_week'] else None, width=2 if row['is_holiday_week'] else 0)),
                    showlegend=showlegend, legendgroup=classification,
                    hovertemplate=f"<b>Week {row['week_number']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Scripts: {row['scripts']}<br>WoW %: {row['wow_pct']:.1f}%<br>Classification: {classification}<extra></extra>"),
                row=1, col=1
            )
    
    # Panel 2: WoW Timeline
    classification_order = ['Meaningfully Below', 'Slightly Below', 'In-Line', 'Slightly Above', 'Meaningfully Above', 'Baseline Building']
    classification_y = {c: i for i, c in enumerate(classification_order)}
    
    for classification in all_classifications:
        df_subset = df_wow[df_wow['classification'] == classification]
        for _, row in df_subset.iterrows():
            fig.add_trace(
                go.Scatter(x=[row['date']], y=[classification_y.get(classification, 0)],
                    mode='markers', name=classification,
                    marker=dict(color=color_map[classification], size=12 if row['is_holiday_week'] else 10, 
                    symbol='diamond' if row['is_holiday_week'] else 'square',
                    line=dict(color='red' if row['is_holiday_week'] else None, width=2 if row['is_holiday_week'] else 0)),
                    showlegend=False, legendgroup=classification,
                    hovertemplate=f"<b>Week {row['week_number']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>WoW %: {row['wow_pct']:.1f}%<br>Classification: {classification}<extra></extra>"),
                row=2, col=1
            )
    
    # Add holiday lines
    holiday_dates = df_wow[df_wow['is_holiday_week']]['date'].unique()
    for holiday_date in holiday_dates:
        fig.add_vline(x=holiday_date, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1)
        fig.add_vline(x=holiday_date, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    
    fig.update_layout(height=700, title_text="WoW Method Analysis", showlegend=True, hovermode='closest')
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Weekly Scripts", row=1, col=1)
    fig.update_yaxes(title_text="Classification", tickmode='array', tickvals=list(range(len(classification_order))),
                    ticktext=classification_order, row=2, col=1)
    
    return fig

def create_zscore_only_chart(df_zscore):
    """Create 2-panel Z-Score-only visualization"""
    color_map = {
        'Baseline Building': '#CCCCCC',
        'In-Line': '#2E86AB',
        'Slightly Above': '#A23B72',
        'Slightly Below': '#F18F01',
        'Meaningfully Above': '#06FFA5',
        'Meaningfully Below': '#FF4444'
    }
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Z-Score Method - Script Trend', 'Z-Score Classification Timeline'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    all_classifications = set(df_zscore['classification'].unique())
    legend_added = set()
    
    # Panel 1: Z-Score Script Trend
    for classification in all_classifications:
        df_subset = df_zscore[df_zscore['classification'] == classification]
        for _, row in df_subset.iterrows():
            symbol = 'diamond' if row['is_holiday_week'] else 'circle'
            size = 12 if row['is_holiday_week'] else 8
            showlegend = classification not in legend_added
            if showlegend:
                legend_added.add(classification)
            
            fig.add_trace(
                go.Scatter(x=[row['date']], y=[row['scripts']],
                    mode='markers', name=classification,
                    marker=dict(color=color_map[classification], size=size, symbol=symbol,
                    line=dict(color='red' if row['is_holiday_week'] else None, width=2 if row['is_holiday_week'] else 0)),
                    showlegend=showlegend, legendgroup=classification,
                    hovertemplate=f"<b>Week {row['week_number']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Scripts: {row['scripts']}<br>Z-Score: {row['z_score']:.2f}<br>Classification: {classification}<extra></extra>"),
                row=1, col=1
            )
    
    # Panel 2: Z-Score Timeline
    classification_order = ['Meaningfully Below', 'Slightly Below', 'In-Line', 'Slightly Above', 'Meaningfully Above', 'Baseline Building']
    classification_y = {c: i for i, c in enumerate(classification_order)}
    
    for classification in all_classifications:
        df_subset = df_zscore[df_zscore['classification'] == classification]
        for _, row in df_subset.iterrows():
            fig.add_trace(
                go.Scatter(x=[row['date']], y=[classification_y.get(classification, 0)],
                    mode='markers', name=classification,
                    marker=dict(color=color_map[classification], size=12 if row['is_holiday_week'] else 10, 
                    symbol='diamond' if row['is_holiday_week'] else 'square',
                    line=dict(color='red' if row['is_holiday_week'] else None, width=2 if row['is_holiday_week'] else 0)),
                    showlegend=False, legendgroup=classification,
                    hovertemplate=f"<b>Week {row['week_number']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Z-Score: {row['z_score']:.2f}<br>Classification: {classification}<extra></extra>"),
                row=2, col=1
            )
    
    # Add holiday lines
    holiday_dates = df_zscore[df_zscore['is_holiday_week']]['date'].unique()
    for holiday_date in holiday_dates:
        fig.add_vline(x=holiday_date, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1)
        fig.add_vline(x=holiday_date, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    
    fig.update_layout(height=700, title_text="Z-Score Method Analysis", showlegend=True, hovermode='closest')
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Weekly Scripts", row=1, col=1)
    fig.update_yaxes(title_text="Classification", tickmode='array', tickvals=list(range(len(classification_order))),
                    ticktext=classification_order, row=2, col=1)
    
    return fig

# ============================================================================
# NEW: TRx-STOCK PERFORMANCE ANALYSIS FUNCTIONS
# ============================================================================

def calculate_trx_category(df_trx, method='zscore'):
    """
    Calculate TRx performance categories for a drug's prescription data.
    Uses the existing scripts.py classification functions.
    
    Args:
        df_trx: DataFrame with 'date' and 'scripts' columns
        method: 'zscore' or 'wow'
    
    Returns:
        DataFrame with classification column added
    """
    # Prepare data
    df = df_trx.copy()
    df = df.sort_values('date').reset_index(drop=True)
    df['week_number'] = range(1, len(df) + 1)
    
    # Flag holidays
    df = flag_holiday_weeks(df)
    
    # Get maturity classification for thresholds
    maturity = classify_drug_maturity(df)
    
    if method == 'zscore':
        df = classify_zscore_method(df, maturity['baseline_window'])
    else:
        df = classify_wow_method(df, maturity['wow_thresholds'])
    
    return df

def get_stock_return(df_stock, start_date, end_date):
    """
    Calculate stock return between two dates.
    
    Args:
        df_stock: DataFrame with 'date' and 'price' columns
        start_date: Start date for return calculation
        end_date: End date for return calculation
    
    Returns:
        float: Percentage return, or None if data not available
    """
    df = df_stock.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Find closest available dates
    start_prices = df[df['date'] >= start_date].sort_values('date')
    end_prices = df[df['date'] <= end_date].sort_values('date', ascending=False)
    
    if len(start_prices) == 0 or len(end_prices) == 0:
        return None
    
    start_price = start_prices.iloc[0]['price']
    end_price = end_prices.iloc[0]['price']
    
    if start_price == 0:
        return None
    
    return ((end_price - start_price) / start_price) * 100

def create_trx_stock_scatter(analysis_results, selected_drugs):
    """
    Create interactive scatter plot of TRx categories vs stock returns.
    
    Args:
        analysis_results: List of dicts with drug analysis results
        selected_drugs: List of drug names to display
    
    Returns:
        Plotly figure
    """
    # Color map for categories
    color_map = {
        'Meaningfully Below': '#FF4444',
        'Slightly Below': '#F18F01',
        'In-Line': '#2E86AB',
        'Slightly Above': '#A23B72',
        'Meaningfully Above': '#06FFA5',
        'Baseline Building': '#CCCCCC'
    }
    
    # Category order for x-axis
    category_order = ['Meaningfully Below', 'Slightly Below', 'In-Line', 'Slightly Above', 'Meaningfully Above']
    category_x = {cat: i for i, cat in enumerate(category_order)}
    
    fig = go.Figure()
    
    # Filter to selected drugs
    filtered_results = [r for r in analysis_results if r['drug'] in selected_drugs]
    
    # Add traces for each drug (for legend toggling)
    drugs_added = set()
    
    for result in filtered_results:
        drug = result['drug']
        ticker = result['ticker']
        
        # Skip if category not in our order (e.g., Baseline Building)
        if result['category'] not in category_x:
            continue
        
        x_val = category_x[result['category']]
        y_val = result['stock_return']
        
        # Add jitter to x for overlapping points
        x_jitter = x_val + np.random.uniform(-0.15, 0.15)
        
        showlegend = drug not in drugs_added
        if showlegend:
            drugs_added.add(drug)
        
        fig.add_trace(go.Scatter(
            x=[x_jitter],
            y=[y_val],
            mode='markers+text',
            name=f"{drug} ({ticker})",
            text=[f"{ticker}<br>{result['trx_date'].strftime('%m/%d')}"],
            textposition='top center',
            textfont=dict(size=9),
            marker=dict(
                size=14,
                color=color_map.get(result['category'], '#888888'),
                line=dict(color='white', width=1),
                symbol='circle'
            ),
            legendgroup=drug,
            showlegend=showlegend,
            hovertemplate=(
                f"<b>{drug} ({ticker})</b><br>"
                f"TRx Week: {result['trx_date'].strftime('%Y-%m-%d')}<br>"
                f"Category: {result['category']}<br>"
                f"Stock Return: {result['stock_return']:.2f}%<br>"
                f"Release Date: {result['release_date'].strftime('%Y-%m-%d')}<br>"
                f"Return Window: {result['release_date'].strftime('%m/%d')} → {result['end_date'].strftime('%m/%d')}<br>"
                "<extra></extra>"
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="TRx Performance Category vs. One-Week Stock Return",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="TRx Performance Category",
            tickmode='array',
            tickvals=list(range(len(category_order))),
            ticktext=category_order,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title="One-Week Stock Return (%)",
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.5)',
            zerolinewidth=1
        ),
        height=600,
        hovermode='closest',
        legend=dict(
            title="Drugs (click to toggle)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add category color bands
    for i, cat in enumerate(category_order):
        fig.add_vrect(
            x0=i-0.4, x1=i+0.4,
            fillcolor=color_map[cat],
            opacity=0.1,
            layer="below",
            line_width=0
        )
    
    return fig


def load_ishara_rapid_data(file):
    """Load and parse the Ishara Rapid Excel file with multi-drug TRx data."""
    df = pd.read_excel(file)
    
    # Find the Week column
    week_col = None
    for col in df.columns:
        if 'week' in str(col).lower():
            week_col = col
            break
    
    if week_col is None:
        week_col = df.columns[0]  # Assume first column is date
    
    # Extract drug data
    drugs_data = {}
    for col in df.columns:
        if col == week_col:
            continue
        # Parse column name to get drug and metric
        col_str = str(col).replace('\n', ' ').strip()
        parts = col_str.split()
        if len(parts) >= 2:
            drug_name = parts[0]
            metric = parts[-1] if len(parts) > 1 else 'TRx'
        else:
            drug_name = col_str
            metric = 'TRx'
        
        # Only include TRx metrics for now
        if 'TRx' in col_str or 'trx' in col_str.lower():
            if drug_name not in drugs_data:
                drugs_data[drug_name] = {}
            
            # Create DataFrame for this drug
            drug_df = pd.DataFrame({
                'date': pd.to_datetime(df[week_col], errors='coerce'),
                'scripts': pd.to_numeric(df[col], errors='coerce')
            }).dropna()
            
            if 'EUTRx' in col_str or 'EUTRX' in col_str:
                drugs_data[drug_name]['EUTRx'] = drug_df
            else:
                drugs_data[drug_name]['TRx'] = drug_df
    
    return drugs_data


def load_stock_data(file, file_type='bloomberg'):
    """
    Load stock price data from various formats.
    
    Args:
        file: Uploaded file object
        file_type: 'bloomberg' for Bloomberg export, 'simple' for date/price CSV
    
    Returns:
        DataFrame with 'date' and 'price' columns
    """
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    
    # Try to auto-detect format
    if file_type == 'bloomberg':
        # Bloomberg format: has metadata rows at top
        # Find the row with "Date" header
        date_row_idx = None
        for idx, row in df.iterrows():
            if 'Date' in str(row.values):
                date_row_idx = idx
                break
        
        if date_row_idx is not None:
            # Re-read starting from the data rows
            df_data = df.iloc[date_row_idx+1:].copy()
            df_data.columns = ['date', 'price', 'volume'] if len(df.columns) >= 3 else ['date', 'price']
        else:
            # Assume first column is date, second is price
            df_data = df.iloc[:, :2].copy()
            df_data.columns = ['date', 'price']
    else:
        # Simple format: first col = date, second col = price
        df_data = df.iloc[:, :2].copy()
        df_data.columns = ['date', 'price']
    
    df_data['date'] = pd.to_datetime(df_data['date'], errors='coerce')
    df_data['price'] = pd.to_numeric(df_data['price'], errors='coerce')
    df_data = df_data.dropna()
    df_data = df_data.sort_values('date')
    
    return df_data


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Drug Script Analysis Tool",
    page_icon="💊",
    layout="wide"
)

# Title
st.title("💊 Drug Script Analysis Tool")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2 = st.tabs(["📊 Drug Script Analysis", "📈 TRx-Stock Performance Analysis"])

# ============================================================================
# TAB 1: EXISTING DRUG SCRIPT ANALYSIS (UNCHANGED)
# ============================================================================

with tab1:
    st.markdown("""
    Analyze prescription data using WoW and Z-Score methods.
    Upload an Excel file or CSV to get started.
    """)
    
    # Sidebar for Tab 1 settings
    with st.sidebar:
        st.header("⚙️ Tab 1 Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with date and script columns",
            key="tab1_upload"
        )
        
        st.divider()
        st.header("📊 Analysis Method")
        analysis_method = st.radio(
            "Select analysis method(s)",
            options=["Both WoW & Z-Score", "WoW Method Only", "Z-Score Method Only"],
            index=0,
            help="Choose which analysis methods to run"
        )
        
        st.divider()
        st.header("🎛️ Custom Thresholds")
        st.markdown("Leave blank for automatic detection")
        
        # WoW thresholds
        st.subheader("WoW Method")
        wow_inline = st.number_input(
            "WoW In-Line Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=1.0,
            help="Percentage change threshold for in-line classification"
        )
        
        wow_slight = st.number_input(
            "WoW Slight Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=1.0,
            help="Percentage change threshold for slight changes"
        )
        
        wow_meaningful = st.number_input(
            "WoW Meaningful Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=1.0,
            help="Percentage change threshold for meaningful changes"
        )
        
        st.divider()
        st.subheader("Z-Score Method")
        zscore_slight = st.number_input(
            "Z-Score Slight Threshold",
            min_value=0.0,
            max_value=5.0,
            value=None,
            step=0.1,
            help="Z-score threshold for slight changes (e.g., 1.0 = 1 standard deviation)"
        )
        
        zscore_meaningful = st.number_input(
            "Z-Score Meaningful Threshold",
            min_value=0.0,
            max_value=5.0,
            value=None,
            step=0.1,
            help="Z-score threshold for meaningful changes (e.g., 2.0 = 2 standard deviations)"
        )
        
        st.divider()
        baseline_window = st.number_input(
            "Baseline Window (weeks)",
            min_value=1,
            max_value=52,
            value=None,
            step=1,
            help="Number of weeks to use for baseline calculation"
        )
    
    # Main content area for Tab 1
    if uploaded_file is not None:
        # Process file
        with st.spinner("Processing your file..."):
            try:
                # Read the file
                if uploaded_file.name.endswith('.xlsx'):
                    df_excel = pd.read_excel(uploaded_file)
                    
                    # Show available columns
                    st.info("📊 Available columns in your Excel file:")
                    st.write(df_excel.columns.tolist())
                    
                    # Let user select columns
                    col1, col2 = st.columns(2)
                    with col1:
                        date_col = st.selectbox("Select Date Column", df_excel.columns)
                    with col2:
                        value_col = st.selectbox("Select Value Column", df_excel.columns)
                    
                    # Process data
                    if st.button("✅ Process Data", type="primary", key="tab1_process"):
                        df_processed = pd.DataFrame()
                        df_processed['date'] = pd.to_datetime(df_excel[date_col], errors='coerce')
                        df_processed['scripts'] = pd.to_numeric(df_excel[value_col], errors='coerce')
                        
                        # Filter out summary rows (Grand Total, Total, etc.)
                        df_processed = df_processed.dropna()
                        
                        # Additional filter: remove rows where date column contains text like "Grand Total"
                        if date_col in df_excel.columns:
                            summary_keywords = ['total', 'grand', 'summary', 'subtotal']
                            mask = df_excel[date_col].astype(str).str.lower().str.contains('|'.join(summary_keywords), na=False)
                            df_processed = df_processed[~mask]
                        
                        # Save to temporary CSV
                        csv_path = "temp_data.csv"
                        df_processed.to_csv(csv_path, index=False)
                        
                        st.success(f"✅ Processed {len(df_processed)} weeks of data")
                        st.session_state['processed_csv'] = csv_path
                else:
                    # CSV file - save and store
                    csv_path = "temp_data.csv"
                    with open(csv_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state['processed_csv'] = csv_path
                    
                    # Show preview
                    df_preview = pd.read_csv(csv_path)
                    st.success("✅ CSV file loaded")
                    st.dataframe(df_preview.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.stop()
    
    # Run analysis button
    if 'processed_csv' in st.session_state:
        st.divider()
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True, key="tab1_run"):
            with st.spinner("Running analysis... This may take a moment."):
                try:
                    # Run the analysis
                    fig, df_wow, df_zscore, differences = main(
                        filepath=st.session_state['processed_csv'],
                        custom_wow_inline=wow_inline if wow_inline else None,
                        custom_wow_slight=wow_slight if wow_slight else None,
                        custom_wow_meaningful=wow_meaningful if wow_meaningful else None,
                        custom_zscore_slight=zscore_slight if zscore_slight else None,
                        custom_zscore_meaningful=zscore_meaningful if zscore_meaningful else None,
                        custom_baseline_window=int(baseline_window) if baseline_window else None
                    )
                    
                    # Display results based on selected method
                    st.success("✅ Analysis Complete!")
                    
                    # Show the appropriate visualization based on method selection
                    if analysis_method == "Both WoW & Z-Score":
                        st.plotly_chart(fig, use_container_width=True)
                    elif analysis_method == "WoW Method Only":
                        wow_fig = create_wow_only_chart(df_wow)
                        st.plotly_chart(wow_fig, use_container_width=True)
                    else:
                        zscore_fig = create_zscore_only_chart(df_zscore)
                        st.plotly_chart(zscore_fig, use_container_width=True)
                    
                    # Display summary statistics
                    st.header("📊 Summary Statistics")
                    
                    if analysis_method == "Both WoW & Z-Score":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("WoW Method")
                            wow_counts = df_wow['classification'].value_counts()
                            st.bar_chart(wow_counts)
                        with col2:
                            st.subheader("Z-Score Method")
                            zscore_counts = df_zscore['classification'].value_counts()
                            st.bar_chart(zscore_counts)
                    elif analysis_method == "WoW Method Only":
                        st.subheader("WoW Method")
                        wow_counts = df_wow['classification'].value_counts()
                        st.bar_chart(wow_counts)
                    else:
                        st.subheader("Z-Score Method")
                        zscore_counts = df_zscore['classification'].value_counts()
                        st.bar_chart(zscore_counts)
                    
                    # Display data tables
                    st.header("📋 Detailed Results")
                    
                    if analysis_method in ["Both WoW & Z-Score", "WoW Method Only"]:
                        with st.expander("📊 WoW Results Table"):
                            st.dataframe(df_wow, use_container_width=True)
                    
                    if analysis_method in ["Both WoW & Z-Score", "Z-Score Method Only"]:
                        with st.expander("📈 Z-Score Results Table"):
                            st.dataframe(df_zscore, use_container_width=True)
                    
                    if analysis_method == "Both WoW & Z-Score" and len(differences) > 0:
                        with st.expander(f"🔍 Disagreements ({len(differences)} weeks)"):
                            st.dataframe(differences, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error running analysis: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("👆 Please upload a file in the sidebar to begin")


# ============================================================================
# TAB 2: TRx-STOCK PERFORMANCE ANALYSIS (NEW)
# ============================================================================

with tab2:
    st.markdown("""
    ### Analyze TRx Performance Categories vs. Stock Returns
    
    **Analysis Logic:**
    1. **TRx Performance Date:** Week ending date (e.g., 10/10/2024)
    2. **Data Release Date:** 7 days after TRx week end (e.g., 10/17/2024)
    3. **Stock Performance Window:** Release date → 7 days forward (e.g., 10/17 → 10/24)
    
    Does knowing a drug's TRx category help predict the following week's stock movement?
    """)
    
    st.divider()
    
    # Drug-to-Ticker mapping (user can customize)
    st.subheader("📋 Step 1: Configure Drug-Stock Mapping")
    
    default_mappings = {
        'NEFFY': 'SPRY',
        'REZDIFFRA': 'MDGL', 
        'VOQUEZNA': 'PHAT',
        'XDEMVY': 'VTRS',
        'XPHOZAH': 'AKBA',
        'ZORYVE': 'ARQT'
    }
    
    with st.expander("🔧 Drug-to-Stock Ticker Mapping", expanded=True):
        st.markdown("Edit the ticker symbols for each drug:")
        
        col1, col2, col3 = st.columns(3)
        drug_ticker_map = {}
        
        drugs = list(default_mappings.keys())
        for i, drug in enumerate(drugs):
            with [col1, col2, col3][i % 3]:
                ticker = st.text_input(
                    drug,
                    value=default_mappings[drug],
                    key=f"ticker_{drug}"
                )
                drug_ticker_map[drug] = ticker
    
    st.divider()
    
    # File uploads
    st.subheader("📁 Step 2: Upload Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**TRx Data (Ishara Rapid Excel)**")
        trx_file = st.file_uploader(
            "Upload TRx data file",
            type=['xlsx', 'csv'],
            help="Excel file with weekly TRx data for multiple drugs",
            key="tab2_trx"
        )
    
    with col2:
        st.markdown("**Stock Price Data**")
        stock_file = st.file_uploader(
            "Upload stock price file",
            type=['xlsx', 'csv'],
            help="Stock price data (Bloomberg export or simple date/price format)",
            key="tab2_stock"
        )
        
        stock_ticker_input = st.text_input(
            "Stock ticker for uploaded file",
            value="SPRY",
            help="Enter the ticker symbol for the uploaded stock data"
        )
    
    st.divider()
    
    # Analysis settings
    st.subheader("⚙️ Step 3: Analysis Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        classification_method = st.selectbox(
            "TRx Classification Method",
            options=["Z-Score", "WoW"],
            index=0,
            help="Method to classify TRx performance"
        )
    
    with col2:
        release_delay = st.number_input(
            "Data Release Delay (days)",
            min_value=1,
            max_value=14,
            value=7,
            help="Days after TRx week end until data is released"
        )
    
    with col3:
        return_window = st.number_input(
            "Stock Return Window (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Days to measure stock return after release"
        )
    
    st.divider()
    
    # Run Analysis
    if st.button("🚀 Run TRx-Stock Analysis", type="primary", use_container_width=True, key="tab2_run"):
        
        if trx_file is None:
            st.error("❌ Please upload a TRx data file")
        elif stock_file is None:
            st.error("❌ Please upload a stock price data file")
        else:
            with st.spinner("Analyzing TRx-Stock relationship..."):
                try:
                    # Load TRx data
                    drugs_data = load_ishara_rapid_data(trx_file)
                    st.success(f"✅ Loaded TRx data for {len(drugs_data)} drugs: {list(drugs_data.keys())}")
                    
                    # Load stock data
                    stock_data = load_stock_data(stock_file, file_type='bloomberg')
                    st.success(f"✅ Loaded {len(stock_data)} stock price records for {stock_ticker_input}")
                    
                    # Store stock data by ticker
                    stock_by_ticker = {stock_ticker_input: stock_data}
                    
                    # Calculate TRx categories and match with stock returns
                    analysis_results = []
                    
                    for drug_name, drug_metrics in drugs_data.items():
                        ticker = drug_ticker_map.get(drug_name)
                        
                        if ticker is None or ticker not in stock_by_ticker:
                            continue
                        
                        # Get TRx data (prefer TRx over EUTRx)
                        if 'TRx' in drug_metrics:
                            df_trx = drug_metrics['TRx']
                        elif 'EUTRx' in drug_metrics:
                            df_trx = drug_metrics['EUTRx']
                        else:
                            continue
                        
                        # Skip if not enough data
                        if len(df_trx) < 8:
                            continue
                        
                        # Calculate classifications
                        method = 'zscore' if classification_method == "Z-Score" else 'wow'
                        df_classified = calculate_trx_category(df_trx, method=method)
                        
                        # Get stock data
                        df_stock = stock_by_ticker[ticker]
                        
                        # For each classified week, calculate forward stock return
                        for _, row in df_classified.iterrows():
                            if row['classification'] == 'Baseline Building':
                                continue
                            
                            trx_date = row['date']
                            release_date = trx_date + timedelta(days=release_delay)
                            end_date = release_date + timedelta(days=return_window)
                            
                            # Get stock return
                            stock_return = get_stock_return(df_stock, release_date, end_date)
                            
                            if stock_return is not None:
                                analysis_results.append({
                                    'drug': drug_name,
                                    'ticker': ticker,
                                    'trx_date': trx_date,
                                    'release_date': release_date,
                                    'end_date': end_date,
                                    'category': row['classification'],
                                    'scripts': row['scripts'],
                                    'stock_return': stock_return
                                })
                    
                    if len(analysis_results) == 0:
                        st.warning("⚠️ No matching data found. Make sure the stock ticker matches your drug-ticker mapping.")
                    else:
                        st.success(f"✅ Generated {len(analysis_results)} data points for analysis")
                        
                        # Drug selection for visualization
                        st.subheader("📊 Results")
                        
                        available_drugs = list(set(r['drug'] for r in analysis_results))
                        selected_drugs = st.multiselect(
                            "Select drugs to display",
                            options=available_drugs,
                            default=available_drugs,
                            help="Toggle individual drugs on/off"
                        )
                        
                        # Create and display scatter plot
                        if len(selected_drugs) > 0:
                            fig = create_trx_stock_scatter(analysis_results, selected_drugs)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary statistics
                            st.subheader("📈 Summary Statistics")
                            
                            df_results = pd.DataFrame(analysis_results)
                            df_selected = df_results[df_results['drug'].isin(selected_drugs)]
                            
                            # Group by category
                            category_stats = df_selected.groupby('category').agg({
                                'stock_return': ['mean', 'std', 'count']
                            }).round(2)
                            category_stats.columns = ['Avg Return (%)', 'Std Dev (%)', 'Count']
                            
                            st.dataframe(category_stats, use_container_width=True)
                            
                            # Full results table
                            with st.expander("📋 Full Results Table"):
                                df_display = df_selected[['drug', 'ticker', 'trx_date', 'category', 'scripts', 'stock_return']].copy()
                                df_display['trx_date'] = df_display['trx_date'].dt.strftime('%Y-%m-%d')
                                df_display['stock_return'] = df_display['stock_return'].round(2)
                                st.dataframe(df_display, use_container_width=True)
                        else:
                            st.info("👆 Select at least one drug to display the chart")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**About:** This tool provides two analysis views:
- **Tab 1:** Individual drug script analysis using WoW and Z-Score methods
- **Tab 2:** Cross-drug TRx-Stock performance correlation analysis
""")
