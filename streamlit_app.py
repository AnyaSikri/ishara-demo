"""
Streamlit Web App for Drug Script Analysis
Converts the notebook analysis into an interactive web interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys

# Import your existing analysis functions
sys.path.append('.')
from scripts import main

# Page configuration
st.set_page_config(
    page_title="Drug Script Analysis Tool",
    page_icon="💊",
    layout="wide"
)

# Title and description
st.title("💊 Drug Script Analysis Tool")
st.markdown("""
Analyze prescription data using WoW and Z-Score methods.
Upload an Excel file or CSV to get started.
""")

# CUSTOMIZE: Change colors, add logo, etc.
# st.markdown("""
# <style>
# .stApp {
#     background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
# }
# </style>
# """, unsafe_allow_html=True)
# st.image("your-logo.png")  # Add your logo

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file with date and script columns"
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

# Main content area
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
                if st.button("✅ Process Data", type="primary"):
                    df_processed = pd.DataFrame()
                    df_processed['date'] = pd.to_datetime(df_excel[date_col], errors='coerce')
                    df_processed['scripts'] = pd.to_numeric(df_excel[value_col], errors='coerce')
                    
                    # Filter out summary rows (Grand Total, Total, etc.)
                    # Remove rows where date is invalid or scripts is NaN
                    df_processed = df_processed.dropna()
                    
                    # Additional filter: remove rows where date column contains text like "Grand Total", "Total", etc.
                    # Check the original date column for these patterns
                    if date_col in df_excel.columns:
                        summary_keywords = ['total', 'grand', 'summary', 'subtotal']
                        mask = df_excel[date_col].astype(str).str.lower().str.contains('|'.join(summary_keywords), na=False)
                        # Invert mask to keep rows NOT containing keywords
                        df_processed = df_processed[~mask]
                    
                    # Save to temporary CSV
                    csv_path = "temp_data.csv"
                    df_processed.to_csv(csv_path, index=False)
                    
                    st.success(f"✅ Processed {len(df_processed)} weeks of data")
                    
                    # Store in session state
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
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
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
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Show the main visualization
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary statistics
                st.header("📊 Summary Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("WoW Method")
                    wow_counts = df_wow['classification'].value_counts()
                    st.bar_chart(wow_counts)
                
                with col2:
                    st.subheader("Z-Score Method")
                    zscore_counts = df_zscore['classification'].value_counts()
                    st.bar_chart(zscore_counts)
                
                # Display data tables
                with st.expander("📋 WoW Results Table"):
                    st.dataframe(df_wow, use_container_width=True)
                
                with st.expander("📋 Z-Score Results Table"):
                    st.dataframe(df_zscore, use_container_width=True)
                
                # Display disagreements
                if len(differences) > 0:
                    with st.expander(f"🔍 Disagreements ({len(differences)} weeks)"):
                        st.dataframe(differences, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")
                st.exception(e)

else:
    st.info("👆 Please upload a file to begin")

# Footer
st.divider()
st.markdown("""
---
**About:** This tool compares two analysis methods: Week-over-Week (WoW) percentage change 
and Z-Score statistical analysis to identify trends in prescription data.
""")

