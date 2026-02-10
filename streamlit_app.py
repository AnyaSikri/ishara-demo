"""Drug Script Analysis Tool - Streamlit Application"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta
from scipy import stats
import sys

sys.path.append('.')
from scripts import main, load_data, flag_holiday_weeks, classify_drug_maturity, classify_wow_method, classify_zscore_method

def display_dataframe(df, max_rows=100):
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        st.caption(f"Showing first {max_rows} of {len(df)} rows")
    else:
        df_display = df
    
    # Convert to HTML and display
    html = df_display.to_html(index=False, classes='dataframe', border=0)
    
    # Add some basic styling
    styled_html = f"""
    <style>
        .dataframe {{
            font-size: 12px;
            border-collapse: collapse;
            width: 100%;
        }}
        .dataframe th {{
            background-color: #f0f2f6;
            padding: 8px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-weight: 600;
        }}
        .dataframe td {{
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
        }}
        .dataframe tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
    {html}
    """
    st.markdown(styled_html, unsafe_allow_html=True)

def create_wow_only_chart(df_wow):
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

# NEW: TRx-STOCK PERFORMANCE ANALYSIS FUNCTIONS

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
    
    # Flag holidays
    df = flag_holiday_weeks(df)
    
    # Get maturity classification for thresholds
    maturity = classify_drug_maturity(df)
    
    # Run classification row by row (matching how main() does it)
    results = []
    
    for idx, row in df.iterrows():
        week_number = idx + 1  # 1-based week numbering
        current_week_scripts = row['scripts']
        
        if method == 'zscore':
            result = classify_zscore_method(
                df,
                current_week_scripts,
                week_number,
                maturity['baseline_window']
            )
            results.append({
                'week_number': week_number,
                'date': row['date'],
                'scripts': current_week_scripts,
                'is_holiday_week': row['is_holiday_week'],
                'holiday_name': row['holiday_name'],
                'classification': result['classification'],
                'z_score': result['z_score']
            })
        else:
            result = classify_wow_method(
                df,
                current_week_scripts,
                week_number,
                maturity['wow_thresholds']
            )
            results.append({
                'week_number': week_number,
                'date': row['date'],
                'scripts': current_week_scripts,
                'is_holiday_week': row['is_holiday_week'],
                'holiday_name': row['holiday_name'],
                'classification': result['classification'],
                'wow_pct': result['wow_pct']
            })
    
    return pd.DataFrame(results)

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

def create_trx_stock_scatter(analysis_results, selected_drugs, lag_data=None):
    """
    Create interactive scatter plot with Z-Score on X-axis vs Stock Returns on Y-axis.
    Points colored by classification category.
    
    Args:
        analysis_results: List of dicts with drug analysis results
        selected_drugs: List of drug names to display
        lag_data: Optional list with z_score data
    
    Returns:
        Plotly figure
    """
    # Color map for categories
    color_map = {
        'Meaningfully Below': '#e74c3c',   # Red
        'Slightly Below': '#f39c12',        # Orange
        'In-Line': '#3498db',               # Blue
        'Slightly Above': '#9b59b6',        # Purple
        'Meaningfully Above': '#2ecc71',    # Green
        'Baseline Building': '#bdc3c7'      # Gray
    }
    
    fig = go.Figure()
    
    # Filter to selected drugs
    filtered_lag = [d for d in (lag_data or []) if d['drug'] in selected_drugs and d.get('z_score') is not None]
    
    if len(filtered_lag) == 0:
        # Fallback: no z-score data available
        fig.add_annotation(
            text="Z-Score data not available. Use Z-Score classification method.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Calculate axis ranges
    all_z = [d['z_score'] for d in filtered_lag]
    all_returns = [d.get('return_lag_1', 0) for d in filtered_lag if d.get('return_lag_1') is not None]
    
    z_min = min(all_z) - 0.5 if all_z else -3
    z_max = max(all_z) + 0.5 if all_z else 3
    z_min, z_max = max(z_min, -4), min(z_max, 4)
    
    ret_min = min(all_returns) - 2 if all_returns else -15
    ret_max = max(all_returns) + 2 if all_returns else 15
    
    # Add colored background zones for Z-score regions
    zones = [
        (-4, -1.0, 'rgba(231,76,60,0.1)', 'Meaningfully Below'),
        (-1.0, -0.3, 'rgba(243,156,18,0.1)', 'Slightly Below'),
        (-0.3, 0.3, 'rgba(52,152,219,0.1)', 'In-Line'),
        (0.3, 1.0, 'rgba(155,89,182,0.1)', 'Slightly Above'),
        (1.0, 4, 'rgba(46,204,113,0.1)', 'Meaningfully Above')
    ]
    
    for x0, x1, color, label in zones:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, layer="below", line_width=0)
    
    # Add zone labels at top
    zone_labels = [
        (-2.5, 'Meaningfully\nBelow', '#e74c3c'),
        (-0.65, 'Slightly\nBelow', '#f39c12'),
        (0, 'In-Line', '#3498db'),
        (0.65, 'Slightly\nAbove', '#9b59b6'),
        (2.5, 'Meaningfully\nAbove', '#2ecc71')
    ]
    
    for x_pos, label, color in zone_labels:
        if z_min <= x_pos <= z_max:
            fig.add_annotation(
                x=x_pos, y=ret_max - 1,
                text=label,
                showarrow=False,
                font=dict(size=9, color=color),
                opacity=0.8
            )
    
    # Plot each data point
    for d in filtered_lag:
        z_score = d['z_score']
        stock_return = d.get('return_lag_1')
        
        if stock_return is None:
            continue
        
        category = d['category']
        drug = d['drug']
        
        fig.add_trace(go.Scatter(
            x=[z_score],
            y=[stock_return],
            mode='markers',
            name=drug,
            marker=dict(
                size=12,
                color=color_map.get(category, '#888'),
                opacity=0.8,
                line=dict(color='white', width=1.5),
                symbol='circle'
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{drug}</b><br>"
                f"Week: {d['trx_date'].strftime('%b %d, %Y')}<br>"
                f"Z-Score: {z_score:.2f}<br>"
                f"Category: {category}<br>"
                f"Stock Return: {stock_return:+.2f}%<br>"
                "<extra></extra>"
            )
        ))
    
    # Add trendline
    if len(filtered_lag) >= 3:
        valid_data = [(d['z_score'], d['return_lag_1']) for d in filtered_lag if d.get('return_lag_1') is not None]
        if len(valid_data) >= 3:
            z_vals = [v[0] for v in valid_data]
            ret_vals = [v[1] for v in valid_data]
            try:
                slope, intercept, r_value, p_value, _ = stats.linregress(z_vals, ret_vals)
                x_line = np.array([min(z_vals), max(z_vals)])
                y_line = slope * x_line + intercept
                
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    name='Trendline',
                    line=dict(color='#2c3e50', width=2, dash='dash'),
                    showlegend=True,
                    hovertemplate=f"Trendline (r={r_value:.3f})<extra></extra>"
                ))
                
                # Add correlation annotation
                corr_color = '#2ecc71' if r_value > 0.15 else ('#e74c3c' if r_value < -0.15 else '#7f8c8d')
                fig.add_annotation(
                    x=z_max - 0.3, y=ret_max - 2,
                    text=f"<b>r = {r_value:.3f}</b>",
                    showarrow=False,
                    font=dict(size=14, color=corr_color),
                    bgcolor="white",
                    borderpad=4
                )
            except:
                pass
    
    # Add legend for categories
    for cat, color in color_map.items():
        if cat != 'Baseline Building':
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name=cat,
                marker=dict(size=10, color=color),
                showlegend=True
            ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6", line_width=2)
    fig.add_vline(x=0, line_dash="solid", line_color="#95a5a6", line_width=1)
    
    # Z-score threshold lines
    fig.add_vline(x=-1.0, line_dash="dot", line_color="#e74c3c", line_width=1, opacity=0.5)
    fig.add_vline(x=1.0, line_dash="dot", line_color="#2ecc71", line_width=1, opacity=0.5)
    
    fig.update_layout(
        title=dict(
            text="<b>TRx Z-Score vs. Stock Return</b>",
            font=dict(size=20),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="TRx Z-Score (How Unusual Were Prescriptions?)", font=dict(size=13)),
            range=[z_min, z_max],
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=True,
            zerolinecolor='#95a5a6',
            zerolinewidth=1,
            dtick=0.5
        ),
        yaxis=dict(
            title=dict(text="Stock Return % (Week After Data Release)", font=dict(size=13)),
            range=[ret_min, ret_max],
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=True,
            zerolinecolor='#95a5a6',
            zerolinewidth=2
        ),
        height=550,
        legend=dict(
            title="<b>Classification</b>",
            yanchor="top", y=0.99,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    return fig


def create_lag_analysis_dashboard(lag_data, selected_drugs):
    """
    Create multi-panel scatter plot dashboard for lag correlation analysis.
    
    8 panels (2 rows × 4 cols) showing lags 1-8 weeks, plus a 9th correlation summary panel.
    
    Args:
        lag_data: List of dicts with z_score, stock returns at various lags, drug info
        selected_drugs: List of drug names to display
    
    Returns:
        Plotly figure with subplots
    """
    
    # Color palette for drugs
    drug_colors = {
        'NEFFY': '#1f77b4',
        'REZDIFFRA': '#ff7f0e', 
        'VOQUEZNA': '#2ca02c',
        'XDEMVY': '#d62728',
        'XPHOZAH': '#9467bd',
        'ZORYVE': '#8c564b',
        'ARISTADA': '#e377c2',
        'AUSTEDO': '#7f7f7f',
        'INGREZZA': '#bcbd22',
        'LINZESS': '#17becf'
    }
    
    # Default color for unknown drugs
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Filter data
    filtered_data = [d for d in lag_data if d['drug'] in selected_drugs]
    
    if len(filtered_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create subplots: 3 rows × 3 cols (8 lag panels + 1 correlation summary)
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f'Lag {i} Week{"s" if i > 1 else ""}' for i in range(1, 9)] + ['Correlation by Lag'],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Calculate axis ranges for consistency
    all_z_scores = [d['z_score'] for d in filtered_data if d['z_score'] is not None]
    all_returns = []
    for lag in range(1, 9):
        all_returns.extend([d.get(f'return_lag_{lag}', None) for d in filtered_data if d.get(f'return_lag_{lag}') is not None])
    
    if len(all_z_scores) == 0 or len(all_returns) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for lag analysis", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    z_min, z_max = min(all_z_scores) - 0.5, max(all_z_scores) + 0.5
    z_min, z_max = max(z_min, -4), min(z_max, 4)  # Clamp to reasonable range
    ret_min, ret_max = min(all_returns) - 2, max(all_returns) + 2
    
    # Track correlations for summary panel
    correlations = []
    
    # Assign colors to drugs
    unique_drugs = list(set(d['drug'] for d in filtered_data))
    drug_color_map = {}
    for i, drug in enumerate(unique_drugs):
        if drug in drug_colors:
            drug_color_map[drug] = drug_colors[drug]
        else:
            drug_color_map[drug] = default_colors[i % len(default_colors)]
    
    # Create each lag panel
    for lag in range(1, 9):
        row = (lag - 1) // 3 + 1
        col = (lag - 1) % 3 + 1
        
        # Get data for this lag
        lag_key = f'return_lag_{lag}'
        valid_data = [d for d in filtered_data if d.get(lag_key) is not None and d['z_score'] is not None]
        
        if len(valid_data) < 2:
            correlations.append(0)
            continue
        
        # Calculate correlation
        z_scores = [d['z_score'] for d in valid_data]
        returns = [d[lag_key] for d in valid_data]
        
        try:
            corr, p_value = stats.pearsonr(z_scores, returns)
        except:
            corr = 0
        
        correlations.append(corr)
        
        # Add scatter points for each drug
        drugs_added = set()
        for d in valid_data:
            drug = d['drug']
            showlegend = (lag == 1) and (drug not in drugs_added)  # Only show legend on first panel
            if showlegend:
                drugs_added.add(drug)
            
            fig.add_trace(
                go.Scatter(
                    x=[d['z_score']],
                    y=[d[lag_key]],
                    mode='markers',
                    name=drug,
                    marker=dict(
                        size=8,
                        color=drug_color_map.get(drug, '#888888'),
                        line=dict(color='white', width=0.5)
                    ),
                    legendgroup=drug,
                    showlegend=showlegend,
                    hovertemplate=(
                        f"<b>{drug}</b><br>"
                        f"TRx Week: {d['trx_date'].strftime('%Y-%m-%d')}<br>"
                        f"Z-Score: {d['z_score']:.2f}<br>"
                        f"Return (Lag {lag}): {d[lag_key]:.2f}%<br>"
                        f"Return End: {d.get(f'end_date_lag_{lag}', 'N/A')}<br>"
                        "<extra></extra>"
                    )
                ),
                row=row, col=col
            )
        
        # Add trendline (linear regression)
        if len(z_scores) >= 3:
            try:
                slope, intercept, _, _, _ = stats.linregress(z_scores, returns)
                x_line = np.array([min(z_scores), max(z_scores)])
                y_line = slope * x_line + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color='rgba(100,100,100,0.5)', width=2, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
            except:
                pass
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, row=row, col=col)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4, row=row, col=col)
        
        # Update panel title with correlation
        corr_color = '#2ca02c' if corr > 0.15 else ('#d62728' if corr < -0.15 else '#888888')
        title_text = f'Lag {lag} Week{"s" if lag > 1 else ""} (r={corr:.2f})'
        
        # Update subplot title
        fig.layout.annotations[lag-1].text = f'<b>Lag {lag}</b> <span style="color:{corr_color}">r={corr:.2f}</span>'
    
    # Add correlation summary bar chart (position 9 = row 3, col 3)
    bar_colors = ['#2ca02c' if c > 0.15 else ('#d62728' if c < -0.15 else '#888888') for c in correlations]
    
    fig.add_trace(
        go.Bar(
            x=[f'Lag {i}' for i in range(1, 9)],
            y=correlations,
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate='%{x}: r=%{y:.3f}<extra></extra>'
        ),
        row=3, col=3
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=3, col=3)
    fig.add_hline(y=0.15, line_dash="dash", line_color="green", opacity=0.3, row=3, col=3)
    fig.add_hline(y=-0.15, line_dash="dash", line_color="red", opacity=0.3, row=3, col=3)
    
    # Update all axes
    for lag in range(1, 9):
        row = (lag - 1) // 3 + 1
        col = (lag - 1) % 3 + 1
        
        fig.update_xaxes(range=[z_min, z_max], title_text="Z-Score" if row == 3 else "", row=row, col=col)
        fig.update_yaxes(range=[ret_min, ret_max], title_text="Return %" if col == 1 else "", row=row, col=col)
    
    # Update correlation panel axes
    fig.update_xaxes(title_text="", row=3, col=3)
    fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=3, col=3)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>TRx Z-Score vs. Forward Stock Returns: Lag Analysis</b>",
            font=dict(size=20),
            x=0.5
        ),
        height=900,
        width=1000,
        showlegend=True,
        legend=dict(
            title="Drugs (click to toggle)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_time_series_comparison(analysis_results, lag_data, stock_data, selected_drugs):
    """
    Create stacked time series showing TRx Z-scores and stock returns over time.
    
    Two panels:
    - Top: TRx Z-score over time (colored by classification)
    - Bottom: Stock price/returns over time
    """
    from plotly.subplots import make_subplots
    
    # Filter data
    filtered_results = [r for r in analysis_results if r['drug'] in selected_drugs]
    filtered_lag = [d for d in lag_data if d['drug'] in selected_drugs]
    
    if len(filtered_lag) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Drug colors
    drug_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    unique_drugs = list(set(d['drug'] for d in filtered_lag))
    drug_color_map = {drug: drug_colors[i % len(drug_colors)] for i, drug in enumerate(unique_drugs)}
    
    # Category colors for markers
    category_colors = {
        'Meaningfully Below': '#e74c3c',
        'Slightly Below': '#f39c12',
        'In-Line': '#3498db',
        'Slightly Above': '#9b59b6',
        'Meaningfully Above': '#2ecc71'
    }
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('<b>TRx Z-Score Over Time</b>', '<b>Stock Returns Over Time</b>'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
        shared_xaxes=True
    )
    
    # Top panel: TRx Z-scores
    for drug in unique_drugs:
        drug_data = sorted([d for d in filtered_lag if d['drug'] == drug], key=lambda x: x['trx_date'])
        
        if len(drug_data) == 0:
            continue
        
        dates = [d['trx_date'] for d in drug_data]
        z_scores = [d['z_score'] for d in drug_data]
        categories = [d['category'] for d in drug_data]
        marker_colors = [category_colors.get(c, '#888') for c in categories]
        
        # Line trace
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=z_scores,
                mode='lines',
                name=drug,
                line=dict(color=drug_color_map[drug], width=2),
                legendgroup=drug,
                showlegend=True,
                hovertemplate=f"<b>{drug}</b><br>Date: %{{x}}<br>Z-Score: %{{y:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Marker trace (colored by category)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=z_scores,
                mode='markers',
                name=f"{drug} (categories)",
                marker=dict(size=10, color=marker_colors, line=dict(color='white', width=1)),
                legendgroup=drug,
                showlegend=False,
                hovertemplate=f"<b>{drug}</b><br>Date: %{{x}}<br>Z-Score: %{{y:.2f}}<br>Category: %{{text}}<extra></extra>",
                text=categories
            ),
            row=1, col=1
        )
    
    # Bottom panel: Stock returns (1-week forward)
    for drug in unique_drugs:
        drug_data = sorted([d for d in filtered_lag if d['drug'] == drug and d.get('return_lag_1') is not None], 
                          key=lambda x: x['trx_date'])
        
        if len(drug_data) == 0:
            continue
        
        dates = [d['trx_date'] for d in drug_data]
        returns = [d['return_lag_1'] for d in drug_data]
        
        # Bar colors based on positive/negative
        bar_colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in returns]
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=returns,
                name=f"{drug} returns",
                marker_color=bar_colors,
                opacity=0.7,
                legendgroup=drug,
                showlegend=False,
                hovertemplate=f"<b>{drug}</b><br>Week: %{{x}}<br>1-Week Return: %{{y:.2f}}%<extra></extra>"
            ),
            row=2, col=1
        )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6", line_width=1, row=2, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="#2ecc71", line_width=1, opacity=0.5, row=1, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="#e74c3c", line_width=1, opacity=0.5, row=1, col=1)
    
    fig.update_layout(
        height=600,
        title=dict(text="<b>TRx Performance & Stock Returns Over Time</b>", x=0.5, font=dict(size=18)),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        plot_bgcolor='#fafbfc',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", row=1, col=1)
    fig.update_yaxes(title_text="Stock Return (%)", row=2, col=1)
    
    return fig


def create_scatter_with_time_color(analysis_results, selected_drugs):
    """
    Scatter plot with points colored by date (gradient from old to recent).
    """
    filtered_results = [r for r in analysis_results if r['drug'] in selected_drugs]
    
    if len(filtered_results) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Category order
    category_order = ['Meaningfully Below', 'Slightly Below', 'In-Line', 'Slightly Above', 'Meaningfully Above']
    category_x = {cat: i for i, cat in enumerate(category_order)}
    category_labels = ['Meaningfully\nBelow', 'Slightly\nBelow', 'In-Line', 'Slightly\nAbove', 'Meaningfully\nAbove']
    
    # Get date range for color scaling
    all_dates = [r['trx_date'] for r in filtered_results]
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = (max_date - min_date).days or 1
    
    fig = go.Figure()
    
    # Add points
    for result in filtered_results:
        if result['category'] not in category_x:
            continue
        
        x_val = category_x[result['category']] + np.random.uniform(-0.25, 0.25)
        y_val = result['stock_return']
        
        # Calculate color based on date (0 = oldest, 1 = most recent)
        days_from_start = (result['trx_date'] - min_date).days
        date_normalized = days_from_start / date_range
        
        fig.add_trace(go.Scatter(
            x=[x_val],
            y=[y_val],
            mode='markers',
            name=result['drug'],
            marker=dict(
                size=14,
                color=date_normalized,
                colorscale='Viridis',
                cmin=0,
                cmax=1,
                showscale=False,
                line=dict(color='white', width=1.5)
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{result['drug']}</b><br>"
                f"Date: {result['trx_date'].strftime('%b %d, %Y')}<br>"
                f"Category: {result['category']}<br>"
                f"Return: {result['stock_return']:+.2f}%<br>"
                "<extra></extra>"
            )
        ))
    
    # Add colorbar manually
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            cmin=0, cmax=1,
            colorbar=dict(
                title="Time",
                tickvals=[0, 0.5, 1],
                ticktext=[min_date.strftime('%b %Y'), '', max_date.strftime('%b %Y')],
                len=0.5,
                y=0.75
            ),
            showscale=True
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Calculate y range
    all_returns = [r['stock_return'] for r in filtered_results if r['category'] in category_x]
    y_min = min(all_returns) - 3 if all_returns else -10
    y_max = max(all_returns) + 3 if all_returns else 10
    
    fig.update_layout(
        title=dict(text="<b>TRx Category vs. Stock Return (Colored by Time)</b>", x=0.5, font=dict(size=18)),
        xaxis=dict(
            title="TRx Performance Category",
            tickmode='array',
            tickvals=list(range(len(category_order))),
            ticktext=category_labels,
            showgrid=False
        ),
        yaxis=dict(
            title="Stock Return (%)",
            range=[y_min, y_max],
            gridcolor='rgba(236,240,241,0.8)'
        ),
        height=500,
        plot_bgcolor='#fafbfc',
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6", line_width=2)
    
    return fig


def create_prescription_stock_line_chart(lag_data, stock_data, selected_drugs):
    """
    Create a dual-axis line chart showing prescriptions and stock price over time.
    
    Left Y-axis: Prescription counts
    Right Y-axis: Stock price
    X-axis: Time
    """
    from plotly.subplots import make_subplots
    
    # Filter lag data
    filtered_data = [d for d in lag_data if d['drug'] in selected_drugs]
    
    if len(filtered_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Drug colors
    drug_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    unique_drugs = list(set(d['drug'] for d in filtered_data))
    
    # Plot prescriptions for each drug (left axis)
    for i, drug in enumerate(unique_drugs):
        drug_data = sorted([d for d in filtered_data if d['drug'] == drug], key=lambda x: x['trx_date'])
        
        if len(drug_data) == 0:
            continue
        
        # Get prescription data from analysis_results (scripts field)
        dates = [d['trx_date'] for d in drug_data]
        
        # We need to get the scripts from the original data
        # For now, use z_score as proxy indicator, or get from lag_data if available
        # Actually, let's look for scripts in the data
        scripts = []
        for d in drug_data:
            # Try to find scripts value
            if 'scripts' in d:
                scripts.append(d['scripts'])
            else:
                scripts.append(None)
        
        # If we don't have scripts, skip this drug's prescription line
        if all(s is None for s in scripts):
            continue
        
        color = drug_colors[i % len(drug_colors)]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=scripts,
                mode='lines+markers',
                name=f'{drug} TRx',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{drug}</b><br>Date: %{{x}}<br>Prescriptions: %{{y:,.0f}}<extra></extra>"
            ),
            secondary_y=False
        )
    
    # Plot stock price (right axis)
    if stock_data is not None and len(stock_data) > 0:
        stock_df = stock_data.sort_values('date')
        
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=stock_df['price'],
                mode='lines',
                name='Stock Price',
                line=dict(color='#2c3e50', width=3),
                opacity=0.7,
                hovertemplate="<b>Stock</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Prescriptions & Stock Price Over Time</b>",
            font=dict(size=18),
            x=0.5
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)'
        ),
        height=450,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=1.05,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="<b>Weekly Prescriptions</b>",
        secondary_y=False,
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        titlefont=dict(color='#1f77b4')
    )
    fig.update_yaxes(
        title_text="<b>Stock Price ($)</b>",
        secondary_y=True,
        showgrid=False,
        titlefont=dict(color='#2c3e50')
    )
    
    return fig


def create_multi_drug_chart(drugs_data, selected_drugs, selected_metric='TRx'):
    """
    Create a chart showing multiple drugs on the same graph.
    
    Args:
        drugs_data: Dict of drug name -> {'TRx': df, 'EUTRx': df}
        selected_drugs: List of drugs to display
        selected_metric: 'TRx' or 'EUTRx'
    """
    drug_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fig = go.Figure()
    
    for i, drug in enumerate(selected_drugs):
        if drug not in drugs_data:
            continue
        
        drug_metrics = drugs_data[drug]
        
        # Get the appropriate metric
        if selected_metric in drug_metrics:
            df = drug_metrics[selected_metric]
        elif 'TRx' in drug_metrics:
            df = drug_metrics['TRx']
        elif 'EUTRx' in drug_metrics:
            df = drug_metrics['EUTRx']
        else:
            continue
        
        df = df.sort_values('date')
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['scripts'],
            mode='lines+markers',
            name=drug,
            line=dict(color=drug_colors[i % len(drug_colors)], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{drug}</b><br>Date: %{{x}}<br>{selected_metric}: %{{y:,.0f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text=f"<b>Multi-Drug {selected_metric} Comparison</b>", x=0.5, font=dict(size=20)),
        xaxis=dict(title="Date", showgrid=True, gridcolor='rgba(236,240,241,0.8)'),
        yaxis=dict(title=f"Weekly {selected_metric}", showgrid=True, gridcolor='rgba(236,240,241,0.8)'),
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        plot_bgcolor='#fafbfc',
        paper_bgcolor='white',
        hovermode='x unified'
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


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Drug Script Analysis Tool",
    page_icon="💊",
    layout="wide"
)

st.title("Drug Script Analysis Tool")

tab1, tab2 = st.tabs(["Drug Script Analysis", "TRx-Stock Performance Analysis"])

with tab1:
    
    # Analysis mode selector
    tab1_mode = st.radio(
        "Analysis Mode",
        options=["Single Drug Analysis", "Multi-Drug Comparison"],
        horizontal=True,
        help="Single Drug: Detailed WoW/Z-Score analysis | Multi-Drug: Compare multiple drugs on same chart"
    )
    
    st.divider()
    
    # Sidebar for Tab 1 settings
    with st.sidebar:
        st.header("Tab 1 Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with date and script columns",
            key="tab1_upload"
        )
        
        st.divider()
        st.header("Analysis Method")
        analysis_method = st.radio(
            "Select analysis method(s)",
            options=["Both WoW & Z-Score", "WoW Method Only", "Z-Score Method Only"],
            index=0,
            help="Choose which analysis methods to run"
        )
        
        st.divider()
        st.header("Custom Thresholds")
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
                    st.info("Available columns in your Excel file:")
                    st.write(df_excel.columns.tolist())
                    
                    if tab1_mode == "Single Drug Analysis":
                        # Single drug mode - select one value column
                        col1, col2 = st.columns(2)
                        with col1:
                            date_col = st.selectbox("Select Date Column", df_excel.columns)
                        with col2:
                            value_col = st.selectbox("Select Value Column", df_excel.columns)
                        
                        # Process data
                        if st.button("Process Data", type="primary", key="tab1_process"):
                            df_processed = pd.DataFrame()
                            df_processed['date'] = pd.to_datetime(df_excel[date_col], errors='coerce')
                            df_processed['scripts'] = pd.to_numeric(df_excel[value_col], errors='coerce')
                            
                            # Filter out summary rows
                            df_processed = df_processed.dropna()
                            
                            if date_col in df_excel.columns:
                                summary_keywords = ['total', 'grand', 'summary', 'subtotal']
                                mask = df_excel[date_col].astype(str).str.lower().str.contains('|'.join(summary_keywords), na=False)
                                df_processed = df_processed[~mask]
                            
                            csv_path = "temp_data.csv"
                            df_processed.to_csv(csv_path, index=False)
                            
                            st.success(f"Processed {len(df_processed)} weeks of data")
                            st.session_state['processed_csv'] = csv_path
                    
                    else:
                        # Multi-drug comparison mode
                        date_col = st.selectbox("Select Date Column", df_excel.columns, key="multi_date")
                        
                        # Get numeric columns (potential drug metrics)
                        numeric_cols = [col for col in df_excel.columns if col != date_col]
                        
                        selected_cols = st.multiselect(
                            "Select Drug Columns to Compare",
                            options=numeric_cols,
                            default=numeric_cols[:min(5, len(numeric_cols))],
                            help="Select multiple drug/metric columns to compare on one chart"
                        )
                        
                        metric_type = st.radio(
                            "Metric Type",
                            options=["TRx", "EUTRx", "NRx", "Other"],
                            horizontal=True
                        )
                        
                        if st.button("Generate Multi-Drug Chart", type="primary", key="tab1_multi"):
                            if len(selected_cols) == 0:
                                st.error("Please select at least one column")
                            else:
                                # Build drugs_data structure
                                multi_drugs_data = {}
                                for col in selected_cols:
                                    col_str = str(col).replace('\n', ' ').strip()
                                    parts = col_str.split()
                                    drug_name = parts[0] if parts else col_str
                                    
                                    drug_df = pd.DataFrame({
                                        'date': pd.to_datetime(df_excel[date_col], errors='coerce'),
                                        'scripts': pd.to_numeric(df_excel[col], errors='coerce')
                                    }).dropna()
                                    
                                    if drug_name not in multi_drugs_data:
                                        multi_drugs_data[drug_name] = {}
                                    multi_drugs_data[drug_name][metric_type] = drug_df
                                
                                st.session_state['multi_drugs_data'] = multi_drugs_data
                                st.session_state['multi_metric'] = metric_type
                                st.success(f"Loaded {len(multi_drugs_data)} drugs for comparison")
                                
                                # Display the chart immediately
                                fig = create_multi_drug_chart(multi_drugs_data, list(multi_drugs_data.keys()), metric_type)
                                st.plotly_chart(fig, use_container_width=True)
                                
                else:
                    # CSV file - save and store (single drug mode only)
                    if tab1_mode == "Single Drug Analysis":
                        csv_path = "temp_data.csv"
                        with open(csv_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state['processed_csv'] = csv_path
                        
                        df_preview = pd.read_csv(csv_path)
                        st.success("CSV file loaded")
                        display_dataframe(df_preview.head())
                    else:
                        st.warning("Multi-drug mode requires an Excel file with multiple columns. Please upload an Excel (.xlsx) file.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.stop()
    
    # Run analysis button (single drug mode only)
    if 'processed_csv' in st.session_state and tab1_mode == "Single Drug Analysis":
        st.divider()
        
        if st.button("Run Analysis", type="primary", use_container_width=True, key="tab1_run"):
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
                    st.success("Analysis Complete!")
                    
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
                    st.header("Summary Statistics")
                    
                    if analysis_method == "Both WoW & Z-Score":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("WoW Method")
                            wow_counts = df_wow['classification'].value_counts()
                            fig_wow = go.Figure(data=[go.Bar(x=wow_counts.index.tolist(), y=wow_counts.values.tolist())])
                            fig_wow.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                            st.plotly_chart(fig_wow, use_container_width=True)
                        with col2:
                            st.subheader("Z-Score Method")
                            zscore_counts = df_zscore['classification'].value_counts()
                            fig_zscore = go.Figure(data=[go.Bar(x=zscore_counts.index.tolist(), y=zscore_counts.values.tolist())])
                            fig_zscore.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                            st.plotly_chart(fig_zscore, use_container_width=True)
                    elif analysis_method == "WoW Method Only":
                        st.subheader("WoW Method")
                        wow_counts = df_wow['classification'].value_counts()
                        fig_wow = go.Figure(data=[go.Bar(x=wow_counts.index.tolist(), y=wow_counts.values.tolist())])
                        fig_wow.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_wow, use_container_width=True)
                    else:
                        st.subheader("Z-Score Method")
                        zscore_counts = df_zscore['classification'].value_counts()
                        fig_zscore = go.Figure(data=[go.Bar(x=zscore_counts.index.tolist(), y=zscore_counts.values.tolist())])
                        fig_zscore.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_zscore, use_container_width=True)
                    
                    # Display data tables
                    st.header("Detailed Results")
                    
                    if analysis_method in ["Both WoW & Z-Score", "WoW Method Only"]:
                        with st.expander("WoW Results Table"):
                            display_dataframe(df_wow)
                    
                    if analysis_method in ["Both WoW & Z-Score", "Z-Score Method Only"]:
                        with st.expander("Z-Score Results Table"):
                            display_dataframe(df_zscore)
                    
                    if analysis_method == "Both WoW & Z-Score" and len(differences) > 0:
                        with st.expander(f"Disagreements ({len(differences)} weeks)"):
                            display_dataframe(differences)
                    
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("Please upload a file in the sidebar to begin")


# TAB 2: TRx-STOCK PERFORMANCE ANALYSIS (NEW)

with tab2:
    st.subheader("Step 1: Configure Drug-Stock Mapping")
    
    default_mappings = {
        'NEFFY': 'SPRY',
        'REZDIFFRA': 'MDGL', 
        'VOQUEZNA': 'PHAT',
        'XDEMVY': 'VTRS',
        'XPHOZAH': 'AKBA',
        'ZORYVE': 'ARQT'
    }
    
    with st.expander("Drug-to-Stock Ticker Mapping", expanded=True):
        
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
    st.subheader("Step 2: Upload Data Files")
    
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
    st.subheader("Step 3: Analysis Settings")
    
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
    if st.button("Run TRx-Stock Analysis", type="primary", use_container_width=True, key="tab2_run"):
        
        if trx_file is None:
            st.error("Please upload a TRx data file")
        elif stock_file is None:
            st.error("Please upload a stock price data file")
        else:
            with st.spinner("Analyzing TRx-Stock relationship..."):
                try:
                    # Load TRx data
                    drugs_data = load_ishara_rapid_data(trx_file)
                    st.success(f"Loaded TRx data for {len(drugs_data)} drugs: {list(drugs_data.keys())}")
                    
                    # Load stock data
                    stock_data = load_stock_data(stock_file, file_type='bloomberg')
                    st.success(f"Loaded {len(stock_data)} stock price records for {stock_ticker_input}")
                    
                    # Store stock data by ticker
                    stock_by_ticker = {stock_ticker_input: stock_data}
                    
                    # Calculate TRx categories and match with stock returns
                    analysis_results = []
                    lag_data = []  # For lag analysis
                    
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
                            
                            # Get stock return for primary analysis
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
                            
                            # Calculate returns at multiple lags (1-8 weeks) for lag analysis
                            z_score = row.get('z_score', None)
                            if z_score is not None:
                                lag_entry = {
                                    'drug': drug_name,
                                    'ticker': ticker,
                                    'trx_date': trx_date,
                                    'release_date': release_date,
                                    'category': row['classification'],
                                    'z_score': z_score
                                }
                                
                                # Calculate returns at each lag
                                for lag_weeks in range(1, 9):
                                    lag_end_date = release_date + timedelta(weeks=lag_weeks)
                                    lag_return = get_stock_return(df_stock, release_date, lag_end_date)
                                    lag_entry[f'return_lag_{lag_weeks}'] = lag_return
                                    if lag_return is not None:
                                        lag_entry[f'end_date_lag_{lag_weeks}'] = lag_end_date.strftime('%Y-%m-%d')
                                
                                lag_data.append(lag_entry)
                    
                    if len(analysis_results) == 0:
                        st.warning("No matching data found. Make sure the stock ticker matches your drug-ticker mapping.")
                    else:
                        st.success(f"Generated {len(analysis_results)} data points for analysis")
                        
                        # Drug selection for visualization
                        st.subheader("Results")
                        
                        available_drugs = list(set(r['drug'] for r in analysis_results))
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            selected_drugs = st.multiselect(
                                "Select drugs to display",
                                options=available_drugs,
                                default=available_drugs,
                                help="Toggle individual drugs on/off"
                            )
                        with col2:
                            viz_type = st.selectbox(
                                "Visualization Type",
                                options=["Category Scatter", "Time-Colored Scatter", "Time Series", "Lag Analysis"],
                                index=0,
                                help="Choose how to visualize the data"
                            )
                        
                        # Create and display selected visualization
                        if len(selected_drugs) > 0:
                            
                            if viz_type == "Category Scatter":
                                fig = create_trx_stock_scatter(analysis_results, selected_drugs, lag_data)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.divider()
                                st.markdown("### Prescriptions & Stock Price Over Time")
                                
                                # Build prescription data from analysis_results
                                prescription_data = []
                                for r in analysis_results:
                                    if r['drug'] in selected_drugs:
                                        prescription_data.append({
                                            'drug': r['drug'],
                                            'trx_date': r['trx_date'],
                                            'scripts': r['scripts']
                                        })
                                
                                if len(prescription_data) > 0:
                                    line_fig = create_prescription_stock_line_chart(prescription_data, stock_data, selected_drugs)
                                    st.plotly_chart(line_fig, use_container_width=True)
                                else:
                                    st.info("No prescription data available for line chart")
                            
                            elif viz_type == "Time-Colored Scatter":
                                fig = create_scatter_with_time_color(analysis_results, selected_drugs)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif viz_type == "Time Series":
                                if len(lag_data) > 0:
                                    fig = create_time_series_comparison(analysis_results, lag_data, stock_data, selected_drugs)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Use Z-Score classification method to see time series.")
                            
                            elif viz_type == "Lag Analysis":
                                if len(lag_data) > 0:
                                    lag_fig = create_lag_analysis_dashboard(lag_data, selected_drugs)
                                    st.plotly_chart(lag_fig, use_container_width=True)
                                else:
                                    st.warning("Use Z-Score classification method for lag analysis.")
                            
                            # Data tables in expanders
                            st.divider()
                            df_results = pd.DataFrame(analysis_results)
                            df_selected = df_results[df_results['drug'].isin(selected_drugs)]
                            
                            with st.expander("Full Results Table"):
                                df_display = df_selected[['drug', 'ticker', 'trx_date', 'category', 'scripts', 'stock_return']].copy()
                                df_display['trx_date'] = df_display['trx_date'].dt.strftime('%Y-%m-%d')
                                df_display['stock_return'] = df_display['stock_return'].round(2)
                                display_dataframe(df_display)
                        else:
                            st.info("Select at least one drug to display the chart")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)

