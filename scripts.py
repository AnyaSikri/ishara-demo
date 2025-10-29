"""
Neffy WoW vs Z-Score Comparison Tool
Analyzes neffy prescription data using two methods side-by-side:
1. Week-over-Week Percentage Change (WoW)
2. Z-Score Statistical Analysis (with 8-week rolling baseline)
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np


def load_data(filepath):
    """
    Load neffy CSV and prepare for analysis
    - Read CSV
    - Rename columns to 'date' and 'scripts'
    - Convert date to datetime
    - Sort by date ascending
    - Return cleaned DataFrame
    """
    df = pd.read_csv(filepath)
    df.columns = ['date', 'scripts']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def classify_drug_maturity(df):
    """
    Classify drug stage based on total weeks of data
    
    Returns dict with:
    - 'stage': 'Brand New Release' | 'Emerging/Expanding' | 'Fully Mature'
    - 'baseline_window': 4 | 8 | 16 weeks
    - 'wow_thresholds': dict with slight/meaningful % thresholds
    
    Rules:
    - < 26 weeks (6 months): Brand New Release
      - baseline_window: 4
      - wow_thresholds: {'slight': 20, 'meaningful': 40}
    
    - 26-103 weeks (6mo - 2yr): Emerging/Expanding  
      - baseline_window: 8
      - wow_thresholds: {'slight': 10, 'meaningful': 25}
    
    - 104+ weeks (2+ years): Fully Mature
      - baseline_window: 16
      - wow_thresholds: {'slight': 5, 'meaningful': 15}
    """
    total_weeks = len(df)
    
    print("\n" + "="*80)
    print("🔍 DEBUG: DRUG MATURITY CLASSIFICATION")
    print("="*80)
    print(f"Total weeks in dataset: {total_weeks}")
    
    if total_weeks < 26:
        result = {
            'stage': 'Brand New Release',
            'baseline_window': 4,
            'wow_thresholds': {'inline': 10, 'slight': 20, 'meaningful': 20}
        }
        print(f"✓ Stage: {result['stage']}")
        print(f"✓ Baseline window: {result['baseline_window']} weeks")
        print(f"✓ WoW thresholds: In-Line=±{result['wow_thresholds']['inline']}%, Slight=±{result['wow_thresholds']['slight']}%, Meaningful=±{result['wow_thresholds']['meaningful']}%")
        return result
    elif total_weeks < 104:
        result = {
            'stage': 'Emerging/Expanding',
            'baseline_window': 8,
            'wow_thresholds': {'inline': 8, 'slight': 15, 'meaningful': 15}
        }
        print(f"✓ Stage: {result['stage']}")
        print(f"✓ Baseline window: {result['baseline_window']} weeks")
        print(f"✓ WoW thresholds: In-Line=±{result['wow_thresholds']['inline']}%, Slight=±{result['wow_thresholds']['slight']}%, Meaningful=±{result['wow_thresholds']['meaningful']}%")
        return result
    else:
        result = {
            'stage': 'Fully Mature',
            'baseline_window': 16,
            'wow_thresholds': {'inline': 5, 'slight': 12, 'meaningful': 12}
        }
        print(f"✓ Stage: {result['stage']}")
        print(f"✓ Baseline window: {result['baseline_window']} weeks")
        print(f"✓ WoW thresholds: In-Line=±{result['wow_thresholds']['inline']}%, Slight=±{result['wow_thresholds']['slight']}%, Meaningful=±{result['wow_thresholds']['meaningful']}%")
        return result


def flag_holiday_weeks(df, buffer_days=2):
    """
    Flag weeks within buffer_days of major US holidays
    
    Major holidays to include:
    - New Year's Day (Jan 1)
    - Memorial Day (last Mon in May, approx May 27)
    - Independence Day (Jul 4)
    - Labor Day (first Mon in Sept, approx Sept 2)
    - Thanksgiving (4th Thurs in Nov, approx Nov 28)
    - Christmas (Dec 25)
    
    Add columns to df:
    - 'is_holiday_week': boolean
    - 'holiday_name': string or None
    
    Return modified DataFrame
    """
    df = df.copy()
    df['is_holiday_week'] = False
    df['holiday_name'] = None
    
    # Define holidays for years in dataset
    years = df['date'].dt.year.unique()
    holidays = []
    
    for year in years:
        holidays.extend([
            (datetime(year, 1, 1), "New Year's Day"),
            (datetime(year, 5, 27), "Memorial Day"),  # Approximate
            (datetime(year, 7, 4), "Independence Day"),
            (datetime(year, 9, 2), "Labor Day"),  # Approximate
            (datetime(year, 11, 28), "Thanksgiving"),  # Approximate
            (datetime(year, 12, 25), "Christmas"),
        ])
    
    # Flag weeks near holidays
    for idx, row in df.iterrows():
        week_date = row['date']
        for holiday_date, holiday_name in holidays:
            days_diff = abs((week_date - holiday_date).days)
            if days_diff <= buffer_days:
                df.at[idx, 'is_holiday_week'] = True
                df.at[idx, 'holiday_name'] = holiday_name
                break
    
    return df


def classify_wow_method(df, current_week_scripts, week_number, thresholds):
    """
    Week-over-Week percentage change classification
    
    Parameters:
    - df: full DataFrame
    - current_week_scripts: this week's count
    - week_number: index position (1-based)
    - thresholds: dict with 'slight' and 'meaningful' % values
    
    Logic:
    1. If week_number == 1: return 'Baseline Building'
    
    2. Get previous week's scripts (week_number - 1)
    
    3. Calculate WoW % change:
       wow_pct = ((current - previous) / previous) * 100
    
    4. Classify based on thresholds:
       - If abs(wow_pct) <= thresholds['slight']: 'In-Line'
       - If abs(wow_pct) <= thresholds['meaningful']: 
         'Slightly Above' if positive, 'Slightly Below' if negative
       - If abs(wow_pct) > thresholds['meaningful']:
         'Meaningfully Above' if positive, 'Meaningfully Below' if negative
    
    Return dict:
    {
        'classification': string,
        'wow_pct': float,
        'previous_week_scripts': int,
        'method': 'WoW'
    }
    """
    # Debug output for first few weeks
    if week_number <= 5:
        print(f"\n🔍 DEBUG WoW - Week {week_number}:")
        print(f"   Current scripts: {current_week_scripts}")
        print(f"   Thresholds being used: inline=±{thresholds['inline']}%, slight=±{thresholds['slight']}%, meaningful=±{thresholds['meaningful']}%")
    
    if week_number == 1:
        if week_number <= 5:
            print(f"   → Classification: 'Baseline Building' (first week)")
        return {
            'classification': 'Baseline Building',
            'wow_pct': 0.0,
            'previous_week_scripts': None,
            'method': 'WoW'
        }
    
    previous_week_scripts = df.iloc[week_number - 2]['scripts']
    wow_pct = ((current_week_scripts - previous_week_scripts) / previous_week_scripts) * 100
    
    if week_number <= 5:
        print(f"   Previous week scripts: {previous_week_scripts}")
        print(f"   WoW % change: {wow_pct:.1f}%")
    
    # Classify based on thresholds (3-tier system)
    if abs(wow_pct) <= thresholds['inline']:
        classification = 'In-Line'
    elif abs(wow_pct) <= thresholds['slight']:
        classification = 'Slightly Above' if wow_pct > 0 else 'Slightly Below'
    elif abs(wow_pct) <= thresholds['meaningful']:
        classification = 'Slightly Above' if wow_pct > 0 else 'Slightly Below'  # Same as above for now
    else:
        classification = 'Meaningfully Above' if wow_pct > 0 else 'Meaningfully Below'
    
    if week_number <= 5:
        print(f"   abs(wow_pct)={abs(wow_pct):.1f} vs inline={thresholds['inline']}, slight={thresholds['slight']}, meaningful={thresholds['meaningful']}")
        print(f"   → Classification: '{classification}'")
    
    return {
        'classification': classification,
        'wow_pct': wow_pct,
        'previous_week_scripts': previous_week_scripts,
        'method': 'WoW'
    }


def classify_zscore_method(df, current_week_scripts, week_number, baseline_window, 
                          custom_zscore_slight=None, custom_zscore_meaningful=None):
    """
    Z-Score statistical classification with holiday-awareness
    
    Parameters:
    - df: full DataFrame with holiday flags
    - current_week_scripts: this week's count
    - week_number: index position (1-based)
    - baseline_window: number of weeks to look back (8 typically)
    - custom_zscore_slight: Custom threshold for slight changes (default: 1.0)
    - custom_zscore_meaningful: Custom threshold for meaningful changes (default: 2.0)
    
    Logic:
    1. If week_number <= 3: return 'Baseline Building'
    
    2. Get historical window (previous baseline_window weeks):
       historical = df.iloc[max(0, week_number - baseline_window - 1) : week_number - 1]
    
    3. EXCLUDE holiday weeks from baseline:
       baseline_data = historical[~historical['is_holiday_week']]
       
       If fewer than 3 non-holiday weeks remain:
         fall back to using all historical weeks
    
    4. Calculate baseline statistics:
       baseline_mean = baseline_data['scripts'].mean()
       baseline_std = baseline_data['scripts'].std()
    
    5. Handle edge case (std == 0):
       If no variation, return 'In-Line' if equal, 'Meaningful Change' if different
    
    6. Calculate z-score:
       z_score = (current_week_scripts - baseline_mean) / baseline_std
    
    7. Classify based on thresholds:
       - abs(z_score) <= 1.0: 'In-Line'
       - abs(z_score) <= 2.0: 'Slightly Above/Below'
       - abs(z_score) > 2.0: 'Meaningfully Above/Below'
    
    Return dict:
    {
        'classification': string,
        'z_score': float,
        'baseline_mean': float,
        'baseline_std': float,
        'baseline_weeks_used': int,
        'holidays_excluded': int,
        'method': 'Z-Score'
    }
    """
    # Debug output for first few weeks
    if week_number <= 5:
        print(f"\n🔍 DEBUG Z-Score - Week {week_number}:")
        print(f"   Current scripts: {current_week_scripts}")
        print(f"   Baseline window: {baseline_window} weeks")
    
    if week_number <= 3:
        if week_number <= 5:
            print(f"   → Classification: 'Baseline Building' (≤3 weeks)")
        return {
            'classification': 'Baseline Building',
            'z_score': 0.0,
            'baseline_mean': None,
            'baseline_std': None,
            'baseline_weeks_used': 0,
            'holidays_excluded': 0,
            'method': 'Z-Score'
        }
    
    # Get historical window
    start_idx = max(0, week_number - baseline_window - 1)
    end_idx = week_number - 1
    historical = df.iloc[start_idx:end_idx].copy()
    
    # Exclude holiday weeks from baseline
    baseline_data = historical[~historical['is_holiday_week']]
    holidays_excluded = len(historical) - len(baseline_data)
    
    # Fall back to all historical if too few non-holiday weeks
    if len(baseline_data) < 3:
        baseline_data = historical
        holidays_excluded = 0
    
    # Calculate baseline statistics
    baseline_mean = baseline_data['scripts'].mean()
    baseline_std = baseline_data['scripts'].std()
    baseline_weeks_used = len(baseline_data)
    
    if week_number <= 5:
        print(f"   Baseline mean: {baseline_mean:.1f}")
        print(f"   Baseline std: {baseline_std:.1f}")
        print(f"   Baseline weeks used: {baseline_weeks_used}")
    
    # Handle edge case: no standard deviation
    if baseline_std == 0 or pd.isna(baseline_std):
        if current_week_scripts == baseline_mean:
            classification = 'In-Line'
        else:
            classification = 'Meaningfully Above' if current_week_scripts > baseline_mean else 'Meaningfully Below'
        z_score = 0.0
    else:
        # Calculate z-score
        z_score = (current_week_scripts - baseline_mean) / baseline_std
        
        # Use custom thresholds or defaults
        threshold_slight = custom_zscore_slight if custom_zscore_slight is not None else 1.0
        threshold_meaningful = custom_zscore_meaningful if custom_zscore_meaningful is not None else 2.0
        
        if week_number <= 5:
            print(f"   Z-score: {z_score:.2f}")
            print(f"   Z-score thresholds: In-Line ≤{threshold_slight}, Slight ≤{threshold_meaningful}, Meaningful >{threshold_meaningful}")
        
        # Classify based on z-score thresholds
        if abs(z_score) <= threshold_slight:
            classification = 'In-Line'
        elif abs(z_score) <= threshold_meaningful:
            classification = 'Slightly Above' if z_score > 0 else 'Slightly Below'
        else:
            classification = 'Meaningfully Above' if z_score > 0 else 'Meaningfully Below'
    
    if week_number <= 5:
        print(f"   → Classification: '{classification}'")
    
    return {
        'classification': classification,
        'z_score': z_score,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'baseline_weeks_used': baseline_weeks_used,
        'holidays_excluded': holidays_excluded,
        'method': 'Z-Score'
    }


def run_dual_analysis(df_with_holidays, custom_zscore_slight=None, custom_zscore_meaningful=None):
    """
    Run both WoW and Z-Score methods on entire dataset
    
    Parameters:
    - df_with_holidays: DataFrame with holiday flags
    - custom_zscore_slight: Custom Z-score threshold for slight changes
    - custom_zscore_meaningful: Custom Z-score threshold for meaningful changes
    
    Steps:
    1. Get drug maturity info
    2. Loop through each week in dataset
    3. For each week:
       - Run WoW classification
       - Run Z-Score classification
       - Combine results into row
    4. Return two DataFrames (one per method) with columns:
       - week_number
       - date
       - scripts
       - is_holiday_week
       - holiday_name
       - classification
       - method-specific metrics (wow_pct or z_score)
    """
    maturity = classify_drug_maturity(df_with_holidays)
    
    print("\n" + "="*80)
    print("🔍 DEBUG: STARTING DUAL ANALYSIS")
    print("="*80)
    print(f"Using WoW thresholds: {maturity['wow_thresholds']}")
    print(f"Using Z-Score baseline window: {maturity['baseline_window']} weeks")
    print(f"Analyzing {len(df_with_holidays)} weeks of data")
    print("\nShowing detailed debug for first 5 weeks:")
    
    wow_results = []
    zscore_results = []
    
    for idx, row in df_with_holidays.iterrows():
        week_number = idx + 1  # 1-based week numbering
        current_week_scripts = row['scripts']
        
        # Run WoW classification
        wow_result = classify_wow_method(
            df_with_holidays,
            current_week_scripts,
            week_number,
            maturity['wow_thresholds']
        )
        
        wow_results.append({
            'week_number': week_number,
            'date': row['date'],
            'scripts': current_week_scripts,
            'is_holiday_week': row['is_holiday_week'],
            'holiday_name': row['holiday_name'],
            'classification': wow_result['classification'],
            'wow_pct': wow_result['wow_pct'],
            'previous_week_scripts': wow_result['previous_week_scripts']
        })
        
        # Run Z-Score classification
        zscore_result = classify_zscore_method(
            df_with_holidays,
            current_week_scripts,
            week_number,
            maturity['baseline_window'],
            custom_zscore_slight=custom_zscore_slight,
            custom_zscore_meaningful=custom_zscore_meaningful
        )
        
        zscore_results.append({
            'week_number': week_number,
            'date': row['date'],
            'scripts': current_week_scripts,
            'is_holiday_week': row['is_holiday_week'],
            'holiday_name': row['holiday_name'],
            'classification': zscore_result['classification'],
            'z_score': zscore_result['z_score'],
            'baseline_mean': zscore_result['baseline_mean'],
            'baseline_std': zscore_result['baseline_std'],
            'baseline_weeks_used': zscore_result['baseline_weeks_used'],
            'holidays_excluded': zscore_result['holidays_excluded']
        })
    
    df_wow = pd.DataFrame(wow_results)
    df_zscore = pd.DataFrame(zscore_results)
    
    print("\n" + "="*80)
    print("🔍 DEBUG: CLASSIFICATION SUMMARY")
    print("="*80)
    print("\nWoW Method Classifications:")
    print(df_wow['classification'].value_counts().to_string())
    print("\nZ-Score Method Classifications:")
    print(df_zscore['classification'].value_counts().to_string())
    
    return df_wow, df_zscore, maturity


def create_comparison_chart(df_wow, df_zscore):
    """
    Create 4-panel comparison visualization using plotly subplots
    
    Layout:
    Row 1, Col 1: WoW Method - Script trend with colored classifications
    Row 1, Col 2: Z-Score Method - Script trend with colored classifications
    Row 2, Col 1: WoW Classification Timeline (horizontal scatter)
    Row 2, Col 2: Z-Score Classification Timeline (horizontal scatter)
    
    Color mapping:
    {
        'Baseline Building': '#CCCCCC',
        'In-Line': '#2E86AB',
        'Slightly Above': '#A23B72',
        'Slightly Below': '#F18F01',
        'Meaningfully Above': '#06FFA5',
        'Meaningfully Below': '#FF4444'
    }
    
    Visual markers:
    - Normal weeks: circles (size 8)
    - Holiday weeks: red-outlined diamonds (size 12)
    - Timeline: squares (size 10)
    - Holiday vertical lines: red dashed
    
    Chart configuration:
    - Height: 900px
    - Vertical spacing: 0.12
    - Title: "Neffy Analysis: WoW % vs Z-Score Methods"
    - Show legend
    - Y-axis labels: "Weekly Scripts" (top), "Classification" (bottom)
    """
    color_map = {
        'Baseline Building': '#CCCCCC',
        'In-Line': '#2E86AB',
        'Slightly Above': '#A23B72',
        'Slightly Below': '#F18F01',
        'Meaningfully Above': '#06FFA5',
        'Meaningfully Below': '#FF4444'
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'WoW Method - Script Trend',
            'Z-Score Method - Script Trend',
            'WoW Classification Timeline',
            'Z-Score Classification Timeline'
        ),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Get unique classifications for legend
    all_classifications = set(df_wow['classification'].unique()) | set(df_zscore['classification'].unique())
    
    # Track which classifications have been added to legend
    legend_added = set()
    
    # Row 1, Col 1: WoW Script Trend
    for classification in all_classifications:
        df_subset = df_wow[df_wow['classification'] == classification]
        
        for _, row in df_subset.iterrows():
            symbol = 'diamond' if row['is_holiday_week'] else 'circle'
            size = 12 if row['is_holiday_week'] else 8
            line_color = 'red' if row['is_holiday_week'] else None
            line_width = 2 if row['is_holiday_week'] else 0
            
            showlegend = classification not in legend_added
            if showlegend:
                legend_added.add(classification)
            
            hover_text = (
                f"Week {row['week_number']}<br>"
                f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                f"Scripts: {row['scripts']}<br>"
                f"WoW %: {row['wow_pct']:.1f}%<br>"
                f"Classification: {classification}"
            )
            if row['is_holiday_week']:
                hover_text += f"<br>Holiday: {row['holiday_name']}"
            
            fig.add_trace(
                go.Scatter(
                    x=[row['date']],
                    y=[row['scripts']],
                    mode='markers',
                    marker=dict(
                        color=color_map[classification],
                        size=size,
                        symbol=symbol,
                        line=dict(color=line_color, width=line_width)
                    ),
                    name=classification,
                    showlegend=showlegend,
                    legendgroup=classification,
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=1, col=1
            )
    
    # Row 1, Col 2: Z-Score Script Trend
    for classification in all_classifications:
        df_subset = df_zscore[df_zscore['classification'] == classification]
        
        for _, row in df_subset.iterrows():
            symbol = 'diamond' if row['is_holiday_week'] else 'circle'
            size = 12 if row['is_holiday_week'] else 8
            line_color = 'red' if row['is_holiday_week'] else None
            line_width = 2 if row['is_holiday_week'] else 0
            
            hover_text = (
                f"Week {row['week_number']}<br>"
                f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                f"Scripts: {row['scripts']}<br>"
                f"Z-Score: {row['z_score']:.2f}<br>"
                f"Classification: {classification}"
            )
            if row['is_holiday_week']:
                hover_text += f"<br>Holiday: {row['holiday_name']}"
            
            fig.add_trace(
                go.Scatter(
                    x=[row['date']],
                    y=[row['scripts']],
                    mode='markers',
                    marker=dict(
                        color=color_map[classification],
                        size=size,
                        symbol=symbol,
                        line=dict(color=line_color, width=line_width)
                    ),
                    name=classification,
                    showlegend=False,
                    legendgroup=classification,
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=1, col=2
            )
    
    # Row 2, Col 1: WoW Classification Timeline
    classification_order = ['Meaningfully Below', 'Slightly Below', 'In-Line', 'Slightly Above', 'Meaningfully Above', 'Baseline Building']
    classification_y = {c: i for i, c in enumerate(classification_order)}
    
    for classification in all_classifications:
        df_subset = df_wow[df_wow['classification'] == classification]
        
        for _, row in df_subset.iterrows():
            symbol = 'diamond' if row['is_holiday_week'] else 'square'
            size = 12 if row['is_holiday_week'] else 10
            line_color = 'red' if row['is_holiday_week'] else None
            line_width = 2 if row['is_holiday_week'] else 0
            
            hover_text = (
                f"Week {row['week_number']}<br>"
                f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                f"WoW %: {row['wow_pct']:.1f}%<br>"
                f"Classification: {classification}"
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[row['date']],
                    y=[classification_y.get(classification, 0)],
                    mode='markers',
                    marker=dict(
                        color=color_map[classification],
                        size=size,
                        symbol=symbol,
                        line=dict(color=line_color, width=line_width)
                    ),
                    name=classification,
                    showlegend=False,
                    legendgroup=classification,
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=2, col=1
            )
    
    # Row 2, Col 2: Z-Score Classification Timeline
    for classification in all_classifications:
        df_subset = df_zscore[df_zscore['classification'] == classification]
        
        for _, row in df_subset.iterrows():
            symbol = 'diamond' if row['is_holiday_week'] else 'square'
            size = 12 if row['is_holiday_week'] else 10
            line_color = 'red' if row['is_holiday_week'] else None
            line_width = 2 if row['is_holiday_week'] else 0
            
            hover_text = (
                f"Week {row['week_number']}<br>"
                f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                f"Z-Score: {row['z_score']:.2f}<br>"
                f"Classification: {classification}"
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[row['date']],
                    y=[classification_y.get(classification, 0)],
                    mode='markers',
                    marker=dict(
                        color=color_map[classification],
                        size=size,
                        symbol=symbol,
                        line=dict(color=line_color, width=line_width)
                    ),
                    name=classification,
                    showlegend=False,
                    legendgroup=classification,
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=2, col=2
            )
    
    # Add holiday vertical lines
    holiday_dates = df_wow[df_wow['is_holiday_week']]['date'].unique()
    for holiday_date in holiday_dates:
        for row_num in [1, 2]:
            for col_num in [1, 2]:
                fig.add_vline(
                    x=holiday_date,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.3,
                    row=row_num,
                    col=col_num
                )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    
    fig.update_yaxes(title_text="Weekly Scripts", row=1, col=1)
    fig.update_yaxes(title_text="Weekly Scripts", row=1, col=2)
    fig.update_yaxes(
        title_text="Classification",
        tickmode='array',
        tickvals=list(range(len(classification_order))),
        ticktext=classification_order,
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Classification",
        tickmode='array',
        tickvals=list(range(len(classification_order))),
        ticktext=classification_order,
        row=2, col=2
    )
    
    fig.update_layout(
        height=900,
        title_text="Neffy Analysis: WoW % vs Z-Score Methods",
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def analyze_differences(df_wow, df_zscore):
    """
    Find and report weeks where methods disagree
    
    Compare 'classification' column between methods
    
    For each disagreement, print:
    - Week number
    - Date
    - Scripts
    - Is holiday week?
    - WoW classification + wow_pct
    - Z-Score classification + z_score
    
    Return DataFrame of differences with all comparison info
    """
    differences = []
    
    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS")
    print("="*80)
    
    for idx in range(len(df_wow)):
        wow_row = df_wow.iloc[idx]
        zscore_row = df_zscore.iloc[idx]
        
        if wow_row['classification'] != zscore_row['classification']:
            differences.append({
                'week_number': wow_row['week_number'],
                'date': wow_row['date'],
                'scripts': wow_row['scripts'],
                'is_holiday_week': wow_row['is_holiday_week'],
                'holiday_name': wow_row['holiday_name'],
                'wow_classification': wow_row['classification'],
                'wow_pct': wow_row['wow_pct'],
                'zscore_classification': zscore_row['classification'],
                'z_score': zscore_row['z_score']
            })
    
    print(f"\nFound {len(differences)} weeks where methods disagree\n")
    
    for diff in differences:
        print(f"Week {diff['week_number']} ({diff['date'].strftime('%Y-%m-%d')}):")
        print(f"  Scripts: {diff['scripts']}")
        if diff['is_holiday_week']:
            print(f"  Holiday Week: {diff['holiday_name']}")
        print(f"  WoW says: '{diff['wow_classification']}' ({diff['wow_pct']:+.1f}%)")
        print(f"  Z-Score says: '{diff['zscore_classification']}' (z={diff['z_score']:+.2f})")
        print()
    
    return pd.DataFrame(differences)


def print_summary_stats(df_wow, df_zscore):
    """
    Print comparison statistics:
    
    For each method:
    - Total weeks analyzed
    - Classification breakdown (count per category)
    - Holiday weeks count
    
    Agreement analysis:
    - Total weeks both methods agree
    - Agreement percentage
    - Most common disagreement pattern
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_weeks = len(df_wow)
    holiday_weeks = df_wow['is_holiday_week'].sum()
    
    print(f"\nTotal weeks analyzed: {total_weeks}")
    print(f"Holiday weeks: {holiday_weeks}")
    
    print("\n--- WoW Method ---")
    wow_counts = df_wow['classification'].value_counts().sort_index()
    for classification, count in wow_counts.items():
        print(f"  {classification}: {count}")
    
    print("\n--- Z-Score Method ---")
    zscore_counts = df_zscore['classification'].value_counts().sort_index()
    for classification, count in zscore_counts.items():
        print(f"  {classification}: {count}")
    
    # Agreement analysis
    agreements = sum(df_wow['classification'] == df_zscore['classification'])
    agreement_pct = (agreements / total_weeks) * 100
    
    print(f"\n--- Agreement Analysis ---")
    print(f"Weeks where methods agree: {agreements}/{total_weeks} ({agreement_pct:.1f}%)")
    print(f"Weeks where methods disagree: {total_weeks - agreements}/{total_weeks} ({100 - agreement_pct:.1f}%)")
    
    # Most common disagreement pattern
    disagreements = df_wow[df_wow['classification'] != df_zscore['classification']].copy()
    if len(disagreements) > 0:
        disagreements['pattern'] = disagreements.apply(
            lambda row: f"WoW:{df_wow.loc[row.name, 'classification']} vs Z-Score:{df_zscore.loc[row.name, 'classification']}",
            axis=1
        )
        most_common = disagreements['pattern'].value_counts().head(1)
        if len(most_common) > 0:
            print(f"\nMost common disagreement pattern:")
            print(f"  {most_common.index[0]} ({most_common.values[0]} times)")


def main(filepath='neffy_scripts_google_cloud.csv', 
         custom_wow_inline=None,
         custom_wow_slight=None, 
         custom_wow_meaningful=None,
         custom_zscore_slight=None,
         custom_zscore_meaningful=None,
         custom_baseline_window=None):
    """
    Complete analysis pipeline with customizable thresholds
    
    Parameters:
    - filepath: path to CSV file
    - custom_wow_inline: WoW % threshold for "inline" change (overrides drug maturity default)
    - custom_wow_slight: WoW % threshold for "slight" change (overrides drug maturity default)
    - custom_wow_meaningful: WoW % threshold for "meaningful" change (overrides drug maturity default)
    - custom_zscore_slight: Z-score threshold for "slight" change (default: 1.0)
    - custom_zscore_meaningful: Z-score threshold for "meaningful" change (default: 2.0)
    - custom_baseline_window: Baseline window in weeks for Z-score (overrides drug maturity default)
    
    Steps:
    1. Load data
    2. Flag holidays
    3. Classify drug maturity (print to console)
    4. Run both analyses
    5. Create comparison visualization
    6. Analyze differences
    7. Print summary statistics
    
    Return: fig, df_wow, df_zscore, differences_df
    
    Print progress messages at each step
    """
    print("="*80)
    print("NEFFY WOW vs Z-SCORE COMPARISON TOOL")
    print("="*80)
    print("\n🐛 DEBUG MODE ENABLED - Verbose output for troubleshooting")
    
    # Display custom threshold info if provided
    if any([custom_wow_inline, custom_wow_slight, custom_wow_meaningful, custom_zscore_slight, custom_zscore_meaningful, custom_baseline_window]):
        print("\n🎛️  CUSTOM THRESHOLDS ENABLED:")
        if custom_wow_inline is not None:
            print(f"   WoW In-Line: ±{custom_wow_inline}%")
        if custom_wow_slight is not None:
            print(f"   WoW Slight: ±{custom_wow_slight}%")
        if custom_wow_meaningful is not None:
            print(f"   WoW Meaningful: ±{custom_wow_meaningful}%")
        if custom_zscore_slight is not None:
            print(f"   Z-Score Slight: ±{custom_zscore_slight}")
        if custom_zscore_meaningful is not None:
            print(f"   Z-Score Meaningful: ±{custom_zscore_meaningful}")
        if custom_baseline_window is not None:
            print(f"   Baseline Window: {custom_baseline_window} weeks")
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    df = load_data(filepath)
    print(f"✓ Loaded {len(df)} weeks of data")
    
    # Step 2: Flag holidays
    print("\n[2/7] Flagging holiday weeks...")
    df_with_holidays = flag_holiday_weeks(df)
    holiday_count = df_with_holidays['is_holiday_week'].sum()
    print(f"✓ Flagged {holiday_count} holiday weeks")
    
    # Step 3: Classify drug maturity (now includes debug output)
    print("\n[3/7] Classifying drug maturity...")
    maturity = classify_drug_maturity(df_with_holidays)
    
    # Override with custom thresholds if provided
    if custom_wow_inline is not None:
        maturity['wow_thresholds']['inline'] = custom_wow_inline
        print(f"   ⚙️  Overriding WoW inline threshold to ±{custom_wow_inline}%")
    if custom_wow_slight is not None:
        maturity['wow_thresholds']['slight'] = custom_wow_slight
        print(f"   ⚙️  Overriding WoW slight threshold to ±{custom_wow_slight}%")
    if custom_wow_meaningful is not None:
        maturity['wow_thresholds']['meaningful'] = custom_wow_meaningful
        print(f"   ⚙️  Overriding WoW meaningful threshold to ±{custom_wow_meaningful}%")
    if custom_baseline_window is not None:
        maturity['baseline_window'] = custom_baseline_window
        print(f"   ⚙️  Overriding baseline window to {custom_baseline_window} weeks")
    
    print(f"✓ Drug Maturity: {maturity['stage']} ({maturity['baseline_window']}-week baseline)")
    print(f"✓ WoW thresholds: In-Line ±{maturity['wow_thresholds']['inline']}%, Slight ±{maturity['wow_thresholds']['slight']}%, Meaningful ±{maturity['wow_thresholds']['meaningful']}%")
    
    # Step 4 & 5: Run both analyses (now includes debug output)
    print("\n[4/7] Running WoW analysis...")
    print("\n[5/7] Running Z-Score analysis...")
    df_wow, df_zscore, _ = run_dual_analysis(df_with_holidays, 
                                               custom_zscore_slight=custom_zscore_slight,
                                               custom_zscore_meaningful=custom_zscore_meaningful)
    print("✓ WoW analysis complete")
    print("✓ Z-Score analysis complete")
    
    # Step 6: Create visualization
    print("\n[6/7] Creating comparison visualization...")
    fig = create_comparison_chart(df_wow, df_zscore)
    print("✓ Visualization created")
    
    # Step 7: Analyze differences
    print("\n[7/7] Analyzing differences...")
    differences_df = analyze_differences(df_wow, df_zscore)
    
    # Print summary statistics
    print_summary_stats(df_wow, df_zscore)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nReturned objects:")
    print("  fig          - Interactive Plotly visualization")
    print("  df_wow       - WoW method results DataFrame")
    print("  df_zscore    - Z-Score method results DataFrame")
    print("  differences  - Weeks where methods disagree")
    print("\nTo display the chart: fig.show()")
    
    return fig, df_wow, df_zscore, differences_df


# Auto-run ready message
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Analysis ready! Run: fig, wow, zscore, diffs = main()")
    print("="*80)
