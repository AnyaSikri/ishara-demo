# %%
# @title Setup - Local Version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os

print("✅ Pharmaceutical analysis setup complete!")
print("📊 Ready to analyze Neffy script data")
print("📁 Make sure your CSV file is in the same directory")

# %%


# Load data from Excel file
def load_pharmaceutical_data_from_excel(file_path):
    """
    Load pharmaceutical script data from Excel file
    
    Parameters:
    - file_path: Path to Excel file (.xlsx or .xls)
    
    Returns:
    - pandas DataFrame with pharmaceutical data
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Display basic info about the loaded data
        print(f"Successfully loaded data from: {file_path}")
        print(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['date', 'scripts']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Missing required columns: {missing_columns}")
            print("Please ensure your Excel file has 'date' and 'scripts' columns")
            return None
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Display sample data
        print(f"\nSample data (first 5 rows):")
        print(df.head().to_string())
        
        # Display date range
        print(f"\nDate range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total weeks: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please check the file path and try again")
        return None
    except Exception as e:
        print(f"❌ Error loading Excel file: {str(e)}")
        return None

# Example usage - replace with your Excel file path
# df = load_pharmaceutical_data_from_excel('your_pharmaceutical_data.xlsx')

# %%
# Create sample Excel data for testing (if you don't have your own file)
def create_sample_pharmaceutical_data():
    """
    Create sample pharmaceutical script data for testing
    This simulates the data structure you'd have from BigQuery
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create date range (50 weeks of data)
    start_date = datetime(2024, 1, 5)  # Start on a Friday (typical week end)
    dates = [start_date + timedelta(weeks=i) for i in range(50)]
    
    # Generate realistic script data with trends and seasonality
    np.random.seed(42)  # For reproducible results
    
    # Base trend (slight growth over time)
    base_trend = np.linspace(2000, 3500, 50)
    
    # Add some randomness and seasonality
    seasonal_pattern = 200 * np.sin(2 * np.pi * np.arange(50) / 12)  # Monthly seasonality
    random_noise = np.random.normal(0, 300, 50)
    
    # Create script counts
    scripts = base_trend + seasonal_pattern + random_noise
    scripts = np.maximum(scripts, 500)  # Ensure minimum of 500 scripts
    
    # Add some holiday impacts (lower scripts around holidays)
    holiday_weeks = [12, 16, 17, 43]  # Thanksgiving, Christmas, New Year, July 4th
    for week in holiday_weeks:
        if week < len(scripts):
            scripts[week] *= 0.7  # 30% reduction for holidays
    
    # Calculate week-over-week growth rate
    wow_growth = []
    for i in range(len(scripts)):
        if i == 0:
            wow_growth.append(None)
        else:
            growth = ((scripts[i] - scripts[i-1]) / scripts[i-1]) * 100
            wow_growth.append(round(growth, 2))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'scripts': scripts.astype(int),
        'WoW_Growth_Rate_Percent': wow_growth
    })
    
    return df

# @title Load Your Neffy Data
print("Choose one of the following options:")
print("1. Load your Neffy CSV file: df = load_neffy_data_from_csv('neffy_scripts_google_cloud.csv')")
print("2. Load your own Excel file: df = load_pharmaceutical_data_from_excel('your_file.xlsx')")
print("3. Use sample data for testing: df = create_sample_pharmaceutical_data()")

def load_neffy_data_from_csv(file_path):
    """
    Load Neffy script data from CSV file (Google Cloud format)
    
    Parameters:
    - file_path: Path to CSV file
    
    Returns:
    - pandas DataFrame with Neffy data
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Display basic info about the loaded data
        print(f"Successfully loaded Neffy data from: {file_path}")
        print(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Rename columns to match expected format
        if 'neffy' in df.columns and 'EUTRX' in df.columns:
            df = df.rename(columns={'neffy': 'date', 'EUTRX': 'scripts'})
            print("✅ Renamed columns: 'neffy' → 'date', 'EUTRX' → 'scripts'")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Display sample data
        print(f"\nSample Neffy data (first 5 rows):")
        print(df.head().to_string())
        
        # Display date range
        print(f"\nNeffy data range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total weeks: {len(df)}")
        print(f"Script range: {df['scripts'].min():,} to {df['scripts'].max():,}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please check the file path and try again")
        return None
    except Exception as e:
        print(f"❌ Error loading CSV file: {str(e)}")
        return None

# Load your Neffy data (choose one option):

# Option 1: Load your actual Neffy CSV file
df = load_neffy_data_from_csv('neffy_scripts_google_cloud.csv')

# Option 2: Use sample data for testing (uncomment if needed)
# df = create_sample_pharmaceutical_data()

# Option 3: Load Excel file (uncomment if needed)
# df = load_pharmaceutical_data_from_excel('your_file.xlsx')

if df is not None:
    print(f"\n✅ Neffy data loaded successfully!")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
else:
    print("❌ Failed to load data. Using sample data instead.")
    df = create_sample_pharmaceutical_data()

# %%
# Data is already loaded as pandas DataFrame from Excel
# Just ensure proper formatting and display basic info

print("📊 PHARMACEUTICAL DATA OVERVIEW")
print("=" * 50)
print(f"Total weeks of data: {len(df)}")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Script range: {df['scripts'].min():,} to {df['scripts'].max():,}")
print(f"Average weekly scripts: {df['scripts'].mean():.0f}")

# Display first few rows
print(f"\nFirst 5 weeks of data:")
print(df.head().to_string(index=False))

# Display last few rows  
print(f"\nLast 5 weeks of data:")
print(df.tail().to_string(index=False))

# Basic statistics
print(f"\n📈 BASIC STATISTICS:")
print(f"Mean scripts per week: {df['scripts'].mean():.0f}")
print(f"Median scripts per week: {df['scripts'].median():.0f}")
print(f"Standard deviation: {df['scripts'].std():.0f}")
print(f"Total scripts (all weeks): {df['scripts'].sum():,}")

# Check for any missing values
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    print(f"\n⚠️  Missing data:")
    for col, count in missing_data.items():
        if count > 0:
            print(f"  {col}: {count} missing values")
else:
    print(f"\n✅ No missing data found")


# %%
# SIMPLE HOLIDAY DETECTION - MAJOR US HOLIDAYS ONLY
# Step-by-step function to flag holiday-impacted weeks

import pandas as pd
from datetime import datetime

def get_major_us_holidays(year):
    """
    Step 1: Define major US holidays for a given year
    Returns dictionary of holiday dates and names
    """

    holidays = {
        # Fixed date holidays
        f'{year}-01-01': 'New Years Day',
        f'{year}-07-04': 'Independence Day',
        f'{year}-12-25': 'Christmas Day',

        # Variable date holidays (using approximations)
        # Memorial Day: Last Monday in May (around May 25-31)
        f'{year}-05-27': 'Memorial Day',  # Approximate

        # Labor Day: First Monday in September (around Sept 2-8)
        f'{year}-09-02': 'Labor Day',  # Approximate

        # Thanksgiving: 4th Thursday in November (around Nov 22-28)
        f'{year}-11-28': 'Thanksgiving',  # Approximate
    }

    return holidays

def flag_holiday_weeks(df, buffer_days=2):
    """
    Step 2: Flag weeks that fall within buffer days of major holidays

    Parameters:
    - df: DataFrame with 'date' column
    - buffer_days: Days before/after holiday to consider impacted (default 2)

    Returns:
    - df with holiday flags added
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Step 2a: Initialize new columns
    df['is_holiday_week'] = False
    df['holiday_name'] = None

    # Step 2b: Get all years in dataset
    years = range(df['date'].dt.year.min(), df['date'].dt.year.max() + 1)

    # Step 2c: Collect all holidays for relevant years
    all_holidays = {}
    for year in years:
        year_holidays = get_major_us_holidays(year)
        all_holidays.update(year_holidays)

    # Step 2d: Convert holiday dates to datetime
    holiday_dates = {}
    for date_str, name in all_holidays.items():
        try:
            holiday_dates[pd.to_datetime(date_str)] = name
        except:
            continue  # Skip invalid dates

    # Step 2e: Check each week against holidays
    for idx, row in df.iterrows():
        week_date = row['date']

        # Check if week falls within buffer of any holiday
        for holiday_date, holiday_name in holiday_dates.items():
            days_difference = abs((week_date - holiday_date).days)

            if days_difference <= buffer_days:
                df.at[idx, 'is_holiday_week'] = True
                df.at[idx, 'holiday_name'] = holiday_name
                break  # Stop at first matching holiday

    return df

def summarize_holiday_impact(df_with_holidays):
    """
    Step 3: Summarize which weeks were flagged as holiday-impacted
    """

    total_weeks = len(df_with_holidays)
    holiday_weeks = df_with_holidays['is_holiday_week'].sum()

    print(f"HOLIDAY IMPACT SUMMARY:")
    print(f"Total weeks: {total_weeks}")
    print(f"Holiday-impacted weeks: {holiday_weeks} ({holiday_weeks/total_weeks*100:.1f}%)")
    print()

    # Show specific holiday weeks
    holiday_data = df_with_holidays[df_with_holidays['is_holiday_week']]

    if len(holiday_data) > 0:
        print("FLAGGED HOLIDAY WEEKS:")
        for _, row in holiday_data.iterrows():
            week_num = row.name + 1
            date_str = row['date'].strftime('%Y-%m-%d')
            scripts = row.get('scripts', 'N/A')
            print(f"Week {week_num} ({date_str}): {row['holiday_name']} - {scripts} scripts")
    else:
        print("No holiday weeks detected in your date range.")

    return holiday_data

def detect_holidays_simple(df, buffer_days=2):
    """
    Step 4: Main function - runs all steps in sequence

    Parameters:
    - df: Your DataFrame with 'date' and 'scripts' columns
    - buffer_days: Days around holiday to flag (default 2)

    Returns:
    - DataFrame with holiday flags added
    """

    print("Step 1: Loading major US holidays...")

    print("Step 2: Checking each week against holidays...")
    df_with_holidays = flag_holiday_weeks(df, buffer_days)

    print("Step 3: Generating summary...")
    holiday_summary = summarize_holiday_impact(df_with_holidays)

    print("Step 4: Holiday detection complete!")

    return df_with_holidays

print("Simple holiday detection loaded!")
print("Run: df_with_holidays = detect_holidays_simple(df)")

# %%
# Run with default 2-day buffer
df_with_holidays = detect_holidays_simple(df)

# Or use 3-day buffer for more conservative flagging
df_with_holidays = detect_holidays_simple(df, buffer_days=3)

# %%
def classify_drug_maturity(df):
    """
    Simple drug maturity classification based on weeks of data
    """
    total_weeks = len(df)

    if total_weeks < 26:  # Less than 6 months
        return {
            'stage': 'Brand New Release',
            'baseline_window': 4
        }
    elif total_weeks < 104:  # Less than 2 years
        return {
            'stage': 'Emerging/Expanding',
            'baseline_window': 8
        }
    else:  # 2+ years
        return {
            'stage': 'Fully Mature',
            'baseline_window': 16
        }

def get_maturity_info(df):
    """
    Get basic maturity info for the drug
    """
    result = classify_drug_maturity(df)
    total_weeks = len(df)

    print(f"Drug Maturity: {result['stage']}")
    print(f"Total weeks of data: {total_weeks}")
    print(f"Recommended baseline window: {result['baseline_window']} weeks")

    return result

# Run this:
result = get_maturity_info(df)

# %%
def classify_weekly_performance_with_holidays(df, current_week_scripts, week_number):
    """
    Classify current week performance, excluding holiday weeks from baseline

    Parameters:
    - df: DataFrame with 'date', 'scripts', and holiday columns
    - current_week_scripts: This week's script count
    - week_number: Which week this is (1, 2, 3, etc.)

    Returns:
    - Classification and confidence level
    """

    # Get drug maturity stage
    maturity = classify_drug_maturity(df)
    stage = maturity['stage']
    baseline_window = maturity['baseline_window']

    # Can't analyze first few weeks - not enough data
    if week_number <= 3:
        return {
            'classification': 'Baseline Building',
            'confidence': 'Low',
            'reason': 'Insufficient historical data for comparison'
        }

    # MODIFIED SECTION: Get historical data excluding holiday weeks
    analysis_window = min(len(df) - 1, baseline_window)
    historical_data = df.iloc[-analysis_window-1:-1].copy()  # Get last N weeks before current

    # Remove holiday weeks from baseline calculation
    if 'is_holiday_week' in historical_data.columns:
        non_holiday_historical = historical_data[~historical_data['is_holiday_week']]

        # Need at least 3 non-holiday weeks for reliable baseline
        if len(non_holiday_historical) < 3:
            # Fall back to all historical data if too few non-holiday weeks
            baseline_data = historical_data
            baseline_note = f"Limited non-holiday data ({len(non_holiday_historical)} weeks), using all historical data"
        else:
            baseline_data = non_holiday_historical
            baseline_note = f"Baseline excludes {len(historical_data) - len(non_holiday_historical)} holiday weeks"
    else:
        # No holiday column, use all data
        baseline_data = historical_data
        baseline_note = "No holiday data available, using all historical weeks"

    if len(baseline_data) == 0:
        return {
            'classification': 'Baseline Building',
            'confidence': 'Low',
            'reason': 'No historical data available'
        }

    # Calculate baseline statistics from non-holiday weeks only
    baseline_mean = baseline_data['scripts'].mean()
    baseline_std = baseline_data['scripts'].std()

    # Handle case where std is 0
    if baseline_std == 0:
        if current_week_scripts == baseline_mean:
            return {
                'classification': 'In-Line',
                'confidence': 'Medium',
                'reason': 'Matches consistent historical performance',
                'baseline_note': baseline_note
            }
        else:
            return {
                'classification': 'Meaningful Change',
                'confidence': 'Medium',
                'reason': 'Deviation from previously consistent performance',
                'baseline_note': baseline_note
            }

    # Calculate z-score
    z_score = (current_week_scripts - baseline_mean) / baseline_std

    # Classification thresholds based on maturity (UPDATED TO 3 CATEGORIES)
    if stage == 'Brand New Release':
        slight_threshold = 1.2
        meaningful_threshold = 1.9
        confidence_level = 'Low'
    elif stage == 'Emerging/Expanding':
        slight_threshold = 1.0
        meaningful_threshold = 1.8
        confidence_level = 'High'
    else:  # Fully Mature
        slight_threshold = 0.8
        meaningful_threshold = 1.5
        confidence_level = 'Very High'

    # Classify based on z-score
    abs_z = abs(z_score)

    if abs_z <= slight_threshold:
        classification = 'In-Line'
    elif abs_z <= meaningful_threshold:
        if z_score > 0:
            classification = 'Slightly Above'
        else:
            classification = 'Slightly Below'
    else:
        if z_score > 0:
            classification = 'Meaningfully Above'
        else:
            classification = 'Meaningfully Below'

    return {
        'classification': classification,
        'confidence': confidence_level,
        'z_score': round(z_score, 2),
        'baseline_mean': round(baseline_mean, 0),
        'baseline_std': round(baseline_std, 0),
        'analysis_window': len(baseline_data),
        'baseline_note': baseline_note,
        'reason': f'{abs_z:.1f} standard deviations from {len(baseline_data)}-week non-holiday baseline'
    }

# %%
def run_holiday_aware_trend_analysis(df_with_holidays):
    """
    Apply holiday-aware classification to entire dataset

    Parameters:
    - df_with_holidays: DataFrame from detect_holidays_simple()

    Returns:
    - DataFrame with trend analysis results for visualization
    """

    results = []

    for idx, row in df_with_holidays.iterrows():
        week_number = idx + 1

        # Get classification for this week using your existing function
        classification_result = classify_weekly_performance_with_holidays(
            df_with_holidays,
            row['scripts'],
            week_number
        )

        # Combine original data with classification results
        week_result = {
            'week_number': week_number,
            'date': row['date'],
            'scripts': row['scripts'],
            'is_holiday_week': row.get('is_holiday_week', False),
            'holiday_name': row.get('holiday_name', None),
            'classification': classification_result['classification'],
            'confidence': classification_result['confidence'],
            'z_score': classification_result.get('z_score', None)
        }

        results.append(week_result)

    return pd.DataFrame(results)

# %%
# HOLIDAY-AWARE TREND VISUALIZATION
# Interactive charts showing trend classifications with holiday markers

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_holiday_trend_visualization(trend_results_with_holidays):
    """
    Create comprehensive visualization of trend analysis with holiday awareness

    Parameters:
    - trend_results_with_holidays: DataFrame from run_holiday_aware_trend_analysis()
    """

    df = trend_results_with_holidays.copy()

    # Color mapping for classifications
    color_map = {
        'Baseline Building': '#CCCCCC',
        'In-Line': '#2E86AB',
        'Slightly Above': '#A23B72',
        'Slightly Below': '#F18F01',
        'Meaningfully Above': '#06FFA5',
        'Meaningfully Below': '#FF4444',
        'Holiday-Impacted': '#8A2BE2'
    }

    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Weekly Scripts with Trend Classifications',
            'Classification Timeline with Holiday Markers',
            'Z-Score Analysis (Distance from Baseline)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )

    # Chart 1: Main trend line with classification colors
    for classification in df['classification'].unique():
        if pd.notna(classification):
            class_data = df[df['classification'] == classification]

            # Different markers for holiday vs non-holiday weeks
            holiday_data = class_data[class_data.get('is_holiday_week', False) == True]
            non_holiday_data = class_data[class_data.get('is_holiday_week', False) == False]

            # Non-holiday weeks (circles)
            if len(non_holiday_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=non_holiday_data['date'],
                        y=non_holiday_data['scripts'],
                        mode='markers+lines',
                        name=classification,
                        marker=dict(
                            color=color_map.get(classification, '#000000'),
                            size=8,
                            symbol='circle'
                        ),
                        line=dict(color=color_map.get(classification, '#000000'), width=1),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Scripts: %{y:,}<br>' +
                                    'Classification: ' + classification + '<br>' +
                                    '<extra></extra>',
                        text=[f'Week {row["week_number"]}' for _, row in non_holiday_data.iterrows()]
                    ),
                    row=1, col=1
                )

            # Holiday weeks (diamonds)
            if len(holiday_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=holiday_data['date'],
                        y=holiday_data['scripts'],
                        mode='markers',
                        name=f'{classification} (Holiday)',
                        marker=dict(
                            color=color_map.get(classification, '#000000'),
                            size=12,
                            symbol='diamond',
                            line=dict(color='red', width=2)
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Scripts: %{y:,}<br>' +
                                    'Classification: ' + classification + '<br>' +
                                    'Holiday: %{customdata}<br>' +
                                    '<extra></extra>',
                        text=[f'Week {row["week_number"]}' for _, row in holiday_data.iterrows()],
                        customdata=[row.get('holiday_name', 'Holiday') for _, row in holiday_data.iterrows()]
                    ),
                    row=1, col=1
                )

    # Chart 2: Classification timeline as horizontal bars
    classification_order = ['Meaningfully Below', 'Slightly Below', 'In-Line', 'Slightly Above', 'Meaningfully Above', 'Baseline Building']

    for i, classification in enumerate(classification_order):
        class_data = df[df['classification'] == classification]
        if len(class_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=class_data['date'],
                    y=[classification] * len(class_data),
                    mode='markers',
                    name=f'{classification} Timeline',
                    marker=dict(
                        color=color_map.get(classification, '#000000'),
                        size=10,
                        symbol='square'
                    ),
                    showlegend=False,
                    hovertemplate='Week %{text}: %{customdata:,} scripts<br>' +
                                'Date: %{x}<br>' +
                                '<extra></extra>',
                    text=[row['week_number'] for _, row in class_data.iterrows()],
                    customdata=[row['scripts'] for _, row in class_data.iterrows()]
                ),
                row=2, col=1
            )

    # Add holiday markers to timeline
    holiday_weeks = df[df.get('is_holiday_week', False) == True]
    if len(holiday_weeks) > 0:
        for _, row in holiday_weeks.iterrows():
            fig.add_vline(
                x=row['date'],
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                row=2, col=1
            )

    # Chart 3: Z-score analysis
    analyzable_weeks = df[df['classification'] != 'Baseline Building']
    if len(analyzable_weeks) > 0:
        # Regular weeks
        non_holiday_z = analyzable_weeks[analyzable_weeks.get('is_holiday_week', False) == False]
        if len(non_holiday_z) > 0:
            fig.add_trace(
                go.Scatter(
                    x=non_holiday_z['date'],
                    y=non_holiday_z['z_score'],
                    mode='markers+lines',
                    name='Z-Score',
                    marker=dict(color='#2E86AB', size=6),
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate='Week %{text}: %{y:.2f} std devs<br>' +
                                'Date: %{x}<br>' +
                                '<extra></extra>',
                    text=[row['week_number'] for _, row in non_holiday_z.iterrows()]
                ),
                row=3, col=1
            )

        # Holiday weeks with special markers
        holiday_z = analyzable_weeks[analyzable_weeks.get('is_holiday_week', False) == True]
        if len(holiday_z) > 0:
            fig.add_trace(
                go.Scatter(
                    x=holiday_z['date'],
                    y=holiday_z['z_score'],
                    mode='markers',
                    name='Z-Score (Holiday)',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='diamond',
                        line=dict(color='darkred', width=1)
                    ),
                    hovertemplate='Week %{text}: %{y:.2f} std devs<br>' +
                                'Date: %{x}<br>' +
                                'Holiday: %{customdata}<br>' +
                                '<extra></extra>',
                    text=[row['week_number'] for _, row in holiday_z.iterrows()],
                    customdata=[row.get('holiday_name', 'Holiday') for _, row in holiday_z.iterrows()]
                ),
                row=3, col=1
            )

        # Add reference lines for z-score chart
        fig.add_hline(y=1.8, line_dash="solid", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=-1.8, line_dash="solid", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="orange", opacity=0.5, row=3, col=1)
        fig.add_hline(y=-1.0, line_dash="dash", line_color="orange", opacity=0.5, row=3, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=3, col=1)

    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Holiday-Aware Trend Analysis Dashboard",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_yaxes(title_text="Weekly Scripts", row=1, col=1)
    fig.update_yaxes(title_text="Classification", row=2, col=1)
    fig.update_yaxes(title_text="Standard Deviations", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    return fig

def create_holiday_impact_summary_chart(trend_results_with_holidays):
    """
    Create summary chart showing holiday vs non-holiday performance
    """

    df = trend_results_with_holidays.copy()

    # Separate holiday vs non-holiday weeks
    holiday_weeks = df[df.get('is_holiday_week', False) == True]
    non_holiday_weeks = df[df.get('is_holiday_week', False) == False]

    # Count classifications for each group
    holiday_counts = holiday_weeks['classification'].value_counts() if len(holiday_weeks) > 0 else pd.Series()
    non_holiday_counts = non_holiday_weeks['classification'].value_counts() if len(non_holiday_weeks) > 0 else pd.Series()

    # Create comparison chart
    fig = go.Figure()

    classifications = ['Meaningfully Below', 'Slightly Below', 'In-Line', 'Slightly Above', 'Meaningfully Above']

    holiday_values = [holiday_counts.get(c, 0) for c in classifications]
    non_holiday_values = [non_holiday_counts.get(c, 0) for c in classifications]

    fig.add_trace(go.Bar(
        name='Non-Holiday Weeks',
        x=classifications,
        y=non_holiday_values,
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='Holiday Weeks',
        x=classifications,
        y=holiday_values,
        marker_color='red'
    ))

    fig.update_layout(
        title='Classification Comparison: Holiday vs Non-Holiday Weeks',
        xaxis_title='Classification',
        yaxis_title='Number of Weeks',
        barmode='group',
        height=400
    )

    return fig

def visualize_holiday_trend_analysis(trend_results_with_holidays):
    """
    Main function to create all visualizations
    """

    print("Creating holiday-aware trend visualizations...")

    # Main comprehensive dashboard
    main_chart = create_holiday_trend_visualization(trend_results_with_holidays)

    # Holiday impact comparison
    comparison_chart = create_holiday_impact_summary_chart(trend_results_with_holidays)

    print("✓ Visualizations created!")
    print("\nChart Legend:")
    print("• Circles = Normal weeks")
    print("• Red-outlined diamonds = Holiday weeks")
    print("• Dashed red lines = Holiday periods")
    print("• Z-score reference lines: ±1.0 (slight), ±1.8 (meaningful)")

    return main_chart, comparison_chart


print("Holiday-aware visualization functions loaded!")
print("Usage: main_chart, comparison_chart = visualize_holiday_trend_analysis(trend_results)")

# %%
# Run holiday detection first
df_with_holidays = detect_holidays_simple(df)

# Then run trend analysis
trend_results = run_holiday_aware_trend_analysis(df_with_holidays)

# Now create visualizations
main_chart, comparison_chart = visualize_holiday_trend_analysis(trend_results)
main_chart.show()

# %%
print_table(trend_results)
save_table(trend_results)

# %%
# Quick pharmaceutical data check
quick_data_overview(trend_results)

# Ask specific questions
ask_gemini_about_data(trend_results, "tell me about week 40", model)

# %%
import pandas as pd
import numpy as np

def create_simple_table(trend_results):
    """
    Create a simple, clean table with the key information
    """
    df = trend_results.copy()

    # Calculate percent change from previous week
    df['percent_change'] = df['scripts'].pct_change() * 100

    # Create simple table with the columns you actually have
    table = pd.DataFrame({
        'Week': df['week_number'],
        'Date': df['date'].dt.strftime('%Y-%m-%d'),
        'Scripts': df['scripts'],
        'Percent_Change': df['percent_change'].round(1),
        'Classification': df['classification'],
        'Confidence': df['confidence'],
        'Z-Score': df['z_score'].round(2),
        'Holiday': df['holiday_name'].fillna('')
    })

    return table

def print_table(trend_results):
    """
    Print the table in a clean format
    """
    table = create_simple_table(trend_results)
    print(table.to_string(index=False))

def save_table(trend_results, filename='trend_analysis.csv'):
    """
    Save the table to CSV file
    """
    table = create_simple_table(trend_results)
    table.to_csv(filename, index=False)
    print(f"Table saved to {filename}")

# Usage:
# print_table(trend_results)
# save_table(trend_results)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

# Optional: Install and import Gemini AI Studio (for AI-powered analysis)
# Uncomment the lines below if you want AI analysis features
# try:
#     import google.generativeai as genai
#     print("Google AI Studio library available")
# except ImportError:
#     !pip install google-generativeai
#     import google.generativeai as genai

def initialize_gemini(api_key=None):
    """
    Initialize Google AI Studio Gemini model for pharmaceutical data analysis
    Get API key from: https://ai.google.dev/
    """
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Set your API key: os.environ['GOOGLE_API_KEY'] = 'your-key'")
        print("Get one from: https://ai.google.dev/")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Test with pharma-specific content
        test = model.generate_content("Analyze pharmaceutical script data")
        print("Gemini initialized successfully for pharma data analysis")
        return model
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return None

def ask_gemini_about_data(df, question, model=None, max_rows=50):
    """
    Ask Gemini about pharmaceutical script data with smart sampling

    Args:
        df: Pharmaceutical DataFrame with columns like 'scripts', 'z_score', 'classification', etc.
        question: Question about the data
        model: Gemini model instance
        max_rows: Maximum rows to send to API
    """
    if model is None:
        model = initialize_gemini()
        if model is None:
            return "Could not initialize Gemini model"

    try:
        # Find specific weeks mentioned in question
        week_mentions = re.findall(r'week\s*(\d+)', question.lower())
        specific_weeks = [int(w) for w in week_mentions]

        # Smart sampling for pharma data
        if len(df) > max_rows:
            # Get representative sample: start, middle, end
            start_sample = df.head(15)
            middle_idx = len(df) // 2
            middle_sample = df.iloc[middle_idx-5:middle_idx+5]
            end_sample = df.tail(10)
            df_sample = pd.concat([start_sample, middle_sample, end_sample]).drop_duplicates()
            note = f"\n[Showing {len(df_sample)} representative rows from {len(df)} total weeks]"
        else:
            df_sample = df.copy()
            note = ""

        # Add specific weeks mentioned in question
        for week_num in specific_weeks:
            if week_num <= len(df):
                # Try multiple ways to find the week
                specific_row = None

                # Method 1: Check if there's a week_number column
                if 'week_number' in df.columns:
                    specific_data = df[df['week_number'] == week_num]
                    if not specific_data.empty:
                        specific_row = specific_data

                # Method 2: Use index as week number (most common)
                elif week_num - 1 < len(df):
                    specific_row = df.iloc[[week_num - 1]]

                # Add the specific week data
                if specific_row is not None:
                    df_sample = pd.concat([df_sample, specific_row]).drop_duplicates()
                    note += f"\n[Added week {week_num} data as requested]"

        # Sort chronologically and format
        df_sample = df_sample.sort_index()
        data_text = df_sample.to_string(max_cols=12, float_format='%.2f')

        # Pharma-specific summary statistics
        pharma_stats = ""
        key_columns = ['scripts', 'z_score', 'classification']
        available_columns = [col for col in key_columns if col in df.columns]

        if available_columns:
            try:
                stats = df[available_columns].describe().round(2)
                pharma_stats = f"\nKey Pharmaceutical Metrics:\n{stats.to_string()}"
            except:
                pharma_stats = f"\nAvailable pharma columns: {available_columns}"

        # Enhanced prompt for pharmaceutical context
        prompt = f"""
        You are analyzing pharmaceutical script tracking data with weekly performance metrics, holiday adjustments, and trend classifications.

        DATASET CONTEXT:
        - Total weeks: {df.shape[0]} rows × {df.shape[1]} columns
        - Columns: {list(df.columns)}
        - Sample shown: {len(df_sample)} weeks{note}

        PHARMACEUTICAL DATA SAMPLE:
        {data_text}

        PERFORMANCE METRICS:
        {pharma_stats}

        QUESTION: {question}

        Please analyze this pharmaceutical script data and provide:
        1. Specific numerical answers (script counts, z-scores, percentages)
        2. Performance classifications context (In-Line, Above/Below baseline)
        3. Holiday impact analysis if relevant
        4. Week-over-week comparisons when applicable
        5. Any concerning patterns or anomalies for script performance
        6. Business implications for pharmaceutical tracking

        Reference specific weeks, dates, and metrics from the data provided.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error analyzing pharmaceutical data: {str(e)}"

def quick_data_overview(df):
    """
    Pharmaceutical-specific data overview
    """
    print("PHARMACEUTICAL DATA OVERVIEW")
    print("=" * 50)
    print(f"Total weeks tracked: {df.shape[0]}")
    print(f"Data columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")

    # Check for key pharmaceutical columns
    pharma_columns = {
        'scripts': 'Script counts',
        'z_score': 'Performance z-scores',
        'classification': 'Performance classifications',
        'is_holiday_week': 'Holiday flags',
        'holiday_name': 'Holiday names',
        'date': 'Week dates'
    }

    print("\nKey pharmaceutical metrics found:")
    for col, description in pharma_columns.items():
        if col in df.columns:
            if col == 'scripts':
                print(f"  {col}: {description} (Range: {df[col].min():.0f} - {df[col].max():.0f})")
            elif col == 'z_score':
                print(f"  {col}: {description} (Range: {df[col].min():.2f} to {df[col].max():.2f})")
            elif col == 'classification' and df[col].dtype == 'object':
                unique_vals = df[col].value_counts()
                print(f"  {col}: {description}")
                for classification, count in unique_vals.head(3).items():
                    print(f"    - {classification}: {count} weeks")
            elif col == 'is_holiday_week':
                holiday_count = df[col].sum() if df[col].dtype == bool else df[col].value_counts().get(True, 0)
                print(f"  {col}: {description} ({holiday_count} holiday weeks)")
            else:
                print(f"  {col}: {description}")

    # Show critical weeks
    print(f"\nFirst 3 weeks:")
    print(df.head(3)[['scripts', 'z_score', 'classification']].to_string() if 'scripts' in df.columns else df.head(3).to_string())

    # Show week 40 if it exists
    if len(df) >= 40:
        print(f"\nWeek 40 data:")
        week_40 = df.iloc[39:40]
        if 'scripts' in df.columns:
            print(week_40[['scripts', 'z_score', 'classification']].to_string())
        else:
            print(week_40.to_string())

def interactive_data_chat(df):
    """
    Interactive pharmaceutical data analysis chat
    """
    model = initialize_gemini()
    if model is None:
        print("Could not initialize Gemini for pharmaceutical analysis")
        return

    print("=" * 60)
    print("PHARMACEUTICAL SCRIPT DATA ANALYSIS CHAT")
    print("=" * 60)
    print(f"Loaded {df.shape[0]} weeks of pharmaceutical script data")
    print(f"Columns: {', '.join(df.columns.tolist())}")

    # Show date range if available
    if 'date' in df.columns:
        print(f"Period: {df['date'].min()} to {df['date'].max()}")

    # Show script performance summary
    if 'scripts' in df.columns:
        print(f"Script range: {df['scripts'].min():.0f} - {df['scripts'].max():.0f}")
    if 'classification' in df.columns:
        classifications = df['classification'].value_counts()
        print(f"Performance classifications: {dict(classifications.head(3))}")

    print("\nCommands:")
    print("  'quit' - Exit chat")
    print("  'info' - Show detailed data info")
    print("  'examples' - Show pharmaceutical analysis examples")
    print("  Or ask questions about your pharmaceutical data")
    print("=" * 60)

    while True:
        try:
            question = input("\nPharmaceutical Analysis Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Pharmaceutical analysis complete!")
                break

            elif question.lower() == 'info':
                quick_data_overview(df)
                continue

            elif question.lower() == 'examples':
                print("\nPharmaceutical analysis examples:")
                examples = [
                    "What happened in week 40?",
                    "Which weeks had the highest script volumes?",
                    "How many weeks were classified as 'Meaningfully Above'?",
                    "What was the impact of holidays on script performance?",
                    "Show me weeks with concerning z-scores below -2",
                    "What's the average script count for non-holiday weeks?",
                    "Which holiday caused the biggest script drop?",
                    "Compare week 25 performance to baseline",
                    "What trends do you see in script volumes over time?",
                    "Are there any weeks I should investigate further?"
                ]
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
                continue

            elif question == '':
                continue

            print("\nAnalyzing pharmaceutical data...")
            answer = ask_gemini_about_data(df, question, model)
            print(f"\nPharmaceutical Analysis:\n{answer}")

        except KeyboardInterrupt:
            print("\n\nPharmaceutical analysis interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError in pharmaceutical analysis: {e}")

def run_example_queries(df, model=None):
    """
    Run pharmaceutical-specific example queries
    """
    if model is None:
        model = initialize_gemini()
        if model is None:
            print("Could not initialize Gemini for pharmaceutical examples")
            return

    print("=" * 60)
    print("PHARMACEUTICAL SCRIPT ANALYSIS EXAMPLES")
    print("=" * 60)

    pharma_questions = [
        "Provide an executive summary of the pharmaceutical script performance data",
        "Which weeks showed 'Meaningfully Above' or 'Meaningfully Below' performance?",
        "How did holiday weeks impact script volumes compared to normal weeks?",
        "What were the top 3 highest performing weeks and what drove the performance?",
        "What was the total script volume for the entire tracking period?",
        "Which specific holidays had the most negative impact on script performance?",
        "What is the overall trend in script volumes - growing, declining, or stable?",
        "What percentage of weeks were impacted by holidays?",
        "Are there any weeks with concerning performance that need investigation?",
        "What insights can you provide for pharmaceutical business planning?"
    ]

    for i, question in enumerate(pharma_questions, 1):
        print(f"\nPharmaceutical Question {i}: {question}")
        print("-" * 50)
        answer = ask_gemini_about_data(df, question, model)
        print(f"Analysis: {answer}")
        print("=" * 60)

def advanced_query(df, question, model=None, context_window=3):
    """
    Advanced pharmaceutical query with context window around specific weeks
    """
    if model is None:
        model = initialize_gemini()
        if model is None:
            return "Could not initialize Gemini for advanced pharmaceutical analysis"

    # Find specific weeks and add surrounding context
    week_mentions = re.findall(r'week\s*(\d+)', question.lower())
    specific_weeks = [int(w) for w in week_mentions]

    # Build context-rich sample
    df_sample = df.head(10).copy()  # Always include first 10 weeks

    # Add context around specific weeks
    for week_num in specific_weeks:
        if week_num <= len(df):
            start_idx = max(0, week_num - 1 - context_window)
            end_idx = min(len(df), week_num - 1 + context_window + 1)
            context_data = df.iloc[start_idx:end_idx]
            df_sample = pd.concat([df_sample, context_data]).drop_duplicates()

    # Enhanced question with pharmaceutical context
    enhanced_question = f"""
    {question}

    Please provide context from surrounding weeks to understand:
    - Performance trends leading up to and following the specified week(s)
    - How the week(s) compare to adjacent weeks
    - Any patterns or anomalies in the pharmaceutical script data
    - Business implications for script performance tracking
    """

    return ask_gemini_about_data(df_sample, enhanced_question, model, max_rows=len(df_sample))

def pharma_insights_summary(df, model=None):
    """
    Generate comprehensive pharmaceutical insights summary
    """
    if model is None:
        model = initialize_gemini()
        if model is None:
            return "Could not initialize Gemini for pharmaceutical insights"

    summary_question = """
    Please provide a comprehensive pharmaceutical business intelligence summary including:

    1. PERFORMANCE OVERVIEW: Overall script volume trends and performance classifications
    2. HOLIDAY IMPACT ANALYSIS: How holidays affected script volumes and which ones had biggest impact
    3. ANOMALY DETECTION: Weeks with unusual performance (high/low z-scores) and potential causes
    4. TREND ANALYSIS: Growth, decline, or stability patterns in script volumes over time
    5. RISK ASSESSMENT: Weeks or patterns that may indicate business risks or opportunities
    6. ACTIONABLE RECOMMENDATIONS: Business insights for pharmaceutical planning and strategy

    Focus on metrics that matter for pharmaceutical business decision-making.
    """

    return ask_gemini_about_data(df, summary_question, model, max_rows=75)

# Auto-setup check
if os.getenv('GOOGLE_API_KEY'):
    print("\nAPI key detected - ready for pharmaceutical analysis!")
else:
    print("\nSet API key: os.environ['GOOGLE_API_KEY'] = 'your-key'")

# %%
# Uncomment and add your API key if you want to use AI analysis
# model = initialize_gemini('your-api-key-here')
print("AI analysis is optional - you can run all other analysis without it")

# %% [markdown]
# Pharmaceutical-Specific Functions:
# 
# *   initialize_gemini() - Setup for pharma analysis
# * ask_gemini_about_data() - Smart week detection + pharma context
# * quick_data_overview() - Pharma metrics summary (scripts, z-scores, classifications)
# * interactive_data_chat() - Pharma-focused interactive session
# * run_example_queries() - 10 pre-built pharmaceutical business questions
# * advanced_query() - Context-aware analysis with surrounding weeks
# * pharma_insights_summary() - Comprehensive business intelligence report

# %%
# Interactive pharmaceutical session (requires AI setup)
# Uncomment if you have set up Gemini AI
# interactive_data_chat(trend_results)
print("Interactive AI chat requires Gemini API setup. All other analysis works without it.")

# %%
# AI-powered analysis (requires Gemini API setup)
# Uncomment if you have set up Gemini AI
# run_example_queries(trend_results)
# advanced_query(trend_results, "compare week 40 to surrounding weeks", model)
# pharma_insights_summary(trend_results, model)

print("AI-powered analysis requires Gemini API setup.")
print("All core pharmaceutical analysis (trends, holidays, z-scores, visualizations) works without AI.")

# %%
# Create example Excel file for your own data
def create_example_excel_file(filename='pharmaceutical_data_template.xlsx'):
    """
    Create an example Excel file with the correct format for pharmaceutical analysis
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create example data structure
    start_date = datetime(2024, 1, 5)
    dates = [start_date + timedelta(weeks=i) for i in range(10)]
    
    example_data = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'scripts': [2000, 2100, 1950, 2200, 2050, 2300, 2150, 2400, 2250, 2500],
        'WoW_Growth_Rate_Percent': [None, 5.0, -7.1, 12.8, -6.8, 12.2, -6.5, 11.6, -6.3, 11.1]
    })
    
    # Save to Excel
    example_data.to_excel(filename, index=False)
    print(f"✅ Example Excel file created: {filename}")
    print(f"📋 File contains {len(example_data)} rows of sample data")
    print(f"📅 Date range: {example_data['date'].iloc[0]} to {example_data['date'].iloc[-1]}")
    
    return filename

def show_data_format_requirements():
    """
    Display the required data file formats (CSV and Excel)
    """
    print("📋 DATA FILE FORMAT REQUIREMENTS")
    print("=" * 50)
    print("Your data file should have these columns:")
    print()
    print("Column A: 'date' (or 'neffy' for CSV)")
    print("  - Format: YYYY-MM-DD (e.g., 2024-01-05)")
    print("  - Description: Week ending date")
    print()
    print("Column B: 'scripts' (or 'EUTRX' for CSV)") 
    print("  - Format: Numbers only (e.g., 2500)")
    print("  - Description: Weekly script/prescription counts")
    print()
    print("Column C: 'WoW_Growth_Rate_Percent' (Optional)")
    print("  - Format: Numbers with decimals (e.g., 5.2)")
    print("  - Description: Week-over-week growth percentage")
    print()
    print("📝 EXAMPLE DATA FORMATS:")
    print()
    print("CSV Format (like your Neffy data):")
    print("neffy,EUTRX")
    print("2024-09-06,3")
    print("2024-09-13,7")
    print("2024-09-20,15")
    print()
    print("Excel Format:")
    print("date        | scripts | WoW_Growth_Rate_Percent")
    print("2024-01-05  | 2000    | ")
    print("2024-01-12  | 2100    | 5.0")
    print("2024-01-19  | 1950    | -7.1")
    print("2024-01-26  | 2200    | 12.8")
    print()
    print("💡 TIPS:")
    print("- CSV: Use 'neffy' and 'EUTRX' column names (auto-renamed)")
    print("- Excel: Use 'date' and 'scripts' column names")
    print("- Dates should be in YYYY-MM-DD format")
    print("- Script counts should be whole numbers")
    print("- Save CSV as .csv, Excel as .xlsx or .xls format")

# Show format requirements
show_data_format_requirements()

# Create example file (uncomment to create)
# create_example_excel_file('my_pharmaceutical_data.xlsx')


# %% [markdown]
# # 🚀 Google Colab Instructions
# 
# ## How to Run This Notebook in Google Colab:
# 
# ### Step 1: Open in Google Colab
# 1. Go to [Google Colab](https://colab.research.google.com/)
# 2. Click "File" → "Upload notebook"
# 3. Upload this `.ipynb` file
# 
# ### Step 2: Run the Setup Cell
# - Run the first cell to install required packages
# - This will install: `plotly`, `pandas`, `numpy`, `matplotlib`, `openpyxl`
# 
# ### Step 3: Upload Your Data
# - Run the "Upload Your Neffy Data File" cell
# - Click "Choose Files" and select your `neffy_scripts_google_cloud.csv` file
# - The system will automatically detect and process your Neffy data
# 
# ### Step 4: Run All Analysis
# - Execute all remaining cells in order
# - The interactive visualizations will display directly in Colab
# - All analysis works without any Google Cloud setup required
# 
# ## ✅ Colab Advantages:
# - **No local installation needed** - Everything runs in the cloud
# - **Interactive visualizations** - Plotly charts display directly in notebook
# - **File upload interface** - Easy to upload your CSV/Excel files
# - **Free GPU/CPU** - Google provides free compute resources
# - **Shareable** - Easy to share results with colleagues
# 
# ## 📊 What You'll Get:
# - Holiday impact analysis for Neffy
# - Z-score performance classifications
# - Interactive trend visualizations
# - Exportable results tables
# - Pharmaceutical business insights
# 
# ## 🔧 Troubleshooting:
# - If file upload fails, try refreshing the page
# - Make sure your CSV has 'neffy' and 'EUTRX' columns
# - All visualizations work in Colab without additional setup
# 

# %%
# @title Two Analysis Methods Comparison
def build_comparison_tables(df):
    """
    Build two separate analysis formats:
    1. Your Method (WoW %) - Week-over-week percentage change
    2. Shaun's Method (Z-Score) - Statistical analysis with 8-week rolling baseline
    """
    
    # Create a copy for analysis
    analysis_df = df.copy()
    
    # ========================================
    # 1. YOUR METHOD (WoW %)
    # ========================================
    print("📊 METHOD 1: Your Method (WoW % Analysis)")
    print("=" * 50)
    
    # Calculate week-over-week percentage change
    analysis_df['wow_percent'] = analysis_df['scripts'].pct_change() * 100
    
    # Create flags based on WoW %
    def classify_wow_performance(wow_percent, is_mature=True):
        if pd.isna(wow_percent):
            return '-'
        
        if is_mature:  # If Neffy is mature
            if abs(wow_percent) <= 5:
                return 'In-Line'
            elif abs(wow_percent) <= 15:
                return 'Slightly Above' if wow_percent > 0 else 'Slightly Below'
            else:
                return 'Meaningfully Above' if wow_percent > 0 else 'Meaningfully Below'
        else:  # If Neffy is new/emerging (more volatile)
            if abs(wow_percent) <= 10:
                return 'In-Line'
            elif abs(wow_percent) <= 25:
                return 'Slightly Above' if wow_percent > 0 else 'Slightly Below'
            else:
                return 'Meaningfully Above' if wow_percent > 0 else 'Meaningfully Below'
    
    analysis_df['your_flag'] = analysis_df['wow_percent'].apply(
        lambda x: classify_wow_performance(x, is_mature=True)
    )
    
    # Create your method table
    your_method_table = analysis_df[['date', 'scripts', 'wow_percent', 'your_flag']].copy()
    your_method_table['week'] = range(1, len(your_method_table) + 1)
    your_method_table['wow_change'] = your_method_table['wow_percent'].apply(
        lambda x: f"{x:+.2f}%" if not pd.isna(x) else '-'
    )
    
    your_method_display = your_method_table[['week', 'scripts', 'wow_change', 'your_flag']].copy()
    
    print("Your Method (WoW %) - Sample Results:")
    print(your_method_display.head(10).to_string(index=False))
    
    # ========================================
    # 2. SHAUN'S METHOD (Z-Score)
    # ========================================
    print(f"\n📊 METHOD 2: Shaun's Method (Z-Score Analysis)")
    print("=" * 50)
    
    # Calculate 8-week rolling statistics
    analysis_df['rolling_mean_8wk'] = analysis_df['scripts'].rolling(window=8, min_periods=1).mean().shift(1)
    analysis_df['rolling_std_8wk'] = analysis_df['scripts'].rolling(window=8, min_periods=1).std().shift(1)
    
    # Calculate Z-score
    analysis_df['z_score'] = (analysis_df['scripts'] - analysis_df['rolling_mean_8wk']) / analysis_df['rolling_std_8wk']
    
    # Create flags based on Z-score
    def classify_zscore_performance(z_score):
        if pd.isna(z_score):
            return '-'
        
        abs_z = abs(z_score)
        if abs_z <= 1.0:
            return 'In-Line'
        elif abs_z <= 2.0:
            return 'Slightly Above' if z_score > 0 else 'Slightly Below'
        else:
            return f'{abs_z:.1f} SD Above' if z_score > 0 else f'{abs_z:.1f} SD Below'
    
    analysis_df['shaun_flag'] = analysis_df['z_score'].apply(classify_zscore_performance)
    
    # Create Shaun's method table
    shaun_method_table = analysis_df[['date', 'scripts', 'rolling_mean_8wk', 'rolling_std_8wk', 'z_score', 'shaun_flag']].copy()
    shaun_method_table['week'] = range(1, len(shaun_method_table) + 1)
    
    # Round for display
    shaun_method_table['8wk_mean'] = shaun_method_table['rolling_mean_8wk'].round(0)
    shaun_method_table['st_dev'] = shaun_method_table['rolling_std_8wk'].round(0)
    shaun_method_table['z_score_display'] = shaun_method_table['z_score'].apply(
        lambda x: f"{x:+.2f}" if not pd.isna(x) else '-'
    )
    
    shaun_method_display = shaun_method_table[['week', 'scripts', '8wk_mean', 'st_dev', 'z_score_display', 'shaun_flag']].copy()
    shaun_method_display.columns = ['Week', 'Scripts', '8-Wk Mean', 'St.Dev', 'Z-Score', "Shaun's Flag"]
    
    print("Shaun's Method (Z-Score) - Sample Results:")
    print(shaun_method_display.head(10).to_string(index=False))
    
    return your_method_table, shaun_method_table, your_method_display, shaun_method_display

# Run the comparison analysis
print("🚀 BUILDING TWO COMPARISON TABLES")
print("=" * 60)

your_table, shaun_table, your_display, shaun_display = build_comparison_tables(df)


# %%
# @title Holiday-Aware Trend Analysis Dashboard Visualization
def create_comparison_dashboard(your_table, shaun_table, df_with_holidays=None):
    """
    Create a comprehensive dashboard comparing both analysis methods
    Similar to the attached image showing trend classifications
    """
    
    # Color mapping for classifications
    color_map = {
        'Baseline Building': '#CCCCCC',
        'In-Line': '#2E86AB',
        'Slightly Above': '#A23B72',
        'Slightly Below': '#F18F01',
        'Meaningfully Above': '#06FFA5',
        'Meaningfully Below': '#FF4444',
        'Holiday-Impacted': '#8A2BE2'
    }
    
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Weekly Scripts with Trend Classifications (Your Method - WoW %)',
            'Weekly Scripts with Trend Classifications (Shaun\'s Method - Z-Score)',
            'Z-Score Analysis Timeline'
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # ========================================
    # CHART 1: YOUR METHOD (WoW %)
    # ========================================
    
    # Group data by classification for your method
    your_classifications = your_table['your_flag'].unique()
    
    for classification in your_classifications:
        if classification == '-':
            continue
            
        class_data = your_table[your_table['your_flag'] == classification]
        
        # Check for holiday weeks if available
        if df_with_holidays is not None and 'is_holiday_week' in df_with_holidays.columns:
            holiday_data = class_data.merge(
                df_with_holidays[['date', 'is_holiday_week']], 
                on='date', 
                how='left'
            )
            holiday_weeks = holiday_data[holiday_data['is_holiday_week'] == True]
            non_holiday_weeks = holiday_data[holiday_data['is_holiday_week'] == False]
            
            # Non-holiday weeks (circles)
            if len(non_holiday_weeks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=non_holiday_weeks['date'],
                        y=non_holiday_weeks['scripts'],
                        mode='markers+lines',
                        name=f'{classification} (WoW)',
                        marker=dict(
                            color=color_map.get(classification, '#000000'),
                            size=8,
                            symbol='circle'
                        ),
                        line=dict(color=color_map.get(classification, '#000000'), width=2),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Scripts: %{y:,}<br>' +
                                    'WoW Change: %{customdata}<br>' +
                                    'Classification: ' + classification + '<br>' +
                                    '<extra></extra>',
                        text=[f'Week {row["week"]}' for _, row in non_holiday_weeks.iterrows()],
                        customdata=[row['wow_change'] for _, row in non_holiday_weeks.iterrows()]
                    ),
                    row=1, col=1
                )
            
            # Holiday weeks (diamonds)
            if len(holiday_weeks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=holiday_weeks['date'],
                        y=holiday_weeks['scripts'],
                        mode='markers',
                        name=f'{classification} (WoW Holiday)',
                        marker=dict(
                            color=color_map.get(classification, '#000000'),
                            size=12,
                            symbol='diamond',
                            line=dict(color='red', width=2)
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Scripts: %{y:,}<br>' +
                                    'WoW Change: %{customdata}<br>' +
                                    'Classification: ' + classification + ' (Holiday)<br>' +
                                    '<extra></extra>',
                        text=[f'Week {row["week"]}' for _, row in holiday_weeks.iterrows()],
                        customdata=[row['wow_change'] for _, row in holiday_weeks.iterrows()]
                    ),
                    row=1, col=1
                )
        else:
            # No holiday data, show all as regular weeks
            fig.add_trace(
                go.Scatter(
                    x=class_data['date'],
                    y=class_data['scripts'],
                    mode='markers+lines',
                    name=f'{classification} (WoW)',
                    marker=dict(
                        color=color_map.get(classification, '#000000'),
                        size=8,
                        symbol='circle'
                    ),
                    line=dict(color=color_map.get(classification, '#000000'), width=2),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Date: %{x}<br>' +
                                'Scripts: %{y:,}<br>' +
                                'WoW Change: %{customdata}<br>' +
                                'Classification: ' + classification + '<br>' +
                                '<extra></extra>',
                    text=[f'Week {row["week"]}' for _, row in class_data.iterrows()],
                    customdata=[row['wow_change'] for _, row in class_data.iterrows()]
                ),
                row=1, col=1
            )
    
    # ========================================
    # CHART 2: SHAUN'S METHOD (Z-Score)
    # ========================================
    
    # Group data by classification for Shaun's method
    shaun_classifications = shaun_table['shaun_flag'].unique()
    
    for classification in shaun_classifications:
        if classification == '-':
            continue
            
        class_data = shaun_table[shaun_table['shaun_flag'] == classification]
        
        # Check for holiday weeks if available
        if df_with_holidays is not None and 'is_holiday_week' in df_with_holidays.columns:
            holiday_data = class_data.merge(
                df_with_holidays[['date', 'is_holiday_week']], 
                on='date', 
                how='left'
            )
            holiday_weeks = holiday_data[holiday_data['is_holiday_week'] == True]
            non_holiday_weeks = holiday_data[holiday_data['is_holiday_week'] == False]
            
            # Non-holiday weeks (circles)
            if len(non_holiday_weeks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=non_holiday_weeks['date'],
                        y=non_holiday_weeks['scripts'],
                        mode='markers+lines',
                        name=f'{classification} (Z-Score)',
                        marker=dict(
                            color=color_map.get(classification, '#000000'),
                            size=8,
                            symbol='circle'
                        ),
                        line=dict(color=color_map.get(classification, '#000000'), width=2),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Scripts: %{y:,}<br>' +
                                    'Z-Score: %{customdata}<br>' +
                                    'Classification: ' + classification + '<br>' +
                                    '<extra></extra>',
                        text=[f'Week {row["week"]}' for _, row in non_holiday_weeks.iterrows()],
                        customdata=[f"{row['z_score']:+.2f}" for _, row in non_holiday_weeks.iterrows()]
                    ),
                    row=2, col=1
                )
            
            # Holiday weeks (diamonds)
            if len(holiday_weeks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=holiday_weeks['date'],
                        y=holiday_weeks['scripts'],
                        mode='markers',
                        name=f'{classification} (Z-Score Holiday)',
                        marker=dict(
                            color=color_map.get(classification, '#000000'),
                            size=12,
                            symbol='diamond',
                            line=dict(color='red', width=2)
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Scripts: %{y:,}<br>' +
                                    'Z-Score: %{customdata}<br>' +
                                    'Classification: ' + classification + ' (Holiday)<br>' +
                                    '<extra></extra>',
                        text=[f'Week {row["week"]}' for _, row in holiday_weeks.iterrows()],
                        customdata=[f"{row['z_score']:+.2f}" for _, row in holiday_weeks.iterrows()]
                    ),
                    row=2, col=1
                )
        else:
            # No holiday data, show all as regular weeks
            fig.add_trace(
                go.Scatter(
                    x=class_data['date'],
                    y=class_data['scripts'],
                    mode='markers+lines',
                    name=f'{classification} (Z-Score)',
                    marker=dict(
                        color=color_map.get(classification, '#000000'),
                        size=8,
                        symbol='circle'
                    ),
                    line=dict(color=color_map.get(classification, '#000000'), width=2),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Date: %{x}<br>' +
                                'Scripts: %{y:,}<br>' +
                                'Z-Score: %{customdata}<br>' +
                                'Classification: ' + classification + '<br>' +
                                '<extra></extra>',
                    text=[f'Week {row["week"]}' for _, row in class_data.iterrows()],
                    customdata=[f"{row['z_score']:+.2f}" for _, row in class_data.iterrows()]
                ),
                row=2, col=1
            )
    
    # ========================================
    # CHART 3: Z-Score Timeline
    # ========================================
    
    # Plot Z-scores over time
    analyzable_weeks = shaun_table[shaun_table['z_score'].notna()]
    
    if len(analyzable_weeks) > 0:
        # Regular weeks
        if df_with_holidays is not None and 'is_holiday_week' in df_with_holidays.columns:
            holiday_data = analyzable_weeks.merge(
                df_with_holidays[['date', 'is_holiday_week']], 
                on='date', 
                how='left'
            )
            non_holiday_z = holiday_data[holiday_data['is_holiday_week'] == False]
            holiday_z = holiday_data[holiday_data['is_holiday_week'] == True]
            
            if len(non_holiday_z) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=non_holiday_z['date'],
                        y=non_holiday_z['z_score'],
                        mode='markers+lines',
                        name='Z-Score',
                        marker=dict(color='#2E86AB', size=6),
                        line=dict(color='#2E86AB', width=2),
                        hovertemplate='Week %{text}: %{y:.2f} std devs<br>' +
                                    'Date: %{x}<br>' +
                                    '<extra></extra>',
                        text=[row['week'] for _, row in non_holiday_z.iterrows()]
                    ),
                    row=3, col=1
                )
            
            # Holiday weeks with special markers
            if len(holiday_z) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=holiday_z['date'],
                        y=holiday_z['z_score'],
                        mode='markers',
                        name='Z-Score (Holiday)',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='diamond',
                            line=dict(color='darkred', width=1)
                        ),
                        hovertemplate='Week %{text}: %{y:.2f} std devs<br>' +
                                    'Date: %{x}<br>' +
                                    'Holiday Impact<br>' +
                                    '<extra></extra>',
                        text=[row['week'] for _, row in holiday_z.iterrows()]
                    ),
                    row=3, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=analyzable_weeks['date'],
                    y=analyzable_weeks['z_score'],
                    mode='markers+lines',
                    name='Z-Score',
                    marker=dict(color='#2E86AB', size=6),
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate='Week %{text}: %{y:.2f} std devs<br>' +
                                'Date: %{x}<br>' +
                                '<extra></extra>',
                    text=[row['week'] for _, row in analyzable_weeks.iterrows()]
                ),
                row=3, col=1
            )
        
        # Add reference lines for z-score chart
        fig.add_hline(y=2.0, line_dash="solid", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=-2.0, line_dash="solid", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="orange", opacity=0.5, row=3, col=1)
        fig.add_hline(y=-1.0, line_dash="dash", line_color="orange", opacity=0.5, row=3, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Holiday-Aware Trend Analysis Dashboard - Method Comparison",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_yaxes(title_text="Weekly Scripts", row=1, col=1)
    fig.update_yaxes(title_text="Weekly Scripts", row=2, col=1)
    fig.update_yaxes(title_text="Standard Deviations", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

# Create the comparison dashboard
print("🎨 Creating Holiday-Aware Trend Analysis Dashboard...")

# Get holiday data if available
holiday_data = None
if 'df_with_holidays' in locals():
    holiday_data = df_with_holidays

dashboard_fig = create_comparison_dashboard(your_table, shaun_table, holiday_data)
dashboard_fig.show()

print("✅ Dashboard created successfully!")
print("\n📊 Chart Legend:")
print("• Circles = Normal weeks")
print("• Red-outlined diamonds = Holiday weeks")
print("• Z-score reference lines: ±1.0 (slight), ±2.0 (meaningful)")
print("• Top chart: Your Method (WoW % analysis)")
print("• Middle chart: Shaun's Method (Z-Score analysis)")
print("• Bottom chart: Z-Score timeline")


# %%
# @title Export Comparison Tables
def export_comparison_results(your_display, shaun_display):
    """
    Export both analysis methods to CSV files for further analysis
    """
    
    # Export Your Method (WoW %)
    your_filename = 'neffy_analysis_your_method_wow_percent.csv'
    your_display.to_csv(your_filename, index=False)
    print(f"✅ Your Method (WoW %) exported to: {your_filename}")
    
    # Export Shaun's Method (Z-Score)
    shaun_filename = 'neffy_analysis_shaun_method_zscore.csv'
    shaun_display.to_csv(shaun_filename, index=False)
    print(f"✅ Shaun's Method (Z-Score) exported to: {shaun_filename}")
    
    # Create summary comparison
    summary_data = {
        'Method': ['Your Method (WoW %)', 'Shaun\'s Method (Z-Score)'],
        'Analysis_Type': ['Week-over-Week Percentage Change', 'Statistical Z-Score Analysis'],
        'Baseline': ['Previous Week', '8-Week Rolling Average'],
        'Thresholds': ['±5% (In-Line), ±15% (Slight), >15% (Meaningful)', '±1.0 SD (In-Line), ±2.0 SD (Slight), >2.0 SD (Meaningful)'],
        'Best_For': ['Quick trend identification', 'Statistical significance testing'],
        'Holiday_Impact': ['High sensitivity to weekly changes', 'More stable, less sensitive to single-week anomalies']
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = 'neffy_analysis_methods_comparison_summary.csv'
    summary_df.to_csv(summary_filename, index=False)
    print(f"✅ Methods comparison summary exported to: {summary_filename}")
    
    return your_filename, shaun_filename, summary_filename

# Export the results
print("📁 EXPORTING COMPARISON RESULTS")
print("=" * 50)

your_file, shaun_file, summary_file = export_comparison_results(your_display, shaun_display)

print(f"\n📊 ANALYSIS COMPLETE!")
print("=" * 30)
print("Files created:")
print(f"• {your_file} - Your WoW % analysis")
print(f"• {shaun_file} - Shaun's Z-Score analysis") 
print(f"• {summary_file} - Methods comparison summary")
print(f"• Interactive dashboard displayed above")

print(f"\n🎯 KEY INSIGHTS:")
print("• Your Method: Great for identifying immediate week-over-week changes")
print("• Shaun's Method: Better for statistical significance and trend stability")
print("• Both methods complement each other for comprehensive analysis")
print("• Holiday impacts are highlighted with diamond markers in visualizations")



