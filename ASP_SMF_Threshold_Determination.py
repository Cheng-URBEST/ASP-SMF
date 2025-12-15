import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from kneed import KneeLocator
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- Plotting Font Settings ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Path Settings ---
# Input folder (contains time series CSV files)
input_folder = Path(r"./data/step1_Threshold_Determination/input_time_series") 
# Output folder (for analysis results and plots)
output_folder_analysis = Path(r"./data/step1_Threshold_Determination/output_analysis")

# --- 2. Helper Functions (Slope Calculation & Knee Point Detection) ---

def compute_slope(x, y):
    """Calculates the slope of a linear fit for the given x and y arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if len(x_valid) < 2: return np.nan
    if np.all(y_valid == y_valid[0]): return 0.0  
    try:
        slope, _ = np.polyfit(x_valid, y_valid, 1)
        return slope
    except Exception:
        return np.nan

def find_knee_on_transformed_curve(x_data, y_data):
    """
    Finds the knee point for negative change rates by transforming the curve 
    and applying the Kneedle algorithm.
    """
    x_flipped = -x_data
    df = pd.DataFrame({'x': x_flipped, 'y': y_data}).sort_values(by='x').reset_index(drop=True)
    x_transformed_intermediate = df['x'].values
    y_transformed_intermediate = df['y'].values 
    y_transformed_final = 1 - y_transformed_intermediate
    kneedle = KneeLocator(x_transformed_intermediate, y_transformed_final, curve='concave', direction='increasing')
    if kneedle.knee is None:
        return np.nan
    return -kneedle.knee

# --- 3. Helper Functions (Decision Tree Analysis & Plotting) ---

def add_decision_annotations(ax, clf, feature_names):
    """Annotates the decision tree thresholds on the given matplotlib axis."""
    tree_ = clf.tree_
    plotted_thresholds = {'h': set(), 'v': set()} 

    annotation_fontsize = 6
    trans_x = ax.get_xaxis_transform()
    trans_y = ax.get_yaxis_transform()

    for i in range(tree_.node_count):
        if tree_.feature[i] != -2:
            feature_idx = tree_.feature[i]
            threshold = tree_.threshold[i]
            feature = feature_names[feature_idx]
            
            if feature == feature_names[0]:  # Vertical line
                if round(threshold, 5) not in plotted_thresholds['v']:
                    ax.axvline(x=threshold, color='green', linestyle='--', linewidth=1.5, zorder=3)
                    ax.text(threshold, 1.01, f'{threshold:.4f}', transform=trans_x, 
                            rotation=45, ha='left', va='bottom', 
                            fontsize=annotation_fontsize, zorder=4, 
                            color='black', clip_on=False)
                    plotted_thresholds['v'].add(round(threshold, 5))
            else:  # Horizontal line
                if round(threshold, 5) not in plotted_thresholds['h']:
                    ax.axhline(y=threshold, color='purple', linestyle='--', linewidth=1.5, zorder=3)
                    ax.text(1.01, threshold, f'{threshold:.4f}', transform=trans_y, 
                            rotation=0, ha='left', va='center', 
                            fontsize=annotation_fontsize, zorder=4, 
                            color='black', clip_on=False)
                    plotted_thresholds['h'].add(round(threshold, 5))

def determine_phase_3(row, pos_threshold, neg_threshold):
    """
    Determines the snow phenology phase (0, 1, or 2) based on snow depth 
    and the 3-day change rate using the calculated thresholds.
    """
    snow_depth = row['ERA5_Land_snow_depth']
    rate = row['rate_3day']
    
    if pd.isna(pos_threshold) or pd.isna(neg_threshold):
        return None

    if rate >= pos_threshold:
        return 2  # Accumulation (Dynamic Phase)
    elif rate <= neg_threshold:
        return 2  # Melt (Dynamic Phase)
    elif snow_depth < 0.05 and neg_threshold < rate < pos_threshold:
        return 0  # No Snow (Phase 0)
    elif snow_depth >= 0.05 and neg_threshold < rate < pos_threshold:
        return 1  # Stable (Phase 1)
    else:
        return None 

def plot_detailed_decision_boundary(clf, X, feature_names, output_path=None):
    """Visualizes the detailed decision boundary and invokes the annotation function."""
    feature_x, feature_y = feature_names[0], feature_names[1]
    
    x_min, x_max = X[feature_x].min() - 0.001, X[feature_x].max() + 0.001
    y_min, y_max = X[feature_y].min() - 0.001, X[feature_y].max() + 0.001
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    Z_input = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_names)
    Z = clf.predict(Z_input)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(14, 12)) 
    cmap = ListedColormap(['#87CEEB', '#FFB6C1'])
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.7, levels=[0.5, 1.5, 2.5], zorder=1)

    ax.set_title('Detailed Decision Boundary (Annotated)', fontsize=18, fontweight='bold', pad=35)
    ax.set_xlabel(feature_x, fontsize=16, fontweight='bold')
    ax.set_ylabel(feature_y, fontsize=16, fontweight='bold')
    
    x_ticks = np.linspace(x_min, x_max, 15) 
    y_ticks = np.linspace(y_min, y_max, 15)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{tick:.3f}' for tick in x_ticks], rotation=30, ha="right", fontsize=12)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.3f}' for tick in y_ticks], fontsize=12)

    legend_elements = [Patch(facecolor='#87CEEB', edgecolor='k', label='Phase 1 (Stable Phase)'),
                       Patch(facecolor='#FFB6C1', edgecolor='k', label='Phase 2 (Rapid Change Phase)')]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=2)
    
    add_decision_annotations(ax, clf, feature_names)
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Detailed decision boundary plot saved to: {output_path.name}")
    plt.close()

# --- 4. Statistical Reporting & Histogram Generation ---

def generate_quantile_report(df, output_path, pos_threshold=None, neg_threshold=None):
    print("Generating quantile statistical report...")
    
    def calculate_quantiles(data, step=5):
        if data.dropna().empty: return [np.nan] * len(np.arange(0, 101, step))
        return np.percentile(data.dropna(), np.arange(0, 101, step))
        
    quantile_results = pd.DataFrame({
        'Percentile': np.arange(0, 101, 5),
        'ERA5_Land_snow_depth': calculate_quantiles(df['ERA5_Land_snow_depth']),
        'rate_3day': calculate_quantiles(df['rate_3day']),
        'rate_3day_positive': calculate_quantiles(df[df['rate_3day'] > 0]['rate_3day']),
        'rate_3day_negative': calculate_quantiles(df[df['rate_3day'] < 0]['rate_3day'])
    })

    if pos_threshold is not None or neg_threshold is not None:
        threshold_rows = []
        if pos_threshold is not None:
            threshold_rows.append({
                'Percentile': 'THRESHOLD_POSITIVE',
                'ERA5_Land_snow_depth': None, 'rate_3day': pos_threshold,
                'rate_3day_positive': pos_threshold, 'rate_3day_negative': None
            })
        if neg_threshold is not None:
            threshold_rows.append({
                'Percentile': 'THRESHOLD_NEGATIVE',
                'ERA5_Land_snow_depth': None, 'rate_3day': neg_threshold,
                'rate_3day_positive': None, 'rate_3day_negative': neg_threshold
            })
        quantile_results = pd.concat([quantile_results, pd.DataFrame(threshold_rows)], ignore_index=True)

    quantile_results.to_excel(output_path, index=False)
    print(f"Quantile statistics (with thresholds) saved to: {output_path.name}")

def generate_rate_histogram(df, output_path, threshold_value=None, is_positive=True): 
    if is_positive:
        title = '3-Day Rate > 0'
        filtered_df = df[df['rate_3day'] > 0]['rate_3day']
        color_dot = '#22B14C'
        xlim = (-0.02, 0.4)
    else:
        title = '3-Day Rate < 0'
        filtered_df = df[df['rate_3day'] < 0]['rate_3day']
        color_dot = '#22B14C'
        xlim = (-0.119, 0.005)

    if filtered_df.empty:
        return

    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(5, 6))
    ax1.hist(filtered_df, bins=40, color='gray', edgecolor='black', alpha=0.7, label='Count')
    ax1.set_xlabel('3-Day Rate', color='black', fontsize=14, fontweight='bold', fontname='Arial')
    ax1.set_xlim(xlim)
    ax1.set_ylabel('Count', color='black', fontsize=14, fontweight='bold', fontname='Arial')
    
    ax2 = ax1.twinx()
    sorted_data = filtered_df.sort_values()
    cumulative = sorted_data.rank(method='average', pct=True)
    ax2.plot(sorted_data, cumulative, color='red', marker='o', linestyle='-', markersize=2, label='Cumulative Frequency')

    if threshold_value is not None:
        threshold_y_value = np.sum(filtered_df <= threshold_value) / len(filtered_df)
        ax2.plot(threshold_value, threshold_y_value, marker='o', color=color_dot, markersize=15, linestyle='none', markeredgecolor='none')

    ax2.set_ylabel('Frequency', color='red', fontsize=14, fontweight='bold', fontname='Arial')
    ax2.set_ylim(0, 1.05)
    plt.title(title, fontsize=14, fontweight='bold', fontname='Arial')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', transparent=False, bbox_inches='tight')
    plt.close()
    print(f"{title} histogram saved: {output_path.name}")


# --- 5. Single File Processing ---

def process_single_file(csv_file):
    try:
        pid = os.getpid()
        # print(f"[Process {pid}] Processing: {csv_file.name}") # Optional logging
        
        df = pd.read_csv(csv_file)
        if 'Date' not in df.columns or 'ERA5_Land_snow_depth' not in df.columns:
            return None

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            return None

        df['ERA5_Land_snow_depth'] = pd.to_numeric(df['ERA5_Land_snow_depth'], errors='coerce')
        
        df = df.sort_values('Date').reset_index(drop=True)
        df['Day_difference'] = (df['Date'] - df['Date'].iloc[0]).dt.days
        df['rate_3day'] = np.nan
        n = len(df)
        
        dates_diff = df['Day_difference'].values
        snow_depths = df['ERA5_Land_snow_depth'].values
        slopes = []
        
        for i in range(n):
            start = max(0, i - 1)
            end = min(n, i + 2)
            x_win = dates_diff[start:end]
            y_win = snow_depths[start:end]
            slopes.append(compute_slope(x_win, y_win))
            
        df['rate_3day'] = slopes
        df.to_csv(csv_file, index=False)
        
        cols_to_return = ['Date', 'ERA5_Land_snow_depth', 'rate_3day']
        extra_cols = ['ERA5_Land_snowfall_sun', 'ERA5_Land_snowmelt_sun']
        existing_extra_cols = [c for c in extra_cols if c in df.columns]
        
        return df[cols_to_return + existing_extra_cols].copy()

    except Exception:
        return None

# --- 6. Decision Tree Analysis Logic ---

def run_decision_tree_analysis(master_df, out_folder, pos_thresh, neg_thresh):
    """Executes decision tree analysis and saves results."""
    
    # 1. Filtering and Feature Engineering
    required_cols = ['ERA5_Land_snowfall_sun', 'ERA5_Land_snowmelt_sun', 'rate_3day', 'ERA5_Land_snow_depth']
    if not all(col in master_df.columns for col in required_cols):
        print("Error: Missing required columns for decision tree.")
        return

    # Apply phenology logic
    master_df['phase_3'] = master_df.apply(lambda row: determine_phase_3(row, pos_thresh, neg_thresh), axis=1)
    
    # Filter
    df_filtered = master_df[(master_df['phase_3'].notna()) & (master_df['phase_3'] != 0)].copy()
    
    if df_filtered.empty:
        print("No valid data (Phase 3 != 0) after threshold filtering.")
        return
        
    df_filtered['sum_fall_melt'] = df_filtered['ERA5_Land_snowfall_sun'] + df_filtered['ERA5_Land_snowmelt_sun'] 
    df_filtered['subtract_fall_melt'] = df_filtered['ERA5_Land_snowfall_sun'] - df_filtered['ERA5_Land_snowmelt_sun']

    # 2. Prepare Training Data
    feature_names = ['subtract_fall_melt', 'sum_fall_melt'] 
    X = df_filtered[feature_names]
    y = df_filtered['phase_3']
    
    if len(y.unique()) < 2:
        print("Error: Only one class present, cannot train classifier.")
        return

    # 3. Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Save concise text report
    tree_rules = export_text(clf, feature_names=feature_names, decimals=5)
    
    txt_output_path = out_folder / "decision_tree_results.txt"
    with open(txt_output_path, "w", encoding='utf-8') as f:
        f.write("Decision Tree Analysis Results\n")
        f.write("==============================\n\n")
        f.write(f"(1) Positive Threshold : {pos_thresh}\n")
        f.write(f"(2) Negative Threshold : {neg_thresh}\n\n")
        f.write("Decision Rules (text form):\n")
        f.write("-" * 30 + "\n")
        f.write(tree_rules)
    
    print(f"Decision tree results saved to: {txt_output_path.name}")

    # 4. Plot 1: Data Distribution
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = {1: 'blue', 2: 'red'}
    ax.scatter(X[feature_names[0]], X[feature_names[1]], c=y.map(colors), alpha=0.5, zorder=2)
    add_decision_annotations(ax, clf, feature_names)
    
    ax.set_title('Data Distribution and Decision Tree Splits', fontsize=16, fontweight='bold', pad=35)
    ax.set_xlabel(feature_names[0], fontsize=14, fontweight='bold')
    ax.set_ylabel(feature_names[1], fontsize=14, fontweight='bold')
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Phase 1 (Stable Phase)', markerfacecolor='blue', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='Phase 2 (Rapid Change Phase)', markerfacecolor='red', markersize=10)]
    ax.legend(handles=legend_elements, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plot1_path = out_folder / 'Data_Distribution_Annotated.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data distribution plot saved: {plot1_path.name}")

    # 5. Plot 2: Detailed Decision Boundary
    detailed_plot_path = out_folder / 'Detailed_Decision_Boundary_Annotated.png'
    plot_detailed_decision_boundary(clf, X, feature_names, output_path=detailed_plot_path)


# --- 7. Main Entry Point ---

if __name__ == '__main__':
    # 1. Preparation
    output_folder_analysis.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_folder.glob("*.csv"))

    if not csv_files:
        print(f"Warning: No CSV files found in {input_folder}")
    else:
        num_processes = multiprocessing.cpu_count()
        print(f"\nProcessing {len(csv_files)} files with {num_processes} processes...")

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = executor.map(process_single_file, csv_files)

        print("\nMerging data...")
        successful_results = [res for res in results if res is not None]

        if successful_results:
            combined_df = pd.concat(successful_results, ignore_index=True)
            print(f"Data merged. Total records: {len(combined_df)}")

            print("\nCalculating Thresholds...")  
            positive_threshold = None
            negative_threshold = None

            # (1) Positive
            positive_rates = combined_df[combined_df['rate_3day'] > 0]['rate_3day'].sort_values()
            if not positive_rates.empty:
                x_pos = positive_rates.values
                y_pos = np.arange(1, len(x_pos) + 1) / len(x_pos)
                kneedle_pos = KneeLocator(x_pos, y_pos, curve='concave', direction='increasing')
                if kneedle_pos.knee is not None:
                    positive_threshold = kneedle_pos.knee
                    print(f"Positive Threshold: {positive_threshold:.8f}")

            # (2) Negative
            negative_rates = combined_df[combined_df['rate_3day'] < 0]['rate_3day'].sort_values()
            if not negative_rates.empty:
                x_neg = negative_rates.values
                y_neg = np.arange(1, len(x_neg) + 1) / len(x_neg)
                negative_threshold = find_knee_on_transformed_curve(x_neg, y_neg)
                print(f"Negative Threshold: {negative_threshold:.8f}")
            
            # 5. Generate Basic Statistical Charts
            quantile_path = output_folder_analysis / "3_day_quantile_statistics.xlsx"
            hist_pos_path = output_folder_analysis / "rate_3day_positive_distribution.png"
            hist_neg_path = output_folder_analysis / "rate_3day_negative_distribution.png"
            
            generate_quantile_report(combined_df, quantile_path, positive_threshold, negative_threshold) 
            generate_rate_histogram(combined_df, hist_pos_path, positive_threshold, is_positive=True)
            generate_rate_histogram(combined_df, hist_neg_path, negative_threshold, is_positive=False)
            
            # 6. Execute Decision Tree Analysis
            if positive_threshold is not None and negative_threshold is not None:
                run_decision_tree_analysis(combined_df, output_folder_analysis, positive_threshold, negative_threshold)
            else:
                print("\nWarning: Could not calculate both thresholds. Skipping decision tree analysis.")

            print("\nAll operations completed.")
        else:
            print("\nAll file processing failed.")