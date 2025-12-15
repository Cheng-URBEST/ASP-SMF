import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================================================================
# 1. Configuration Parameters
# ==============================================================================
# These thresholds should be updated based on the output of the
# "ASP_SMF_Threshold_Determination.py" script for your specific study area.
THRESHOLDS = {
    # Threshold for 3-day positive change rate (m/day) to identify Rapid Accumulation Phase.
    'RATE_ACCUMULATION': 0.03396963,

    # Threshold for 3-day negative change rate (m/day) to identify Rapid Melt Phase.
    'RATE_MELT': -0.00793871,

    # Lower bound for Net Mass Flux (Snowfall - Snowmelt) in m/day.
    'SUBTRACT_FALL_MELT_LOWER': -0.0028,

    # Upper bound for Net Mass Flux (Snowfall - Snowmelt) in m/day.
    'SUBTRACT_FALL_MELT_UPPER': 0.0078,

    # Threshold for Total Mass Flux (Snowfall + Snowmelt) in m/day.
    'SUM_FALL_MELT': 0.0000,

    # Binary snow detection threshold (0.02 m) 
    'SNOW_DEPTH_STABLE': 0.02,
}

# ==============================================================================
# 2. Helper Functions - Phenology Segmentation and Consolidation
# ==============================================================================

def compute_slope(x, y):
    try:
        temp_df = pd.DataFrame({'x': x, 'y': y})
        temp_df.dropna(inplace=True)
        if len(temp_df) < 2: return np.nan
        if temp_df['y'].nunique() == 1: return 0.0
        slope, _ = np.polyfit(temp_df['x'], temp_df['y'], 1)
        return slope
    except np.linalg.LinAlgError:
        return np.nan

def determine_phase_4(row, thresholds):
    snow_depth = row['ERA5_Land_snow_depth']
    rate = row['rate_3day']
    if rate >= thresholds['RATE_ACCUMULATION']: return 2
    elif rate <= thresholds['RATE_MELT']: return 3
    elif snow_depth < thresholds['SNOW_DEPTH_STABLE'] and thresholds['RATE_MELT'] < rate < thresholds['RATE_ACCUMULATION']: return 0
    elif snow_depth >= thresholds['SNOW_DEPTH_STABLE'] and thresholds['RATE_MELT'] < rate < thresholds['RATE_ACCUMULATION']: return 1
    else: return np.nan

def determine_phase_3(row, thresholds):
    snow_depth = row['ERA5_Land_snow_depth']
    rate = row['rate_3day']
    if rate >= thresholds['RATE_ACCUMULATION']: return 2
    elif rate <= thresholds['RATE_MELT']: return 2
    elif snow_depth < thresholds['SNOW_DEPTH_STABLE'] and thresholds['RATE_MELT'] < rate < thresholds['RATE_ACCUMULATION']: return 0
    elif snow_depth >= thresholds['SNOW_DEPTH_STABLE'] and thresholds['RATE_MELT'] < rate < thresholds['RATE_ACCUMULATION']: return 1
    else: return np.nan

def calculate_fall_melt_metrics(df):
    df['ERA5_Land_sum_fall_melt'] = df['ERA5_Land_snowfall_sun'] + df['ERA5_Land_snowmelt_sun']
    df['ERA5_Land_subtract_fall_melt'] = df['ERA5_Land_snowfall_sun'] - df['ERA5_Land_snowmelt_sun']
    return df

def rle_indices(values):
    values = values.tolist()
    if len(values) == 0: return []
    runs = []
    start_idx = 0
    current_val = values[0]
    for i in range(1, len(values)):
        if values[i] != current_val:
            runs.append((start_idx, i - 1, current_val))
            start_idx = i
            current_val = values[i]
    runs.append((start_idx, len(values) - 1, current_val))
    return runs

def clamp_window(start, end, length):
    return max(start, 0), min(end, length - 1)

def get_windowed_sub_dfs(df, center_indices, window_size_for_count, window_size_for_mean):
    n = len(df)
    c_center = (min(center_indices) + max(center_indices)) / 2.0
    half_count = window_size_for_count // 2
    big_start, big_end = clamp_window(int(np.floor(c_center - half_count)), int(np.floor(c_center + half_count)), n)
    sub_df_big = df.iloc[big_start:big_end + 1]
    half_mean = window_size_for_mean // 2
    small_start, small_end = clamp_window(int(np.floor(c_center - half_mean)), int(np.floor(c_center + half_mean)), n)
    sub_df_small = df.iloc[small_start:small_end + 1]
    return sub_df_big, sub_df_small

def apply_run_merge_logic(phase_array, df, run_length_to_process, window_count, count_threshold, window_mean, thresholds):
    phase = phase_array.copy()
    runs = rle_indices(phase)
    n = len(df)
    for start_idx, end_idx, val in runs:
        if (end_idx - start_idx + 1) != run_length_to_process: continue
        if start_idx == 0:
            if end_idx + 1 < n: phase[start_idx:end_idx + 1] = phase[end_idx + 1]
            continue
        if end_idx == n - 1:
            if start_idx - 1 >= 0: phase[start_idx:end_idx + 1] = phase[start_idx - 1]
            continue
        sub_df_big, sub_df_small = get_windowed_sub_dfs(df, list(range(start_idx, end_idx + 1)), window_count, window_mean)
        counts = sub_df_big['phase_3_combine'].value_counts()
        dominant_phases = counts[counts >= count_threshold]
        if not dominant_phases.empty:
            new_val = dominant_phases.idxmax()
        else:
            mean_sub_fall = sub_df_small['ERA5_Land_subtract_fall_melt'].mean()
            mean_sum_fall = sub_df_small['ERA5_Land_sum_fall_melt'].mean()
            mean_snow_depth = sub_df_small['ERA5_Land_snow_depth'].mean()
            if val == 0:
                is_stable = (thresholds['SUBTRACT_FALL_MELT_LOWER'] < mean_sub_fall <= thresholds['SUBTRACT_FALL_MELT_UPPER']) or \
                            (mean_sum_fall <= thresholds['SUM_FALL_MELT'])
                new_val = 1 if is_stable else 2
            else: 
                new_val = 0 if mean_snow_depth <= thresholds['SNOW_DEPTH_STABLE'] else 1
        phase[start_idx:end_idx + 1] = new_val
    return phase

def final_short_run_cleanup(phase_array, df, thresholds):
    phase = phase_array.copy()
    runs = rle_indices(phase)
    for idx, (start_idx, end_idx, run_val) in enumerate(runs):
        if (end_idx - start_idx + 1) <= 5:
            left_val = runs[idx - 1][2] if idx > 0 else None
            right_val = runs[idx + 1][2] if idx < len(runs) - 1 else None
            new_val = -1
            if left_val is not None and left_val == right_val:
                new_val = left_val
            else:
                mean_snow = df.iloc[start_idx:end_idx + 1]['ERA5_Land_snow_depth'].mean()
                if run_val == 0: new_val = 2
                elif run_val == 1: new_val = 0 if mean_snow <= thresholds['SNOW_DEPTH_STABLE'] else 2
                elif run_val == 2: new_val = 0 if mean_snow <= thresholds['SNOW_DEPTH_STABLE'] else 1
            if new_val != -1:
                phase[start_idx:end_idx + 1] = new_val
    return phase

def final_check_and_correct(df, thresholds):
    df['group'] = (df['phase_3_combine'] != df['phase_3_combine'].shift()).cumsum()
    df['run_length'] = df.groupby('group')['phase_3_combine'].transform('count')
    df['phase_3_combine_check'] = df['phase_3_combine']
    short_run_groups = df[df['run_length'] <= 5]['group'].unique()
    for group_id in short_run_groups:
        group_indices = df[df['group'] == group_id].index
        start_idx, end_idx = group_indices[0], group_indices[-1]
        if start_idx == 0 or end_idx == len(df) - 1: continue
        phase = df.loc[start_idx, 'phase_3_combine']
        prev_phase = df.loc[start_idx - 1, 'phase_3_combine']
        next_phase = df.loc[end_idx + 1, 'phase_3_combine']
        new_val = -1
        if prev_phase == next_phase:
            new_val = prev_phase
        else:
            avg_snow_depth = df.loc[start_idx:end_idx, 'ERA5_Land_snow_depth'].mean()
            if phase == 0: new_val = 2
            elif phase == 1: new_val = 0 if avg_snow_depth <= thresholds['SNOW_DEPTH_STABLE'] else 2
            elif phase == 2: new_val = 0 if avg_snow_depth <= thresholds['SNOW_DEPTH_STABLE'] else 1
        if new_val != -1:
            df.loc[start_idx:end_idx, 'phase_3_combine_check'] = new_val
    df.drop(['group', 'run_length'], axis=1, inplace=True)
    return df

# ==============================================================================
# 3. Helper Functions - Shape Model Fitting (Grid Search & RMA-SLSQP)
# ==============================================================================

def rle_pandas_style(series):
    n = len(series)
    if n == 0: return [], []
    else:
        changes = series != series.shift()
        run_lengths = changes.cumsum()
        return series.groupby(run_lengths).first().values, series.groupby(run_lengths).count().values

def modify_step1_phase(df, max_iterations=6):
    iteration = 0
    while iteration < max_iterations:
        phases, counts = rle_pandas_style(df['step1_phase'])
        to_modify = []
        for i, (phase, count) in enumerate(zip(phases, counts)):
            if phase in [1, 2]:
                segment_start = sum(counts[:i])
                segment_end = segment_start + count
                segment = df.iloc[segment_start:segment_end]
                valid_S1 = segment['S1_dry_IMS'].notna().sum()
                if valid_S1 <= 3 and count <= 14:
                    to_modify.append(i)
        if not to_modify: break
        to_modify_sorted = sorted(to_modify, key=lambda x: counts[x])
        for run_idx in to_modify_sorted:
            if run_idx >= len(phases): continue
            phase = phases[run_idx]
            if phase not in [1, 2]: continue
            prev_idx, next_idx = run_idx - 1, run_idx + 1
            prev_length = counts[prev_idx] if prev_idx >= 0 else np.inf
            next_length = counts[next_idx] if next_idx < len(phases) else np.inf
            if prev_length < next_length and prev_idx >= 0: new_phase = phases[prev_idx]
            elif next_idx < len(phases): new_phase = phases[next_idx]
            else: new_phase = phase
            start = sum(counts[:run_idx])
            end = start + counts[run_idx]
            df.loc[start:end-1, 'step1_phase'] = new_phase
        iteration += 1
    return df

def shift_series(series, beta):
    return series.shift(beta)

def fit_alpha_gamma_with_constraints(X, Y):
    if len(X) <= 1: return np.nan, np.nan, np.inf
    mask = ~np.isnan(X) & ~np.isnan(Y)
    X_clean, Y_clean = X[mask], Y[mask]
    if len(X_clean) <= 1: return np.nan, np.nan, np.inf
    def objective(params):
        alpha, gamma = params
        residuals = alpha * X_clean + gamma - Y_clean
        return np.sum(residuals**2) / (alpha**2 + 1)
    x0 = [1.0, 0.0]
    bounds = [(0.3, 2.0), (-2.0, 2.0)]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds)
    if not res.success: return np.nan, np.nan, np.inf
    alpha_opt, gamma_opt = res.x
    y_pred = alpha_opt * X_clean + gamma_opt
    y_pred[y_pred < 0] = 0
    rmse = np.sqrt(mean_squared_error(Y_clean, y_pred))
    return alpha_opt, gamma_opt, rmse

def fill_step3_missing(df, Target_snow='S1_dry_IMS', snow_col='ERA5_Land_snow_depth'):
    df['step3_copyERA5_Land'] = np.nan
    mask = df[Target_snow].notna() & df[snow_col].notna()
    if mask.sum() < 2:
        df['step3_omi'] = np.nan
        df['step3_copyERA5_Land'] = df[snow_col]
        return df
    best_rmse, best_beta, best_alpha, best_gamma = np.inf, np.nan, np.nan, np.nan
    for beta in range(-6, 7):
        shifted_snow = df[snow_col].shift(beta)
        mask_beta = mask & shifted_snow.notna()
        if mask_beta.sum() < 2: continue
        X, Y = shifted_snow[mask_beta].values, df[Target_snow][mask_beta].values
        alpha_opt, gamma_opt, rmse = fit_alpha_gamma_with_constraints(X, Y)
        if rmse < best_rmse:
            best_rmse, best_beta, best_alpha, best_gamma = rmse, beta, alpha_opt, gamma_opt
    if np.isfinite(best_rmse):
        best_shifted_snow = df[snow_col].shift(best_beta)
        mask_step3 = ((df['S1_dry_IMS'] > 0.01) | df['S1_dry_IMS'].isna()) & (df['step2_opt'].isna())
        pred = np.nan * np.ones(len(df), dtype=float)
        valid_pred_mask = mask_step3 & best_shifted_snow.notna()
        pred[valid_pred_mask] = best_alpha * best_shifted_snow[valid_pred_mask] + best_gamma
        pred[pred < 0] = 0.0
        df['step3_omi'] = pred
    else:
        df['step3_omi'] = np.nan
        df['step3_copyERA5_Land'] = df[snow_col]
    return df

# ==============================================================================
# 4. Helper Function - Plotting Module
# ==============================================================================

def plot_snow_depth_timeseries(df, output_path):
    """
    Generates a time series plot comparing ERA5, S1, and Optimized Snow Depth.
    
    Args:
        df (pd.DataFrame): DataFrame containing date and snow depth columns.
        output_path (str): Path to save the output PNG file.
    """
    print(f"--> Generating time series plot...")
    
    # Ensure Date column is datetime objects
    plot_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(plot_df['Date']):
        plot_df['Date'] = pd.to_datetime(plot_df['Date'], errors='coerce')
    plot_df.sort_values('Date', inplace=True)
    
    # Use standard style
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Plot ERA5-Land (Background - Blue Dashed Line)
    # Using 'alpha' to make it less dominant
    ax.plot(plot_df['Date'], plot_df['ERA5_Land_snow_depth'], 
            label='ERA5-Land (Reanalysis)', color='royalblue', 
            linestyle='--', alpha=0.6, linewidth=1.5)
    
    # 2. Plot Optimized Snow Depth (Result - Green Solid Line)
    # Thicker line to highlight the result
    if 'Optimized_Snow_Depth' in plot_df.columns:
        ax.plot(plot_df['Date'], plot_df['Optimized_Snow_Depth'], 
                label='ASP-SMF (Optimized)', color='forestgreen', 
                linestyle='-', linewidth=2.0)
    
    # 3. Plot Sentinel-1 (Anchors - Red Dots)
    # Check if S1_dry exists, if not, try S1_dry_IMS
    s1_col = 'S1_dry' if 'S1_dry' in plot_df.columns else 'S1_dry_IMS'
    
    if s1_col in plot_df.columns:
        # Filter for non-null values to avoid clutter
        s1_data = plot_df.dropna(subset=[s1_col])
        if not s1_data.empty:
            ax.scatter(s1_data['Date'], s1_data[s1_col], 
                       label='Sentinel-1 (Anchors)', color='red', 
                       marker='o', s=60, zorder=5, edgecolors='white', linewidth=0.5)

    # Aesthetics
    ax.set_title('Snow Depth Time Series Optimization', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Snow Depth (m)', fontsize=12, fontweight='bold')
    
    # Date formatting on X-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    
    # Grid and Legend
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    
    # Tight layout and Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"--> Plot saved to: {output_path}")

# ==============================================================================
# 5. Core Processing Flow
# ==============================================================================

def process_single_file_complete_flow(file_path):
    print(f"--> Reading file: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Read failed: {e}")
        return

    # --- Save original columns to preserve them later ---
    original_columns = df.columns.tolist()

    # ==========================================================================
    # PART 0: Preprocessing - Generate S1_dry_IMS column
    # ==========================================================================
    
    # Check if necessary columns exist to generate S1_dry_IMS
    if 'S1_dry' in df.columns and 'IMS' in df.columns:
        # Copy S1_dry data
        df['S1_dry_IMS'] = df['S1_dry'].copy()
        # Logic: If S1_dry is null AND IMS == 2 (usually indicating no snow/land), fill with 0
        condition = (df['S1_dry'].isnull()) & (df['IMS'] == 2)
        df.loc[condition, 'S1_dry_IMS'] = 0
        print("--> [Preprocessing] Successfully generated S1_dry_IMS from S1_dry and IMS.")
    elif 'S1_dry_IMS' in df.columns:
        print("--> [Preprocessing] S1_dry_IMS already exists. Using it directly.")
    else:
        print("Error: Missing required columns ('S1_dry' and 'IMS') to generate S1_dry_IMS, and 'S1_dry_IMS' does not exist.")
        return

    # ==========================================================================
    # PART 1: Phenology Segmentation
    # ==========================================================================
    required_cols = ['Date', 'ERA5_Land_snow_depth', 'ERA5_Land_snowfall_sun', 'ERA5_Land_snowmelt_sun']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns (for phenology calculation): {required_cols}")
        return

    df['ERA5_Land_snow_depth'] = df['ERA5_Land_snow_depth'].clip(lower=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
    df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Day_difference'] = (df['Date'] - df['Date'].iloc[0]).dt.days
    df['rate_3day'] = np.nan
    n = len(df)
    for i in range(n):
        window = df.iloc[max(0, i - 1):min(n, i + 2)]
        slope = compute_slope(window['Day_difference'].values, window['ERA5_Land_snow_depth'].values)
        df.at[i, 'rate_3day'] = slope
    df.drop(columns=['Day_difference'], inplace=True)

    df['phase_4'] = df.apply(lambda row: determine_phase_4(row, THRESHOLDS), axis=1)
    df['phase_3'] = df.apply(lambda row: determine_phase_3(row, THRESHOLDS), axis=1)
    df = calculate_fall_melt_metrics(df)
    df['phase_3_combine'] = df['phase_3'].copy()
    
    merge_configs = [(1, 3, 2, 3), (2, 6, 3, 4), (3, 9, 4, 5), (4, 12, 5, 6), (5, 15, 6, 7)]
    for length, win_count, thresh, win_mean in merge_configs:
        df['phase_3_combine'] = apply_run_merge_logic(
            df['phase_3_combine'].values, df, length, win_count, thresh, win_mean, THRESHOLDS
        )
    df['phase_3_combine'] = final_short_run_cleanup(df['phase_3_combine'].values, df, THRESHOLDS)
    df = final_check_and_correct(df, THRESHOLDS)
    print("    Phase 1 Completed (Phase Check Generated)")

    # ==========================================================================
    # PART 2: Shape Model Fitting
    # ==========================================================================
    
    # S1_dry_IMS is already generated or confirmed in PART 0
    
    df['step1_phase'] = df['phase_3_combine_check'].copy()
    df = modify_step1_phase(df)

    # Step 2: Local Optimization (Grid Search)
    phases, counts = rle_pandas_style(df['step1_phase'])
    opt_counter = 0
    alpha_values = np.arange(0.3, 2.1, 0.1)
    gamma_values = np.arange(-2, 2.1, 0.1)

    for i, (phase, count) in enumerate(zip(phases, counts)):
        if phase not in [1, 2]: continue
        
        segment_start = sum(counts[:i])
        segment_end = segment_start + count
        segment = df.iloc[segment_start:segment_end]
        
        valid_S1 = segment['S1_dry_IMS'].notna().sum()
        valid_gt_0 = (segment['S1_dry_IMS'] > 0.008).sum()

        if valid_S1 > 2 and valid_gt_0 >= 2:
            col_name = f'opt_p{int(phase)}_{count}_{opt_counter}'
            opt_counter += 1
            best_rmse = np.inf
            best_alpha, best_gamma, best_beta = np.nan, np.nan, np.nan
            
            for beta in range(-6, 7):
                shifted_ERA5 = shift_series(segment['ERA5_Land_snow_depth'], beta)
                temp_df = pd.DataFrame({'ERA5_Land_shifted': shifted_ERA5, 'S1': segment['S1_dry_IMS']})
                temp_df['ERA5_Land_shifted'] = temp_df['ERA5_Land_shifted'].where(temp_df['ERA5_Land_shifted'] >= 0, 0)
                mask = temp_df['ERA5_Land_shifted'].notna() & temp_df['S1'].notna()
                if mask.sum() < 2: continue
                
                ERA5_valid = temp_df.loc[mask, 'ERA5_Land_shifted'].values
                S1_valid = temp_df.loc[mask, 'S1'].values

                for alpha in alpha_values:
                    for gamma in gamma_values:
                        S1_pred = alpha * ERA5_valid + gamma
                        S1_pred = np.where(S1_pred >= 0, S1_pred, 0)
                        rmse = mean_squared_error(S1_valid, S1_pred) ** 0.5
                        if rmse < best_rmse:
                            best_rmse, best_alpha, best_gamma, best_beta = rmse, alpha, gamma, beta

            if not np.isnan(best_alpha):
                shifted_full = shift_series(segment['ERA5_Land_snow_depth'], best_beta)
                pred = best_alpha * shifted_full + best_gamma
                pred = pred.apply(lambda x: x if x >= 0 else 0)
                df[col_name] = np.nan
                df.loc[segment_start:segment_end - 1, col_name] = pred.values

    opt_cols = [c for c in df.columns if c.startswith('opt_p')]
    df['step2_opt'] = df[opt_cols].mean(axis=1, skipna=True) if opt_cols else np.nan

    # Step 3 (Restriction: Skip Step 3 if Step 2 produced no results)
    if df['step2_opt'].notna().any():
        df = fill_step3_missing(df)
    else:
        # If Step 2 failed completely, skip Step 3
        print("    [Info] Step 2 produced no valid results. Skipping Step 3 (Global Optimization).")
        df['step3_omi'] = np.nan

    # ==========================================================================
    # PART 3: Final Result Generation (Pure Combine + Fallback)
    # ==========================================================================
    
    target_col_name = "Optimized_Snow_Depth"
    df[target_col_name] = np.nan

    # 1. Pure Combine Logic
    # Priority A: S1 near zero -> 0
    df.loc[df['S1_dry_IMS'] <= 0.01, target_col_name] = 0
    
    # Priority B: Step 2 Optimized Values
    cond_step2 = ((df['S1_dry_IMS'] > 0.01) | df['S1_dry_IMS'].isna()) & df['step2_opt'].notna()
    df.loc[cond_step2, target_col_name] = df.loc[cond_step2, 'step2_opt']
    
    # Priority C: Step 3 Optimized Values (Only if A and B don't cover)
    cond_step3 = ((df['S1_dry_IMS'] > 0.01) | df['S1_dry_IMS'].isna()) & df['step2_opt'].isna()
    if df['step3_omi'].notna().any():
         df.loc[cond_step3 & df['step3_omi'].notna(), target_col_name] = df.loc[cond_step3 & df['step3_omi'].notna(), 'step3_omi']

    # 2. Missing Value Imputation (Fallback to ERA5)
    df[target_col_name] = df[target_col_name].fillna(df['ERA5_Land_snow_depth'])

    print("    Phases 2 & 3 Completed (Optimization & Clean Merge)")

    # ==========================================================================
    # PART 4: Clean up columns and save
    # ==========================================================================
    
    # Prepare columns to keep: Original + Generated S1_dry_IMS + Final Result
    cols_to_keep = original_columns + ['S1_dry_IMS', target_col_name]
    
    # Deduplicate columns (prevent S1_dry_IMS from being added if already in original)
    cols_to_keep_unique = []
    seen = set()
    for c in cols_to_keep:
        if c not in seen:
            cols_to_keep_unique.append(c)
            seen.add(c)

    # Filter for final existing columns
    final_cols = [c for c in df.columns if c in cols_to_keep_unique]
    df_final = df[final_cols]

    # Save CSV
    dirname = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    output_path = os.path.join(dirname, f"{name_no_ext}_Optimized.csv")
    
    df_final.to_csv(output_path, index=False)
    print(f"--> Processing complete! File saved to: {output_path}")
    
    # ==========================================================================
    # PART 5: Plotting (New Addition)
    # ==========================================================================
    plot_output_path = os.path.join(dirname, f"{name_no_ext}_Optimized_Plot.png")
    # Using df_final which contains the optimized result, but we might need 'S1_dry' from original df
    # So we merge 'S1_dry' back into df_final if it exists in original df but not in df_final
    if 'S1_dry' in df.columns and 'S1_dry' not in df_final.columns:
        df_final = df_final.copy()
        df_final['S1_dry'] = df['S1_dry']
    
    plot_snow_depth_timeseries(df_final, plot_output_path)


def main():
    target_file_path = r"./data/step2_ASP_SMF_Workflow/Sample_timeSeries2.csv"
    if os.path.exists(target_file_path):
        process_single_file_complete_flow(target_file_path)
    else:
        print(f"Error: File not found {target_file_path}")

if __name__ == "__main__":
    main()