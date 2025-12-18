import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Ensures graphs work in the browser
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --- 1. Page Configuration ---
st.set_page_config(page_title="Early Warning System: CSD", layout="wide")
st.title("🌊 Early Warning System: Critical Slowing Down")
st.markdown("""
This dashboard allows you to detect **Early Warning Signals (EWS)** in time-series data. 
It uses **Critical Slowing Down (CSD)** indicators: **Variance** and **Autocorrelation**.
""")

# --- 2. Sidebar: Data Upload & Parameters ---
st.sidebar.header("1. Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Default parameters
default_sigma = 30
default_window = 14
default_threshold = 1.0

if uploaded_file is not None:
    # --- 3. Data Loading ---
    df = pd.read_csv(uploaded_file)
    
    # Column Selection
    st.sidebar.subheader("Select Columns")
    
    # 1. Date Column: Allow all columns (since dates often load as strings first)
    # We try to find a column with 'date' or 'time' in the name to set as default
    all_cols = df.columns.tolist()
    default_date_idx = next((i for i, col in enumerate(all_cols) if 'date' in col.lower() or 'time' in col.lower()), 0)
    
    date_col = st.sidebar.selectbox("Date Column", all_cols, index=default_date_idx, 
                                    help="Select the column containing timestamps.")

    # 2. Value Column: Filter for NUMERIC types only (float, int)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.error("❌ No numeric columns found! Please upload a dataset with numerical values.")
        st.stop()
        
    # Try to find a column with 'level', 'height', or 'value' for default
    default_val_idx = next((i for i, col in enumerate(numeric_cols) if any(x in col.lower() for x in ['level', 'height', 'value', 'fsm'])), 0)
    
    value_col = st.sidebar.selectbox("Value Column", numeric_cols, index=default_val_idx,
                                     help="Select the numeric variable to analyze (e.g. Water Level).")
    
    # Convert Date
    try:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        st.sidebar.success("Date parsed successfully!")
    except Exception as e:
        st.sidebar.error(f"Date parsing error: {e}")

    # --- 4. Parameter Tuning ---
    st.sidebar.subheader("Tuning Parameters")
    
    # Sliders for interactive adjustment
    sigma_days = st.sidebar.slider("Trend Smoothing (Sigma in Days)", 
                                   min_value=7, max_value=60, value=default_sigma, 
                                   help="Higher values = Smoother trend (ignores more short-term noise).")
    
    window_days = st.sidebar.slider("Analysis Window (Days)", 
                                    min_value=3, max_value=30, value=default_window,
                                    help="Size of the sliding window to measure variance.")
    
    # Create two columns for thresholds in the sidebar
    col_var, col_ac = st.sidebar.columns(2)
    with col_var:
        threshold = st.number_input("Variance Threshold", 
                                    min_value=0.1, max_value=10.0, value=default_threshold, step=0.1,
                                    help="Alarm if Variance > X")
    with col_ac:
        threshold_ac = st.number_input("Autocorr Threshold", 
                                       min_value=0.1, max_value=1.0, value=0.8, step=0.05,
                                       help="Alarm if Memory > X")

    # Calculations
    sigma_hours = 24 * sigma_days
    window_hours = 24 * window_days
    
    # 1. Trend (Gaussian)
    trend = gaussian_filter1d(df[value_col], sigma=sigma_hours)
    
    # 2. Residuals
    residuals = df[value_col] - trend
    
    # 3. CSD Signals
    rolling_variance = residuals.rolling(window=window_hours).var()
    rolling_autocorr = residuals.rolling(window=window_hours).corr(residuals.shift(1))
    
    # 4. Alarms (Variance Based)
    alarms = rolling_variance > threshold
    
    # --- 5. Main Dashboard Display ---
    
    # UPDATED: Added Tab 5 for Event Deep Dive
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analysis Plots", 
        "📋 Data & Statistics", 
        "⏱️ Lead Time Analysis", 
        "🧠 Memory Analysis",
        "🔍 Event Deep Dive"
    ])
    
    # --- TAB 1: PLOTS ---
    with tab1:
        st.subheader("Step 1: Trend Extraction")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df.index, df[value_col], color='gray', alpha=0.5, label='Raw Data')
        ax1.plot(df.index, trend, color='blue', linewidth=2, label=f'Trend (Sigma={sigma_days}d)')
        ax1.legend()
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)
        
        st.subheader("Step 2: Early Warning Signal (Variance)")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(rolling_variance.index, rolling_variance, color='orange', label='Rolling Variance')
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
        ax2.fill_between(rolling_variance.index, 0, rolling_variance.max(), 
                         where=alarms, color='red', alpha=0.1, label='Alarm Triggered')
        ax2.legend()
        ax2.set_ylabel("Variance")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
        
        st.subheader("Step 3: Stress Test (Alarms vs. Reality)")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(df.index, df[value_col], color='gray', alpha=0.6, label='Raw Data')
        ax3.fill_between(df.index, df[value_col].min(), df[value_col].max(), 
                         where=alarms, color='red', alpha=0.3, label='Warning Active')
        ax3.set_title("When would the alarm have rung?")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    # --- TAB 2: DATA ---
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Data Points", len(df))
            st.metric("Total Alarm Events", int(alarms.astype(int).diff().eq(1).sum()))
        with col2:
            st.write("### Preview of Alarms")
            st.dataframe(df[alarms].head(10))

    # --- TAB 3: LEAD TIME ANALYSIS ---
    with tab3:
        st.subheader("Performance Evaluation: Multiple Events")
        
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            default_height = float(df[value_col].quantile(0.99))
            min_flood_height = st.number_input("Minimum Flood Height (m)", 
                                               value=default_height, 
                                               format="%.2f",
                                               help="Peaks below this level are ignored.")
        with col_param2:
            min_distance_days = st.number_input("Min Days Between Floods", value=30, step=1)
        
        peaks, _ = find_peaks(df[value_col], height=min_flood_height, distance=min_distance_days*24)
        
        if len(peaks) > 0:
            st.success(f"Found **{len(peaks)}** flood events based on these settings.")
            results = []
            for p_idx in peaks:
                peak_date = df.index[p_idx]
                peak_level = df[value_col].iloc[p_idx]
                window_start = peak_date - pd.Timedelta(days=30)
                pre_peak_data = rolling_variance.loc[window_start:peak_date]
                alarms_triggered = pre_peak_data[pre_peak_data > threshold]
                
                if not alarms_triggered.empty:
                    first_alarm = alarms_triggered.index[0]
                    lead_time = (peak_date - first_alarm).days
                    status = "✅ Detected" if lead_time > 0 else "⚠️ Late Warning"
                else:
                    first_alarm = None
                    lead_time = 0
                    status = "❌ Missed"
                
                results.append({
                    "Flood Date": peak_date.date(),
                    "Peak Level": round(peak_level, 2),
                    "First Alarm": first_alarm.date() if first_alarm else "-",
                    "Lead Time (Days)": lead_time,
                    "Status": status
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.applymap(
                lambda x: 'color: green' if x == '✅ Detected' else ('color: red' if x == '❌ Missed' else ''), 
                subset=['Status']
            ), use_container_width=True)
        else:
            st.warning("No flood peaks found.")

    # --- TAB 4: MEMORY ANALYSIS ---
    with tab4:
        st.subheader("Signal 2: Critical Slowing Down (Memory)")
        st.markdown("""
        **What is this?**
        Before a crash, systems become "sluggish." This shows up as high **Autocorrelation**.
        """)
        
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.plot(rolling_autocorr.index, rolling_autocorr, color='purple', linewidth=1.5, label='Autocorrelation (Lag-1)')
        ax4.axhline(y=threshold_ac, color='magenta', linestyle='--', linewidth=2, label=f'Warning Level ({threshold_ac})')
        
        high_memory = rolling_autocorr > threshold_ac
        if high_memory.any():
            ax4.fill_between(rolling_autocorr.index, 0, 1, where=high_memory, color='purple', alpha=0.2, label='High Memory Zone')
            
        ax4.set_ylim(-0.2, 1.1)
        ax4.set_ylabel("Autocorrelation")
        ax4.legend(loc='lower left')
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

    # --- TAB 5: EVENT DEEP DIVE (NEW) ---
    with tab5:
        st.subheader("🔍 Event Deep Dive Analysis")
        st.markdown("Manually zoom into a specific event to check the exact timing of the alarm vs. the flood peak.")

        # 1. Inputs for Deep Dive
        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        
        # Try to set sensible defaults (e.g., the Jan 2021 flood if in range)
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        default_start = pd.to_datetime('2020-12-01').date()
        default_end = pd.to_datetime('2021-02-15').date()
        
        # Ensure defaults are within valid data range
        if default_start < min_date or default_start > max_date: default_start = min_date
        if default_end > max_date or default_end < min_date: default_end = max_date

        with col_d1:
            zoom_start = st.date_input("Analysis Start Date", value=default_start, min_value=min_date, max_value=max_date)
        with col_d2:
            zoom_end = st.date_input("Analysis End Date", value=default_end, min_value=min_date, max_value=max_date)
        with col_d3:
            # Allow local threshold adjustment
            local_threshold = st.number_input("Event Threshold", min_value=0.1, max_value=10.0, value=float(threshold), step=0.1)

        # 2. Slice Data
        ts_start = pd.Timestamp(zoom_start)
        ts_end = pd.Timestamp(zoom_end)
        
        if ts_start >= ts_end:
            st.error("Error: Start Date must be before End Date.")
        else:
            # Subset for Variance and Water Level
            wl_subset = df.loc[ts_start:ts_end, value_col]
            var_subset = rolling_variance.loc[ts_start:ts_end]

            if wl_subset.empty:
                st.warning("No data found in this range.")
            else:
                # 3. Find Peak and Alarm
                # Find Peak in this window
                flood_peak_date = wl_subset.idxmax()
                flood_peak_level = wl_subset.max()
                
                # Find First Alarm in this window (before the peak)
                # We filter variance for values > local_threshold
                alarms_subset = var_subset[var_subset > local_threshold]
                
                # Look for alarm ONLY before or on the peak date
                valid_alarms = alarms_subset[alarms_subset.index <= flood_peak_date]
                
                if not valid_alarms.empty:
                    alarm_date = valid_alarms.index[0]
                    lead_time = flood_peak_date - alarm_date
                    lead_days = lead_time.days
                    is_detected = True
                else:
                    alarm_date = None
                    lead_days = 0
                    is_detected = False

                # 4. Display Stats
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Flood Peak Date", flood_peak_date.strftime('%Y-%m-%d'))
                c2.metric("Flood Peak Level", f"{flood_peak_level:.2f} m")
                c3.metric("Alarm Date", alarm_date.strftime('%Y-%m-%d') if is_detected else "-")
                c4.metric("Lead Time", f"{lead_days} Days" if is_detected else "Missed", delta=lead_days if is_detected else None)
                st.markdown("---")

                # 5. Plotting
                fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Top: Water Level
                ax5a.plot(wl_subset.index, wl_subset, color='gray', label='Water Level')
                ax5a.scatter(flood_peak_date, flood_peak_level, color='black', s=100, label='Flood Peak', zorder=5)
                ax5a.set_title(f'Water Level ({zoom_start} to {zoom_end})')
                ax5a.set_ylabel('Level (m)')
                ax5a.legend(loc='upper left')
                ax5a.grid(True, alpha=0.3)
                
                # Bottom: Variance
                ax5b.plot(var_subset.index, var_subset, color='orange', linewidth=2, label='Variance')
                ax5b.axhline(y=local_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {local_threshold}')
                
                if is_detected:
                    ax5b.scatter(alarm_date, local_threshold, color='red', s=100, zorder=5)
                    # Add Annotation Arrow
                    # Calculate reasonable text offset based on time range
                    offset_days = max(1, (ts_end - ts_start).days // 15)
                    ax5b.annotate(f'Warning: {alarm_date.strftime("%b %d")}\n({lead_days} Days Before)', 
                                 xy=(alarm_date, local_threshold), 
                                 xytext=(alarm_date - pd.Timedelta(days=offset_days), local_threshold + (var_subset.max()*0.1)),
                                 arrowprops=dict(facecolor='red', shrink=0.05), 
                                 color='red', fontweight='bold')
                
                ax5b.set_title(f'Variance Signal (Threshold = {local_threshold})')
                ax5b.set_ylabel('Variance')
                ax5b.set_xlabel('Date')
                ax5b.legend(loc='upper left')
                ax5b.grid(True, alpha=0.3)
                
                st.pyplot(fig5)
                plt.close(fig5)

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
    st.markdown("Expected format: A CSV with a Date column and a Value column.")