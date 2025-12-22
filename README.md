# CSD-analysis-for-early-Warning using Water Level Time Series Data
analyse variance of time series data to detect system instability and  flags early warning

# Application Requirements: Early Warning System for Critical Slowing Down (CSD)

## 1. Project Overview
A web-based dashboard to detect Early Warning Signals (EWS) in time-series data (specifically water levels) using Critical Slowing Down (CSD) indicators: Variance and Autocorrelation. The tool allows users to upload data, tune parameters, and retrospectively analyze flood events to determine warning lead times.

## 2. Functional Requirements

### 2.1 Data Input
* **File Upload:** Support for CSV file uploads.
* **Column Selection:**
    * **Date Column:** Auto-detect columns containing "date" or "time". Allow manual override.
    * **Value Column:** Auto-detect numeric columns containing "level", "height", or "value". strictly filter for numeric types only.
* **Data Cleaning:** Automatic parsing of dates (day-first supported), conversion of values to numeric, and removal of NaN rows.

### 2.2 User Configuration (Sidebar)
* **Trend Smoothing:** Slider to adjust Sigma (in days) for the Gaussian filter (Range: 7-60 days).
* **Analysis Window:** Slider to adjust the rolling window size (in days) for metric calculation (Range: 3-30 days).
* **Thresholds:**
    * **Variance Threshold:** Input field for the variance alarm cutoff.
    * **Autocorrelation Threshold:** Input field for the lag-1 autocorrelation cutoff (Default: 0.8).

### 2.3 Visualization & Analysis (Tabs)
The application must include five (5) distinct tabs:

* **Tab 1: Analysis Plots**
    * Plot Raw Data vs. Calculated Trend (Gaussian).
    * Plot Rolling Variance signal vs. Threshold.
    * "Stress Test" visualization: Highlight alarm periods (red zones) overlaid on the raw water level data.

* **Tab 2: Data & Statistics**
    * Display total data points and count of distinct alarm events.
    * Table view of data rows where alarms were triggered.

* **Tab 3: Lead Time Analysis (Performance Evaluation)**
    * **Parameters:** Inputs for "Minimum Flood Height" (default to 99th percentile) and "Min Days Between Floods".
    * **Logic:** Detect peaks using `scipy.signal.find_peaks`. Check a 30-day pre-peak window for alarms.
    * **Output:** A styled dataframe listing Flood Date, Peak Level, First Alarm Date, Lead Time (Days), and Status (Detected/Missed/Late).

* **Tab 4: Memory Analysis**
    * Calculate and plot Rolling Lag-1 Autocorrelation.
    * Visualize "High Memory Zones" where autocorrelation exceeds the threshold.

* **Tab 5: Event Deep Dive**
    * **Zoom:** Date pickers to select a specific start and end date for analysis.
    * **Local Tuning:** Input for a local variance threshold specific to this view.
    * **Logic:** Auto-calculate the lead time for the maximum peak within the selected window based on the local threshold.
    * **Visualization:** Detailed dual plots (Water Level and Variance) with annotation arrows showing the exact warning date and days of lead time.

## 3. Technical Requirements

### 3.1 Software Stack
* **Language:** Python 3.x
* **Framework:** Streamlit

### 3.2 Key Libraries
* `pandas`: For data manipulation and time-series handling.
* `matplotlib`: For static plotting (must use 'Agg' backend for web compatibility).
* `scipy`:
    * `ndimage.gaussian_filter1d`: For trend extraction.
    * `signal.find_peaks`: For identifying flood events.

### 3.3 Performance Constraints
* Calculations (rolling windows) must be optimized for responsiveness.
* Graphs must render using `matplotlib.pyplot` but handle memory by closing figures after rendering.
