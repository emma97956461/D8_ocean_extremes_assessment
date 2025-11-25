Here's the updated README.md file with comprehensive documentation for the new extreme event analysis functions:

# 🌊 Oceanic Region Analysis Toolkit

A Python toolkit for analyzing sea surface temperature (SST) anomalies across oceanic regions using probability density functions (PDFs), extreme value analysis, and marine heatwave (MHW) detection.

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Preparation](#-data-preparation)
- [Usage Guide](#-usage-guide)
- [Regional Definitions](#-regional-definitions)
- [Extreme Event Analysis](#-extreme-event-analysis)
- [Performance Optimization](#-performance-optimization)
- [Examples](#-examples)
- [Citation](#-citation)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **Regional PDF Analysis**: Compute probability density functions for 13 predefined oceanic regions
- **Global & Seasonal Analysis**: Analyze SST anomalies globally or by season/hemisphere
- **Extreme Value Detection**: Identify and analyze extreme temperature events
- **Marine Heatwave (MHW) Analysis**: Detect, characterize, and analyze MHW events
- **Intensity Analysis**: Compute MHW intensity metrics (average, maximum, median)
- **Model-Specific Grid Support**: Automatic mask generation for different model grids
- **Performance Optimized**: Multiple processing modes for datasets of any size
- **Flexible Input**: Works with xarray DataArrays or (dataset, variable) tuples
- **Visualization Ready**: Built-in plotting functions with consistent styling

---

## 📦 Installation

```bash
pip install numpy matplotlib cartopy xarray scipy dask geopandas pathlib
```

**Repository Structure:**
```
├── DV8_PDFs.py              # Core PDF analysis functions
├── DV8_extremes.py          # Extreme value analysis functions
├── DV8_PDFs.ipynb           # Tutorial notebook for PDF analysis
├── DV8_extremes.ipynb       # Tutorial notebook for extremes
├── region_masks.zarr/       # Pre-computed regional masks
├── model_masks/            # Model-specific mask directory
└── README.md
```

---

## 🚀 Quick Start

### Basic PDF Analysis
```python
import xarray as xr
from DV8_extremes import *

# Load your SST anomaly data
models = {
    'Model1': sst_anomaly_data1,
    'Model2': sst_anomaly_data2
}

# Run global analysis
global_pdfs = quick_global_analysis(models, bins=100, xlim=(-5, 5))

# Run regional analysis
regional_pdfs, masks = quick_regional_analysis(models, method='ultrafast')
```

### Extreme Event Analysis
```python
# Load extreme event data (boolean arrays: True = extreme event)
extreme_events = {
    'Model1': extreme_events_data1,  # shape: (time, lat, lon)
    'Model2': extreme_events_data2
}

# Quick regional extremes analysis
fig, ax, regional_data = quick_regional_extremes_analysis(
    extreme_events,
    plot_type='barchart',
    per_grid_cell=True,
    normalize=True
)
```

### MHW Event Analysis
```python
# Comprehensive MHW analysis
mhw_events, regional_mhw, masks = quick_mhw_events_analysis(
    extreme_events,
    min_duration=5,           # Minimum MHW duration in days
    max_gap=2,               # Maximum gap to merge events
    plot_maps=True,
    plot_regional=True
)

# Selective MHW analysis
mhw_events, regional_mhw, masks = selective_mhw_analysis(
    extreme_events,
    plots_to_show=['regional_summary', 'events_map']
)
```

---

## 🔧 Data Preparation

Your SST data should have dimensions `(time, lat, lon)` and be preprocessed as follows:

### 1. Filter Latitudes (Ice-Free Oceans)

```python
# Retain latitudes between 50°S and 70°N
sst_data = sst_data.where(
    (sst_data.lat >= -50) & (sst_data.lat <= 70), 
    drop=True
)
```

### 2. Remove Sea Ice Contamination

```python
# Remove gridcells with sea ice (SST ≈ -1.75°C in OSTIA, ICON, IFS-FESOM)
sst_data = sst_data.where(sst_data > -1.7)
```

### 3. Standardize Grid (if needed)

```python
# Convert longitude from 0→360 to -180→180
if lon_range == (0, 360):
    sst_data = sst_data.assign_coords(
        lon=(((sst_data.lon + 180) % 360) - 180)
    )
    sst_data = sst_data.sortby('lon')
```

### 4. Extreme Event Detection

For extreme event analysis, you need boolean arrays indicating extreme conditions:

```python
# Example: Detect extremes as values above 95th percentile
threshold = sst_data.quantile(0.95, dim='time')
extreme_events = sst_data > threshold
```

---

## 📖 Usage Guide

### Global PDF Analysis

```python
# Compute global probability density functions
pdfs = quick_global_analysis(
    models_dict=models,
    bins=100,           # Number of histogram bins
    xlim=(-5, 5),       # Temperature anomaly range
    log_scale=False     # Use linear or log scale
)
```

### Regional PDF Analysis

```python
# Analyze specific oceanic regions
regional_pdfs, masks = quick_regional_analysis(
    models_dict=models,
    method='ultrafast',              # 'fast' or 'ultrafast'
    regions=['Pacific_Equatorial',   # Optional: specify regions
             'North_Atlantic_MiddleLats']
)
```

### Extreme Event Analysis

```python
# Compute regional extremes
regional_data, masks = compute_regional_extremes(
    models_dict=extreme_events,
    time_dim='time',
    normalize=True,      # Convert to days/year
    per_grid_cell=True,  # Normalize by grid cell count
    regions=None         # All regions by default
)

# Quick visualization
fig, ax = plot_regional_extremes_barchart(regional_data)
fig, ax = plot_regional_extremes_heatmap(regional_data)
```

### MHW Event Analysis

```python
# Detect MHW events from extreme event data
mhw_events_dict = compute_mhw_events_for_models(
    extreme_events,
    min_duration=5,      # Minimum event duration (days)
    max_gap=2,          # Maximum gap to merge events
    max_events_per_cell=100
)

# Compute regional MHW statistics
regional_mhw_data = compute_regional_mhw_events(
    mhw_events_dict,
    masks_dict,
    normalize=True,
    per_grid_cell=True
)

# Plot MHW statistics
fig, ax = plot_mhw_event_count_map(mhw_events_dict, 'Model1')
fig, ax = plot_mhw_avg_duration_map(mhw_events_dict, 'Model1')
```

### Intensity Analysis

```python
# Compute MHW intensity using SSTA data
intensity_dict = {}
for model_name, mhw_ds in mhw_events_dict.items():
    intensity_dict[model_name] = compute_event_intensity_vectorized(
        mhw_ds,
        ssta_data[model_name]  # Your original SST anomaly data
    )

# Plot intensity maps
fig, ax = plot_avg_intensity_map(intensity_dict['Model1'])
fig, ax = plot_max_intensity_map(intensity_dict['Model1'])
fig, ax = plot_duration_intensity_scatter(mhw_ds, intensity_ds)
```

---

## 🗺️ Regional Definitions

The toolkit includes **13 oceanic regions** based on the Global Oceans and Seas shapefile:

| Region | Latitude Range | Description |
|--------|----------------|-------------|
| `Southern_Ocean` | -50°S to -40°S | Southern ocean areas |
| `Pacific_Equatorial` | -10° to 10° | Equatorial Pacific |
| `Atlantic_Equatorial` | -10° to 10° | Equatorial Atlantic |
| `Indian_Equatorial` | -10° to 10° | Equatorial Indian Ocean |
| `North_Pacific_SubTropics` | 10°N to 30°N | North Pacific subtropics |
| `North_Pacific_MiddleLats` | 30°N to 70°N | North Pacific mid-latitudes |
| `South_Pacific_SubTropics` | -40°S to -10°S | South Pacific subtropics |
| `North_Atlantic_SubTropics` | 10°N to 30°N | North Atlantic subtropics |
| `North_Atlantic_MiddleLats` | 30°N to 70°N | North Atlantic mid-latitudes |
| `South_Atlantic_SubTropics` | -40°S to -10°S | South Atlantic subtropics |
| `Indian_NorthSubTropics` | 10°N to 30°N | Indian Ocean north subtropics |
| `Indian_SouthSubTropics` | -40°S to -10°S | Indian Ocean south subtropics |
| `Mediterranean_Sea` | — | Mediterranean Sea region |

**Note:** Masks are mutually exclusive (no overlapping grid points) and optimized for SST variance analysis.

---

## 🌡️ Extreme Event Analysis

### Key Functions

#### 1. Regional Extreme Frequency
```python
regional_data, masks = compute_regional_extremes(
    extreme_events_dict,
    normalize=True,      # Output in days/year
    per_grid_cell=True   # Average per grid cell
)
```

#### 2. MHW Event Detection
```python
mhw_events = compute_mhw_events_for_models(
    extreme_events_dict,
    min_duration=5,      # Minimum MHW duration
    max_gap=2,          # Merge events with gaps ≤ 2 days
    max_events_per_cell=100
)
```

#### 3. MHW Intensity Analysis
```python
intensity_data = compute_event_intensity_vectorized(
    mhw_events_ds,
    ssta_data,          # Original SST anomaly data
    time_dim='time'
)
```

#### 4. Quick Analysis Wrappers
```python
# All-in-one analysis
mhw_events, regional_mhw, masks = quick_mhw_events_analysis(extreme_events)

# Selective plotting
mhw_events, regional_mhw, masks = selective_mhw_analysis(
    extreme_events,
    plots_to_show=['regional_summary', 'events_map', 'duration_map']
)
```

### Output Metrics

- **Event Count**: Number of MHW events per region/grid cell
- **Event Duration**: Length of MHW events in days
- **Total Event Days**: Cumulative MHW days
- **Average Duration**: Mean duration of MHW events
- **Intensity Metrics**: Average, maximum, and median SSTA during events

### Visualization Options

- **Spatial Maps**: Event counts, average duration, intensity
- **Regional Barcharts**: Compare metrics across regions and models
- **Heatmaps**: Matrix visualization of regional metrics
- **Scatter Plots**: Duration vs intensity relationships
- **Grid Plots**: Multi-panel comparisons across metrics

---

## ⚡ Performance Optimization

### Processing Modes

| Mode | Best For | Memory Usage | Speed |
|------|----------|--------------|-------|
| `'ultrafast'` | Large datasets (>1GB) | Low | Very Fast |
| `'fast'` | Medium datasets | Medium | Fast |
| Default | Small datasets | Higher | Moderate |

### Model-Specific Masks

```python
# Masks are automatically created and cached for each model grid
masks_dict = create_model_specific_masks(
    models_dict,
    shapefile_path='/path/to/shapefile.shp',
    mask_save_dir='/path/to/mask/directory'
)

# Masks are saved as Zarr files for fast reloading
```

### Memory Management

```python
# For large datasets, use progressive processing
intensity_data = compute_intensity_progressive(
    mhw_events_ds,
    ssta_data,
    batch_size=100  # Process 100 latitudes at a time
)

# Use Dask for parallel computation
results = xr.apply_ufunc(
    compute_function,
    data,
    dask='parallelized',
    output_dtypes=[float]
)
```

**Additional Tips:**
- Pre-compute masks and save to Zarr format for reuse
- Use Dask for lazy loading of large datasets
- Process models sequentially rather than all at once
- Use `selective_mhw_analysis` to plot only specific visualizations

---

## 📚 Examples

### Basic Workflow
```python
import xarray as xr
from DV8_extremes import *

# 1. Load and prepare data
sst = xr.open_dataset('sst_data.nc')['sst_anomaly']
sst = sst.where((sst.lat >= -50) & (sst.lat <= 70), drop=True)
sst = sst.where(sst > -1.7)

# 2. Detect extreme events (95th percentile)
threshold = sst.quantile(0.95, dim='time')
extreme_events = sst > threshold

models = {'Observations': extreme_events}

# 3. Analyze extremes
regional_data, masks = compute_regional_extremes(models)

# 4. MHW analysis
mhw_events, regional_mhw, masks = quick_mhw_events_analysis(models)

# 5. Intensity analysis
intensity_data = compute_event_intensity_vectorized(
    mhw_events['Observations'],
    sst
)
```

### Advanced Workflow
```python
# Multi-model comparison with custom regions
extreme_events_dict = {
    'Model_A': extremes_a,
    'Model_B': extremes_b,
    'Observations': extremes_obs
}

# Custom region selection
regions_of_interest = [
    'Pacific_Equatorial',
    'North_Atlantic_MiddleLats', 
    'Southern_Ocean'
]

# Comprehensive analysis
mhw_events, regional_mhw, masks = selective_mhw_analysis(
    extreme_events_dict,
    plots_to_show=['regional_summary', 'events_map', 'regional_grid'],
    regions=regions_of_interest,
    min_duration=5,
    max_gap=2
)

# Intensity comparison
intensity_dict = {}
for model_name, mhw_ds in mhw_events.items():
    intensity_dict[model_name] = compute_event_intensity_vectorized(
        mhw_ds,
        ssta_data[model_name]
    )

# Plot multi-model intensity maps
fig, axes = plot_multi_model_intensity_maps(
    intensity_dict,
    plot_type='avg_intensity'
)
```

---

## 📝 Citation

When using this toolkit with shapefile-based regional masks, please cite:

```bibtex
@misc{marineregions2021,
  author = {{Flanders Marine Institute}},
  title = {Global Oceans and Seas, version 1},
  year = {2021},
  url = {https://www.marineregions.org/},
  doi = {10.14284/542}
}
```

For MHW analysis methodology, consider citing:

```bibtex
@article{hobday2016hierarchy,
  title={A hierarchical approach to defining marine heatwaves},
  author={Hobday, Alistair J and Alexander, Lisa V and Perkins, Sarah E and Smale, Dan A and Straub, Sandra C and Oliver, Eric CJ and Benthuysen, Jessica A and Burrows, Michael T and Donat, Markus G and Feng, Ming and others},
  journal={Progress in Oceanography},
  volume={141},
  pages={227--238},
  year={2016},
  publisher={Elsevier}
}
```

---

## 🔍 Troubleshooting

### Common Issues

**Memory Errors with Large Datasets**
```python
# Solution: Use progressive processing and ultrafast mode
intensity_data = compute_intensity_progressive(mhw_ds, ssta_data, batch_size=50)
regional_data, masks = compute_regional_extremes(models, method='ultrafast')
```

**MHW Detection Too Slow**
```python
# Reduce maximum events per cell
mhw_events = compute_mhw_events_for_models(
    extreme_events,
    max_events_per_cell=50  # Default: 100
)
```

**Missing Regions in Output**
- Verify your data covers the required latitude ranges (-50°S to 70°N)
- Check that sea ice filtering hasn't removed entire regions
- Ensure extreme event detection hasn't created empty regions

**Shapefile Not Found**
- The toolkit includes pre-computed masks in `region_masks.zarr/`
- Model-specific masks are automatically generated and cached
- If creating custom masks, download the shapefile from [marineregions.org](https://www.marineregions.org/)

**Dimension Errors**
- Ensure data dimensions are `(time, lat, lon)`
- Check coordinate names match: `'lat'`, `'lon'`, `'time'`
- Verify extreme event data is boolean (`True`/`False`)

### Performance Tips

1. **Use Zarr Storage**: Masks are automatically saved in efficient Zarr format
2. **Batch Processing**: Large datasets can be processed in batches
3. **Selective Plotting**: Use `selective_mhw_analysis` to avoid unnecessary plots
4. **Model-Specific Masks**: Masks are cached per model grid for fast reloading

### Getting Help

- Check function docstrings: `help(compute_regional_extremes)`
- Review example notebooks for complete workflows
- Verify data preprocessing steps are correctly applied
- Use the `quick_` wrapper functions for standard analyses

---

