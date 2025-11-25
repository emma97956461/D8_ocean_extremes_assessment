# 🌊 Oceanic Region Analysis Toolkit

> **Data Source Notice**: This toolkit is designed to analyze SST anomalies and SST extremes obtained from **[marEx](https://github.com/wienkers/marEx/tree/main)** (Marine Extremes). The marEx package provides the processed sea surface temperature data, extreme event detection, and marine heatwave identification that serve as input to this regional analysis toolkit.

---

## 🔗 marEx Integration

This toolkit works seamlessly with output from the [marEx package](https://github.com/wienkers/marEx/tree/main), which provides:

- **SST Anomaly Calculation**: Processed sea surface temperature anomalies
- **Extreme Event Detection**: Boolean arrays identifying extreme SST days
- **MHW Identification**: Marine heatwave event detection and characterization
- **Data Preprocessing**: Quality control, filtering, and standardization

### Typical Workflow:
1. **Use marEx** to process raw SST data and detect extremes/MHWs
2. **Use this toolkit** to analyze regional distributions and characteristics
3. **Compare results** across oceanic regions and models

For detailed information on data processing and extreme event detection, refer to the [marEx documentation](https://github.com/wienkers/marEx/tree/main).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Mask Creation System](#mask-creation-system)
- [DV8_PDFs.py - PDF Analysis Functions](#dv8_pdfspy---pdf-analysis-functions)
- [DV8_extremes.py - Extreme Event Analysis](#dv8_extremespy---extreme-event-analysis)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Regional Definitions](#regional-definitions)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)
- [Citation](#citation)
- [Troubleshooting](#troubleshooting)

---

## Overview

This toolkit consists of two main modules with distinct purposes:

- **`DV8_PDFs.py`** - Analyzes **SST anomaly distributions** using probability density functions (PDFs)
- **`DV8_extremes.py`** - Analyzes **extreme events and MHWs** using event detection and characterization

Both modules share the same regional mask system but store masks separately to avoid conflicts.

---

## 🎭 Mask Creation System

### Separate Mask Directories

The toolkit creates **model-specific masks** that are automatically cached for efficiency:

| Analysis Type | Mask Directory | Purpose |
|---------------|----------------|---------|
| **PDF Analysis** | `pdf_model_masks/` | SST anomaly distribution analysis |
| **Extreme Events** | `model_masks/` | Extreme event and MHW analysis |

### How Mask Creation Works

1. **Automatic Grid Detection**: For each model, the toolkit analyzes the grid coordinates (latitudes & longitudes)
2. **Unique Hashing**: Creates a unique hash based on grid characteristics (size, coordinate ranges)
3. **Shapefile Processing**: Uses the Global Oceans and Seas shapefile to create precise regional masks
4. **Mutual Exclusivity**: Ensures no grid point belongs to multiple regions using priority ordering
5. **Zarr Caching**: Saves masks in efficient Zarr format for fast reloading

### Key Mask Functions

```python
# Both files contain these core mask functions:
create_model_specific_masks()      # Main entry point
create_*_shapefile_mask()          # PDF or extremes specific
ensure_mutually_exclusive_masks()  # Prevent region overlaps
```

### Regions Created

13 oceanic regions with priority ordering:
1. `Southern_Ocean` (-50°S to -40°S)
2. `Pacific_Equatorial`, `Atlantic_Equatorial`, `Indian_Equatorial` (-10° to 10°)
3. Subtropical and mid-latitude regions (10°-70°N/S)
4. `Mediterranean_Sea`

---

## 📊 DV8_PDFs.py - PDF Analysis Functions

**Purpose**: Analyze the distribution of SST anomalies using probability density functions.

### Core Functions

#### 1. Global PDF Analysis
```python
quick_global_analysis(models_dict, bins=100, xlim=(-5, 5))
```
- Computes PDFs for entire dataset (all regions combined)
- Returns histogram-based probability densities
- Includes basic statistics (mean, std, data points)

#### 2. Regional PDF Analysis
```python
quick_regional_analysis(models_dict, method='fast', regions=None)
```
- Computes PDFs for each of the 13 oceanic regions separately
- **Methods**: `'fast'` (region-by-region) or `'ultrafast'` (model-by-model)
- Uses PDF-specific masks from `pdf_model_masks/`

#### 3. Seasonal PDF Analysis
```python
quick_global_seasonal_analysis(models_dict, by_hemisphere=False)
quick_regional_seasonal_analysis(models_dict, regions=None)
```
- **Global seasonal**: PDFs for DJF, MAM, JJA, SON (optionally by hemisphere)
- **Regional seasonal**: Seasonal PDFs for each region (excludes equatorial regions)
- Excludes equatorial regions from seasonal analysis since they don't have strong seasons

#### 4. Mask Visualization
```python
quick_visualize_masks(masks_dict, model_name)
plot_combined_regions_mask(masks_dict, model_name)
plot_model_masks(masks_dict, model_name)
```
- Visualize regional masks for quality control
- Combined view (all regions) and individual region plots

### Key Features
- **Classic histogram method** with fixed temperature ranges
- **Dask-optimized** for large datasets
- **Flexible input**: Accepts xarray DataArrays or (dataset, variable) tuples
- **Automatic mask management** with PDF-specific caching

---

## 🌡️ DV8_extremes.py - Extreme Event Analysis

**Purpose**: Detect and analyze extreme SST events and marine heatwaves (MHWs).

### Core Functions

#### 1. Extreme Event Frequency
```python
compute_regional_extremes(models_dict, normalize=True, per_grid_cell=True)
quick_regional_extremes_analysis(models_dict, plot_type='barchart')
```
- Counts extreme event days in each region
- Normalization options: days/year, per grid cell
- Visualization: barcharts, heatmaps, single-model plots

#### 2. MHW Event Detection
```python
compute_mhw_events_for_models(extreme_events_dict, min_duration=5, max_gap=2)
quick_mhw_events_analysis(models_dict, plot_maps=True, plot_regional=True)
selective_mhw_analysis(models_dict, plots_to_show=['regional_summary'])
```
- Detects MHW events from extreme event data
- **Parameters**: Minimum duration, maximum gap for merging events
- **Output**: Event count, duration, start/end times for each grid cell

#### 3. MHW Intensity Analysis
```python
compute_event_intensity_vectorized(mhw_events_ds, ssta_data)
compute_event_intensity_map_blocks(mhw_events_ds, ssta_data)
```
- Computes intensity metrics using original SSTA data:
  - `avg_intensity`: Mean SSTA during events
  - `max_intensity`: Peak SSTA during events  
  - `median_intensity`: Median SSTA during events
- **Optimized versions**: Vectorized and map_blocks for large datasets

#### 4. Regional MHW Statistics
```python
compute_regional_mhw_events(mhw_events_dict, masks_dict)
```
- Aggregates MHW statistics by region:
  - Event count, total event days, average duration
  - Normalized by grid cell count or regional totals

#### 5. Comprehensive Visualization
```python
plot_mhw_event_count_map(mhw_events_dict, model_name)
plot_mhw_avg_duration_map(mhw_events_dict, model_name)
plot_regional_mhw_events_barchart(regional_mhw_data, metric='event_count')
plot_duration_intensity_scatter(mhw_events_ds, intensity_ds)
```
- Spatial maps of event metrics
- Regional comparisons across models
- Duration-intensity relationships
- Multi-model intensity comparisons

### Key Features
- **Structured MHW detection** with duration and gap parameters
- **Intensity computation** using original SSTA values
- **Comprehensive regional statistics** for MHW characteristics
- **Advanced visualization** for spatial and comparative analysis
- **Performance optimized** with Dask parallelization

---

## 📦 Installation

```bash
pip install numpy matplotlib cartopy xarray scipy dask geopandas pathlib
```

**Repository Structure:**
```
├── DV8_PDFs.py              # PDF analysis functions
├── DV8_extremes.py          # Extreme event analysis functions  
├── DV8_PDFs.ipynb           # PDF analysis tutorial
├── DV8_extremes.ipynb       # Extreme events tutorial
├── pdf_model_masks/         # PDF-specific masks (auto-created)
├── model_masks/            # Extreme event masks (auto-created)
└── README.md
```

---

## 🚀 Quick Start

### PDF Analysis (SST Anomaly Distributions)
```python
from DV8_PDFs import *

# Load SST anomaly data
models = {
    'Model1': sst_anomaly_data1,  # xarray DataArray with (time, lat, lon)
    'Model2': sst_anomaly_data2
}

# Quick analyses
global_pdfs = quick_global_analysis(models)
regional_pdfs, masks = quick_regional_analysis(models, method='ultrafast')
seasonal_pdfs = quick_global_seasonal_analysis(models, by_hemisphere=True)
```

### Extreme Event Analysis (MHWs)
```python
from DV8_extremes import *

# Load extreme event data (boolean: True = extreme day)
extreme_events = {
    'Model1': extreme_events_data1,  # shape: (time, lat, lon)
    'Model2': extreme_events_data2
}

# Quick analyses
regional_data, masks = compute_regional_extremes(extreme_events)
mhw_events, regional_mhw, masks = quick_mhw_events_analysis(extreme_events)

# Intensity analysis (requires original SSTA data)
intensity_data = compute_event_intensity_vectorized(mhw_events['Model1'], ssta_data)
```

---

## 🔧 Data Preparation

### SST Data Requirements
- **Dimensions**: `(time, lat, lon)`
- **Coordinates**: `lat`, `lon`, `time`
- **Values**: SST anomalies in °C

### Preprocessing Steps
```python
# 1. Filter latitudes (ice-free oceans)
sst_data = sst_data.where((sst_data.lat >= -50) & (sst_data.lat <= 70), drop=True)

# 2. Remove sea ice contamination
sst_data = sst_data.where(sst_data > -1.7)

# 3. Standardize longitude if needed
if lon_range == (0, 360):
    sst_data = sst_data.assign_coords(lon=(((sst_data.lon + 180) % 360) - 180))
    sst_data = sst_data.sortby('lon')

# 4. For extreme events: create boolean mask
threshold = sst_data.quantile(0.95, dim='time')
extreme_events = sst_data > threshold
```

---

## 🗺️ Regional Definitions

13 mutually exclusive oceanic regions:

| Region | Latitude Range | Analysis Type |
|--------|----------------|---------------|
| `Southern_Ocean` | -50°S to -40°S | Both |
| `Pacific_Equatorial` | -10° to 10° | Both |
| `Atlantic_Equatorial` | -10° to 10° | Both |
| `Indian_Equatorial` | -10° to 10° | Both |
| `North_Pacific_SubTropics` | 10°N to 30°N | Both |
| `North_Pacific_MiddleLats` | 30°N to 70°N | Both |
| `South_Pacific_SubTropics` | -40°S to -10°S | Both |
| `North_Atlantic_SubTropics` | 10°N to 30°N | Both |
| `North_Atlantic_MiddleLats` | 30°N to 70°N | Both |
| `South_Atlantic_SubTropics` | -40°S to -10°S | Both |
| `Indian_NorthSubTropics` | 10°N to 30°N | Both |
| `Indian_SouthSubTropics` | -40°S to -10°S | Both |
| `Mediterranean_Sea` | — | Both |

**Note**: Equatorial regions are excluded from seasonal analysis in regional PDFs.

---

## ⚡ Performance Optimization

### PDF Analysis Modes
```python
# For large datasets (>1GB)
regional_pdfs, masks = quick_regional_analysis(models, method='ultrafast')

# For medium datasets  
regional_pdfs, masks = quick_regional_analysis(models, method='fast')
```

### Extreme Event Optimization
```python
# Progressive intensity computation for large datasets
intensity_data = compute_intensity_progressive(mhw_ds, ssta_data, batch_size=100)

# Reduce event storage if needed
mhw_events = compute_mhw_events_for_models(extreme_events, max_events_per_cell=50)
```

### Mask Caching
- Masks are automatically created and cached per model grid
- Separate directories prevent conflicts between PDF and extremes analysis
- Zarr format enables fast reloading of pre-computed masks

---

## 📚 Examples

### Complete PDF Workflow
```python
from DV8_PDFs import *

# 1. Load and prepare SST anomaly data
sst = xr.open_dataset('sst_data.nc')['sst_anomaly']
sst = sst.where((sst.lat >= -50) & (sst.lat <= 70), drop=True)
sst = sst.where(sst > -1.7)

# 2. Create model dictionary
models = {'Observations': sst}

# 3. Run analyses
global_pdf = quick_global_analysis(models)
regional_pdfs, masks = quick_regional_analysis(models, method='ultrafast')
seasonal_pdfs = quick_global_seasonal_analysis(models, by_hemisphere=True)

# 4. Visualize masks
quick_visualize_masks(masks, 'Observations')
```

### Complete Extreme Events Workflow
```python
from DV8_extremes import *

# 1. Detect extreme events (95th percentile)
threshold = sst.quantile(0.95, dim='time')
extreme_events_data = sst > threshold
models = {'Observations': extreme_events_data}

# 2. Regional extreme frequency
regional_data, masks = compute_regional_extremes(models, normalize=True)

# 3. MHW detection and analysis
mhw_events, regional_mhw, masks = selective_mhw_analysis(
    models,
    plots_to_show=['regional_summary', 'events_map'],
    min_duration=5,
    max_gap=2
)

# 4. Intensity analysis
intensity_data = compute_event_intensity_vectorized(
    mhw_events['Observations'], 
    sst  # Original SSTA data
)
```

---

## 📝 Citation

When using the shapefile-based regional masks, please cite:

```bibtex
@misc{marineregions2021,
  author = {{Flanders Marine Institute}},
  title = {Global Oceans and Seas, version 1},
  year = {2021},
  url = {https://www.marineregions.org/},
  doi = {10.14284/542}
}
```

For MHW analysis methodology:

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

**Mask Creation Failures**
- Verify shapefile exists at expected path
- Check model grid coordinates are properly defined
- Ensure sufficient disk space for mask caching

**Memory Errors**
```python
# PDF analysis: use ultrafast mode
regional_pdfs, masks = quick_regional_analysis(models, method='ultrafast')

# Extremes analysis: use progressive processing
intensity_data = compute_intensity_progressive(mhw_ds, ssta_data, batch_size=50)
```

**Missing Regions**
- Confirm data covers required latitude range (-50°S to 70°N)
- Check sea ice filtering hasn't removed entire regions
- Verify extreme event detection has sufficient data

**Performance Issues**
- Use `ultrafast` method for large datasets
- Process specific regions instead of all regions
- Use selective plotting to avoid unnecessary visualizations

### Getting Help

- Check function docstrings: `help(quick_global_analysis)`
- Review tutorial notebooks: `DV8_PDFs.ipynb` and `DV8_extremes.ipynb`
- Verify data meets preprocessing requirements
- Ensure proper coordinate names and dimensions

---

## 🔄 Module Comparison

| Feature | DV8_PDFs.py | DV8_extremes.py |
|---------|-------------|-----------------|
| **Primary Purpose** | SST anomaly distributions | Extreme events & MHWs |
| **Main Output** | Probability density functions | Event counts, durations, intensities |
| **Mask Directory** | `pdf_model_masks/` | `model_masks/` |
| **Key Functions** | `quick_global_analysis()`, `quick_regional_analysis()` | `compute_mhw_events()`, `compute_event_intensity()` |
| **Seasonal Analysis** | Includes global & regional (excl. equatorial) | Not available |
| **Data Requirement** | SST anomaly values | Boolean extreme event arrays + SSTA for intensity |
