# 🌊 Oceanic Region Analysis Toolkit

A Python toolkit for analyzing sea surface temperature (SST) anomalies across oceanic regions using probability density functions (PDFs) and extreme value analysis.

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Preparation](#-data-preparation)
- [Usage Guide](#-usage-guide)
- [Regional Definitions](#-regional-definitions)
- [Performance Optimization](#-performance-optimization)
- [Examples](#-examples)
- [Citation](#-citation)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **Regional PDF Analysis**: Compute probability density functions for 13 predefined oceanic regions
- **Global & Seasonal Analysis**: Analyze SST anomalies globally or by season/hemisphere
- **Extreme Value Detection**: Identify and analyze extreme temperature events
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
└── README.md
```

---

## 🚀 Quick Start

```python
import xarray as xr
from DV8_PDFs import *

# Load your SST anomaly data
models = {
    'Model1': sst_anomaly_data1,
    'Model2': sst_anomaly_data2
}

# Run global analysis
global_pdfs = quick_global_analysis(models, bins=100, xlim=(-5, 5))

# Run regional analysis
regional_pdfs, masks = quick_regional_analysis(models, method='ultrafast')

# Seasonal analysis by hemisphere
seasonal_pdfs = quick_global_seasonal_analysis(models, by_hemisphere=True)
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

### Seasonal Analysis

```python
# Global seasonal PDFs
seasonal = quick_global_seasonal_analysis(
    models_dict=models,
    by_hemisphere=True,  # Split Northern/Southern hemispheres
    seasons='DJF'        # Specific season or 'all'
)
```

### Working with Masks

```python
# Load pre-computed masks
masks = create_oceanic_regions_mask(lats, lons, method='shapefile')

# Visualize regions
plot_region_masks(lats, lons, masks)

# Verify no overlaps
verify_mask_exclusivity(masks)
```

---

## 🗺️ Regional Definitions

The toolkit includes **13 oceanic regions** based on the Global Oceans and Seas shapefile:

| Region | Latitude Range | Description |
|--------|----------------|-------------|
| `Southern_Ocean` | ≤ -40°S | Southern ocean areas |
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

## ⚡ Performance Optimization

Choose the right processing mode for your dataset:

| Mode | Best For | Memory Usage | Speed |
|------|----------|--------------|-------|
| `'ultrafast'` | Large datasets (>1GB) | Low | Very Fast |
| `'fast'` | Medium datasets | Medium | Fast |
| Default | Small datasets | Higher | Moderate |

```python
# For large datasets
pdfs, masks = quick_regional_analysis(models, method='ultrafast')

# For maximum speed, process specific regions only
pdfs, masks = quick_regional_analysis(
    models, 
    method='ultrafast',
    regions=['Pacific_Equatorial', 'North_Atlantic_MiddleLats']
)
```

**Additional Tips:**
- Pre-compute masks and save to Zarr format for reuse
- Use Dask for lazy loading of large datasets
- Process models sequentially rather than all at once

---

## 📚 Examples

Detailed tutorials are available in the Jupyter notebooks:

- **`DV8_PDFs.ipynb`**: Complete workflow for PDF analysis with visualizations
- **`DV8_extremes.ipynb`**: Extreme value analysis and threshold detection

### Example Workflow

```python
import xarray as xr
from DV8_PDFs import *

# 1. Load and prepare data
sst = xr.open_dataset('sst_data.nc')['sst_anomaly']
sst = sst.where((sst.lat >= -50) & (sst.lat <= 70), drop=True)
sst = sst.where(sst > -1.7)

# 2. Create model dictionary
models = {'Observations': sst}

# 3. Analyze
global_pdf = quick_global_analysis(models)
regional_pdfs, masks = quick_regional_analysis(models)
seasonal_pdfs = quick_global_seasonal_analysis(models, by_hemisphere=True)

# 4. Visualize results
plot_region_masks(sst.lat, sst.lon, masks)
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

---

## 🔍 Troubleshooting

### Common Issues

**Memory Errors**
```python
# Solution: Use ultrafast mode
regional_pdfs, masks = quick_regional_analysis(models, method='ultrafast')
```

**Missing Regions**
- Verify your data covers the required latitude/longitude ranges (-50°S to 70°N)
- Check that sea ice filtering hasn't removed entire regions

**Shapefile Not Found**
- The toolkit includes pre-computed masks in `region_masks.zarr/`
- If creating custom masks, download the shapefile from [marineregions.org](https://www.marineregions.org/)

**Dimension Errors**
- Ensure data dimensions are `(time, lat, lon)`
- Check coordinate names match: `'lat'`, `'lon'`, `'time'`

### Getting Help

- Check function docstrings: `help(quick_global_analysis)`
- Review example notebooks for complete workflows
- Verify data preprocessing steps are correctly applied

---

## 📄 License

Please refer to the repository license file for usage terms.

## 🤝 Contributing

Questions, bug reports, and feature requests are welcome. Please open an issue or refer to the example notebooks for guidance.