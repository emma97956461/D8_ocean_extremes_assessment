import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import dask.array as da

def create_oceanic_regions_mask(lats, lons):
    """
    Create non-overlapping masks for oceanic regions based on specified coordinates
    """
    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons
    
    masks = {
        'equatorial': np.zeros_like(lon_grid, dtype=bool),
        'southern_ocean': np.zeros_like(lon_grid, dtype=bool),
        'eastern_boundary': np.zeros_like(lon_grid, dtype=bool),
        'western_boundary': np.zeros_like(lon_grid, dtype=bool),
        'north_atlantic': np.zeros_like(lon_grid, dtype=bool)
    }
    
    # EQUATORIAL REGION
    masks['equatorial'] = (lat_grid >= -15) & (lat_grid <= 15)
    
    # SOUTHERN OCEAN
    masks['southern_ocean'] = (lat_grid >= -90) & (lat_grid <= -40)
    
    # EASTERN BOUNDARY REGIONS
    eastern_boxes = [
        (lat_grid >= 30) & (lat_grid <= 60) & (lon_grid >= -165) & (lon_grid <= -110),
        (lat_grid >= 15) & (lat_grid <= 40) & (lon_grid >= -40) & (lon_grid <= 0),
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= -105) & (lon_grid <= -68),
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= 0) & (lon_grid <= 25),
        (lat_grid >= -40) & (lat_grid <= -20) & (lon_grid >= 90) & (lon_grid <= 140)
    ]
    
    masks['eastern_boundary'] = np.zeros_like(lon_grid, dtype=bool)
    for box in eastern_boxes:
        masks['eastern_boundary'] = masks['eastern_boundary'] | box
    
    # WESTERN BOUNDARY REGIONS
    western_boxes = [
        (lat_grid >= 25) & (lat_grid <= 60) & (lon_grid >= -81) & (lon_grid <= -40),
        (lat_grid >= 25) & (lat_grid <= 60) & (lon_grid >= 120) & (lon_grid <= 170),
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= -60) & (lon_grid <= -28),
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= 140) & (lon_grid <= 180)
    ]
    
    masks['western_boundary'] = np.zeros_like(lon_grid, dtype=bool)
    for box in western_boxes:
        masks['western_boundary'] = masks['western_boundary'] | box
    
    # NORTH ATLANTIC
    masks['north_atlantic'] = (lat_grid >= 50) & (lat_grid <= 70) & (lon_grid >= -40) & (lon_grid <= 25)
    
    # Ensure no overlaps
    masks['equatorial'] = masks['equatorial'] & ~masks['southern_ocean']
    masks['north_atlantic'] = masks['north_atlantic'] & ~masks['southern_ocean'] & ~masks['equatorial']
    masks['eastern_boundary'] = masks['eastern_boundary'] & ~masks['southern_ocean'] & ~masks['equatorial'] & ~masks['north_atlantic']
    masks['western_boundary'] = masks['western_boundary'] & ~masks['southern_ocean'] & ~masks['equatorial'] & ~masks['north_atlantic'] & ~masks['eastern_boundary']
    
    return masks

def extract_region_data(data_array, lats, lons, masks, time_idx=None):
    """
    Extract data for each region from the input data array
    """
    region_data = {}
    
    for region, mask in masks.items():
        if data_array.ndim == 3:
            if time_idx is not None:
                region_data[region] = data_array[time_idx, mask]
            else:
                region_data[region] = data_array[:, mask].flatten()
        else:
            region_data[region] = data_array[mask]
    
    return region_data

def compute_pdfs(region_data, bins=50, density=True, trim_tails=False, tail_percentile=1):
    """
    Compute PDFs for each region's data with optional tail trimming
    """
    pdfs = {}
    stats = {}
    
    for region, data in region_data.items():
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) > 0:
            if trim_tails and len(clean_data) > 0:
                lower_bound = np.percentile(clean_data, tail_percentile)
                upper_bound = np.percentile(clean_data, 100 - tail_percentile)
                clean_data = clean_data[(clean_data >= lower_bound) & (clean_data <= upper_bound)]
            
            counts, bin_edges = np.histogram(clean_data, bins=bins, density=density)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            pdfs[region] = {
                'counts': counts,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'data': clean_data
            }
            
            stats[region] = {
                'mean': np.mean(clean_data),
                'std': np.std(clean_data),
                'min': np.min(clean_data),
                'max': np.max(clean_data),
                'n_points': len(clean_data),
                'trimmed': trim_tails
            }
        else:
            pdfs[region] = None
            stats[region] = None
    
    return pdfs, stats

def compute_global_pdf_kde(data_array, variable_name, model_name, xlim=(-5, 5), points=1000):
    """
    Compute PDF using KDE for smoother results
    """
    print(f"Computing KDE PDF for {model_name} - {variable_name}")
    
    n_total = int(data_array.count().compute())
    mean_val = float(data_array.mean().compute())
    std_val = float(data_array.std().compute())
    
    print(f"Total data points: {n_total:,}")
    print(f"Using KDE with {points} points")
    
    sample_size = min(100000, n_total)
    step = max(1, n_total // sample_size)
    
    sample = data_array.data.ravel()[::step].compute()
    clean_sample = sample[~np.isnan(sample)]
    
    print(f"Sample size for KDE: {len(clean_sample):,}")
    
    kde = gaussian_kde(clean_sample)
    x_points = np.linspace(xlim[0], xlim[1], points)
    pdf_values = kde(x_points)
    
    stats = {
        'mean': mean_val,
        'std': std_val,
        'n_points': n_total,
        'sample_size': len(clean_sample)
    }
    
    return {
        'x_points': x_points,
        'pdf_values': pdf_values,
        'stats': stats
    }

def compute_percentiles_dask_friendly(data_array, percentiles):
    """
    Compute percentiles in a Dask-friendly way
    """
    flat_data = data_array.data.ravel()
    flat_data_clean = flat_data[~da.isnan(flat_data)]
    
    thresholds = {}
    for p in percentiles:
        thresholds[p] = float(da.percentile(flat_data_clean, p).compute())
    
    return thresholds

def plot_region_pdfs(pdfs, colors=None, xlabel="SSTA (°C)", title="PDFs by Oceanic Region"):
    """
    Plot PDFs for all regions
    """
    if colors is None:
        colors = {
            'eastern_boundary': 'red',
            'western_boundary': 'blue', 
            'equatorial': 'green',
            'southern_ocean': 'purple',
            'north_atlantic': 'orange'
        }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for region, pdf_info in pdfs.items():
        if pdf_info is not None:
            ax1.plot(pdf_info['bin_centers'], pdf_info['counts'], 
                    color=colors[region], label=region.replace('_', ' ').title(), 
                    linewidth=2, alpha=0.8)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Probability Density')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for region, pdf_info in pdfs.items():
        if pdf_info is not None:
            normalized_pdf = pdf_info['counts'] / np.max(pdf_info['counts'])
            ax2.plot(pdf_info['bin_centers'], normalized_pdf,
                    color=colors[region], label=region.replace('_', ' ').title(),
                    linewidth=2, alpha=0.8)
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Normalized Probability Density')
    ax2.set_title('Normalized PDFs (for comparison)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_region_statistics(stats):
    """
    Plot basic statistics for each region
    """
    regions = [r for r in stats.keys() if stats[r] is not None]
    means = [stats[region]['mean'] for region in regions]
    stds = [stats[region]['std'] for region in regions]
    n_points = [stats[region]['n_points'] for region in regions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x_pos = np.arange(len(regions))
    ax1.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7, color='lightblue')
    ax1.bar(x_pos + 0.2, stds, 0.4, label='Std Dev', alpha=0.7, color='lightcoral')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r.replace('_', '\n').title() for r in regions], rotation=45)
    ax1.set_ylabel('Value')
    ax1.set_title('Mean and Standard Deviation by Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x_pos, n_points, alpha=0.7, color='skyblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r.replace('_', '\n').title() for r in regions], rotation=45)
    ax2.set_ylabel('Number of Data Points')
    ax2.set_title('Number of Grid Points in Each Region')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_region_masks(lats, lons, masks, title="Oceanic Regions Mask"):
    """
    Plot the oceanic regions mask
    """
    colors = {
        'eastern_boundary': 'red',
        'western_boundary': 'blue', 
        'equatorial': 'green',
        'southern_ocean': 'purple',
        'north_atlantic': 'orange'
    }
    
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons

    for region, mask in masks.items():
        masked_data = np.where(mask, 1, np.nan)
        ax.pcolormesh(lon_grid, lat_grid, masked_data,
                      cmap=ListedColormap([colors[region]]),
                      alpha=1,
                      transform=ccrs.PlateCarree(),
                      shading='auto')

    ax.coastlines(zorder=8)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=10)
    ax.set_global()

    legend_patches = [mpatches.Patch(color=color, label=r.replace('_',' ').title())
                      for r, color in colors.items()]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    return fig, ax

def analyze_multi_model_extreme_thresholds(models_dict, percentiles=[90, 95, 99, 99.5, 99.9], xlim=(-10, 10)):
    """
    Analyze extreme detection thresholds for multiple models
    """
    print("MULTI-MODEL EXTREME DETECTION THRESHOLD ANALYSIS")
    print("=" * 55)
    print(f"Using xlim: [{xlim[0]}, {xlim[1]}]")
    
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    percentile_colors = ['orange', 'green', 'purple', 'brown', 'black', 'cyan', 'magenta', 'yellow']
    
    thresholds = {}
    for model_name, (ds, var_name) in models_dict.items():
        thresholds[model_name] = compute_percentiles_dask_friendly(ds[var_name], percentiles)
        print(f"{model_name}: {thresholds[model_name]}")
    
    # Individual model PDFs with percentile thresholds
    n_models = len(models_dict)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes1 = np.array([axes1])
    axes1 = axes1.ravel()
    
    print("\nComputing KDE for smooth PDFs...")
    for i, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
        ax = axes1[i]
        color = model_colors[i % len(model_colors)]
        
        pdf_data = compute_global_pdf_kde(ds[var_name], "SST Anomaly", model_name, xlim=xlim)
        
        ax.plot(pdf_data['x_points'], pdf_data['pdf_values'], 
                color=color, label=f'{model_name} PDF', linewidth=2)
        
        for j, (p, threshold) in enumerate(thresholds[model_name].items()):
            p_color = percentile_colors[j % len(percentile_colors)]
            if xlim[0] <= threshold <= xlim[1]:
                ax.axvline(threshold, color=p_color, linestyle='--', alpha=0.7, 
                          label=f'{p}th: {threshold:.3f}°C')
        
        ax.set_xlabel('SST Anomaly (°C)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{model_name}: Extreme Detection Thresholds')
        ax.set_xlim(max(xlim[0], -2), min(xlim[1], 5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    for i in range(len(models_dict), len(axes1)):
        fig1.delaxes(axes1[i])
    
    plt.tight_layout()
    plt.show()
    
    # Threshold comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(percentiles))
    width = 0.8 / n_models
    
    for i, (model_name, model_thresholds) in enumerate(thresholds.items()):
        color = model_colors[i % len(model_colors)]
        model_vals = [model_thresholds[p] for p in percentiles]
        offset = (i - (n_models-1)/2) * width
        ax2.bar(x_pos + offset, model_vals, width, 
               label=model_name, alpha=0.7, color=color)
    
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('Threshold (°C)')
    ax2.set_title('Extreme Detection Thresholds by Model')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{p}th' for p in percentiles])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Comparative analysis table
    print(f"\nComparative Analysis:")
    header = f"{'Percentile':>10} | " + " | ".join([f"{model:>8}" for model in models_dict.keys()])
    print(header)
    print("-" * (len(header) + 10))
    
    for p in percentiles:
        row = f"{p:10.1f} | " + " | ".join([f"{thresholds[model][p]:8.3f}" for model in models_dict.keys()])
        print(row)
    
    return thresholds

def create_model_specific_masks(models_dict):
    """
    Create masks for each model based on their specific grid
    """
    print("Creating model-specific masks...")
    masks_dict = {}
    
    for model_name, (ds, var_name) in models_dict.items():
        print(f"Creating masks for {model_name}...")
        lats = ds.lat.values
        lons = ds.lon.values
        masks_dict[model_name] = create_oceanic_regions_mask(lats, lons)
        print(f"  {model_name} grid: {lats.shape} x {lons.shape}")
    
    return masks_dict

def compute_regional_pdf_kde(data_array, mask, region_name, model_name, xlim=(-5, 5), points=1000, max_sample_size=50000):
    """
    Memory-efficient regional PDF computation using KDE
    """
    print(f"  Computing PDF for {model_name} - {region_name}")
    
    try:
        total_points = int(data_array.count().compute())
        
        if data_array.ndim == 3:
            temporal_mean = data_array.mean(dim='time', skipna=True)
            regional_data = temporal_mean.where(mask).values.flatten()
        else:
            regional_data = data_array.where(mask).values.flatten()
        
        clean_data = regional_data[~np.isnan(regional_data)]
        
        if len(clean_data) == 0:
            print(f"    Warning: No data for {region_name} in {model_name}")
            return None
        
        if len(clean_data) > max_sample_size:
            step = len(clean_data) // max_sample_size
            clean_data = clean_data[::step]
        
        print(f"    Using {len(clean_data):,} points for KDE")
        
        kde = gaussian_kde(clean_data)
        x_points = np.linspace(xlim[0], xlim[1], points)
        pdf_values = kde(x_points)
        
        stats = {
            'mean': np.mean(clean_data),
            'std': np.std(clean_data),
            'min': np.min(clean_data),
            'max': np.max(clean_data),
            'n_points': len(clean_data),
            'total_points': total_points
        }
        
        return {
            'x_points': x_points,
            'pdf_values': pdf_values,
            'stats': stats,
            'region_name': region_name,
            'model_name': model_name
        }
        
    except Exception as e:
        print(f"    Error processing {model_name} - {region_name}: {e}")
        return None

def analyze_regional_extreme_thresholds(models_dict, regions=None, percentiles=[90, 95, 99, 99.5, 99.9], xlim=(-5, 5)):
    """
    Regional equivalent of analyze_multi_model_extreme_thresholds
    Shows PDFs with percentile thresholds for each region
    """
    print("REGIONAL EXTREME DETECTION THRESHOLD ANALYSIS")
    print("=" * 55)
    print(f"Using xlim: [{xlim[0]}, {xlim[1]}]")
    
    masks_dict = create_model_specific_masks(models_dict)
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    percentile_colors = ['orange', 'green', 'purple', 'brown', 'black', 'cyan', 'magenta', 'yellow']
    
    regional_thresholds = {}
    regional_pdfs = {}
    
    print("Computing regional thresholds and PDFs...")
    for region_name in regions:
        print(f"\nProcessing {region_name}...")
        regional_thresholds[region_name] = {}
        regional_pdfs[region_name] = {}
        
        for model_idx, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
            print(f"  {model_name}...")
            
            mask = masks_dict[model_name][region_name]
            data_array = ds[var_name]
            
            try:
                if data_array.ndim == 3:
                    regional_data = data_array.where(mask).mean(dim='time', skipna=True)
                else:
                    regional_data = data_array.where(mask)
                
                values = regional_data.values.flatten()
                clean_data = values[~np.isnan(values)]
                
                if len(clean_data) == 0:
                    continue
                
                thresholds = {}
                for p in percentiles:
                    thresholds[p] = np.percentile(clean_data, p)
                
                regional_thresholds[region_name][model_name] = thresholds
                
                sample_size = min(10000, len(clean_data))
                if len(clean_data) > sample_size:
                    step = len(clean_data) // sample_size
                    sample_data = clean_data[::step]
                else:
                    sample_data = clean_data
                
                kde = gaussian_kde(sample_data)
                x_points = np.linspace(xlim[0], xlim[1], 500)
                pdf_values = kde(x_points)
                
                regional_pdfs[region_name][model_name] = {
                    'x_points': x_points,
                    'pdf_values': pdf_values,
                    'color': model_colors[model_idx % len(model_colors)],
                    'stats': {
                        'mean': np.mean(clean_data),
                        'std': np.std(clean_data),
                        'n_points': len(clean_data)
                    }
                }
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Individual region PDFs with percentile thresholds
    n_regions = len(regions)
    n_cols = min(2, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_regions == 1:
        axes1 = np.array([axes1])
    axes1 = axes1.ravel()
    
    for region_idx, region_name in enumerate(regions):
        ax = axes1[region_idx]
        
        for model_name, pdf_data in regional_pdfs[region_name].items():
            ax.plot(pdf_data['x_points'], pdf_data['pdf_values'], 
                   color=pdf_data['color'], label=model_name, linewidth=2, alpha=0.8)
            
            if region_name in regional_thresholds and model_name in regional_thresholds[region_name]:
                thresholds = regional_thresholds[region_name][model_name]
                for j, (p, threshold) in enumerate(thresholds.items()):
                    if j < len(percentile_colors) and xlim[0] <= threshold <= xlim[1]:
                        ax.axvline(threshold, color=percentile_colors[j], linestyle='--', alpha=0.7,
                                  label=f'{p}th: {threshold:.3f}°C' if model_name == list(models_dict.keys())[0] else "")
        
        ax.set_xlabel('SST Anomaly (°C)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{region_name.replace("_", " ").title()}')
        ax.set_xlim(max(xlim[0], -2), min(xlim[1], 5))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for i in range(len(regions), len(axes1)):
        fig1.delaxes(axes1[i])
    
    plt.tight_layout()
    plt.show()
    
    # Threshold comparison bar chart
    fig2, axes2 = plt.subplots(len(percentiles), 1, figsize=(12, 4 * len(percentiles)))
    if len(percentiles) == 1:
        axes2 = [axes2]
    
    for p_idx, p in enumerate(percentiles):
        ax = axes2[p_idx]
        
        x_pos = np.arange(len(regions))
        width = 0.8 / len(models_dict)
        
        for model_idx, model_name in enumerate(models_dict.keys()):
            color = model_colors[model_idx % len(model_colors)]
            thresholds = []
            
            for region_name in regions:
                if (region_name in regional_thresholds and 
                    model_name in regional_thresholds[region_name]):
                    thresholds.append(regional_thresholds[region_name][model_name][p])
                else:
                    thresholds.append(0)
            
            offset = (model_idx - (len(models_dict)-1)/2) * width
            ax.bar(x_pos + offset, thresholds, width, 
                  label=model_name, alpha=0.7, color=color)
        
        ax.set_xlabel('Region')
        ax.set_ylabel('Threshold (°C)')
        ax.set_title(f'{p}th Percentile Thresholds by Region')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([r.replace('_', '\n').title() for r in regions], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparative analysis table
    print(f"\nREGIONAL THRESHOLD COMPARISON:")
    print("=" * 80)
    
    for p in percentiles:
        print(f"\n{p}th Percentile Thresholds (°C):")
        header = f"{'Region':<20} | " + " | ".join([f"{model:>10}" for model in models_dict.keys()])
        print(header)
        print("-" * len(header))
        
        for region_name in regions:
            row = f"{region_name.replace('_', ' ').title():<20} | "
            for model_name in models_dict.keys():
                if (region_name in regional_thresholds and 
                    model_name in regional_thresholds[region_name]):
                    threshold = regional_thresholds[region_name][model_name][p]
                    row += f"{threshold:>10.3f} | "
                else:
                    row += f"{'N/A':>10} | "
            print(row)
    
    return regional_thresholds, regional_pdfs, masks_dict

def quick_regional_analysis(models_dict, regions=None, xlim=(-5, 5)):
    """
    Quick regional analysis that's guaranteed to be memory efficient
    """
    print("QUICK REGIONAL ANALYSIS")
    print("=" * 30)
    
    masks_dict = create_model_specific_masks(models_dict)
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    regional_thresholds, regional_pdfs, masks_dict = analyze_regional_extreme_thresholds(
        models_dict, regions=regions, xlim=xlim
    )
    
    return regional_pdfs, regional_thresholds, masks_dict