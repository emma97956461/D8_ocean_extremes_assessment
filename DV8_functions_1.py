import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import dask.array as da

def create_old_oceanic_regions_mask(lats, lons):
    """
    OLD VERSION - Create non-overlapping masks for oceanic regions based on specified coordinates
    Parameters:
    - lats: latitude array (1D or 2D)
    - lons: longitude array (1D or 2D)
    
    Returns:
    - masks: dictionary of boolean masks for each region
    """
    # Create coordinate grids if not already 2D
    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons
    
    # Initialize all masks as False
    masks = {
        'equatorial': np.zeros_like(lon_grid, dtype=bool),
        'southern_ocean': np.zeros_like(lon_grid, dtype=bool),
        'eastern_boundary': np.zeros_like(lon_grid, dtype=bool),
        'western_boundary': np.zeros_like(lon_grid, dtype=bool),
        'north_atlantic': np.zeros_like(lon_grid, dtype=bool)
    }
    
    # 1. EQUATORIAL REGION (GREEN)
    # lat min=-15, lat max=15 (all longitudes)
    masks['equatorial'] = (lat_grid >= -15) & (lat_grid <= 15)
    
    # 2. SOUTHERN OCEAN (PURPLE)
    # lat min=-90, lat max=-40 (all longitudes)
    masks['southern_ocean'] = (lat_grid >= -90) & (lat_grid <= -40)
    
    # 3. EASTERN BOUNDARY REGIONS (RED) - 5 boxes
    eastern_boxes = [
        # Box 1: lat min=30, lat max=60, lon min=165W, lon max=110W
        (lat_grid >= 30) & (lat_grid <= 60) & (lon_grid >= -165) & (lon_grid <= -110),
        # Box 2: lat min=15, lat max=40, lon min=40W, lon max=0
        (lat_grid >= 15) & (lat_grid <= 40) & (lon_grid >= -40) & (lon_grid <= 0),
        # Box 3: lat min=-40, lat max=-15, lon min=105W, lon max=68W
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= -105) & (lon_grid <= -68),
        # Box 4: lat min=-40, lat max=-15, lon min=0, lon max=25E
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= 0) & (lon_grid <= 25),
        # Box 5: lat min=-40, lat max=-20, lon min=90E, lon max=140E
        (lat_grid >= -40) & (lat_grid <= -20) & (lon_grid >= 90) & (lon_grid <= 140)
    ]
    
    # Combine all eastern boxes
    masks['eastern_boundary'] = np.zeros_like(lon_grid, dtype=bool)
    for box in eastern_boxes:
        masks['eastern_boundary'] = masks['eastern_boundary'] | box
    
    # 4. WESTERN BOUNDARY REGIONS (BLUE) - 4 boxes
    western_boxes = [
        # Box 1: lat min=25, lat max=60, lon min=81W, lon max=40W
        (lat_grid >= 25) & (lat_grid <= 60) & (lon_grid >= -81) & (lon_grid <= -40),
        # Box 2: lat min=25, lat max=60, lon min=120E, lon max=170E
        (lat_grid >= 25) & (lat_grid <= 60) & (lon_grid >= 120) & (lon_grid <= 170),
        # Box 3: lat min=-40, lat max=-15, lon min=60W, lon max=28W
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= -60) & (lon_grid <= -28),
        # Box 4: lat min=-40, lat max=-15, lon min=140E, lon max=180
        (lat_grid >= -40) & (lat_grid <= -15) & (lon_grid >= 140) & (lon_grid <= 180)
    ]
    
    # Combine all western boxes
    masks['western_boundary'] = np.zeros_like(lon_grid, dtype=bool)
    for box in western_boxes:
        masks['western_boundary'] = masks['western_boundary'] | box
    
    # 5. NORTH ATLANTIC (ORANGE)
    # lat min=50, lat max=70, lon min=40W, lon max=25E
    masks['north_atlantic'] = (lat_grid >= 50) & (lat_grid <= 70) & (lon_grid >= -40) & (lon_grid <= 25)
    
    # Ensure no overlaps by using a priority system
    # Priority order: southern_ocean > equatorial > north_atlantic > eastern_boundary > western_boundary
    masks['equatorial'] = masks['equatorial'] & ~masks['southern_ocean']
    masks['north_atlantic'] = masks['north_atlantic'] & ~masks['southern_ocean'] & ~masks['equatorial']
    masks['eastern_boundary'] = masks['eastern_boundary'] & ~masks['southern_ocean'] & ~masks['equatorial'] & ~masks['north_atlantic']
    masks['western_boundary'] = masks['western_boundary'] & ~masks['southern_ocean'] & ~masks['equatorial'] & ~masks['north_atlantic'] & ~masks['eastern_boundary']
    
    return masks


def create_oceanic_regions_mask(lats, lons):
    """
    NEW VERSION - Create non-overlapping masks for 14 oceanic regions with proper land boundaries
    Parameters:
    - lats: latitude array (1D or 2D)
    - lons: longitude array (1D or 2D)
    
    Returns:
    - masks: dictionary of boolean masks for each of the 14 regions
    """
    # Create coordinate grids if not already 2D
    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons
    
    # Initialize all masks as False
    masks = {
        # Southern Ocean (1 region)
        'southern_ocean': np.zeros_like(lon_grid, dtype=bool),
        
        # Mid-latitudes South (3 regions)
        'mid_lat_south_pacific': np.zeros_like(lon_grid, dtype=bool),
        'mid_lat_south_atlantic': np.zeros_like(lon_grid, dtype=bool),
        'mid_lat_south_indian': np.zeros_like(lon_grid, dtype=bool),
        
        # Equatorial (3 regions)
        'equatorial_pacific': np.zeros_like(lon_grid, dtype=bool),
        'equatorial_atlantic': np.zeros_like(lon_grid, dtype=bool),
        'equatorial_indian': np.zeros_like(lon_grid, dtype=bool),
        
        # Mid-latitudes North (3 regions)
        'mid_lat_north_pacific': np.zeros_like(lon_grid, dtype=bool),
        'mid_lat_north_atlantic': np.zeros_like(lon_grid, dtype=bool),
        'mid_lat_north_indian': np.zeros_like(lon_grid, dtype=bool),
        
        # Northern Latitudes (2 regions - no Indian)
        'northern_lat_pacific': np.zeros_like(lon_grid, dtype=bool),
        'northern_lat_atlantic': np.zeros_like(lon_grid, dtype=bool),
        
        # Additional regions (2 regions)
        'mediterranean_sea': np.zeros_like(lon_grid, dtype=bool),
        'baltic_sea': np.zeros_like(lon_grid, dtype=bool)
    }
    
    # Define ocean basin boundaries with land-based borders
    
    # 1. Mediterranean Sea - carefully bounded to avoid Atlantic spillage
    # Strait of Gibraltar at ~-5.5°W, Mediterranean extends to ~40°E
    mediterranean_lat = (lat_grid >= 30) & (lat_grid <= 45)
    mediterranean_lon = (lon_grid >= -5.5) & (lon_grid <= 40)
    
    # Refine Mediterranean to exclude Atlantic approaches
    # Exclude areas west of Gibraltar that might be considered Atlantic
    masks['mediterranean_sea'] = mediterranean_lat & mediterranean_lon
    
    # 2. Baltic Sea
    baltic_lat = (lat_grid >= 54) & (lat_grid <= 66)
    baltic_lon = (lon_grid >= 10) & (lon_grid <= 30)
    masks['baltic_sea'] = baltic_lat & baltic_lon
    
    # Pacific Ocean boundaries:
    # Western border: Longitude of Tasmania (~147°E) for Mid Lat South Pacific
    # Eastern border: -116° for Northern Pacific, -80° for others
    pacific_mask_general = (
        ((lon_grid >= 147) & (lon_grid <= 180)) |  # Western Pacific
        ((lon_grid >= -180) & (lon_grid <= -80))   # Eastern Pacific (general)
    )
    
    # Northern Pacific has different eastern border
    pacific_mask_northern = (
        ((lon_grid >= 147) & (lon_grid <= 180)) |  # Western Pacific  
        ((lon_grid >= -180) & (lon_grid <= -116))  # Eastern Pacific (northern)
    )
    
    # Atlantic Ocean boundaries with specific longitudes:
    # Equatorial Atlantic: -74°W to 20°E
    atlantic_equatorial = (lon_grid >= -74) & (lon_grid <= 20)
    
    # Mid Lat South Atlantic: -69°W to 20°E  
    atlantic_mid_south = (lon_grid >= -69) & (lon_grid <= 20)
    
    # Mid Lat North Atlantic: complex shape due to Gulf of Mexico
    # Use wider bounds but we'll refine with latitude-dependent boundaries
    atlantic_mid_north_general = (lon_grid >= -98) & (lon_grid <= 20)
    
    # Northern Atlantic: similar to mid north but exclude Mediterranean/Baltic
    atlantic_northern = (lon_grid >= -60) & (lon_grid <= 20)
    
    # Indian Ocean boundaries:
    indian_mask = (lon_grid >= 20) & (lon_grid <= 147)
    
    # 1. SOUTHERN OCEAN (-90 to -40, all longitudes)
    masks['southern_ocean'] = (lat_grid >= -90) & (lat_grid <= -40)
    
    # 2. MID-LATITUDES SOUTH (-40 to -10)
    lat_south = (lat_grid > -40) & (lat_grid <= -10)
    masks['mid_lat_south_pacific'] = lat_south & pacific_mask_general
    masks['mid_lat_south_atlantic'] = lat_south & atlantic_mid_south
    masks['mid_lat_south_indian'] = lat_south & indian_mask
    
    # 3. EQUATORIAL (-10 to +10)
    lat_equatorial = (lat_grid > -10) & (lat_grid <= 10)
    masks['equatorial_pacific'] = lat_equatorial & pacific_mask_general
    masks['equatorial_atlantic'] = lat_equatorial & atlantic_equatorial
    masks['equatorial_indian'] = lat_equatorial & indian_mask
    
    # 4. MID-LATITUDES NORTH (+10 to +30)
    lat_mid_north = (lat_grid > 10) & (lat_grid <= 30)
    
    # For Mid Lat North Atlantic, use complex boundaries to handle Gulf of Mexico
    # Start with general bounds, then refine
    mid_north_atlantic_general = lat_mid_north & atlantic_mid_north_general
    
    # Refine Gulf of Mexico area - exclude Pacific side
    # In the Gulf region (approx 18N-30N, -98W to -81W), ensure we're on Atlantic side
    gulf_mask = (
        (lat_grid >= 18) & (lat_grid <= 30) & 
        (lon_grid >= -98) & (lon_grid <= -81)
    )
    
    # Only include Gulf area if it's clearly Atlantic (east of -90W in northern part)
    gulf_atlantic = gulf_mask & (lon_grid >= -90)
    
    # Combine general Atlantic with refined Gulf area
    masks['mid_lat_north_atlantic'] = (
        (mid_north_atlantic_general & ~gulf_mask) |  # Non-Gulf Atlantic
        gulf_atlantic  # Gulf of Mexico (Atlantic side)
    )
    
    masks['mid_lat_north_pacific'] = lat_mid_north & pacific_mask_general
    masks['mid_lat_north_indian'] = lat_mid_north & indian_mask
    
    # 5. NORTHERN LATITUDES (+30 to +70) - NO INDIAN OCEAN
    lat_north = (lat_grid > 30) & (lat_grid <= 70)
    masks['northern_lat_pacific'] = lat_north & pacific_mask_northern
    masks['northern_lat_atlantic'] = lat_north & atlantic_northern
    
    # Exclude Mediterranean and Baltic from Atlantic regions
    atlantic_regions = ['mid_lat_south_atlantic', 'equatorial_atlantic', 
                       'mid_lat_north_atlantic', 'northern_lat_atlantic']
    
    for region in atlantic_regions:
        masks[region] = masks[region] & ~masks['mediterranean_sea'] & ~masks['baltic_sea']
    
    # Ensure no overlaps
    used_mask = np.zeros_like(lon_grid, dtype=bool)
    
    # Process in priority order
    priority_order = [
        'southern_ocean',
        'mid_lat_south_pacific', 'mid_lat_south_atlantic', 'mid_lat_south_indian',
        'equatorial_pacific', 'equatorial_atlantic', 'equatorial_indian',
        'mid_lat_north_pacific', 'mid_lat_north_atlantic', 'mid_lat_north_indian',
        'northern_lat_pacific', 'northern_lat_atlantic',
        'mediterranean_sea', 'baltic_sea'
    ]
    
    for region in priority_order:
        masks[region] = masks[region] & ~used_mask
        used_mask = used_mask | masks[region]
    
    return masks

# Update the color mapping for 14 regions
def get_region_colors_14():
    """
    Get color mapping for the 14 regions
    """
    return {
        # Southern Ocean
        'southern_ocean': 'purple',
        
        # Mid-latitudes South
        'mid_lat_south_pacific': 'lightblue',
        'mid_lat_south_atlantic': 'blue',
        'mid_lat_south_indian': 'darkblue',
        
        # Equatorial
        'equatorial_pacific': 'lightgreen',
        'equatorial_atlantic': 'green',
        'equatorial_indian': 'darkgreen',
        
        # Mid-latitudes North
        'mid_lat_north_pacific': 'yellow',
        'mid_lat_north_atlantic': 'orange',
        'mid_lat_north_indian': 'red',
        
        # Northern Latitudes (no Indian)
        'northern_lat_pacific': 'pink',
        'northern_lat_atlantic': 'magenta',
        
        # Additional regions
        'mediterranean_sea': 'cyan',
        'baltic_sea': 'teal'
    }

# Update the plot function to handle 14 regions
def plot_region_masks(lats, lons, masks, title="Oceanic Regions Mask"):
    """
    Plot region masks - automatically handles different region counts
    """
    # Create coordinate grids if not already 2D
    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons

    # Auto-detect which mask type we're using and set colors accordingly
    if 'mediterranean_sea' in masks and 'baltic_sea' in masks:
        colors = get_region_colors_14()
        print("Using 14-region mask colors")
    elif 'equatorial_pacific' in masks:
        if 'northern_lat_indian' in masks:
            colors = {
                # Southern Ocean
                'southern_ocean': 'purple',
                
                # Mid-latitudes South
                'mid_lat_south_pacific': 'lightblue',
                'mid_lat_south_atlantic': 'blue', 
                'mid_lat_south_indian': 'darkblue',
                
                # Equatorial
                'equatorial_pacific': 'lightgreen',
                'equatorial_atlantic': 'green',
                'equatorial_indian': 'darkgreen',
                
                # Mid-latitudes North  
                'mid_lat_north_pacific': 'yellow',
                'mid_lat_north_atlantic': 'orange',
                'mid_lat_north_indian': 'red',
                
                # Northern Latitudes
                'northern_lat_pacific': 'pink',
                'northern_lat_atlantic': 'magenta',
                'northern_lat_indian': 'brown'
            }
            print("Using 13-region mask colors")
        else:
            colors = {
                # Southern Ocean
                'southern_ocean': 'purple',
                
                # Mid-latitudes South
                'mid_lat_south_pacific': 'lightblue',
                'mid_lat_south_atlantic': 'blue', 
                'mid_lat_south_indian': 'darkblue',
                
                # Equatorial
                'equatorial_pacific': 'lightgreen',
                'equatorial_atlantic': 'green',
                'equatorial_indian': 'darkgreen',
                
                # Mid-latitudes North  
                'mid_lat_north_pacific': 'yellow',
                'mid_lat_north_atlantic': 'orange',
                'mid_lat_north_indian': 'red',
                
                # Northern Latitudes (no Indian)
                'northern_lat_pacific': 'pink',
                'northern_lat_atlantic': 'magenta'
            }
            print("Using 12-region mask colors")
    else:
        # Old 5-region mask
        colors = {
            'eastern_boundary': 'red',
            'western_boundary': 'blue', 
            'equatorial': 'green',
            'southern_ocean': 'purple',
            'north_atlantic': 'orange'
        }
        print("Using old 5-region mask colors")
    
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

    for region, mask in masks.items():
        if region in colors:  # Only plot regions that have colors defined
            masked_data = np.where(mask, 1, np.nan)
            ax.pcolormesh(lon_grid, lat_grid, masked_data,
                          cmap=ListedColormap([colors[region]]),
                          alpha=1,
                          transform=ccrs.PlateCarree(),
                          shading='auto')

    ax.coastlines(zorder=8)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=10)
    ax.set_global()

    # Legend
    legend_patches = [mpatches.Patch(color=color, label=r.replace('_',' ').title())
                      for r, color in colors.items()]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
              ncol=4)
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    return fig, ax





def extract_region_data(data_array, lats, lons, masks, time_idx=None):
    """
    Extract data for each region from the input data array
    
    Parameters:
    - data_array: 2D or 3D array of data (time, lat, lon) or (lat, lon)
    - lats: latitude coordinates
    - lons: longitude coordinates  
    - masks: dictionary of region masks from create_oceanic_regions_mask
    - time_idx: specific time index to use (if None, uses all times)
    
    Returns:
    - region_data: dictionary with region names as keys and flattened data arrays as values
    """
    region_data = {}
    
    for region, mask in masks.items():
        if data_array.ndim == 3:  # Multiple time steps (time, lat, lon)
            if time_idx is not None:
                # Extract data for specific time
                region_data[region] = data_array[time_idx, mask]
            else:
                # Extract all data across time and space
                # Reshape to (time * spatial_points)
                region_data[region] = data_array[:, mask].flatten()
        else:  # Single time step (lat, lon)
            region_data[region] = data_array[mask]
    
    return region_data

def compute_pdfs(region_data, bins=50, density=True):
    """
    Compute PDFs for each region's data
    
    Parameters:
    - region_data: dictionary with region names as keys and data arrays as values
    - bins: number of bins for histogram
    - density: if True, normalize to form probability density
    
    Returns:
    - pdfs: dictionary with PDF information for each region
    - stats: dictionary with statistical summary for each region
    """
    pdfs = {}
    stats = {}
    
    for region, data in region_data.items():
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) > 0:
            # Compute histogram
            counts, bin_edges = np.histogram(clean_data, bins=bins, density=density)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            pdfs[region] = {
                'counts': counts,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'data': clean_data
            }
            
            # Basic statistics
            stats[region] = {
                'mean': np.mean(clean_data),
                'std': np.std(clean_data),
                'min': np.min(clean_data),
                'max': np.max(clean_data),
                'n_points': len(clean_data)
            }
        else:
            pdfs[region] = None
            stats[region] = None
    
    return pdfs, stats

def compute_pdfs_for_variable(data_array, lats, lons, masks, variable_name, model_name):
    """
    Compute PDFs for a specific variable and model across all regions
    
    Parameters:
    - data_array: 2D or 3D array of data
    - lats: latitude coordinates
    - lons: longitude coordinates
    - masks: dictionary of region masks
    - variable_name: name of the variable for labeling
    - model_name: name of the model for labeling
    
    Returns:
    - pdfs: dictionary with PDF information for each region
    - stats: dictionary with statistical summary for each region
    """
    print(f"Computing PDFs for {model_name} - {variable_name}")
    print("="*50)
    
    # Extract region data
    region_data = extract_region_data(data_array, lats, lons, masks, time_idx=None)
    
    # Compute PDFs
    pdfs, stats = compute_pdfs(region_data, bins=50, density=True)
    
    # Plot PDFs
    fig, axes = plot_region_pdfs(pdfs, 
                               xlabel=f"{variable_name} ({model_name})", 
                               title=f"PDFs of {variable_name} by Region ({model_name})")
    plt.show()
    
    # Plot statistics
    fig, axes = plot_region_statistics(stats)
    plt.show()
    
    # Print summary
    print(f"\n{model_name} - {variable_name} Region Statistics:")
    print("="*60)
    for region, stat in stats.items():
        if stat:
            print(f"{region.replace('_', ' ').title():20} | "
                  f"Mean: {stat['mean']:7.3f} | "
                  f"Std: {stat['std']:6.3f} | "
                  f"Points: {stat['n_points']:8,d}")
    
    return pdfs, stats

def compare_models_pdfs(ostia_pdfs, icon_pdfs, variable_name, regions=None):
    """
    Compare PDFs between OSTIA and ICON for each region
    
    Parameters:
    - ostia_pdfs: PDFs from OSTIA model
    - icon_pdfs: PDFs from ICON model  
    - variable_name: name of variable for labeling
    - regions: list of regions to compare (default: all regions)
    """
    if regions is None:
        regions = ['western_boundary', 'eastern_boundary', 'equatorial', 'north_atlantic', 'southern_ocean']
    
    colors = {'OSTIA': 'blue', 'ICON': 'red'}
    
    # Create subplots
    n_regions = len(regions)
    n_cols = min(3, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_regions == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    for i, region in enumerate(regions):
        ax = axes[i]
        
        # Plot OSTIA PDF
        if ostia_pdfs.get(region) is not None:
            ostia_info = ostia_pdfs[region]
            ax.plot(ostia_info['bin_centers'], ostia_info['counts'], 
                   color=colors['OSTIA'], label='OSTIA', linewidth=2)
        
        # Plot ICON PDF  
        if icon_pdfs.get(region) is not None:
            icon_info = icon_pdfs[region]
            ax.plot(icon_info['bin_centers'], icon_info['counts'],
                   color=colors['ICON'], label='ICON', linewidth=2)
        
        ax.set_title(region.replace('_', ' ').title())
        ax.set_xlabel(variable_name)
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(regions), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Comparison: {variable_name} PDFs by Region (OSTIA vs ICON)', fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_models_stats(ostia_stats, icon_stats, variable_name):
    """
    Compare statistics between OSTIA and ICON
    
    Parameters:
    - ostia_stats: statistics from OSTIA model
    - icon_stats: statistics from ICON model
    - variable_name: name of variable for labeling
    """
    regions = [r for r in ostia_stats.keys() if ostia_stats[r] and icon_stats.get(r)]
    
    # Prepare data for plotting
    ostia_means = [ostia_stats[r]['mean'] for r in regions]
    icon_means = [icon_stats[r]['mean'] for r in regions]
    ostia_stds = [ostia_stats[r]['std'] for r in regions]
    icon_stds = [icon_stats[r]['std'] for r in regions]
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x_pos = np.arange(len(regions))
    width = 0.35
    
    # Means comparison
    ax1.bar(x_pos - width/2, ostia_means, width, label='OSTIA', alpha=0.7, color='blue')
    ax1.bar(x_pos + width/2, icon_means, width, label='ICON', alpha=0.7, color='red')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r.replace('_', '\n').title() for r in regions], rotation=45)
    ax1.set_ylabel('Mean Value')
    ax1.set_title(f'Mean {variable_name} by Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Std comparison
    ax2.bar(x_pos - width/2, ostia_stds, width, label='OSTIA', alpha=0.7, color='blue')
    ax2.bar(x_pos + width/2, icon_stds, width, label='ICON', alpha=0.7, color='red')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r.replace('_', '\n').title() for r in regions], rotation=45)
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title(f'Std Dev of {variable_name} by Region')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model Comparison: {variable_name}', fontsize=16)
    plt.tight_layout()
    plt.show()


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
    
    # Plot 1: Individual PDFs
    for region, pdf_info in pdfs.items():
        if pdf_info is not None:
            ax1.plot(pdf_info['bin_centers'], pdf_info['counts'], 
                    color=colors[region], label=region.replace('_', ' ').title(), 
                    linewidth=2)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Probability Density')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overlaid PDFs (normalized for comparison)
    for region, pdf_info in pdfs.items():
        if pdf_info is not None:
            # Normalize for better comparison
            normalized_pdf = pdf_info['counts'] / np.max(pdf_info['counts'])
            ax2.plot(pdf_info['bin_centers'], normalized_pdf,
                    color=colors[region], label=region.replace('_', ' ').title(),
                    linewidth=2)
    
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
    
    # Mean and std dev
    x_pos = np.arange(len(regions))
    ax1.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7, color='lightblue')
    ax1.bar(x_pos + 0.2, stds, 0.4, label='Std Dev', alpha=0.7, color='lightcoral')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r.replace('_', '\n').title() for r in regions], rotation=45)
    ax1.set_ylabel('Value')
    ax1.set_title('Mean and Standard Deviation by Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Number of data points
    ax2.bar(x_pos, n_points, alpha=0.7, color='skyblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r.replace('_', '\n').title() for r in regions], rotation=45)
    ax2.set_ylabel('Number of Data Points')
    ax2.set_title('Number of Grid Points in Each Region')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def compute_pdfs_robust(region_data, bins=50, density=True, trim_tails=True, tail_percentile=1):
    """
    Compute PDFs with options to handle long tails
    
    Parameters:
    - region_data: dictionary with region data
    - bins: number of bins
    - density: whether to normalize to probability density
    - trim_tails: whether to trim extreme tails for better visualization
    - tail_percentile: percentile to trim from each tail (e.g., 1 = trim 1% from each side)
    """
    pdfs = {}
    stats = {}
    
    for region, data in region_data.items():
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) > 0:
            # Trim tails if requested
            if trim_tails and len(clean_data) > 0:
                lower_bound = np.percentile(clean_data, tail_percentile)
                upper_bound = np.percentile(clean_data, 100 - tail_percentile)
                clean_data = clean_data[(clean_data >= lower_bound) & (clean_data <= upper_bound)]
            
            # Compute histogram
            counts, bin_edges = np.histogram(clean_data, bins=bins, density=density)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            pdfs[region] = {
                'counts': counts,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'data': clean_data
            }
            
            # Basic statistics
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

def plot_region_pdfs_improved(pdfs, colors=None, xlabel="SSTA (°C)", title="PDFs by Oceanic Region", 
                             xlim=None, ylim=None):
    """
    Improved PDF plotting with better axis control
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
    
    # Plot 1: Individual PDFs
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
    
    # Set axis limits if provided
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)
    
    # Plot 2: Overlaid PDFs (normalized for comparison)
    for region, pdf_info in pdfs.items():
        if pdf_info is not None:
            # Normalize for better comparison
            normalized_pdf = pdf_info['counts'] / np.max(pdf_info['counts'])
            ax2.plot(pdf_info['bin_centers'], normalized_pdf,
                    color=colors[region], label=region.replace('_', ' ').title(),
                    linewidth=2, alpha=0.8)
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Normalized Probability Density')
    ax2.set_title('Normalized PDFs (for comparison)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set axis limits if provided
    if xlim is not None:
        ax2.set_xlim(xlim)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def compute_pdfs_for_variable_improved(data_array, lats, lons, masks, variable_name, model_name, 
                                     trim_tails=True, tail_percentile=1, xlim=None):
    """
    Improved version with better tail handling and axis control
    """
    print(f"Computing PDFs for {model_name} - {variable_name}")
    if trim_tails:
        print(f"Trimming {tail_percentile}% from each tail for better visualization")
    print("="*50)
    
    # Extract region data
    region_data = extract_region_data(data_array, lats, lons, masks, time_idx=None)
    
    # Compute PDFs with tail trimming
    pdfs, stats = compute_pdfs_robust(region_data, bins=50, density=True, 
                                    trim_tails=trim_tails, tail_percentile=tail_percentile)
    
    # Plot PDFs with controlled axis limits
    fig, axes = plot_region_pdfs_improved(pdfs, 
                                        xlabel=f"{variable_name} ({model_name})", 
                                        title=f"PDFs of {variable_name} by Region ({model_name})",
                                        xlim=xlim)
    plt.show()
    
    # Plot statistics
    fig, axes = plot_region_statistics(stats)
    plt.show()
    
    # Print summary
    print(f"\n{model_name} - {variable_name} Region Statistics:")
    if trim_tails:
        print(f"(Tails trimmed: {tail_percentile}% from each side)")
    print("="*60)
    for region, stat in stats.items():
        if stat:
            print(f"{region.replace('_', ' ').title():20} | "
                  f"Mean: {stat['mean']:7.3f} | "
                  f"Std: {stat['std']:6.3f} | "
                  f"Points: {stat['n_points']:8,d}")
    
    return pdfs, stats


def compute_region_means_anomalies(anomaly_da, mask_da_stacked):
    """
    Compute mean SST anomaly per region for a given anomaly DataArray and stacked masks.
    
    Parameters:
    - anomaly_da: xarray.DataArray, dims = ('time', 'lat', 'lon') - SST anomalies
    - mask_da_stacked: xarray.DataArray, dims = ('region','lat','lon')
    
    Returns:
    - region_means: xarray.DataArray, dims = ('time', 'region')
    """
    # Expand anomaly to have 'region' dim
    anomaly_expanded = anomaly_da.expand_dims({'region': mask_da_stacked.region})
    
    # Apply masks
    masked_anomaly = anomaly_expanded.where(mask_da_stacked)
    
    # Compute mean over lat/lon
    region_means = masked_anomaly.mean(dim=('lat','lon'))
    
    return region_means

def create_mask_dataarray(lats, lons, masks):
    """
    Create a stacked mask DataArray from the masks dictionary
    
    Parameters:
    - lats: latitude coordinates
    - lons: longitude coordinates
    - masks: dictionary of region masks from create_oceanic_regions_mask
    
    Returns:
    - mask_da_stacked: xarray.DataArray, dims = ('region','lat','lon')
    """
    # Convert masks dictionary to xarray DataArray
    mask_arrays = []
    region_names = []
    
    for region, mask in masks.items():
        mask_arrays.append(mask)
        region_names.append(region)
    
    # Stack into DataArray
    mask_da = xr.DataArray(
        np.stack(mask_arrays, axis=0),
        dims=('region', 'lat', 'lon'),
        coords={
            'region': region_names,
            'lat': lats,
            'lon': lons
        }
    )
    
    return mask_da

def compute_anomaly_statistics(region_means_dict, model_names):
    """
    Compute statistics for region means across models
    
    Parameters:
    - region_means_dict: dictionary with model names as keys and region_means DataArrays as values
    - model_names: list of model names for plotting
    
    Returns:
    - temporal_means: dictionary with temporal means for each model
    - temporal_stds: dictionary with temporal standard deviations for each model
    """
    temporal_means = {}
    temporal_stds = {}
    
    for model_name, region_means in region_means_dict.items():
        # Compute temporal mean for each region
        temporal_means[model_name] = region_means.mean(dim='time').values
        
        # Compute temporal std for each region (variability)
        temporal_stds[model_name] = region_means.std(dim='time').values
    
    return temporal_means, temporal_stds

def plot_region_anomaly_comparison(temporal_means, temporal_stds, region_names, model_names, 
                                  plot_type='means', title_suffix=''):
    """
    Plot comparison of region anomaly statistics across models
    """
    if plot_type == 'means':
        data_dict = temporal_means
        ylabel = 'Mean SST Anomaly (°C)'
        title = f'Mean SST Anomaly by Region {title_suffix}'
    else:  # stds
        data_dict = temporal_stds
        ylabel = 'SST Anomaly Std Dev (°C)'
        title = f'SST Anomaly Variability by Region {title_suffix}'
    
    # Add global region (mean of all regions)
    all_region_names = list(region_names) + ['Global']
    
    # Prepare data for plotting
    all_data = {}
    for model_name in model_names:
        model_data = data_dict[model_name]
        global_mean = np.mean(model_data)  # Global as mean of all regions
        all_data[model_name] = np.append(model_data, global_mean)
    
    # Plot
    x = np.arange(len(all_region_names))
    width = 0.8 / len(model_names)
    
    plt.figure(figsize=(12, 6))
    
    for i, model_name in enumerate(model_names):
        offset = (i - (len(model_names)-1)/2) * width
        plt.bar(x + offset, all_data[model_name], width, 
               label=model_name, alpha=0.8)
    
    plt.xticks(x, [r.replace('_',' ').title() for r in all_region_names], rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    return all_data


def compute_region_means_chunked(anomaly_da, mask_da_stacked, time_chunk_size=100):
    """
    Compute region means in chunks to avoid memory issues
    
    Parameters:
    - anomaly_da: xarray.DataArray with Dask backend
    - mask_da_stacked: mask DataArray
    - time_chunk_size: number of time steps to process at once
    """
    # Get total time steps
    n_time = anomaly_da.sizes['time']
    
    # Initialize list to store results
    chunks = []
    
    # Process in chunks
    for start_idx in range(0, n_time, time_chunk_size):
        end_idx = min(start_idx + time_chunk_size, n_time)
        print(f"Processing time steps {start_idx} to {end_idx-1}...")
        
        # Select time chunk
        anomaly_chunk = anomaly_da.isel(time=slice(start_idx, end_idx))
        
        # Compute region means for this chunk
        region_means_chunk = compute_region_means_anomalies(anomaly_chunk, mask_da_stacked)
        
        # Compute this chunk and store
        chunks.append(region_means_chunk.compute())
    
    # Combine all chunks
    region_means_combined = xr.concat(chunks, dim='time')
    
    return region_means_combined

def save_region_means_direct(region_means_da, filename):
    """
    Save region means directly without computing first
    """
    # Clear time encoding to avoid issues
    if 'time' in region_means_da.coords:
        region_means_da['time'].encoding.clear()
        region_means_da['time'].attrs.pop('units', None)
        region_means_da['time'].attrs.pop('calendar', None)
    
    # Save directly using Dask's chunked writing
    print(f"Saving {filename}...")
    region_means_da.to_netcdf(filename, mode='w')
    print("Save completed!")


def diagnose_region_means(region_means_dict, model_names):
    """
    Diagnostic function to check region means data
    """
    print("DIAGNOSTIC INFORMATION:")
    print("="*50)
    
    for model_name in model_names:
        if model_name in region_means_dict:
            data = region_means_dict[model_name]
            print(f"\n{model_name}:")
            print(f"  Shape: {data.shape}")
            print(f"  Dimensions: {data.dims}")
            print(f"  Regions: {list(data.region.values) if 'region' in data.coords else 'No region dim'}")
            print(f"  Time steps: {data.sizes['time'] if 'time' in data.dims else 'No time dim'}")
            
            if 'time' in data.dims and 'region' in data.dims:
                # Check for NaN values
                nan_count = np.isnan(data.values).sum()
                print(f"  NaN values: {nan_count}/{data.size} ({nan_count/data.size*100:.1f}%)")
                
                # Check mean values
                temporal_means = data.mean(dim='time')
                print(f"  Temporal means by region:")
                for region in data.region.values:
                    region_mean = temporal_means.sel(region=region).values
                    print(f"    {region}: {region_mean:.4f}")
        else:
            print(f"\n{model_name}: NOT FOUND in region_means_dict")

def compare_data_structures(ds1, ds2, name1="Dataset1", name2="Dataset2"):
    """
    Compare the structure of two datasets
    """
    print(f"COMPARING {name1} vs {name2}:")
    print("="*40)
    
    print(f"{name1} shape: {ds1.shape}")
    print(f"{name2} shape: {ds2.shape}")
    
    print(f"{name1} dims: {ds1.dims}")
    print(f"{name2} dims: {ds2.dims}")
    
    if hasattr(ds1, 'region') and hasattr(ds2, 'region'):
        print(f"{name1} regions: {list(ds1.region.values)}")
        print(f"{name2} regions: {list(ds2.region.values)}")
    
    print(f"{name1} time range: {ds1.time.min().values} to {ds1.time.max().values}")
    print(f"{name2} time range: {ds2.time.min().values} to {ds2.time.max().values}")



def compute_global_pdf_simple(data_array, variable_name, model_name, bins=100, xlim=(-5, 5)):
    """
    Simple PDF computation using fixed range - no quantiles needed!
    """
    print(f"Computing global PDF for {model_name} - {variable_name}")
    
    # Just get the total count (this is cheap with Dask)
    n_total = int(data_array.count().compute())
    print(f"Total data points: {n_total:,}")
    print(f"Using fixed range: [{xlim[0]}, {xlim[1]}]")
    
    # Compute histogram with the fixed range we want
    counts, bin_edges = da.histogram(data_array.data, bins=bins, range=xlim, density=True)
    counts = counts.compute()
    bin_edges = bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Get basic stats using Dask (these are memory-efficient)
    mean_val = float(data_array.mean().compute())
    std_val = float(data_array.std().compute())
    
    stats = {
        'mean': mean_val,
        'std': std_val,
        'min': xlim[0],  # We're using our fixed range
        'max': xlim[1],
        'n_points': n_total
    }
    
    return {
        'counts': counts,
        'bin_centers': bin_centers,
        'stats': stats
    }

def plot_multi_model_pdf_comparison(models_dict, bins=100, title_suffix=""):
    """
    Plot PDF comparison for multiple models
    
    Parameters:
    - models_dict: dictionary with model names as keys and (ds, variable_name) tuples as values
                  e.g., {'OSTIA': (ds_ostia, 'dat_anomaly'), 'ICON': (ds_icon, 'dat_anomaly')}
    - bins: number of bins for histogram
    - title_suffix: additional text for the title
    """
    print("MULTI-MODEL GLOBAL PDF COMPARISON")
    print("=" * 45)
    
    # Define a color cycle for models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Compute PDFs for all models
    pdfs = {}
    for i, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
        color = colors[i % len(colors)]
        pdfs[model_name] = compute_global_pdf_simple(ds[var_name], "SST Anomaly", model_name, bins=bins)
        pdfs[model_name]['color'] = color
    
    # Plot all models
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_name, pdf_data in pdfs.items():
        ax.plot(pdf_data['bin_centers'], pdf_data['counts'], 
                color=pdf_data['color'], label=model_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('SST Anomaly (°C)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Global SST Anomaly PDF Comparison{title_suffix}')
    ax.set_xlim(-5, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics for all models
    print(f"\nBasic Statistics:")
    print(f"{'Model':<15} | {'Mean':>8} | {'Std':>8} | {'Data Points':>12}")
    print("-" * 60)
    for model_name, pdf_data in pdfs.items():
        stats = pdf_data['stats']
        print(f"{model_name:<15} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>12,}")
    
    return fig, ax, pdfs

def compute_global_pdf_kde(data_array, variable_name, model_name, xlim=(-5, 5), points=1000):
    """
    Compute PDF using KDE for smoother results
    """
    print(f"Computing KDE PDF for {model_name} - {variable_name}")
    
    # Get basic stats
    n_total = int(data_array.count().compute())
    mean_val = float(data_array.mean().compute())
    std_val = float(data_array.std().compute())
    
    print(f"Total data points: {n_total:,}")
    print(f"Using KDE with {points} points")
    
    # Take a representative sample for KDE (much more memory efficient)
    sample_size = min(100000, n_total)
    step = max(1, n_total // sample_size)
    
    # Sample the data
    sample = data_array.data.ravel()[::step].compute()
    clean_sample = sample[~np.isnan(sample)]
    
    print(f"Sample size for KDE: {len(clean_sample):,}")
    
    # Compute KDE
    kde = gaussian_kde(clean_sample)
    
    # Create evaluation points
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

def plot_multi_model_kde_comparison(models_dict, xlim=(-10, 10), title_suffix=""):
    """
    Plot KDE-based PDF comparison for multiple models with customizable xlim
    """
    print("MULTI-MODEL GLOBAL PDF COMPARISON (KDE)")
    print("=" * 50)
    print(f"Using xlim: [{xlim[0]}, {xlim[1]}]")
    
    # Define a color cycle for models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Compute KDE PDFs for all models
    pdfs = {}
    for i, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
        color = colors[i % len(colors)]
        pdfs[model_name] = compute_global_pdf_kde(ds[var_name], "SST Anomaly", model_name, xlim=xlim)
        pdfs[model_name]['color'] = color
    
    # Plot all models
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_name, pdf_data in pdfs.items():
        ax.plot(pdf_data['x_points'], pdf_data['pdf_values'], 
                color=pdf_data['color'], label=model_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('SST Anomaly (°C)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Global SST Anomaly PDF Comparison (KDE){title_suffix}')
    ax.set_xlim(xlim[0], xlim[1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics for all models
    print(f"\nBasic Statistics:")
    print(f"{'Model':<15} | {'Mean':>8} | {'Std':>8} | {'Data Points':>12}")
    print("-" * 60)
    for model_name, pdf_data in pdfs.items():
        stats = pdf_data['stats']
        print(f"{model_name:<15} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>12,}")
    
    return fig, ax, pdfs

def analyze_multi_model_extreme_thresholds(models_dict, percentiles=[90, 95, 99, 99.5, 99.9], xlim=(-10, 10)):
    """
    Analyze extreme detection thresholds for multiple models with customizable xlim
    and improved individual model PDF plots
    """
    print("MULTI-MODEL EXTREME DETECTION THRESHOLD ANALYSIS")
    print("=" * 55)
    print(f"Using xlim: [{xlim[0]}, {xlim[1]}]")
    
    # Define colors for models and percentiles
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    percentile_colors = ['orange', 'green', 'purple', 'brown', 'black', 'cyan', 'magenta', 'yellow']
    
    # Compute percentiles for all models
    print("Computing percentiles for all models...")
    thresholds = {}
    for model_name, (ds, var_name) in models_dict.items():
        thresholds[model_name] = compute_percentiles_dask_friendly(ds[var_name], percentiles)
        print(f"{model_name}: {thresholds[model_name]}")
    
    # FIGURE 1: Individual model PDFs with percentile thresholds (the good version!)
    n_models = len(models_dict)
    n_cols = min(2, n_models)  # Max 2 columns for readability
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes1 = np.array([axes1])
    axes1 = axes1.ravel()
    
    print("\nComputing KDE for smooth PDFs...")
    for i, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
        ax = axes1[i]
        color = model_colors[i % len(model_colors)]
        
        # Compute KDE PDF
        pdf_data = compute_global_pdf_kde(ds[var_name], "SST Anomaly", model_name, xlim=xlim)
        
        # Plot PDF
        ax.plot(pdf_data['x_points'], pdf_data['pdf_values'], 
                color=color, label=f'{model_name} PDF', linewidth=2)
        
        # Add percentile lines with labels
        for j, (p, threshold) in enumerate(thresholds[model_name].items()):
            p_color = percentile_colors[j % len(percentile_colors)]
            if xlim[0] <= threshold <= xlim[1]:  # Only plot if within xlim
                ax.axvline(threshold, color=p_color, linestyle='--', alpha=0.7, 
                          label=f'{p}th: {threshold:.3f}°C')
        
        ax.set_xlabel('SST Anomaly (°C)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{model_name}: Extreme Detection Thresholds')
        ax.set_xlim(max(xlim[0], -2), min(xlim[1], 5))  # Focus on tail region
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(models_dict), len(axes1)):
        fig1.delaxes(axes1[i])
    
    plt.tight_layout()
    plt.show()
    
    # FIGURE 2: Threshold comparison bar chart (keep this one)
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

def comprehensive_multi_model_analysis(models_dict, percentiles=[90, 95, 99, 99.5, 99.9], xlim=(-10, 10)):
    """
    Comprehensive analysis for multiple models with customizable xlim
    """
    print("COMPREHENSIVE MULTI-MODEL ANALYSIS")
    print("=" * 45)
    print(f"Using xlim: [{xlim[0]}, {xlim[1]}]")
    
    # Define colors
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Compute PDFs and thresholds
    pdfs = {}
    thresholds = {}
    
    for i, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
        color = model_colors[i % len(model_colors)]
        pdfs[model_name] = compute_global_pdf_kde(ds[var_name], "SST Anomaly", model_name, xlim=xlim)
        pdfs[model_name]['color'] = color
        thresholds[model_name] = compute_percentiles_dask_friendly(ds[var_name], percentiles)
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Full PDF comparison
    for model_name, pdf_data in pdfs.items():
        ax1.plot(pdf_data['x_points'], pdf_data['pdf_values'], 
                color=pdf_data['color'], label=model_name, linewidth=2)
    ax1.set_xlabel('SST Anomaly (°C)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Global SST Anomaly PDFs - Full Distribution')
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual model PDFs with percentiles (using the good layout)
    n_models = len(models_dict)
    colors = ['orange', 'green', 'purple', 'brown', 'black']
    
    # For simplicity in the comprehensive plot, just show first 4 models
    models_to_plot = list(models_dict.items())[:4]
    for i, (model_name, (ds, var_name)) in enumerate(models_to_plot):
        pdf_data = pdfs[model_name]
        color = model_colors[i % len(model_colors)]
        
        ax2.plot(pdf_data['x_points'], pdf_data['pdf_values'], 
                color=color, label=model_name, linewidth=2)
        
        # Add percentile lines
        for j, (p, threshold) in enumerate(thresholds[model_name].items()):
            if j < len(colors) and xlim[0] <= threshold <= xlim[1]:
                ax2.axvline(threshold, color=colors[j], linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('SST Anomaly (°C)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('PDFs with Extreme Detection Thresholds\n(Dashed lines show percentiles)')
    ax2.set_xlim(max(xlim[0], -2), min(xlim[1], 5))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Threshold comparison
    x_pos = np.arange(len(percentiles))
    width = 0.8 / n_models
    
    for i, (model_name, model_thresholds) in enumerate(thresholds.items()):
        color = model_colors[i % len(model_colors)]
        model_vals = [model_thresholds[p] for p in percentiles]
        offset = (i - (n_models-1)/2) * width
        ax3.bar(x_pos + offset, model_vals, width, 
               label=model_name, alpha=0.7, color=color)
    
    ax3.set_xlabel('Percentile')
    ax3.set_ylabel('Threshold (°C)')
    ax3.set_title('Extreme Detection Thresholds by Model')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{p}th' for p in percentiles])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relative to first model
    first_model = list(models_dict.keys())[0]
    x_pos = np.arange(len(percentiles))
    
    for i, (model_name, model_thresholds) in enumerate(thresholds.items()):
        if model_name != first_model:
            color = model_colors[i % len(model_colors)]
            relative_diffs = [(model_thresholds[p] - thresholds[first_model][p]) / thresholds[first_model][p] * 100 
                            for p in percentiles]
            ax4.bar(x_pos + (i-1)*0.15, relative_diffs, 0.15, 
                   label=model_name, alpha=0.7, color=color)
    
    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('Relative Difference (%)')
    ax4.set_title(f'Thresholds Relative to {first_model}\n(Positive = higher threshold)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{p}th' for p in percentiles])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print(f"\nKEY INSIGHTS:")
    print("=" * 40)
    first_model = list(models_dict.keys())[0]
    for model_name in list(models_dict.keys())[1:]:
        print(f"\nComparison: {model_name} vs {first_model}:")
        for p in percentiles:
            diff = thresholds[model_name][p] - thresholds[first_model][p]
            if diff > 0:
                implication = f"detects FEWER extremes at {p}th percentile"
            else:
                implication = f"detects MORE extremes at {p}th percentile"
            print(f"  - {model_name} {implication} ({thresholds[first_model][p]:.3f}°C vs {thresholds[model_name][p]:.3f}°C)")
    
    return pdfs, thresholds

# Keep all the other functions the same as before:
def compute_global_pdf_simple(data_array, variable_name, model_name, bins=100, xlim=(-5, 5)):
    """
    Simple PDF computation using fixed range - no quantiles needed!
    """
    print(f"Computing global PDF for {model_name} - {variable_name}")
    
    # Just get the total count (this is cheap with Dask)
    n_total = int(data_array.count().compute())
    print(f"Total data points: {n_total:,}")
    print(f"Using fixed range: [{xlim[0]}, {xlim[1]}]")
    
    # Compute histogram with the fixed range we want
    counts, bin_edges = da.histogram(data_array.data, bins=bins, range=xlim, density=True)
    counts = counts.compute()
    bin_edges = bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Get basic stats using Dask (these are memory-efficient)
    mean_val = float(data_array.mean().compute())
    std_val = float(data_array.std().compute())
    
    stats = {
        'mean': mean_val,
        'std': std_val,
        'min': xlim[0],  # We're using our fixed range
        'max': xlim[1],
        'n_points': n_total
    }
    
    return {
        'counts': counts,
        'bin_centers': bin_centers,
        'stats': stats
    }

def compute_global_pdf_kde(data_array, variable_name, model_name, xlim=(-5, 5), points=1000):
    """
    Compute PDF using KDE for smoother results
    """
    print(f"Computing KDE PDF for {model_name} - {variable_name}")
    
    # Get basic stats
    n_total = int(data_array.count().compute())
    mean_val = float(data_array.mean().compute())
    std_val = float(data_array.std().compute())
    
    print(f"Total data points: {n_total:,}")
    print(f"Using KDE with {points} points")
    
    # Take a representative sample for KDE (much more memory efficient)
    sample_size = min(100000, n_total)
    step = max(1, n_total // sample_size)
    
    # Sample the data
    sample = data_array.data.ravel()[::step].compute()
    clean_sample = sample[~np.isnan(sample)]
    
    print(f"Sample size for KDE: {len(clean_sample):,}")
    
    # Compute KDE
    kde = gaussian_kde(clean_sample)
    
    # Create evaluation points
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
    Compute percentiles in a Dask-friendly way by flattening first
    """
    # Flatten the data and remove NaNs using Dask operations
    flat_data = data_array.data.ravel()
    flat_data_clean = flat_data[~da.isnan(flat_data)]
    
    # Compute percentiles on the cleaned flat data
    thresholds = {}
    for p in percentiles:
        thresholds[p] = float(da.percentile(flat_data_clean, p).compute())
    
    return thresholds


def analyze_points_above_thresholds(models_dict, percentiles=[90, 95, 99, 99.5, 99.9]):
    """
    Analyze how many points are above each percentile threshold for each model
    """
    print("POINTS ABOVE THRESHOLD ANALYSIS")
    print("=" * 50)
    
    # Define colors for models
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Compute percentiles and counts for all models
    thresholds = {}
    counts_above = {}
    percentages_above = {}
    total_points = {}
    
    print("Computing thresholds and counting points above thresholds...")
    
    for i, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
        print(f"Processing {model_name}...")
        
        # Get the data array
        data_array = ds[var_name]
        
        # Compute total number of points (excluding NaNs)
        total_points[model_name] = int(data_array.count().compute())
        
        # Compute percentiles
        thresholds[model_name] = compute_percentiles_dask_friendly(data_array, percentiles)
        
        # For each percentile, count points above threshold
        counts_above[model_name] = {}
        percentages_above[model_name] = {}
        
        for p in percentiles:
            threshold = thresholds[model_name][p]
            
            # Count points above threshold using Dask (memory efficient)
            points_above = (data_array > threshold).sum().compute()
            counts_above[model_name][p] = int(points_above)
            
            # Calculate percentage
            percentage = (points_above / total_points[model_name]) * 100
            percentages_above[model_name][p] = float(percentage)
            
            print(f"  {p}th percentile: {points_above:,} points above ({percentage:.4f}%)")
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Absolute counts above thresholds
    x_pos = np.arange(len(percentiles))
    width = 0.8 / len(models_dict)
    
    for i, model_name in enumerate(models_dict.keys()):
        color = model_colors[i % len(model_colors)]
        counts = [counts_above[model_name][p] for p in percentiles]
        offset = (i - (len(models_dict)-1)/2) * width
        
        ax1.bar(x_pos + offset, counts, width, label=model_name, alpha=0.7, color=color)
        
        # Add value labels on top of bars
        for j, count in enumerate(counts):
            ax1.text(x_pos[j] + offset, count + max(counts)*0.01, f'{count/1e6:.1f}M', 
                    ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax1.set_xlabel('Percentile Threshold')
    ax1.set_ylabel('Number of Points Above Threshold')
    ax1.set_title('Absolute Count of Extreme Points\n(Higher bars = more extremes detected)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{p}th' for p in percentiles])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 2: Percentage of points above thresholds
    for i, model_name in enumerate(models_dict.keys()):
        color = model_colors[i % len(model_colors)]
        percentages = [percentages_above[model_name][p] for p in percentiles]
        
        ax2.plot(x_pos, percentages, 'o-', color=color, label=model_name, linewidth=2, markersize=6)
        
        # Add value labels
        for j, percentage in enumerate(percentages):
            ax2.text(x_pos[j], percentage + 1, f'{percentage:.2f}%', 
                    ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Percentile Threshold')
    ax2.set_ylabel('Percentage of Points Above Threshold (%)')
    ax2.set_title('Percentage of Extreme Points\n(Expected: 100 - percentile)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{p}th' for p in percentiles])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relative to expected percentage (deviation from ideal)
    for i, model_name in enumerate(models_dict.keys()):
        color = model_colors[i % len(model_colors)]
        deviations = []
        
        for p in percentiles:
            expected_percentage = 100 - p
            actual_percentage = percentages_above[model_name][p]
            deviation = actual_percentage - expected_percentage
            deviations.append(deviation)
        
        ax3.plot(x_pos, deviations, 's-', color=color, label=model_name, linewidth=2, markersize=6)
        
        # Add value labels
        for j, deviation in enumerate(deviations):
            ax3.text(x_pos[j], deviation + (0.1 if deviation >= 0 else -0.1), f'{deviation:+.2f}%', 
                    ha='center', va='bottom' if deviation >= 0 else 'top', fontsize=8)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Ideal (0 deviation)')
    ax3.set_xlabel('Percentile Threshold')
    ax3.set_ylabel('Deviation from Expected Percentage (%)')
    ax3.set_title('Deviation from Expected Extreme Percentage\n(Positive = more extremes than expected)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{p}th' for p in percentiles])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison table as a heatmap
    # Prepare data for heatmap
    data_for_heatmap = []
    model_names = list(models_dict.keys())
    
    for model_name in model_names:
        row = [percentages_above[model_name][p] for p in percentiles]
        data_for_heatmap.append(row)
    
    data_for_heatmap = np.array(data_for_heatmap)
    
    im = ax4.imshow(data_for_heatmap, cmap='YlOrRd', aspect='auto')
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(percentiles)):
            text = ax4.text(j, i, f'{data_for_heatmap[i, j]:.2f}%',
                           ha="center", va="center", color="black", fontsize=10)
    
    ax4.set_xticks(range(len(percentiles)))
    ax4.set_xticklabels([f'{p}th' for p in percentiles])
    ax4.set_yticks(range(len(model_names)))
    ax4.set_yticklabels(model_names)
    ax4.set_title('Percentage of Points Above Threshold\n(Heatmap View)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Percentage (%)')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed table
    print(f"\nDETAILED ANALYSIS:")
    print("=" * 100)
    header = f"{'Model':<12} | {'Percentile':>10} | {'Threshold':>10} | {'Points Above':>15} | {'% Above':>10} | {'Expected %':>10} | {'Deviation':>10}"
    print(header)
    print("-" * len(header))
    
    for model_name in models_dict.keys():
        print(f"{model_name:<12} | {'':>10} | {'':>10} | {'':>15} | {'':>10} | {'':>10} | {'':>10}")
        for p in percentiles:
            threshold = thresholds[model_name][p]
            count_above = counts_above[model_name][p]
            percentage_above = percentages_above[model_name][p]
            expected_percentage = 100 - p
            deviation = percentage_above - expected_percentage
            
            print(f"{'':<12} | {p:>10} | {threshold:>10.3f} | {count_above:>15,} | {percentage_above:>9.3f}% | {expected_percentage:>9.1f}% | {deviation:>+9.3f}%")
        print("-" * len(header))
    
    # Summary insights
    print(f"\nKEY INSIGHTS:")
    print("=" * 60)
    
    for model_name in models_dict.keys():
        print(f"\n{model_name}:")
        total = total_points[model_name]
        print(f"  Total data points: {total:,}")
        
        for p in percentiles:
            expected = 100 - p
            actual = percentages_above[model_name][p]
            deviation = actual - expected
            
            if deviation > 1:
                insight = f"  {p}th: {deviation:+.2f}% more extremes than expected ({actual:.2f}% vs {expected:.1f}%)"
            elif deviation < -1:
                insight = f"  {p}th: {deviation:+.2f}% fewer extremes than expected ({actual:.2f}% vs {expected:.1f}%)"
            else:
                insight = f"  {p}th: Close to expected ({actual:.2f}% vs {expected:.1f}%)"
            
            print(insight)
    
    return {
        'thresholds': thresholds,
        'counts_above': counts_above,
        'percentages_above': percentages_above,
        'total_points': total_points
    }


def quick_points_analysis(models_dict, percentiles=[90, 95, 99, 99.5, 99.9]):
    """
    Quick analysis of points above thresholds with minimal output
    """
    print("QUICK POINTS ABOVE THRESHOLD ANALYSIS")
    print("=" * 45)
    
    results = analyze_points_above_thresholds(models_dict, percentiles)
    
    # Print condensed summary
    print(f"\nCONDENSED SUMMARY:")
    print("=" * 80)
    print(f"{'Model':<12} | {'Total Points':>15} | " + " | ".join([f'{p}th %' for p in percentiles]))
    print("-" * 80)
    
    for model_name in models_dict.keys():
        total = results['total_points'][model_name]
        percentages = [f"{results['percentages_above'][model_name][p]:6.2f}%" for p in percentiles]
        print(f"{model_name:<12} | {total:>15,} | " + " | ".join(percentages))
    
    return results




# REGIONAL

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



def compute_regional_pdf_kde_memory_efficient(data_array, mask, region_name, model_name, xlim=(-5, 5), points=1000, max_sample_size=50000):
    """
    Memory-efficient version: Use Dask operations and sampling
    """
    print(f"  Computing PDF for {model_name} - {region_name}")
    
    try:
        # Use Dask to count points first (cheap operation)
        total_points = int(data_array.count().compute())
        
        # Sample the data if it's too large
        if total_points > max_sample_size:
            # Use systematic sampling with Dask
            step = total_points // max_sample_size
            # Create a sample mask
            sample_indices = da.arange(total_points)[::step]
            # This is complex with masked data, so let's use a different approach
            
            # Instead, let's compute basic stats and use KDE on a sample
            print(f"    Large dataset ({total_points:,} points), taking sample...")
            
            # Get a flattened version of the data with mask applied
            if data_array.ndim == 3:
                # For 3D data, we need to be more careful
                # Let's compute mean over time first to reduce dimensions
                temporal_mean = data_array.mean(dim='time', skipna=True)
                regional_data = temporal_mean.where(mask).values.flatten()
            else:
                regional_data = data_array.where(mask).values.flatten()
            
            # Remove NaNs and sample
            clean_data = regional_data[~np.isnan(regional_data)]
            if len(clean_data) > max_sample_size:
                step = len(clean_data) // max_sample_size
                clean_data = clean_data[::step]
                
        else:
            # Small enough to process directly
            if data_array.ndim == 3:
                temporal_mean = data_array.mean(dim='time', skipna=True)
                regional_data = temporal_mean.where(mask).values.flatten()
            else:
                regional_data = data_array.where(mask).values.flatten()
            
            clean_data = regional_data[~np.isnan(regional_data)]
        
        if len(clean_data) == 0:
            print(f"    Warning: No data for {region_name} in {model_name}")
            return None
        
        print(f"    Using {len(clean_data):,} points for KDE")
        
        # Compute KDE
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

def plot_regional_pdf_comparison_memory_efficient(models_dict, masks_dict, regions=None, xlim=(-5, 5)):
    """
    Memory-efficient regional PDF comparison
    """
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    print("MEMORY-EFFICIENT REGIONAL PDF COMPARISON")
    print("=" * 50)
    
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create subplots
    n_regions = len(regions)
    n_cols = min(2, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_regions == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    # Process one region at a time
    all_pdfs = {}
    
    for region_idx, region_name in enumerate(regions):
        print(f"\nProcessing region: {region_name}")
        ax = axes[region_idx]
        region_pdfs = {}
        
        for model_idx, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
            color = model_colors[model_idx % len(model_colors)]
            
            # Get model-specific mask
            mask = masks_dict[model_name][region_name]
            
            # Compute PDF with memory efficiency
            pdf_data = compute_regional_pdf_kde_memory_efficient(
                ds[var_name], mask, region_name, model_name, xlim=xlim
            )
            
            if pdf_data is not None:
                region_pdfs[model_name] = pdf_data
                pdf_data['color'] = color
                
                # Plot PDF
                ax.plot(pdf_data['x_points'], pdf_data['pdf_values'],
                       color=color, label=model_name, linewidth=2, alpha=0.8)
        
        all_pdfs[region_name] = region_pdfs
        
        ax.set_xlabel('SST Anomaly (°C)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{region_name.replace("_", " ").title()}')
        ax.set_xlim(xlim[0], xlim[1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(regions), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nREGIONAL SUMMARY:")
    print("=" * 80)
    for region_name in regions:
        print(f"\n{region_name.replace('_', ' ').title()}:")
        print(f"{'Model':<12} | {'Mean':>8} | {'Std':>8} | {'Sample':>10} | {'Total':>12}")
        print("-" * 60)
        
        for model_name, pdf_data in all_pdfs[region_name].items():
            stats = pdf_data['stats']
            total_pts = stats.get('total_points', stats['n_points'])
            print(f"{model_name:<12} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>10,} | {total_pts:>12,}")
    
    return all_pdfs



def quick_regional_analysis(models_dict, regions=None, xlim=(-5, 5)):
    """
    Quick regional analysis that's guaranteed to be memory efficient
    """
    print("QUICK REGIONAL ANALYSIS")
    print("=" * 30)
    print("This version is optimized for memory usage")
    
    # Create model-specific masks
    print("Creating masks...")
    masks_dict = create_model_specific_masks(models_dict)
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    # 1. Regional PDFs
    print("\n1. Computing regional PDFs...")
    pdfs = plot_regional_pdf_comparison_memory_efficient(
        models_dict, masks_dict, regions, xlim=xlim
    )
    
    # 2. Regional extreme thresholds - use the main function but simplified
    print("\n2. Computing regional extreme thresholds...")
    regional_thresholds, regional_pdfs, regional_masks = analyze_regional_extreme_thresholds(
        models_dict,
        regions=regions,
        percentiles=[90, 95, 99],  # Use fewer percentiles for quick analysis
        xlim=xlim,
        plot_percentile_lines=False  # Turn off percentile lines for quicker plotting
    )
    
    # Format the return to match the original structure
    extremes = {
        'thresholds': regional_thresholds,
        'pdfs': regional_pdfs
    }
    
    return pdfs, extremes, masks_dict



def analyze_regional_extreme_thresholds(models_dict, regions=None, percentiles=[90, 95, 99, 99.5, 99.9], xlim=(-5, 5), plot_percentile_lines=True):
    """
    Regional equivalent of analyze_multi_model_extreme_thresholds
    Shows PDFs with optional percentile thresholds for each region
    
    Parameters:
    - plot_percentile_lines: if True, shows percentile lines on the PDFs
    """
    print("REGIONAL EXTREME DETECTION THRESHOLD ANALYSIS")
    print("=" * 55)
    print(f"Using xlim: [{xlim[0]}, {xlim[1]}]")
    print(f"Plot percentile lines: {plot_percentile_lines}")
    
    # Create model-specific masks
    masks_dict = create_model_specific_masks(models_dict)
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    # Define colors
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    percentile_colors = ['orange', 'green', 'purple', 'brown', 'black', 'cyan', 'magenta', 'yellow']
    
    # Compute thresholds and PDFs for all regions
    regional_thresholds = {}
    regional_pdfs = {}
    
    print("Computing regional thresholds and PDFs...")
    for region_name in regions:
        print(f"\nProcessing {region_name}...")
        regional_thresholds[region_name] = {}
        regional_pdfs[region_name] = {}
        
        for model_idx, (model_name, (ds, var_name)) in enumerate(models_dict.items()):
            print(f"  {model_name}...")
            
            # Get model-specific mask
            mask = masks_dict[model_name][region_name]
            data_array = ds[var_name]
            
            # Compute percentiles for this region and model
            try:
                if data_array.ndim == 3:
                    # For 3D data, compute temporal mean first
                    regional_data = data_array.where(mask).mean(dim='time', skipna=True)
                else:
                    regional_data = data_array.where(mask)
                
                # Flatten and remove NaNs
                values = regional_data.values.flatten()
                clean_data = values[~np.isnan(values)]
                
                if len(clean_data) == 0:
                    continue
                
                # Compute percentiles
                thresholds = {}
                for p in percentiles:
                    thresholds[p] = np.percentile(clean_data, p)
                
                regional_thresholds[region_name][model_name] = thresholds
                
                # Compute KDE PDF for this region and model
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
    
    # FIGURE 1: Individual region PDFs with optional percentile thresholds
    n_regions = len(regions)
    n_cols = min(2, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_regions == 1:
        axes1 = np.array([axes1])
    axes1 = axes1.ravel()
    
    for region_idx, region_name in enumerate(regions):
        ax = axes1[region_idx]
        
        # Plot PDFs for each model in this region
        for model_name, pdf_data in regional_pdfs[region_name].items():
            ax.plot(pdf_data['x_points'], pdf_data['pdf_values'], 
                   color=pdf_data['color'], label=model_name, linewidth=2, alpha=0.8)
            
            # Add percentile lines for this model (if enabled)
            if plot_percentile_lines and region_name in regional_thresholds and model_name in regional_thresholds[region_name]:
                thresholds = regional_thresholds[region_name][model_name]
                for j, (p, threshold) in enumerate(thresholds.items()):
                    if j < len(percentile_colors) and xlim[0] <= threshold <= xlim[1]:
                        ax.axvline(threshold, color=percentile_colors[j], linestyle='--', alpha=0.7,
                                  label=f'{p}th: {threshold:.3f}°C' if model_name == list(models_dict.keys())[0] else "")
        
        ax.set_xlabel('SST Anomaly (°C)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{region_name.replace("_", " ").title()}')
        ax.set_xlim(max(xlim[0], -2), min(xlim[1], 5))  # Focus on tail region
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(regions), len(axes1)):
        fig1.delaxes(axes1[i])
    
    plt.tight_layout()
    plt.show()
    
    # FIGURE 2: Threshold comparison bar chart (always show this)
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




def analyze_regional_points_above_thresholds(models_dict, regions=None, percentiles=[90, 95, 99, 99.5, 99.9]):
    """
    Regional version of analyze_points_above_thresholds - analyzes points above thresholds for each region
    """
    print("REGIONAL POINTS ABOVE THRESHOLD ANALYSIS")
    print("=" * 55)
    
    # Create model-specific masks
    masks_dict = create_model_specific_masks(models_dict)
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    # Define colors for models
    model_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Compute thresholds and counts for all regions and models
    regional_results = {}
    
    print("Computing regional thresholds and counting points above thresholds...")
    
    for region_name in regions:
        print(f"\nProcessing {region_name}...")
        regional_results[region_name] = {
            'thresholds': {},
            'counts_above': {},
            'percentages_above': {},
            'total_points': {}
        }
        
        for model_name, (ds, var_name) in models_dict.items():
            print(f"  {model_name}...")
            
            # Get model-specific mask
            mask = masks_dict[model_name][region_name]
            data_array = ds[var_name]
            
            try:
                # Apply mask and compute regional data
                if data_array.ndim == 3:
                    regional_data = data_array.where(mask).mean(dim='time', skipna=True)
                else:
                    regional_data = data_array.where(mask)
                
                # Flatten and remove NaNs
                values = regional_data.values.flatten()
                clean_data = values[~np.isnan(values)]
                
                if len(clean_data) == 0:
                    continue
                
                total_points = len(clean_data)
                regional_results[region_name]['total_points'][model_name] = total_points
                
                # Compute percentiles
                thresholds = {}
                for p in percentiles:
                    thresholds[p] = np.percentile(clean_data, p)
                regional_results[region_name]['thresholds'][model_name] = thresholds
                
                # Count points above thresholds
                counts_above = {}
                percentages_above = {}
                for p in percentiles:
                    threshold = thresholds[p]
                    count_above = np.sum(clean_data > threshold)
                    counts_above[p] = count_above
                    percentages_above[p] = (count_above / total_points) * 100
                
                regional_results[region_name]['counts_above'][model_name] = counts_above
                regional_results[region_name]['percentages_above'][model_name] = percentages_above
                
                print(f"    {total_points:,} points")
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Create visualizations for each region
    for region_name in regions:
        print(f"\n--- {region_name.replace('_', ' ').title()} Region ---")
        
        # Create plot for this region
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Absolute counts above thresholds
        x_pos = np.arange(len(percentiles))
        width = 0.8 / len(models_dict)
        
        for i, model_name in enumerate(models_dict.keys()):
            if model_name in regional_results[region_name]['counts_above']:
                color = model_colors[i % len(model_colors)]
                counts = [regional_results[region_name]['counts_above'][model_name][p] for p in percentiles]
                offset = (i - (len(models_dict)-1)/2) * width
                
                ax1.bar(x_pos + offset, counts, width, label=model_name, alpha=0.7, color=color)
        
        ax1.set_xlabel('Percentile Threshold')
        ax1.set_ylabel('Number of Points Above Threshold')
        ax1.set_title(f'{region_name.replace("_", " ").title()}: Absolute Count of Extreme Points')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{p}th' for p in percentiles])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Percentage of points above thresholds
        for i, model_name in enumerate(models_dict.keys()):
            if model_name in regional_results[region_name]['percentages_above']:
                color = model_colors[i % len(model_colors)]
                percentages = [regional_results[region_name]['percentages_above'][model_name][p] for p in percentiles]
                ax2.plot(x_pos, percentages, 'o-', color=color, label=model_name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Percentile Threshold')
        ax2.set_ylabel('Percentage of Points Above Threshold (%)')
        ax2.set_title(f'{region_name.replace("_", " ").title()}: Percentage of Extreme Points')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{p}th' for p in percentiles])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Deviation from expected
        for i, model_name in enumerate(models_dict.keys()):
            if model_name in regional_results[region_name]['percentages_above']:
                color = model_colors[i % len(model_colors)]
                deviations = []
                for p in percentiles:
                    expected = 100 - p
                    actual = regional_results[region_name]['percentages_above'][model_name][p]
                    deviations.append(actual - expected)
                ax3.plot(x_pos, deviations, 's-', color=color, label=model_name, linewidth=2, markersize=6)
        
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Percentile Threshold')
        ax3.set_ylabel('Deviation from Expected (%)')
        ax3.set_title(f'{region_name.replace("_", " ").title()}: Deviation from Expected')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{p}th' for p in percentiles])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Heatmap
        data_for_heatmap = []
        model_names_in_region = []
        for model_name in models_dict.keys():
            if model_name in regional_results[region_name]['percentages_above']:
                model_names_in_region.append(model_name)
                row = [regional_results[region_name]['percentages_above'][model_name][p] for p in percentiles]
                data_for_heatmap.append(row)
        
        if data_for_heatmap:
            data_for_heatmap = np.array(data_for_heatmap)
            im = ax4.imshow(data_for_heatmap, cmap='YlOrRd', aspect='auto')
            for i in range(len(model_names_in_region)):
                for j in range(len(percentiles)):
                    ax4.text(j, i, f'{data_for_heatmap[i, j]:.2f}%', ha="center", va="center", color="black", fontsize=10)
            
            ax4.set_xticks(range(len(percentiles)))
            ax4.set_xticklabels([f'{p}th' for p in percentiles])
            ax4.set_yticks(range(len(model_names_in_region)))
            ax4.set_yticklabels(model_names_in_region)
            ax4.set_title(f'{region_name.replace("_", " ").title()}: Percentage Above Threshold')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.show()
        
        # Print table for this region
        print(f"\n{region_name.replace('_', ' ').title()} Region Analysis:")
        header = f"{'Model':<12} | {'Percentile':>10} | {'Threshold':>10} | {'Points Above':>15} | {'% Above':>10} | {'Expected %':>10} | {'Deviation':>10}"
        print(header)
        print("-" * len(header))
        
        for model_name in models_dict.keys():
            if model_name in regional_results[region_name]['thresholds']:
                print(f"{model_name:<12} | {'':>10} | {'':>10} | {'':>15} | {'':>10} | {'':>10} | {'':>10}")
                for p in percentiles:
                    threshold = regional_results[region_name]['thresholds'][model_name][p]
                    count_above = regional_results[region_name]['counts_above'][model_name][p]
                    percentage_above = regional_results[region_name]['percentages_above'][model_name][p]
                    expected = 100 - p
                    deviation = percentage_above - expected
                    
                    print(f"{'':<12} | {p:>10} | {threshold:>10.3f} | {count_above:>15,} | {percentage_above:>9.3f}% | {expected:>9.1f}% | {deviation:>+9.3f}%")
                print("-" * len(header))
    
    return regional_results