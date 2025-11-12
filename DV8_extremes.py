import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from pathlib import Path
from getpass import getuser
import geopandas as gpd


# ===================================================================================
# HELPER FUNCTIONS FROM DV8_PDFs.py
# ===================================================================================

def extract_data_array(model_value):
    """
    Extract data array from either (dataset, variable_name) tuple or direct data array
    """
    if isinstance(model_value, tuple):
        ds, var_name = model_value
        return ds[var_name]
    else:
        return model_value


def create_shapefile_oceanic_regions_mask(lats, lons, example_sst=None, shapefile_path=None, mask_file=None):
    """
    Create oceanic regions mask using shapefile-based approach
    """
    # Create example_sst from lats and lons
    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons
    
    if example_sst == None:
        example_sst= xr.open_dataset('/home/b/b382616/scratch/mhws/DV8/example_sst.nc')
        example_sst=example_sst.dat_anomaly
    # Default path to shapefile if not provided
    if shapefile_path is None:
        shapefile_path = Path('/scratch') / getuser()[0] / getuser() / 'mhws' / 'DV8' / 'goas_v01.shp'
    
    if mask_file is None:
        save_path = Path('/scratch') / getuser()[0] / getuser() / 'mhws' / 'DV8'
        save_path.mkdir(parents=True, exist_ok=True)
        mask_file = save_path / "region_masks.zarr"
    
    # Check if masks already exist
    if mask_file.exists():
        print("Loading existing masks...")
        region_masks_ds = xr.open_zarr(str(mask_file))
        region_masks = {var: region_masks_ds[var] for var in region_masks_ds.data_vars}
        return region_masks
    
    print("Creating masks from shapefile...")
    region_masks = {}

    # Prepare example SST
    example_sst = example_sst.rio.write_crs("EPSG:4326")

    # Load shapefile
    oceans = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    # ----- Southern Ocean -----
    southern_oceans = oceans[oceans["name"].str.contains("South") | (oceans["name"]=="Indian Ocean")]
    mask_southern = example_sst.rio.clip(southern_oceans.geometry, southern_oceans.crs, drop=False)
    mask_southern_bool = ~xr.ufuncs.isnan(mask_southern)
    region_masks["Southern_Ocean"] = mask_southern_bool & (mask_southern_bool.lat <= -40) & (mask_southern_bool.lat >= -50)

    # ----- Mid/High latitude masks -----
    lat_bands = {
        "MidSouth": (-40, -10),
        "MidNorth": (10, 30),
        "Northern": (30, 70)
    }

    oceans_to_mask = ["North Pacific Ocean", "South Pacific Ocean",
                      "North Atlantic Ocean", "South Atlantic Ocean",
                      "Indian Ocean"]

    for ocean in oceans_to_mask:
        ocean_gdf = oceans[oceans["name"] == ocean]
        if ocean_gdf.empty:
            continue
        mask_da = example_sst.rio.clip(ocean_gdf.geometry, ocean_gdf.crs, drop=False)
        mask_bool = ~xr.ufuncs.isnan(mask_da)

        for band_name, (lat_min, lat_max) in lat_bands.items():
            if ocean == "Indian Ocean" and band_name == "Northern":
                continue
            if band_name == "Northern" and not (ocean.startswith("North") or ocean=="Indian Ocean"):
                continue
            if band_name == "MidSouth" and not (ocean.startswith("South") or ocean=="Indian Ocean"):
                continue
            if band_name == "MidNorth" and ocean.startswith("South"):
                continue

            region_masks[f"{ocean}_{band_name}"] = mask_bool & (mask_bool.lat >= lat_min) & (mask_bool.lat <= lat_max)

    # ----- Equatorial masks -----
    equatorial_oceans = ["Pacific", "Atlantic", "Indian"]
    equatorial_lat = (-10, 10)

    for eq_ocean_name in equatorial_oceans:
        gdf = oceans[oceans["name"].str.contains(eq_ocean_name)]
        if gdf.empty:
            continue
        mask_da = example_sst.rio.clip(gdf.geometry, gdf.crs, drop=False)
        mask_bool = ~xr.ufuncs.isnan(mask_da)
        region_masks[f"{eq_ocean_name}_Equatorial"] = mask_bool & (mask_bool.lat >= equatorial_lat[0]) & (mask_bool.lat <= equatorial_lat[1])

    # ----- Small seas -----
    small_seas = {
        "Mediterranean_Sea": "Mediterranean Region",
        "Baltic_Sea": "Baltic Sea",
        "South_China_Eastern_Archipelagic_Seas": "South China and Easter Archipelagic Seas"
    }

    for key, name in small_seas.items():
        gdf = oceans[oceans["name"] == name]
        if gdf.empty:
            continue
        mask_da = example_sst.rio.clip(gdf.geometry, gdf.crs, drop=False)
        mask_bool = ~xr.ufuncs.isnan(mask_da)
        region_masks[key] = mask_bool

    # ----- MODIFICATIONS -----
    print("Applying region modifications...")
    
    # 1. Pacific Equatorial modification
    if ('Pacific_Equatorial' in region_masks and 
        'South_China_Eastern_Archipelagic_Seas' in region_masks and 
        'Indian_Equatorial' in region_masks):
        
        pacific = region_masks['Pacific_Equatorial']
        south_china = region_masks['South_China_Eastern_Archipelagic_Seas']
        indian = region_masks['Indian_Equatorial']

        lat_mask = (pacific.lat >= -10) & (pacific.lat <= 10)
        pacific_eq = pacific.where(lat_mask, False)
        south_china_eq = south_china.where(lat_mask, False)

        lon_mask = (indian.lon >= 120) & (indian.lon <= 142)
        indian_eq = indian.where(lon_mask, False)

        combined_mask = pacific_eq | south_china_eq | indian_eq
        
        region_masks['Pacific_Equatorial'] = combined_mask
        region_masks['Indian_Equatorial'] = indian.where(~lon_mask, False)

    # 2. North Pacific Subtropics modification
    if ('North Pacific Ocean_MidNorth' in region_masks and 
        'South_China_Eastern_Archipelagic_Seas' in region_masks):
        
        north_pacific = region_masks['North Pacific Ocean_MidNorth']
        south_china = region_masks['South_China_Eastern_Archipelagic_Seas']

        lat_mask = (north_pacific.lat >= 10) & (north_pacific.lat <= 30)

        north_pacific_band = north_pacific.where(lat_mask, False)
        south_china_band = south_china.where(lat_mask, False)

        region_masks['North Pacific Ocean_MidNorth'] = north_pacific_band | south_china_band

    # 3. Indian South Subtropics modification
    if 'Indian Ocean_MidSouth' in region_masks:
        indian_mid_south = region_masks['Indian Ocean_MidSouth']

        lat_mask = (indian_mid_south.lat >= -11) & (indian_mid_south.lat <= -10)
        lon_mask = (indian_mid_south.lon >= 105) & (indian_mid_south.lon <= 130)

        box_mask = lat_mask & lon_mask

        region_masks['Indian Ocean_MidSouth'] = indian_mid_south | box_mask

    # 4. Remove Baltic Sea and South China Sea
    regions_to_remove = ['Baltic_Sea', 'South_China_Eastern_Archipelagic_Seas']
    for region in regions_to_remove:
        if region in region_masks:
            del region_masks[region]

    # Mapping old mask names to new names
    rename_map = {
        "Indian Ocean_MidNorth": "Indian_NorthSubTropics",
        "Indian Ocean_MidSouth": "Indian_SouthSubTropics",
        "North Atlantic Ocean_MidNorth": "North_Atlantic_SubTropics",
        "North Atlantic Ocean_Northern": "North_Atlantic_MiddleLats",
        "North Pacific Ocean_MidNorth": "North_Pacific_SubTropics",
        "North Pacific Ocean_Northern": "North_Pacific_MiddleLats",
        "South Atlantic Ocean_MidSouth": "South_Atlantic_SubTropics",
        "South Pacific Ocean_MidSouth": "South_Pacific_SubTropics"
    }
    
    region_masks = {rename_map.get(k, k): v for k, v in region_masks.items()}

    # Ensure masks are mutually exclusive
    region_masks = ensure_mutually_exclusive_masks(region_masks)

    # Save masks to Zarr
    region_masks_ds = xr.Dataset(region_masks)
    region_masks_ds.to_zarr(str(mask_file))
    print(f"Masks saved to {mask_file}")

    return region_masks


def ensure_mutually_exclusive_masks(region_masks, priority_order=None):
    """
    Ensure that no latitude/longitude point belongs to more than one mask.
    """
    print("Ensuring masks are mutually exclusive...")
    
    if priority_order is None:
        priority_order = [
            'Southern_Ocean',
            'Pacific_Equatorial',
            'Atlantic_Equatorial',
            'Indian_Equatorial',
            'North_Pacific_SubTropics',
            'North_Pacific_MiddleLats',
            'South_Pacific_SubTropics',
            'North_Atlantic_SubTropics',
            'North_Atlantic_MiddleLats',
            'South_Atlantic_SubTropics',
            'Indian_NorthSubTropics',
            'Indian_SouthSubTropics',
            'Mediterranean_Sea'
        ]
    
    bool_masks = {}
    for region_name, mask in region_masks.items():
        bool_masks[region_name] = mask.values if hasattr(mask, 'values') else mask
    
    unique_masks = bool_masks.copy()
    
    total_conflicts = 0
    
    for i, high_priority_region in enumerate(priority_order):
        if high_priority_region not in unique_masks:
            continue
            
        for j, low_priority_region in enumerate(priority_order[i+1:], i+1):
            if low_priority_region not in unique_masks:
                continue
                
            overlap = unique_masks[high_priority_region] & unique_masks[low_priority_region]
            conflict_count = np.sum(overlap)
            
            if conflict_count > 0:
                total_conflicts += conflict_count
                unique_masks[low_priority_region] = unique_masks[low_priority_region] & ~overlap
    
    print(f"Total conflicts resolved: {total_conflicts}")
    
    result_masks = {}
    for region_name, bool_mask in unique_masks.items():
        result_masks[region_name] = xr.DataArray(
            bool_mask,
            dims=('lat', 'lon'),
            coords={
                'lat': region_masks[region_name].lat if hasattr(region_masks[region_name], 'lat') else region_masks[region_name].coords['lat'],
                'lon': region_masks[region_name].lon if hasattr(region_masks[region_name], 'lon') else region_masks[region_name].coords['lon']
            },
            name=region_name
        )
    
    return result_masks


def create_oceanic_regions_mask(lats, lons, method='shapefile', shapefile_path=None):
    """
    Create oceanic regions mask with choice of method
    """
    if method == 'shapefile':
        return create_shapefile_oceanic_regions_mask(lats, lons, shapefile_path=shapefile_path)
    else:
        raise ValueError("Method must be 'shapefile'")


def create_model_specific_masks(models_dict):
    """
    Create masks for each model based on their specific grid
    """
    print("Creating model-specific masks...")
    masks_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Creating masks for {model_name}...")
        
        data_array = extract_data_array(model_value)
        
        lats = data_array.lat.values
        lons = data_array.lon.values
        masks_dict[model_name] = create_oceanic_regions_mask(lats, lons)
        print(f"  {model_name} grid: {lats.shape} x {lons.shape}")
    
    return masks_dict


def get_region_colors_shapefile():
    """
    Get color mapping for shapefile-based regions
    """
    return {
        'Southern_Ocean': 'purple',
        'North_Pacific_SubTropics': 'lightblue',
        'North_Pacific_MiddleLats': 'blue',
        'South_Pacific_SubTropics': 'darkblue',
        'Pacific_Equatorial': 'lightgreen',
        'North_Atlantic_SubTropics': 'yellow',
        'North_Atlantic_MiddleLats': 'orange',
        'South_Atlantic_SubTropics': 'red',
        'Atlantic_Equatorial': 'green',
        'Indian_SouthSubTropics': 'pink',
        'Indian_NorthSubTropics': 'magenta',
        'Indian_Equatorial': 'darkgreen',
        'Mediterranean_Sea': 'cyan'
    }


# ===================================================================================
# EXTREME EVENT FREQUENCY ANALYSIS
# ===================================================================================

def compute_extreme_frequency(extreme_events_da, time_dim='time'):
    """
    Compute frequency of extreme events per grid cell
    
    Parameters:
    -----------
    extreme_events_da : xarray.DataArray
        Boolean array with True where extreme events occurred
        Shape: (time, lat, lon) or (lat, lon)
    time_dim : str
        Name of time dimension
    
    Returns:
    --------
    frequency : xarray.DataArray
        Number of extreme days per grid cell
    """
    if time_dim in extreme_events_da.dims:
        frequency = extreme_events_da.sum(dim=time_dim)
    else:
        frequency = extreme_events_da.astype(int)  # Convert boolean to 0/1
    
    return frequency


# ===================================================================================
# UPDATED EXTREME EVENT FREQUENCY ANALYSIS
# ===================================================================================

def plot_extreme_frequency_maps(models_dict, time_dim='time', normalize=True, 
                                figsize=(15, 8), cmap='viridis', 
                                central_longitude=180, lat_range=(-90, 90),
                                contour_above_mean=True, **kwargs):
    """
    Plot maps of extreme event frequency for multiple models with improved styling
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and:
        - (dataset, 'extreme_events') tuples, OR
        - extreme_events DataArrays directly
    time_dim : str
        Name of time dimension
    normalize : bool
        If True, normalize by number of years to get days/year
    figsize : tuple
        Figure size
    cmap : str
        Colormap for frequency plots
    central_longitude : float
        Central longitude for map projection
    lat_range : tuple
        Latitude range (min, max) for plotting
    contour_above_mean : bool
        If True, add contour lines for values above the mean
    **kwargs : additional arguments for plotting
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    print("PLOTTING EXTREME EVENT FREQUENCY MAPS")
    print("=" * 50)
    
    figs = []
    
    for model_name, model_value in models_dict.items():
        print(f"Processing {model_name}...")
        
        # Extract extreme events data
        if isinstance(model_value, tuple):
            ds, var_name = model_value
            extreme_events = ds['extreme_events']
        else:
            extreme_events = model_value
        
        # Compute frequency
        frequency = compute_extreme_frequency(extreme_events, time_dim=time_dim)
        
        # Normalize to days/year if requested
        if normalize and time_dim in extreme_events.dims:
            n_days = len(extreme_events[time_dim])
            n_years = n_days / 365.25
            frequency_yr = frequency / n_years
            units = "days/year"
            title_units = "per year"
        else:
            frequency_yr = frequency
            units = "days"
            title_units = "total"
        
        # Get coordinates
        lons = frequency_yr.lon.values
        lats = frequency_yr.lat.values
        
        # Apply latitude range
        lat_min, lat_max = lat_range
        if 'lat' in frequency_yr.dims:
            frequency_yr = frequency_yr.sel(lat=slice(lat_min, lat_max))
        
        # Calculate statistics
        total_days = float(frequency.sum().values)
        mean_days = float(frequency_yr.mean().values)
        max_days = float(frequency_yr.max().values)
        
        print(f"  Total extreme days: {total_days:,.0f}")
        print(f"  Mean frequency: {mean_days:.2f} {units}")
        print(f"  Max frequency: {max_days:.1f} {units}")
        
        # Create figure with Pacific-centered projection
        fig, ax = plt.subplots(1, figsize=figsize, 
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
        
        # Plot the main data
        vmin_count, vmax_count = 0, np.nanmax(frequency_yr.values)
        
        # Handle 1D vs 2D coordinates
        if frequency_yr.ndim == 2:
            if lons.ndim == 1 and lats.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                plot_data = frequency_yr.values
            else:
                lon_grid, lat_grid = lons, lats
                plot_data = frequency_yr.values
        else:
            raise ValueError("Frequency data should be 2D (lat, lon)")
        
        # Apply latitude mask to coordinates for plotting
        lat_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max)
        plot_data = np.where(lat_mask, plot_data, np.nan)
        
        h = ax.pcolormesh(lon_grid, lat_grid, plot_data, 
                         cmap=cmap, transform=ccrs.PlateCarree(),
                         vmin=vmin_count, vmax=vmax_count, **kwargs)
        
        # ADD ISOLINES FOR HIGH VALUES (above the mean)
        if contour_above_mean:
            mean_count = np.nanmean(frequency_yr.values)
            print(f"  Mean value: {mean_count:.1f} {units}")
            
            # Create a masked array where we only keep values above the mean
            frequency_high = frequency_yr.where(frequency_yr > mean_count)
            
            # Calculate appropriate levels for contours using only the high range
            high_range = vmax_count - mean_count
            
            # Use fewer, more meaningful levels for better readability
            if high_range < 10:
                levels = np.arange(np.ceil(mean_count), vmax_count + 1, 1)  # Every 1 day
            elif high_range < 20:
                levels = np.arange(np.ceil(mean_count), vmax_count + 2, 2)  # Every 2 days
            else:
                levels = np.arange(np.ceil(mean_count), vmax_count + 5, 5)  # Every 5 days
            
            # Filter out too many levels if needed
            if len(levels) > 8:
                levels = levels[::2]  # Take every other level
            
            print(f"  High-value contour levels (above mean): {levels}")
            
            # In the contour section of plot_extreme_frequency_maps, replace this part:
            
            # Only add contours if we have valid levels
            if len(levels) > 0:
                # Add contour lines using the masked data (only shows where values > mean)
                contours = ax.contour(lon_grid, lat_grid, frequency_high, 
                                    levels=levels, 
                                    colors='red',           # More visible color
                                    linewidths=2,           # Thicker lines
                                    alpha=0.9,              # Less transparent
                                    transform=ccrs.PlateCarree(),
                                    zorder=250)             # Higher zorder than land
            
                # CONTOUR LABELS with higher zorder
                labels = ax.clabel(contours, contours.levels, 
                                inline=True, 
                                fmt='%1.0f', 
                                fontsize=12,                    # Larger font
                                colors='black',                 # Black text
                                inline_spacing=8)               # More space around text
                
                # Ensure labels are on top of everything
                if labels:
                    for text_obj in labels:
                        text_obj.set_zorder(300)  # Very high zorder for labels
        
        # Add map features
        ax.coastlines(linewidth=0.8, color='grey')
        ax.add_feature(cfeature.LAND, zorder=200, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.set_extent([-180, 180, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Set title with statistics
        ax.set_title(f'Extreme Event Frequency\n'
                    f'Total: {total_days:,.0f} days | Mean: {mean_days:.1f} {title_units} | Max: {max_days:.1f} {title_units}', 
                    fontsize=20, pad=15)
        
        # Add model name in red (moved up to avoid overlap with colorbar)
        ax.text(0.9, 1.16, model_name, transform=ax.transAxes,
               fontsize=25, fontweight='bold', color='red', ha='center')
        
        # Add colorbar with integer ticks
        cbar = fig.colorbar(h, ax=ax, orientation='vertical', fraction=0.025, pad=0.015)
        cbar.set_label(f'Days {title_units}', fontsize=14)
        min_count = int(np.ceil(vmin_count))
        max_count = int(np.floor(vmax_count))
        cbar_ticks = np.linspace(min_count, max_count, min(10, max_count-min_count+1), dtype=int, endpoint=True)
        cbar.set_ticks(cbar_ticks)
        
        # Add a note about the contours if used
        if contour_above_mean:
            ax.text(0.02, -0.1, f'Contours show values > mean ({mean_count:.0f} {units})', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   zorder=250)
        
        fig.tight_layout()
        figs.append(fig)
    
    return figs

# ===================================================================================
# SEPARATE HISTOGRAMS FOR TOTAL AND YEARLY EXTREMES
# ===================================================================================

def plot_total_extremes_histogram(models_dict, time_dim='time', 
                                  figsize=(12, 8), color='skyblue', 
                                  title="Total Extreme Days by Model"):
    """
    Plot histogram of total number of extreme events per model
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and extreme_events data
    time_dim : str
        Name of time dimension
    figsize : tuple
        Figure size
    color : str
        Bar color
    title : str
        Plot title
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The histogram figure
    ax : matplotlib.axes.Axes
        The axes object
    totals_dict : dict
        Dictionary with model names and their total extreme counts
    """
    print("PLOTTING TOTAL EXTREMES HISTOGRAM")
    print("=" * 40)
    
    totals_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Processing {model_name}...")
        
        # Extract extreme events data
        if isinstance(model_value, tuple):
            ds, var_name = model_value
            extreme_events = ds['extreme_events']
        else:
            extreme_events = model_value
        
        # Compute total extreme days (NOT normalized)
        total_extremes = extreme_events.sum().compute().values
        totals_dict[model_name] = total_extremes
        
        # Calculate time period info for context
        n_days = len(extreme_events[time_dim]) if time_dim in extreme_events.dims else 1
        n_years = n_days / 365.25
        
        print(f"  Total: {total_extremes:,.0f} days over {n_years:.1f} years")
    
    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(totals_dict.keys())
    totals = list(totals_dict.values())
    
    bars = ax.bar(models, totals, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(totals)*0.01,
                f'{total:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Total Extreme Days', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if many models
    if len(models) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax, totals_dict

def plot_yearly_extremes_histogram(models_dict, time_dim='time',
                                   figsize=(12, 8), color='lightcoral',
                                   title="Average Extreme Days per Year by Model"):
    """
    Plot histogram of average number of extreme events per year per model
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and extreme_events data
    time_dim : str
        Name of time dimension
    figsize : tuple
        Figure size
    color : str
        Bar color
    title : str
        Plot title
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The histogram figure
    ax : matplotlib.axes.Axes
        The axes object
    yearly_dict : dict
        Dictionary with model names and their yearly average extreme counts
    """
    print("PLOTTING YEARLY AVERAGE EXTREMES HISTOGRAM")
    print("=" * 50)
    
    yearly_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Processing {model_name}...")
        
        # Extract extreme events data
        if isinstance(model_value, tuple):
            ds, var_name = model_value
            extreme_events = ds['extreme_events']
        else:
            extreme_events = model_value
        
        # Compute total extreme days
        total_extremes = extreme_events.sum().compute().values
        
        # Calculate number of years
        if time_dim in extreme_events.dims:
            n_days = len(extreme_events[time_dim])
            n_years = n_days / 365.25
            yearly_avg = total_extremes / n_years
        else:
            n_years = 1
            yearly_avg = total_extremes
        
        yearly_dict[model_name] = yearly_avg
        print(f"  Yearly average: {yearly_avg:.1f} days/year over {n_years:.1f} years")
    
    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(yearly_dict.keys())
    yearly_avgs = list(yearly_dict.values())
    
    bars = ax.bar(models, yearly_avgs, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for bar, avg in zip(bars, yearly_avgs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(yearly_avgs)*0.01,
                f'{avg:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Extreme Days per Year', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if many models
    if len(models) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax, yearly_dict

def plot_both_extremes_histograms(models_dict, time_dim='time',
                                  figsize=(15, 6), 
                                  color_total='skyblue',
                                  color_yearly='lightcoral'):
    """
    Plot both total and yearly extremes histograms side by side
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and extreme_events data
    time_dim : str
        Name of time dimension
    figsize : tuple
        Figure size for the combined plot
    color_total : str
        Color for total extremes bars
    color_yearly : str
        Color for yearly extremes bars
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The combined figure
    axes : tuple
        (ax_total, ax_yearly) axes objects
    results_dict : dict
        Dictionary with both total and yearly results
    """
    print("PLOTTING BOTH TOTAL AND YEARLY EXTREMES")
    print("=" * 50)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Compute data for both plots
    totals_dict = {}
    yearly_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Processing {model_name}...")
        
        # Extract extreme events data
        if isinstance(model_value, tuple):
            ds, var_name = model_value
            extreme_events = ds['extreme_events']
        else:
            extreme_events = model_value
        
        # Compute total extreme days
        total_extremes = extreme_events.sum().compute().values
        totals_dict[model_name] = total_extremes
        
        # Calculate yearly average
        if time_dim in extreme_events.dims:
            n_days = len(extreme_events[time_dim])
            n_years = n_days / 365.25
            yearly_avg = total_extremes / n_years
        else:
            n_years = 1
            yearly_avg = total_extremes
        
        yearly_dict[model_name] = yearly_avg
        print(f"  Total: {total_extremes:,.0f} days, Yearly: {yearly_avg:.1f} days/year")
    
    models = list(totals_dict.keys())
    totals = list(totals_dict.values())
    yearly_avgs = list(yearly_dict.values())
    
    # Plot 1: Total extremes
    bars1 = ax1.bar(models, totals, color=color_total, alpha=0.7, edgecolor='black', linewidth=1)
    for bar, total in zip(bars1, totals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(totals)*0.01,
                f'{total:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Total Extreme Days', fontsize=12)
    ax1.set_title('Total Extreme Days by Model', fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Yearly averages
    bars2 = ax2.bar(models, yearly_avgs, color=color_yearly, alpha=0.7, edgecolor='black', linewidth=1)
    for bar, avg in zip(bars2, yearly_avgs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(yearly_avgs)*0.01,
                f'{avg:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Average Extreme Days per Year', fontsize=12)
    ax2.set_title('Average Extreme Days per Year by Model', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if many models
    if len(models) > 4:
        for ax in [ax1, ax2]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    results_dict = {
        'totals': totals_dict,
        'yearly_averages': yearly_dict
    }
    
    return fig, (ax1, ax2), results_dict

# ===================================================================================
# UPDATED QUICK ANALYSIS FUNCTIONS
# ===================================================================================

def quick_total_extremes_comparison(models_dict, time_dim='time'):
    """
    Quick comparison of total extremes across models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with extreme_events data
    time_dim : str
        Name of time dimension
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Histogram figure
    ax : matplotlib.axes.Axes
        Axes object
    totals_dict : dict
        Dictionary with model totals
    """
    print("QUICK TOTAL EXTREMES COMPARISON")
    print("=" * 35)
    
    fig, ax, totals_dict = plot_total_extremes_histogram(
        models_dict, time_dim=time_dim
    )
    
    return fig, ax, totals_dict

def quick_yearly_extremes_comparison(models_dict, time_dim='time'):
    """
    Quick comparison of yearly average extremes across models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with extreme_events data
    time_dim : str
        Name of time dimension
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Histogram figure
    ax : matplotlib.axes.Axes
        Axes object
    yearly_dict : dict
        Dictionary with model yearly averages
    """
    print("QUICK YEARLY EXTREMES COMPARISON")
    print("=" * 40)
    
    fig, ax, yearly_dict = plot_yearly_extremes_histogram(
        models_dict, time_dim=time_dim
    )
    
    return fig, ax, yearly_dict

def quick_both_extremes_comparison(models_dict, time_dim='time'):
    """
    Quick comparison of both total and yearly extremes across models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with extreme_events data
    time_dim : str
        Name of time dimension
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Combined histogram figure
    axes : tuple
        (ax_total, ax_yearly) axes objects
    results_dict : dict
        Dictionary with both total and yearly results
    """
    print("QUICK BOTH TOTAL AND YEARLY EXTREMES COMPARISON")
    print("=" * 55)
    
    fig, axes, results_dict = plot_both_extremes_histograms(
        models_dict, time_dim=time_dim
    )
    
    return fig, axes, results_dict



# ===================================================================================
# REGIONAL EXTREMES ANALYSIS
# ===================================================================================

def compute_regional_extremes(models_dict, time_dim='time', normalize=True, regions=None, per_grid_cell=True):
    """
    Compute extreme events for each region and model
    
    Parameters:
    -----------
    per_grid_cell : bool
        If True, return average per grid cell. If False, return regional total.
    """
    print("COMPUTING REGIONAL EXTREMES")
    print("=" * 35)
    
    # Create model-specific masks
    masks_dict = create_model_specific_masks(models_dict)
    
    # Get regions to analyze
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    regional_data = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Processing {model_name}...")
        
        # Extract extreme events data
        extreme_events = extract_data_array(model_value)
        
        # Calculate number of years for normalization
        if normalize and time_dim in extreme_events.dims:
            n_days = len(extreme_events[time_dim])
            n_years = n_days / 365.25
        else:
            n_years = 1
        
        regional_data[model_name] = {}
        
        for region_name in regions:
            if region_name not in masks_dict[model_name]:
                print(f"  Warning: Region {region_name} not found for {model_name}, skipping...")
                continue
            
            # Get mask for this region and model
            mask = masks_dict[model_name][region_name]
            
            # Count number of grid cells in this region
            n_gridcells = mask.sum().compute().values
            if n_gridcells == 0:
                print(f"  Warning: Region {region_name} has 0 grid cells, skipping...")
                continue
            
            # Apply mask to extreme events data
            regional_extremes = extreme_events.where(mask)
            
            # Compute total extreme days in this region
            total_regional_extremes = regional_extremes.sum().compute().values
            
            # NORMALIZE by number of grid cells if requested
            if per_grid_cell:
                regional_value = total_regional_extremes / n_gridcells
            else:
                regional_value = total_regional_extremes
            
            # Normalize to days/year if requested
            if normalize and time_dim in extreme_events.dims:
                regional_avg = regional_value / n_years
                if per_grid_cell:
                    units = "days/year/gridcell"
                else:
                    units = "days/year"
            else:
                regional_avg = regional_value
                if per_grid_cell:
                    units = "total days/gridcell"
                else:
                    units = "total days"
            
            regional_data[model_name][region_name] = regional_avg
            
            print(f"  {region_name}: {regional_avg:.1f} {units} ({(n_gridcells)} grid cells)")
    
    return regional_data, masks_dict



def plot_regional_extremes_barchart(regional_data, figsize=(16, 10), 
                                    cmap='tab20', title="Average Extreme Events per Year by Region and Model"):
    """
    Plot barchart of regional extremes for all models
    
    Parameters:
    -----------
    regional_data : dict
        Output from compute_regional_extremes
    figsize : tuple
        Figure size
    cmap : str
        Colormap for different regions
    title : str
        Plot title
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The bar chart figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    print("PLOTTING REGIONAL EXTREMES BARCHART")
    print("=" * 45)
    
    # Get models and regions
    models = list(regional_data.keys())
    regions = list(regional_data[models[0]].keys())
    
    # Create color map for regions
    colors = plt.cm.get_cmap(cmap, len(regions))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot settings
    bar_width = 0.8 / len(models)  # Dynamic width based on number of models
    x_pos = np.arange(len(regions))
    
    # Plot bars for each model
    for i, model_name in enumerate(models):
        model_values = [regional_data[model_name][region] for region in regions]
        
        # Calculate position for this model's bars
        offset = (i - len(models)/2 + 0.5) * bar_width
        positions = x_pos + offset
        
        bars = ax.bar(positions, model_values, bar_width, 
                     label=model_name, 
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Oceanic Regions', fontsize=12)
    ax.set_ylabel('Average Extreme Days per Year', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.replace('_', ' ').title() for r in regions], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add some statistics to the plot - MOVED BELOW THE LEGEND
    total_stats = []
    for model_name in models:
        model_totals = [regional_data[model_name][region] for region in regions]
        total_extremes = sum(model_totals)
        avg_extremes = np.mean(model_totals)
        total_stats.append(f"{model_name}: Total={total_extremes:.0f}, Avg={avg_extremes:.1f}")
    
    stats_text = "\n".join(total_stats)
    
    # Position the stats text box below the legend (adjust y position as needed)
    ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, ax

def plot_regional_extremes_heatmap(regional_data, figsize=(14, 10), 
                                   cmap='YlOrRd', title="Regional Extreme Events Heatmap"):
    """
    Plot heatmap of regional extremes across models
    
    Parameters:
    -----------
    regional_data : dict
        Output from compute_regional_extremes
    figsize : tuple
        Figure size
    cmap : str
        Colormap for heatmap
    title : str
        Plot title
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The heatmap figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    print("PLOTTING REGIONAL EXTREMES HEATMAP")
    print("=" * 40)
    
    # Get models and regions
    models = list(regional_data.keys())
    regions = list(regional_data[models[0]].keys())
    
    # Create data matrix for heatmap
    data_matrix = np.zeros((len(regions), len(models)))
    
    for i, region in enumerate(regions):
        for j, model in enumerate(models):
            data_matrix[i, j] = regional_data[model][region]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(regions)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticklabels([r.replace('_', ' ').title() for r in regions])
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Average Extreme Days per Year', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(regions)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9,
                          fontweight='bold')
    
    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    return fig, ax

def plot_regional_comparison_single_model(regional_data, model_name, 
                                          figsize=(12, 8), color='steelblue',
                                          title_template="Regional Extreme Events - {}"):
    """
    Plot regional extremes for a single model
    
    Parameters:
    -----------
    regional_data : dict
        Output from compute_regional_extremes
    model_name : str
        Name of model to plot
    figsize : tuple
        Figure size
    color : str
        Bar color
    title_template : str
        Title template (will be formatted with model_name)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The bar chart figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    if model_name not in regional_data:
        raise ValueError(f"Model '{model_name}' not found in regional data")
    
    print(f"PLOTTING REGIONAL EXTREMES FOR {model_name}")
    print("=" * (35 + len(model_name)))
    
    model_data = regional_data[model_name]
    regions = list(model_data.keys())
    values = list(model_data.values())
    
    # Sort regions by value (descending)
    sorted_indices = np.argsort(values)[::-1]
    regions_sorted = [regions[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(len(regions_sorted)), values_sorted, color=color, 
                  alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values_sorted):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values_sorted)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Oceanic Regions', fontsize=12)
    ax.set_ylabel('Average Extreme Days per Year', fontsize=12)
    ax.set_title(title_template.format(model_name), fontsize=14, pad=20)
    ax.set_xticks(range(len(regions_sorted)))
    ax.set_xticklabels([r.replace('_', ' ').title() for r in regions_sorted], 
                       rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    total_extremes = sum(values_sorted)
    avg_extremes = np.mean(values_sorted)
    max_region = regions_sorted[0]
    max_value = values_sorted[0]
    
    stats_text = f"Total: {total_extremes:.0f} days/year\nAverage: {avg_extremes:.1f} days/year\nMax: {max_value:.1f} days/year ({max_region.replace('_', ' ').title()})"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, ax

# ===================================================================================
# QUICK ANALYSIS FUNCTIONS FOR REGIONAL EXTREMES
# ===================================================================================

def quick_regional_extremes_analysis(models_dict, time_dim='time', plot_type='barchart', 
                                     regions=None, per_grid_cell=True, **kwargs):
    """
    Quick analysis of regional extremes across models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with extreme_events data
    time_dim : str
        Name of time dimension
    plot_type : str
        Type of plot: 'barchart', 'heatmap', or 'single'
    regions : list, optional
        Specific regions to analyze
    per_grid_cell : bool
        If True, normalize by number of grid cells in each region
    **kwargs : additional arguments for plotting functions
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The plot figure
    ax : matplotlib.axes.Axes
        The axes object
    regional_data : dict
        Dictionary with regional extremes data
    """
    print("QUICK REGIONAL EXTREMES ANALYSIS")
    print("=" * 35)
    
    # Compute regional data with per_grid_cell parameter
    regional_data, masks_dict = compute_regional_extremes(
        models_dict, time_dim=time_dim, regions=regions, per_grid_cell=per_grid_cell
    )
    
    # Create appropriate plot
    if plot_type == 'barchart':
        fig, ax = plot_regional_extremes_barchart(regional_data, **kwargs)
    elif plot_type == 'heatmap':
        fig, ax = plot_regional_extremes_heatmap(regional_data, **kwargs)
    elif plot_type == 'single':
        # For single model plot, use first model by default
        model_name = list(regional_data.keys())[0]
        fig, ax = plot_regional_comparison_single_model(regional_data, model_name, **kwargs)
    else:
        raise ValueError("plot_type must be 'barchart', 'heatmap', or 'single'")
    
    return fig, ax, regional_data



def quick_all_regional_plots(models_dict, time_dim='time', regions=None, figsize_multiplier=1.0):
    """
    Create all regional plots for comprehensive analysis
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with extreme_events data
    time_dim : str
        Name of time dimension
    regions : list, optional
        Specific regions to analyze
    figsize_multiplier : float
        Multiplier for figure sizes
    
    Returns:
    --------
    figs : dict
        Dictionary of figure objects
    axes : dict
        Dictionary of axes objects
    regional_data : dict
        Dictionary with regional extremes data
    """
    print("COMPREHENSIVE REGIONAL EXTREMES ANALYSIS")
    print("=" * 50)
    
    # Compute regional data once
    regional_data, masks_dict = compute_regional_extremes(
        models_dict, time_dim=time_dim, regions=regions
    )
    
    figs = {}
    axes = {}
    
    # 1. Combined barchart
    figs['barchart'], axes['barchart'] = plot_regional_extremes_barchart(
        regional_data, 
        figsize=(16 * figsize_multiplier, 10 * figsize_multiplier)
    )
    
    # 2. Heatmap
    figs['heatmap'], axes['heatmap'] = plot_regional_extremes_heatmap(
        regional_data,
        figsize=(14 * figsize_multiplier, 10 * figsize_multiplier)
    )
    
    # 3. Individual model plots
    figs['individual'] = {}
    axes['individual'] = {}
    
    for model_name in regional_data.keys():
        figs['individual'][model_name], axes['individual'][model_name] = \
            plot_regional_comparison_single_model(
                regional_data, model_name,
                figsize=(12 * figsize_multiplier, 8 * figsize_multiplier)
            )
    
    return figs, axes, regional_data