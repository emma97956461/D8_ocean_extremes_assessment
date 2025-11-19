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
# HELPER FUNCTIONS
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
        # Get coordinates from original mask
        original_mask = region_masks[region_name]
        result_masks[region_name] = xr.DataArray(
            bool_mask,
            dims=('lat', 'lon'),
            coords={
                'lat': original_mask.lat if hasattr(original_mask, 'lat') else original_mask.coords['lat'],
                'lon': original_mask.lon if hasattr(original_mask, 'lon') else original_mask.coords['lon']
            },
            name=region_name
        )
    
    return result_masks


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
# MODEL-SPECIFIC MASK CREATION FUNCTIONS
# ===================================================================================

def create_model_specific_shapefile_mask(data_array, model_name=None, shapefile_path=None, mask_save_dir=None):
    """
    Create oceanic regions mask for a specific model's grid using shapefile-based approach
    """
    # Get coordinates from the model data
    lats = data_array.lat.values
    lons = data_array.lon.values
    
    # Create example_sst from model's grid
    if lats.ndim == 1 and lons.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons
    
    # Create a dummy DataArray with the model's grid
    example_sst = xr.DataArray(
        np.ones(lat_grid.shape),
        dims=['lat', 'lon'],
        coords={'lat': (['lat', 'lon'], lat_grid), 'lon': (['lat', 'lon'], lon_grid)}
    )
    
    # Default path to shapefile if not provided
    if shapefile_path is None:
        shapefile_path = Path('/scratch') / getuser()[0] / getuser() / 'mhws' / 'DV8' / 'goas_v01.shp'
    
    # Create model-specific mask file path
    if mask_save_dir is None:
        mask_save_dir = Path('/scratch') / getuser()[0] / getuser() / 'mhws' / 'DV8' / 'model_masks'
    
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique identifier for this model's grid
    grid_hash = f"model_{lats.shape[0]}x{lons.shape[0]}_{hash(str(lats[:5].tobytes()) + str(lons[:5].tobytes()))}"
    mask_file = mask_save_dir / f"{grid_hash}_region_masks.zarr"
    
    # Check if masks already exist for this specific grid
    if mask_file.exists():
        print(f"Loading existing masks for model grid {lats.shape} x {lons.shape}...")
        region_masks_ds = xr.open_zarr(str(mask_file))
        region_masks = {var: region_masks_ds[var] for var in region_masks_ds.data_vars}
        return region_masks
    
    print(f"Creating masks from shapefile for model grid {lats.shape} x {lons.shape}...")
    
    # Try to load model-specific example SST if available
    model_example_sst = None
    if model_name is not None:
        # Construct possible example SST file names
        shapefile_dir = Path(shapefile_path).parent
        possible_files = [
            shapefile_dir / f"example_sst_{model_name}.nc",
            shapefile_dir / f"example_sst_{model_name.upper()}.nc", 
            shapefile_dir / f"example_sst_{model_name.lower()}.nc",
            shapefile_dir / f"example_sst_{model_name.replace('-', '_')}.nc",
            shapefile_dir / f"example_sst_{model_name.replace('_', '')}.nc",
        ]
        
        # Also try variations for common model name formats
        if 'IFS' in model_name or 'FESOM' in model_name:
            possible_files.extend([
                shapefile_dir / "example_sst_IFSFESOM.nc",
                shapefile_dir / "example_sst_ifsfesom.nc",
            ])
        
        for example_file in possible_files:
            if example_file.exists():
                print(f"  Found model-specific example SST: {example_file}")
                try:
                    model_example_sst = xr.open_dataset(str(example_file))
                    # Extract the first data variable if it's a Dataset
                    if isinstance(model_example_sst, xr.Dataset):
                        data_vars = list(model_example_sst.data_vars.keys())
                        model_example_sst = model_example_sst[data_vars[0]]
                    
                    # Ensure the DataArray has proper spatial dimensions
                    if hasattr(model_example_sst, 'lat') and hasattr(model_example_sst, 'lon'):
                        # Rename dimensions to standard names if needed
                        dims_mapping = {}
                        for dim in model_example_sst.dims:
                            if 'lat' in dim.lower() and dim != 'lat':
                                dims_mapping[dim] = 'lat'
                            elif 'lon' in dim.lower() and dim != 'lon':
                                dims_mapping[dim] = 'lon'
                        
                        if dims_mapping:
                            model_example_sst = model_example_sst.rename(dims_mapping)
                        
                        # Set spatial dimensions for rioxarray
                        model_example_sst = model_example_sst.rio.set_spatial_dims('lon', 'lat')
                    
                    break
                except Exception as e:
                    print(f"  Warning: Could not load {example_file}: {e}")
                    continue
    
    # Use model-specific example SST if available, otherwise use the dummy one
    if model_example_sst is not None:
        print(f"  Using model-specific example SST for {model_name}")
        example_sst = model_example_sst
    else:
        print(f"  Using generated grid for {model_name}")
    
    # Ensure the example_sst has proper CRS and spatial dimensions
    try:
        example_sst = example_sst.rio.set_spatial_dims('lon', 'lat')
        example_sst = example_sst.rio.write_crs("EPSG:4326")
    except Exception as e:
        print(f"  Warning: Could not set CRS for example_sst: {e}")
        print("  Continuing without CRS...")
    
    region_masks = {}
    
    # Load shapefile
    oceans = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    # ----- Southern Ocean -----
    print("  Creating Southern Ocean mask...")
    southern_oceans = oceans[oceans["name"].str.contains("South") | (oceans["name"]=="Indian Ocean")]
    try:
        mask_southern = example_sst.rio.clip(southern_oceans.geometry, southern_oceans.crs, drop=False)
        mask_southern_bool = ~xr.ufuncs.isnan(mask_southern)
        region_masks["Southern_Ocean"] = mask_southern_bool & (mask_southern_bool.lat <= -40) & (mask_southern_bool.lat >= -50)
    except Exception as e:
        print(f"  Error creating Southern Ocean mask: {e}")
        # Fallback: create a simple mask based on latitude
        lat_mask = (example_sst.lat >= -50) & (example_sst.lat <= -40)
        region_masks["Southern_Ocean"] = xr.where(lat_mask, True, False)

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
        print(f"  Creating mask for {ocean}...")
        ocean_gdf = oceans[oceans["name"] == ocean]
        if ocean_gdf.empty:
            continue
            
        try:
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
        except Exception as e:
            print(f"  Error creating mask for {ocean}: {e}")

    # ----- Equatorial masks -----
    equatorial_oceans = ["Pacific", "Atlantic", "Indian"]
    equatorial_lat = (-10, 10)

    for eq_ocean_name in equatorial_oceans:
        print(f"  Creating equatorial mask for {eq_ocean_name}...")
        gdf = oceans[oceans["name"].str.contains(eq_ocean_name)]
        if gdf.empty:
            continue
            
        try:
            mask_da = example_sst.rio.clip(gdf.geometry, gdf.crs, drop=False)
            mask_bool = ~xr.ufuncs.isnan(mask_da)
            region_masks[f"{eq_ocean_name}_Equatorial"] = mask_bool & (mask_bool.lat >= equatorial_lat[0]) & (mask_bool.lat <= equatorial_lat[1])
        except Exception as e:
            print(f"  Error creating equatorial mask for {eq_ocean_name}: {e}")

    # ----- Small seas -----
    small_seas = {
        "Mediterranean_Sea": "Mediterranean Region",
        "Baltic_Sea": "Baltic Sea",
        "South_China_Eastern_Archipelagic_Seas": "South China and Easter Archipelagic Seas"
    }

    for key, name in small_seas.items():
        print(f"  Creating mask for {name}...")
        gdf = oceans[oceans["name"] == name]
        if gdf.empty:
            continue
            
        try:
            mask_da = example_sst.rio.clip(gdf.geometry, gdf.crs, drop=False)
            mask_bool = ~xr.ufuncs.isnan(mask_da)
            region_masks[key] = mask_bool
        except Exception as e:
            print(f"  Error creating mask for {name}: {e}")

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

    # Save model-specific masks to Zarr
    region_masks_ds = xr.Dataset(region_masks)
    region_masks_ds.to_zarr(str(mask_file))
    print(f"Model-specific masks saved to {mask_file}")

    return region_masks


def create_model_specific_masks(models_dict, shapefile_path=None, mask_save_dir=None, models_example_sst=None):
    """
    Create masks for each model based on their specific grid
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and model data as values
    shapefile_path : str or Path, optional
        Path to the shapefile
    mask_save_dir : str or Path, optional
        Directory to save model-specific masks
    models_example_sst : dict, optional
        Dictionary with model names as keys and example SST data as values
        If not provided, will try to load from files automatically
    
    Returns:
    --------
    masks_dict : dict
        Dictionary with model names as keys and their region masks as values
    """
    print("Creating model-specific masks...")
    masks_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Creating masks for {model_name}...")
        
        data_array = extract_data_array(model_value)
        
        # Create masks specifically for this model's grid
        masks_dict[model_name] = create_model_specific_shapefile_mask(
            data_array, 
            model_name=model_name,
            shapefile_path=shapefile_path,
            mask_save_dir=mask_save_dir
        )
        
        # Print grid information
        lats = data_array.lat.values
        lons = data_array.lon.values
        print(f"  {model_name} grid: {lats.shape} x {lons.shape}")
        print(f"  {model_name} regions: {list(masks_dict[model_name].keys())}")
    
    return masks_dict


# ===================================================================================
# MISSING FUNCTIONS THAT NEED TO BE ADDED
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


def compute_regional_extremes(models_dict, time_dim='time', normalize=True, regions=None, per_grid_cell=True, shapefile_path=None, mask_save_dir=None):
    """
    Compute extreme events for each region and model using model-specific masks
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and model data as values
    time_dim : str
        Name of time dimension
    normalize : bool
        If True, normalize by number of years
    regions : list, optional
        Specific regions to analyze
    per_grid_cell : bool
        If True, return average per grid cell. If False, return regional total.
    shapefile_path : str or Path, optional
        Path to the shapefile for mask creation
    mask_save_dir : str or Path, optional
        Directory to save model-specific masks
    
    Returns:
    --------
    regional_data : dict
        Dictionary with regional extreme data for each model
    masks_dict : dict
        Dictionary with model-specific masks
    """
    print("COMPUTING REGIONAL EXTREMES WITH MODEL-SPECIFIC MASKS")
    print("=" * 50)
    
    # Create model-specific masks
    masks_dict = create_model_specific_masks(
        models_dict, 
        shapefile_path=shapefile_path,
        mask_save_dir=mask_save_dir
    )
    
    # Get regions to analyze (use first model's regions as reference)
    if regions is None:
        first_model = list(masks_dict.keys())[0]
        regions = list(masks_dict[first_model].keys())
    
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
            
            print(f"  {region_name}: {regional_avg:.1f} {units} ({n_gridcells} grid cells)")
    
    return regional_data, masks_dict


# ===================================================================================
# MASK VISUALIZATION FUNCTIONS 
# ===================================================================================

def plot_model_masks(masks_dict, model_name, figsize=(15, 10), central_longitude=180):
    """
    Plot all region masks for a specific model
    """
    if model_name not in masks_dict:
        raise ValueError(f"Model '{model_name}' not found in masks dictionary")
    
    model_masks = masks_dict[model_name]
    regions = list(model_masks.keys())
    
    # Calculate grid size for subplots
    n_regions = len(regions)
    n_cols = 4
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get region colors
    region_colors = get_region_colors_shapefile()
    
    for idx, region_name in enumerate(regions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        mask = model_masks[region_name]
        
        # Get color for this region
        color = region_colors.get(region_name, 'gray')
        
        # Plot the mask
        if hasattr(mask, 'lat') and hasattr(mask, 'lon'):
            if mask.lat.ndim == 1 and mask.lon.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(mask.lon, mask.lat)
                mask_data = mask.values
            else:
                lon_grid, lat_grid = mask.lon, mask.lat
                mask_data = mask.values
        else:
            # Fallback if coordinates are not clear
            lon_grid, lat_grid = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
            mask_data = mask.values
        
        # Plot mask regions in color, non-mask areas in light blue (ocean)
        im = ax.pcolormesh(lon_grid, lat_grid, np.where(mask_data, 1, 0),
                          cmap=ListedColormap(['lightblue', color]),
                          transform=ccrs.PlateCarree(),
                          vmin=0, vmax=1)
        
        # Add map features
        ax.coastlines(linewidth=0.5, color='black')
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
        ax.set_global()
        ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
        
        # Add title
        ax.set_title(region_name.replace('_', ' ').title(), fontsize=10, pad=5)
    
    # Hide unused subplots
    for idx in range(len(regions), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'Oceanic Region Masks - {model_name}', fontsize=16, y=0.95)
    plt.tight_layout()
    
    return fig, axes

def plot_combined_regions_mask(masks_dict, model_name, figsize=(12, 8), central_longitude=180):
    """
    Plot a combined map showing all regions with different colors
    """
    if model_name not in masks_dict:
        raise ValueError(f"Model '{model_name}' not found in masks dictionary")
    
    model_masks = masks_dict[model_name]
    regions = list(model_masks.keys())
    
    fig, ax = plt.subplots(figsize=figsize,
                          subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Get region colors
    region_colors = get_region_colors_shapefile()
    
    # Create a combined array where each region has a unique value
    first_mask = model_masks[regions[0]]
    combined_data = np.zeros(first_mask.shape, dtype=int)
    
    # Assign unique values to each region
    for idx, region_name in enumerate(regions):
        mask = model_masks[region_name]
        combined_data = np.where(mask.values, idx + 1, combined_data)
    
    # Create colormap for all regions
    colors = [region_colors.get(region, 'gray') for region in regions]
    cmap = ListedColormap(colors)
    
    # Get coordinates
    if hasattr(first_mask, 'lat') and hasattr(first_mask, 'lon'):
        if first_mask.lat.ndim == 1 and first_mask.lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(first_mask.lon, first_mask.lat)
        else:
            lon_grid, lat_grid = first_mask.lon, first_mask.lat
    else:
        lon_grid, lat_grid = np.meshgrid(np.arange(first_mask.shape[1]), np.arange(first_mask.shape[0]))
    
    # Plot combined data
    im = ax.pcolormesh(lon_grid, lat_grid, combined_data,
                      cmap=cmap,
                      vmin=0.5, vmax=len(regions) + 0.5,
                      transform=ccrs.PlateCarree())
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Create legend
    legend_patches = []
    for region_name, color in zip(regions, colors):
        patch = mpatches.Patch(color=color, label=region_name.replace('_', ' ').title())
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, 
              loc='center left', 
              bbox_to_anchor=(1.05, 0.5),
              frameon=True,
              fancybox=True,
              shadow=True)
    
    ax.set_title(f'Combined Oceanic Regions - {model_name}', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    return fig, ax


def quick_visualize_masks(masks_dict, model_name=None):
    """
    Quick visualization of masks for a model or all models
    """
    if model_name is None:
        model_name = list(masks_dict.keys())[0]
    
    print(f"Visualizing masks for {model_name}")
    print("=" * 50)
    
    # 1. Combined regions plot
    fig1, ax1 = plot_combined_regions_mask(masks_dict, model_name)
    plt.show()
    
    # 2. Individual masks plot
    fig2, axes2 = plot_model_masks(masks_dict, model_name)
    plt.show()
    
    return fig1, fig2







# ===================================================================================
# EXTREME EVENT FREQUENCY ANALYSIS (UPDATED FOR MODEL-SPECIFIC MASKS)
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


def compute_regional_extremes(models_dict, time_dim='time', normalize=True, regions=None, per_grid_cell=True, shapefile_path=None, mask_save_dir=None):
    """
    Compute extreme events for each region and model using model-specific masks
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and model data as values
    time_dim : str
        Name of time dimension
    normalize : bool
        If True, normalize by number of years
    regions : list, optional
        Specific regions to analyze
    per_grid_cell : bool
        If True, return average per grid cell. If False, return regional total.
    shapefile_path : str or Path, optional
        Path to the shapefile for mask creation
    mask_save_dir : str or Path, optional
        Directory to save model-specific masks
    
    Returns:
    --------
    regional_data : dict
        Dictionary with regional extreme data for each model
    masks_dict : dict
        Dictionary with model-specific masks
    """
    print("COMPUTING REGIONAL EXTREMES WITH MODEL-SPECIFIC MASKS")
    print("=" * 50)
    
    # Create model-specific masks
    masks_dict = create_model_specific_masks(
        models_dict, 
        shapefile_path=shapefile_path,
        mask_save_dir=mask_save_dir
    )
    
    # Get regions to analyze (use first model's regions as reference)
    if regions is None:
        first_model = list(masks_dict.keys())[0]
        regions = list(masks_dict[first_model].keys())
    
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
            
            print(f"  {region_name}: {regional_avg:.1f} {units} ({n_gridcells} grid cells)")
    
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
    
    # Add some statistics to the plot
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




def quick_regional_extremes_analysis(models_dict, time_dim='time', plot_type='barchart', 
                                     regions=None, per_grid_cell=True, shapefile_path=None, 
                                     mask_save_dir=None, **kwargs):
    """
    Quick analysis of regional extremes across models using model-specific masks
    
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
    shapefile_path : str or Path, optional
        Path to the shapefile for mask creation
    mask_save_dir : str or Path, optional
        Directory to save model-specific masks
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
    print("QUICK REGIONAL EXTREMES ANALYSIS WITH MODEL-SPECIFIC MASKS")
    print("=" * 60)
    
    # Extract plotting-specific kwargs (remove computation parameters)
    plot_kwargs = kwargs.copy()
    # Remove parameters that are only for computation, not plotting
    computation_params = ['normalize', 'time_dim', 'regions', 'per_grid_cell', 'shapefile_path', 'mask_save_dir']
    for param in computation_params:
        plot_kwargs.pop(param, None)
    
    # Compute regional data with model-specific masks
    regional_data, masks_dict = compute_regional_extremes(
        models_dict, 
        time_dim=time_dim, 
        regions=regions, 
        per_grid_cell=per_grid_cell,
        shapefile_path=shapefile_path,
        mask_save_dir=mask_save_dir,
        normalize=kwargs.get('normalize', True)  # Get normalize from kwargs or default to True
    )
    
    # Create appropriate plot using local functions
    if plot_type == 'barchart':
        fig, ax = plot_regional_extremes_barchart(regional_data, **plot_kwargs)
    elif plot_type == 'heatmap':
        fig, ax = plot_regional_extremes_heatmap(regional_data, **plot_kwargs)
    elif plot_type == 'single':
        # For single model plot, use first model by default
        model_name = list(regional_data.keys())[0]
        fig, ax = plot_regional_comparison_single_model(regional_data, model_name, **plot_kwargs)
    else:
        raise ValueError("plot_type must be 'barchart', 'heatmap', or 'single'")
    
    return fig, ax, regional_data