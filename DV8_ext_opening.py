import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from pathlib import Path
from getpass import getuser
from scipy import ndimage
import geopandas as gpd
import dask.array as da
import matplotlib.dates as mdates
import pandas as pd
from datetime import timedelta


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
    EXCLUDES the new equatorial Pacific sub-regions from mutual exclusivity check
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
            # Note: Western_Equatorial_Pacific and Eastern_Equatorial_Pacific are excluded
        ]
    
    # Separate the equatorial Pacific sub-regions from the main regions
    main_region_masks = {}
    equatorial_pacific_subregions = {}
    
    for region_name, mask in region_masks.items():
        if region_name in ['Western_Equatorial_Pacific', 'Eastern_Equatorial_Pacific']:
            equatorial_pacific_subregions[region_name] = mask
        else:
            main_region_masks[region_name] = mask
    
    bool_masks = {}
    for region_name, mask in main_region_masks.items():
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
        original_mask = main_region_masks[region_name]
        result_masks[region_name] = xr.DataArray(
            bool_mask,
            dims=('lat', 'lon'),
            coords={
                'lat': original_mask.lat if hasattr(original_mask, 'lat') else original_mask.coords['lat'],
                'lon': original_mask.lon if hasattr(original_mask, 'lon') else original_mask.coords['lon']
            },
            name=region_name
        )
    
    # Add back the equatorial Pacific sub-regions (they can overlap)
    result_masks.update(equatorial_pacific_subregions)
    
    return result_masks

def get_region_colors_shapefile():
    """
    Get color mapping for shapefile-based regions
    """
    return {
        'Southern_Ocean': 'purple',
        'North_Pacific_SubTropics': 'brown',
        'North_Pacific_MiddleLats': 'blue',
        'South_Pacific_SubTropics': 'darkblue',
        'Pacific_Equatorial': 'lightgreen',
        'Western_Equatorial_Pacific': 'red',  # Different color for Western
        'Eastern_Equatorial_Pacific': 'blue',  # Different color for Eastern
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
        mask_save_dir = Path('/scratch') / getuser()[0] / getuser() / 'mhws' / 'DV8' / 'extremes_model_masks'
    
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique identifier using model name and grid statistics
    if model_name:
        # Use model name with grid stats (min, max of coordinates)
        lat_stats = (np.round(np.min(lats), 4), np.round(np.max(lats), 4))
        lon_stats = (np.round(np.min(lons), 4), np.round(np.max(lons), 4))
        grid_hash = f"extremes_{model_name}_{lats.shape[0]}x{lons.shape[0]}_{hash(lat_stats + lon_stats)}"
    else:
        # Fallback to original method
        lat_sample = lats.flat[:10] if hasattr(lats, 'flat') else lats[:10]
        lon_sample = lons.flat[:10] if hasattr(lons, 'flat') else lons[:10]
        lat_repr = tuple(np.round(lat_sample, 6))
        lon_repr = tuple(np.round(lon_sample, 6))
        grid_hash = f"extremes_model_{lats.shape[0]}x{lons.shape[0]}_{hash(lat_repr + lon_repr)}"
    
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

    # NEW: Split Pacific Equatorial into Western and Eastern regions
    if 'Pacific_Equatorial' in region_masks:
        print("  Creating Western and Eastern Equatorial Pacific masks...")
        pacific_eq = region_masks['Pacific_Equatorial']
        
        # Create a copy of the mask with 0-360 longitude coordinates
        pacific_eq_360 = pacific_eq.copy()
        
        # Convert longitudes from -180-180 to 0-360
        if hasattr(pacific_eq_360, 'lon'):
            # For 1D coordinates
            if pacific_eq_360.lon.ndim == 1:
                lon_360 = pacific_eq_360.lon.values.copy()
                lon_360[lon_360 < 0] += 360
                pacific_eq_360 = pacific_eq_360.assign_coords(lon=lon_360)
            else:
                # For 2D coordinates
                lon_360 = pacific_eq_360.lon.values.copy()
                lon_360[lon_360 < 0] += 360
                pacific_eq_360['lon'] = (('lat', 'lon'), lon_360)
        
        # Now split at 150°W = 210° in 0-360 system
        # Western Equatorial Pacific: longitudes > 210° (150°W to 180°E)
        # Eastern Equatorial Pacific: longitudes <= 210° (120°E to 150°W)
        eastern_mask_360 = pacific_eq_360 & (pacific_eq_360.lon > 210)
        western_mask_360 = pacific_eq_360 & (pacific_eq_360.lon <= 210)
        
        # Convert back to -180-180 system
        western_mask = western_mask_360.copy()
        eastern_mask = eastern_mask_360.copy()
        
        if hasattr(western_mask, 'lon'):
            # Convert back to -180-180
            if western_mask.lon.ndim == 1:
                lon_180 = western_mask.lon.values.copy()
                lon_180[lon_180 > 180] -= 360
                western_mask = western_mask.assign_coords(lon=lon_180)
                eastern_mask = eastern_mask.assign_coords(lon=lon_180)
            else:
                lon_180 = western_mask.lon.values.copy()
                lon_180[lon_180 > 180] -= 360
                western_mask['lon'] = (('lat', 'lon'), lon_180)
                eastern_mask['lon'] = (('lat', 'lon'), lon_180)
        
        region_masks['Western_Equatorial_Pacific'] = western_mask
        region_masks['Eastern_Equatorial_Pacific'] = eastern_mask
        
        # Debug: Print the sizes of the new regions
        western_count = np.sum(western_mask.values)
        eastern_count = np.sum(eastern_mask.values)
        total_count = np.sum(pacific_eq.values)
        print(f"    Western points: {western_count}, Eastern points: {eastern_count}, Total: {total_count}")

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
    UPDATED: Shows Western and Eastern Equatorial Pacific with hashed lines in different colors
    FIXED: Handle contourf collections properly
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
    
    # Define priority order for base regions (excluding the overlapping equatorial Pacific sub-regions)
    base_priority_order = [
        'Southern_Ocean',
        'North_Pacific_SubTropics', 'North_Pacific_MiddleLats', 'South_Pacific_SubTropics',
        'North_Atlantic_SubTropics', 'North_Atlantic_MiddleLats', 'South_Atlantic_SubTropics', 
        'Indian_NorthSubTropics', 'Indian_SouthSubTropics',
        'Pacific_Equatorial', 'Atlantic_Equatorial', 'Indian_Equatorial',
        'Mediterranean_Sea'
    ]
    
    # Filter to only include base regions that actually exist for this model
    available_base_regions = [r for r in base_priority_order if r in regions]
    
    for idx, region_name in enumerate(available_base_regions):
        mask = model_masks[region_name]
        # Only assign this region where no previous region has been assigned
        region_pixels = mask.values & (combined_data == 0)
        combined_data[region_pixels] = idx + 1
    
    # Create colormap for base regions
    base_colors = [region_colors.get(region, 'gray') for region in available_base_regions]
    cmap = ListedColormap(['white'] + base_colors)
    
    # Get coordinates
    if hasattr(first_mask, 'lat') and hasattr(first_mask, 'lon'):
        if first_mask.lat.ndim == 1 and first_mask.lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(first_mask.lon, first_mask.lat)
        else:
            lon_grid, lat_grid = first_mask.lon, first_mask.lat
    else:
        lon_grid, lat_grid = np.meshgrid(np.arange(first_mask.shape[1]), np.arange(first_mask.shape[0]))
    
    # Plot base regions
    im = ax.pcolormesh(lon_grid, lat_grid, combined_data,
                      cmap=cmap,
                      vmin=0, vmax=len(available_base_regions) + 0.5,
                      transform=ccrs.PlateCarree())
    
    # Overlay Western and Eastern Equatorial Pacific with hashed patterns and different colors
    hatch_patterns = {
        'Western_Equatorial_Pacific': '////',
        'Eastern_Equatorial_Pacific': '\\\\\\\\'
    }
    
    # Define specific colors for the hatch lines (different from the base colors)
    hatch_colors = {
        'Western_Equatorial_Pacific': 'red',    # Red hatch lines for Western
        'Eastern_Equatorial_Pacific': 'blue'    # Blue hatch lines for Eastern
    }
    
    for region_name, hatch_pattern in hatch_patterns.items():
        if region_name in model_masks:
            mask = model_masks[region_name]
            mask_data = mask.values if hasattr(mask, 'values') else mask
            
            # Create a contour for the hashed region
            if mask_data.any():  # Only plot if there are True values
                # Create a masked array for contouring
                contour_data = np.where(mask_data, 1, 0).astype(float)
                
                try:
                    # Use contourf to create the hashed pattern
                    contourf = ax.contourf(lon_grid, lat_grid, contour_data, 
                                          levels=[0.5, 1.5], 
                                          colors='none',  # No fill color
                                          hatches=[hatch_pattern],
                                          transform=ccrs.PlateCarree(),
                                          alpha=0)
                    
                    # Set the edge color for the hatch - use the specific hatch color
                    hatch_color = hatch_colors.get(region_name, 'gray')
                    
                    # Handle different return types
                    if hasattr(contourf, 'collections'):
                        # Standard matplotlib contourf
                        for collection in contourf.collections:
                            collection.set_edgecolor(hatch_color)
                            collection.set_linewidth(0.8)  # Slightly thicker for visibility
                    elif hasattr(contourf, '__iter__'):
                        # Some versions return an iterable of collections
                        for collection in contourf:
                            if hasattr(collection, 'set_edgecolor'):
                                collection.set_edgecolor(hatch_color)
                                collection.set_linewidth(0.8)
                    else:
                        # Try to access _collections for GeoContourSet objects
                        try:
                            for collection in contourf._collections:
                                collection.set_edgecolor(hatch_color)
                                collection.set_linewidth(0.8)
                        except AttributeError:
                            print(f"Warning: Could not set hatch properties for {region_name}")
                            
                except Exception as e:
                    print(f"Warning: Could not create hatched pattern for {region_name}: {e}")
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Create legend (include "No Region" for background and hashed regions)
    legend_patches = [mpatches.Patch(color='white', label='No Region')]
    
    # Add base regions
    for region_name, color in zip(available_base_regions, base_colors):
        patch = mpatches.Patch(color=color, label=region_name.replace('_', ' ').title())
        legend_patches.append(patch)
    
    # Add hashed regions for equatorial Pacific sub-regions with their specific hatch colors
    for region_name, hatch_pattern in hatch_patterns.items():
        if region_name in regions:
            hatch_color = hatch_colors.get(region_name, 'gray')
            # Use the hatch color for the facecolor in the legend
            patch = mpatches.Patch(facecolor=hatch_color, hatch=hatch_pattern, 
                                 label=region_name.replace('_', ' ').title())
            legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, 
              loc='center left', 
              bbox_to_anchor=(1.05, 0.5),
              frameon=True,
              fancybox=True,
              shadow=True)
    
    ax.set_title(f'Extremes Combined Oceanic Regions - {model_name}\n(Western/Eastern Equatorial Pacific shown with colored hashes)', 
                 fontsize=14, pad=20)
    
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
                                    cmap='tab20', title="Average Extreme Days per Year by Region and Model"):
    """
    Plot barchart of regional extremes for all models
    
    Parameters:
    -----------
    regional_data : dict
        Output from compute_regional_extremes
    figsize : tuple
        Figure size
    cmap : str
        Colormap for different models
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
    
    # Create color map for MODELS (not regions)
    colors = plt.cm.get_cmap(cmap, len(models))
    
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
        
        # Get ONE color for this entire model (not one per region)
        model_color = colors(i)
        
        bars = ax.bar(positions, model_values, bar_width, 
                     label=model_name, 
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5,
                     color=model_color)  # Single color for all bars of this model
    
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
                                   cmap='YlOrRd', title="Regional Extreme Days Heatmap"):
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

# =========================================================================================================================================================
# MARINE HEATWAVES
# =========================================================================================================================================================

# ===================================================================================
# MHW EVENT DETECTION FUNCTIONS 
# ===================================================================================

def detect_mhw_events_structured(time_series, time_coords=None, min_duration=5, max_gap=2, max_events=100):
    """
    MODIFIED: Open-then-Close Logic.
    Retains event_count, durations, start_times, and end_times.
    """
    # Initialize empty return arrays
    empty_ret = (np.int32(0), np.full(max_events, -1, dtype=np.int32), 
                 np.full(max_events, np.datetime64('NaT'), dtype='datetime64[ns]'), 
                 np.full(max_events, np.datetime64('NaT'), dtype='datetime64[ns]'))

    if np.all(~time_series):
        return empty_ret
    
    # --- STEP 1: OPENING (Duration Filter) ---
    labeled_periods, num_periods = ndimage.label(time_series)
    if num_periods == 0: return empty_ret
    
    period_lengths = ndimage.sum(time_series, labeled_periods, range(1, num_periods + 1))
    period_indices = ndimage.find_objects(labeled_periods)
    
    # Filter periods: Only keep those >= min_duration (The 'Opening' step)
    valid_initial_periods = []
    for slc, length in zip(period_indices, period_lengths):
        if slc is not None and length >= min_duration:
            valid_initial_periods.append({
                'start': slc[0].start,
                'end': slc[0].stop - 1,
                'length': int(length)
            })
            
    if not valid_initial_periods:
        return empty_ret

    # --- STEP 2: CLOSING (Bridge Survivors) ---
    # According to Hobday: [5hot, 1cool, 2hot] = 5 days because the '2' was deleted first.
    merged_events = []
    current_event = valid_initial_periods[0]

    for i in range(1, len(valid_initial_periods)):
        next_p = valid_initial_periods[i]
        gap = next_p['start'] - current_event['end'] - 1
        
        if gap <= max_gap:
            # Bridge them
            current_event['end'] = next_p['end']
            current_event['length'] += gap + next_p['length']
        else:
            # Seal current and move to next
            merged_events.append(current_event)
            current_event = next_p
    
    merged_events.append(current_event)

    # --- STEP 3: FORMATTING (Retaining original variables) ---
    event_count = min(len(merged_events), max_events)
    durations = np.full(max_events, -1, dtype=np.int32)
    start_times = np.full(max_events, np.datetime64('NaT'), dtype='datetime64[ns]')
    end_times = np.full(max_events, np.datetime64('NaT'), dtype='datetime64[ns]')
    
    for i in range(event_count):
        ev = merged_events[i]
        durations[i] = ev['length']
        if time_coords is not None:
            start_times[i] = time_coords[ev['start']]
            end_times[i] = time_coords[ev['end']]
            
    return np.int32(event_count), durations, start_times, end_times

def create_mhw_event_dataset(o_ex, min_duration=5, max_gap=2, max_events_per_cell=100):
    print("Creating MHW event dataset (Strict Hobday Logic)...")
    time_coords = o_ex.time.values
    
    results = xr.apply_ufunc(
        detect_mhw_events_structured,
        o_ex,
        input_core_dims=[['time']],
        output_core_dims=[[], ['event_index'], ['event_index'], ['event_index']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.int32, np.int32, 'datetime64[ns]', 'datetime64[ns]'],
        dask_gufunc_kwargs={'output_sizes': {'event_index': max_events_per_cell}},
        kwargs={
            'time_coords': time_coords,
            'min_duration': min_duration,
            'max_gap': max_gap,
            'max_events': max_events_per_cell
        }
    )
    
    # Extract the .data property from each result to avoid the TypeError
    mhw_ds = xr.Dataset({
        'event_count': (['lat', 'lon'], results[0].data),
        'event_durations': (['lat', 'lon', 'event_index'], results[1].data),
        'event_start_times': (['lat', 'lon', 'event_index'], results[2].data),
        'event_end_times': (['lat', 'lon', 'event_index'], results[3].data)
    }, coords={
        'lat': o_ex.lat, 
        'lon': o_ex.lon, 
        'event_index': np.arange(max_events_per_cell)
    })
    
    return mhw_ds

def compute_mhw_events_for_models(models_dict, min_duration=5, max_gap=2, max_events_per_cell=100):
    """
    Compute MHW event datasets for all models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and extreme events DataArrays as values
    min_duration : int
        Minimum event duration in days
    max_gap : int
        Maximum gap between events to merge
    max_events_per_cell : int
        Maximum number of events per grid cell
    
    Returns:
    --------
    mhw_events_dict : dict
        Dictionary with model names as keys and MHW event datasets as values
    """
    print("COMPUTING MHW EVENTS FOR ALL MODELS")
    print("=" * 40)
    
    mhw_events_dict = {}
    
    for model_name, model_data in models_dict.items():
        print(f"Processing {model_name}...")
        
        # Extract extreme events data
        extreme_events = extract_data_array(model_data)
        
        # Create MHW event dataset
        mhw_ds = create_mhw_event_dataset(
            extreme_events,
            min_duration=min_duration,
            max_gap=max_gap,
            max_events_per_cell=max_events_per_cell
        )
        
        mhw_events_dict[model_name] = mhw_ds
        
        # Print some statistics
        total_events = mhw_ds.event_count.sum().compute().values
        avg_duration = mhw_ds.event_durations.where(mhw_ds.event_durations > 0).mean().compute().values
        print(f"  Total events: {total_events}")
        print(f"  Average duration: {avg_duration:.1f} days")
    
    return mhw_events_dict

# ===================================================================================
# REGIONAL MHW EVENT ANALYSIS
# ===================================================================================

def compute_regional_mhw_events(mhw_events_dict, masks_dict, regions=None, 
                                normalize=True, time_dim='time', per_grid_cell=True):
    """
    Compute MHW event statistics for each region and model
    
    Parameters:
    -----------
    mhw_events_dict : dict
        Dictionary with model names as keys and MHW event datasets as values
    masks_dict : dict
        Dictionary with model-specific masks
    regions : list, optional
        Specific regions to analyze
    normalize : bool
        If True, normalize by number of years
    time_dim : str
        Name of time dimension in original data (for normalization)
    per_grid_cell : bool
        If True, return average per grid cell. If False, return regional total.
    
    Returns:
    --------
    regional_mhw_data : dict
        Dictionary with regional MHW event statistics for each model
    """
    print("COMPUTING REGIONAL MHW EVENT STATISTICS")
    print("=" * 50)
    
    # Get regions to analyze (use first model's regions as reference)
    if regions is None:
        first_model = list(masks_dict.keys())[0]
        regions = list(masks_dict[first_model].keys())
    
    regional_mhw_data = {}
    
    for model_name, mhw_ds in mhw_events_dict.items():
        print(f"Processing {model_name}...")
        
        if model_name not in masks_dict:
            print(f"  Warning: No masks found for {model_name}, skipping...")
            continue
        
        regional_mhw_data[model_name] = {}
        
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
            
            # Apply mask to event data
            masked_event_count = mhw_ds.event_count.where(mask)
            masked_durations = mhw_ds.event_durations.where(mask)
            
            # Compute regional statistics
            total_events = masked_event_count.sum().compute().values
            total_event_days = masked_durations.where(masked_durations > 0).sum().compute().values
            
            # Calculate average duration (only for cells with events)
            if total_events > 0:
                avg_duration = total_event_days / total_events
            else:
                avg_duration = 0
            
            # NORMALIZE by number of grid cells if requested
            if per_grid_cell:
                events_per_cell = total_events / n_gridcells
                event_days_per_cell = total_event_days / n_gridcells
            else:
                events_per_cell = total_events
                event_days_per_cell = total_event_days
            
            # Store results
            regional_mhw_data[model_name][region_name] = {
                'event_count': events_per_cell,
                'total_event_days': event_days_per_cell,
                'avg_duration': avg_duration,
                'n_gridcells': n_gridcells
            }
            
            print(f"  {region_name}: {events_per_cell:.1f} events, {avg_duration:.1f} days avg duration")
    
    return regional_mhw_data

# ===================================================================================
# MHW EVENT PLOTTING FUNCTIONS (UPDATED WITH VMIN/VMAX)
# ===================================================================================

def plot_mhw_event_count_map(mhw_events_dict, model_name=None, figsize=(12, 8), 
                            central_longitude=180, cmap='viridis', 
                            title_template="MHW Event Count - {}",
                            vmin=None, vmax=None):
    """
    Plot map of MHW event counts for a model
    
    Parameters:
    -----------
    mhw_events_dict : dict
        Dictionary with model names as keys and MHW event datasets as values
    model_name : str, optional
        Specific model to plot. If None, uses first model.
    figsize : tuple
        Figure size
    central_longitude : float
        Central longitude for map projection
    cmap : str
        Colormap for event counts
    title_template : str
        Title template (will be formatted with model_name)
    vmin, vmax : float, optional
        Colorbar limits
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The map figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    if model_name is None:
        model_name = list(mhw_events_dict.keys())[0]
    
    if model_name not in mhw_events_dict:
        raise ValueError(f"Model '{model_name}' not found in MHW events dictionary")
    
    mhw_ds = mhw_events_dict[model_name]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize,
                          subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Plot event counts with vmin/vmax
    im = mhw_ds.event_count.plot(ax=ax, transform=ccrs.PlateCarree(),
                                cmap=cmap, add_colorbar=True,
                                cbar_kwargs={'label': 'Number of MHW Events'},
                                vmin=vmin, vmax=vmax)
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Add title
    total_events = mhw_ds.event_count.sum().compute().values
    ax.set_title(title_template.format(model_name) + f"\nTotal Events: {total_events:,}", 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    
    return fig, ax

def plot_mhw_avg_duration_map(mhw_events_dict, model_name=None, figsize=(12, 8),
                             central_longitude=180, cmap='plasma',
                             title_template="Average MHW Duration - {}",
                             vmin=None, vmax=None):
    """
    Plot map of average MHW duration for a model
    
    Parameters:
    -----------
    mhw_events_dict : dict
        Dictionary with model names as keys and MHW event datasets as values
    model_name : str, optional
        Specific model to plot. If None, uses first model.
    figsize : tuple
        Figure size
    central_longitude : float
        Central longitude for map projection
    cmap : str
        Colormap for duration
    title_template : str
        Title template (will be formatted with model_name)
    vmin, vmax : float, optional
        Colorbar limits
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The map figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    if model_name is None:
        model_name = list(mhw_events_dict.keys())[0]
    
    if model_name not in mhw_events_dict:
        raise ValueError(f"Model '{model_name}' not found in MHW events dictionary")
    
    mhw_ds = mhw_events_dict[model_name]
    
    # Calculate average duration per grid cell
    # Only consider cells with events (event_count > 0)
    event_mask = mhw_ds.event_count > 0
    avg_duration = (mhw_ds.event_durations.where(mhw_ds.event_durations > 0)
                   .mean(dim='event_index')
                   .where(event_mask))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize,
                          subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Plot average duration with vmin/vmax
    im = avg_duration.plot(ax=ax, transform=ccrs.PlateCarree(),
                          cmap=cmap, add_colorbar=True,
                          cbar_kwargs={'label': 'Average Duration (days)'},
                          vmin=vmin, vmax=vmax)
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Add title
    global_avg = avg_duration.mean().compute().values
    ax.set_title(title_template.format(model_name) + f"\nGlobal Average: {global_avg:.1f} days", 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    
    return fig, ax

def plot_regional_mhw_events_barchart(regional_mhw_data, metric='event_count', 
                                     figsize=(16, 10), cmap='tab20',
                                     title_template="Regional MHW {} by Model"):
    """
    Plot barchart of regional MHW events for all models
    
    Parameters:
    -----------
    regional_mhw_data : dict
        Output from compute_regional_mhw_events
    metric : str
        Metric to plot: 'event_count', 'total_event_days', or 'avg_duration'
    figsize : tuple
        Figure size
    cmap : str
        Colormap for different regions
    title_template : str
        Title template (will be formatted with metric)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The bar chart figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    print(f"PLOTTING REGIONAL MHW {metric.upper()} BARCHART")
    print("=" * (45 + len(metric)))
    
    # Get models and regions
    models = list(regional_mhw_data.keys())
    regions = list(regional_mhw_data[models[0]].keys())
    
    # Create color map for regions
    colors = plt.cm.get_cmap(cmap, len(regions))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot settings
    bar_width = 0.8 / len(models)  # Dynamic width based on number of models
    x_pos = np.arange(len(regions))
    
    # Metric labels and units
    metric_labels = {
        'event_count': 'Event Count',
        'total_event_days': 'Total Event Days', 
        'avg_duration': 'Average Duration (days)'
    }
    
    # Plot bars for each model
    for i, model_name in enumerate(models):
        model_values = [regional_mhw_data[model_name][region][metric] for region in regions]
        
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
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(title_template.format(metric_labels.get(metric, metric)), fontsize=14, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.replace('_', ' ').title() for r in regions], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig, ax

def plot_regional_mhw_comparison_grid(regional_mhw_data, models=None, regions=None,
                                     figsize=(20, 15), metrics=None):
    """
    Plot grid of regional MHW comparisons across models and metrics
    
    Parameters:
    -----------
    regional_mhw_data : dict
        Output from compute_regional_mhw_events
    models : list, optional
        Specific models to include. If None, uses all.
    regions : list, optional
        Specific regions to include. If None, uses all.
    figsize : tuple
        Figure size
    metrics : list, optional
        Metrics to plot. If None, uses ['event_count', 'total_event_days', 'avg_duration']
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The grid figure
    axes : numpy.ndarray of matplotlib.axes.Axes
        The axes objects
    """
    if metrics is None:
        metrics = ['event_count', 'total_event_days', 'avg_duration']
    
    if models is None:
        models = list(regional_mhw_data.keys())
    
    if regions is None:
        regions = list(regional_mhw_data[models[0]].keys())
    
    # Metric labels
    metric_labels = {
        'event_count': 'Event Count',
        'total_event_days': 'Total Event Days',
        'avg_duration': 'Avg Duration (days)'
    }
    
    # Create subplot grid
    n_metrics = len(metrics)
    n_models = len(models)
    
    fig, axes = plt.subplots(n_metrics, n_models, figsize=figsize, squeeze=False)
    
    # Plot each metric for each model
    for i, metric in enumerate(metrics):
        for j, model_name in enumerate(models):
            ax = axes[i, j]
            
            if model_name not in regional_mhw_data:
                ax.set_visible(False)
                continue
            
            # Get data for this model and metric
            model_data = [regional_mhw_data[model_name][region][metric] for region in regions]
            
            # Create bar plot
            bars = ax.bar(range(len(regions)), model_data, color='steelblue', alpha=0.7)
            
            # Customize subplot
            ax.set_xticks(range(len(regions)))
            ax.set_xticklabels([r.replace('_', ' ').title() for r in regions], 
                              rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add labels and titles
            if j == 0:
                ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
            if i == 0:
                ax.set_title(model_name, fontsize=12, pad=10)
            
            # Add value labels on bars
            for bar, value in zip(bars, model_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(model_data)*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=7)
    
    fig.suptitle('Regional MHW Event Statistics Across Models and Metrics', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    
    return fig, axes

def quick_mhw_events_analysis(models_dict, shapefile_path=None, mask_save_dir=None,
                             min_duration=5, max_gap=2, max_events_per_cell=100,
                             plot_maps=True, plot_regional=True, vmin_event_count=None, vmax_event_count=None,
                             vmin_avg_duration=None, vmax_avg_duration=None):
    """
    Quick comprehensive analysis of MHW events across models and regions
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and extreme events DataArrays as values
    shapefile_path : str or Path, optional
        Path to shapefile for mask creation
    mask_save_dir : str or Path, optional
        Directory to save model-specific masks
    min_duration : int
        Minimum event duration in days
    max_gap : int
        Maximum gap between events to merge
    max_events_per_cell : int
        Maximum number of events per grid cell
    plot_maps : bool
        Whether to plot spatial maps
    plot_regional : bool
        Whether to plot regional comparisons
    vmin_event_count, vmax_event_count : float, optional
        Colorbar limits for event count maps
    vmin_avg_duration, vmax_avg_duration : float, optional
        Colorbar limits for average duration maps
    
    Returns:
    --------
    mhw_events_dict : dict
        Dictionary with MHW event datasets
    regional_mhw_data : dict
        Dictionary with regional MHW statistics
    masks_dict : dict
        Dictionary with model-specific masks
    """
    print("QUICK MHW EVENTS ANALYSIS")
    print("=" * 25)
    
    # Step 1: Compute MHW event datasets
    mhw_events_dict = compute_mhw_events_for_models(
        models_dict,
        min_duration=min_duration,
        max_gap=max_gap,
        max_events_per_cell=max_events_per_cell
    )
    
    # Step 2: Create model-specific masks
    masks_dict = create_model_specific_masks(
        models_dict,
        shapefile_path=shapefile_path,
        mask_save_dir=mask_save_dir
    )
    
    # Step 3: Compute regional statistics
    regional_mhw_data = compute_regional_mhw_events(mhw_events_dict, masks_dict)
    
    # Step 4: Create plots
    if plot_maps:
        print("\nCREATING SPATIAL MAPS...")
        for model_name in mhw_events_dict.keys():
            # Event count map with vmin/vmax
            fig1, ax1 = plot_mhw_event_count_map(mhw_events_dict, model_name, 
                                                vmin=vmin_event_count, vmax=vmax_event_count)
            plt.show()
            
            # Average duration map with vmin/vmax
            fig2, ax2 = plot_mhw_avg_duration_map(mhw_events_dict, model_name,
                                                 vmin=vmin_avg_duration, vmax=vmax_avg_duration)
            plt.show()
    
    if plot_regional:
        print("\nCREATING REGIONAL COMPARISONS...")
        # Regional barcharts for different metrics
        for metric in ['event_count', 'total_event_days', 'avg_duration']:
            fig, ax = plot_regional_mhw_events_barchart(regional_mhw_data, metric=metric)
            plt.show()
        
        # Comprehensive grid plot
        fig, axes = plot_regional_mhw_comparison_grid(regional_mhw_data)
        plt.show()
    
    return mhw_events_dict, regional_mhw_data, masks_dict

def selective_mhw_analysis(models_dict, plots_to_show=['regional_summary'], **kwargs):
    """
    Wrapper for selective MHW analysis plotting
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and extreme events DataArrays as values
    plots_to_show : list
        List of plot types to show. Options:
        - 'events_map': MHW event count maps
        - 'duration_map': MHW average duration maps  
        - 'regional_events': Regional event count barchart
        - 'regional_duration': Regional duration barchart
        - 'regional_event_days': Regional total event days barchart
        - 'regional_grid': Regional comparison grid
        - 'regional_summary': Regional summary (the one you want)
    **kwargs : additional arguments passed to quick_mhw_events_analysis
    
    Returns:
    --------
    mhw_events_dict : dict
        Dictionary with MHW event datasets
    regional_mhw_data : dict
        Dictionary with regional MHW statistics
    masks_dict : dict
        Dictionary with model-specific masks
    """
    print("SELECTIVE MHW ANALYSIS - SHOWING ONLY SPECIFIED PLOTS")
    print("=" * 50)
    print(f"Plots to show: {plots_to_show}")
    
    # Extract vmin/vmax parameters for maps
    vmin_event_count = kwargs.pop('vmin_event_count', None)
    vmax_event_count = kwargs.pop('vmax_event_count', None)
    vmin_avg_duration = kwargs.pop('vmin_avg_duration', None)
    vmax_avg_duration = kwargs.pop('vmax_avg_duration', None)
    
    # First run the analysis but suppress all automatic plotting
    mhw_events_dict, regional_mhw_data, masks_dict = quick_mhw_events_analysis(
        models_dict,
        plot_maps=False,    # Turn off all map plots
        plot_regional=False, # Turn off all regional plots
        **{k: v for k, v in kwargs.items() if k not in ['plot_maps', 'plot_regional']}
    )
    
    # Now manually plot only what we want based on plots_to_show
    if 'events_map' in plots_to_show:
        print("\nCreating MHW event count maps...")
        for model_name in mhw_events_dict.keys():
            fig, ax = plot_mhw_event_count_map(mhw_events_dict, model_name,
                                              vmin=vmin_event_count, vmax=vmax_event_count)
            plt.show()
    
    if 'duration_map' in plots_to_show:
        print("\nCreating MHW average duration maps...")
        for model_name in mhw_events_dict.keys():
            fig, ax = plot_mhw_avg_duration_map(mhw_events_dict, model_name,
                                               vmin=vmin_avg_duration, vmax=vmax_avg_duration)
            plt.show()
    
    if 'regional_events' in plots_to_show:
        print("\nCreating regional event count barchart...")
        fig, ax = plot_regional_mhw_events_barchart(regional_mhw_data, metric='event_count')
        plt.show()
    
    if 'regional_duration' in plots_to_show:
        print("\nCreating regional duration barchart...")
        fig, ax = plot_regional_mhw_events_barchart(regional_mhw_data, metric='avg_duration')
        plt.show()
    
    if 'regional_event_days' in plots_to_show:
        print("\nCreating regional event days barchart...")
        fig, ax = plot_regional_mhw_events_barchart(regional_mhw_data, metric='total_event_days')
        plt.show()
    
    if 'regional_grid' in plots_to_show:
        print("\nCreating regional comparison grid...")
        fig, axes = plot_regional_mhw_comparison_grid(regional_mhw_data)
        plt.show()
    
    if 'regional_summary' in plots_to_show:
        print("\nCreating regional summary...")
        # For the "avg number of events per region per model", use the event_count barchart
        fig, ax = plot_regional_mhw_events_barchart(regional_mhw_data, metric='event_count')
        plt.show()
    
    return mhw_events_dict, regional_mhw_data, masks_dict

# ==========================================================================================================================
# INTENSITY -> MAGNITUDE FUNCTIONS (RENAMED AND UPDATED)
# ==========================================================================================================================

def compute_event_magnitude_vectorized(mhw_events_ds, ssta_data, time_dim='time'):
    """
    Vectorized computation of MHW magnitude using Dask and xarray operations
    """
    print("Computing MHW magnitude statistics (vectorized Dask approach)...")
    
    # Extract coordinates
    event_start_times = mhw_events_ds.event_start_times
    event_end_times = mhw_events_ds.event_end_times
    ssta_times = ssta_data[time_dim].values
    
    # Get dimensions
    n_lats, n_lons, n_events = len(mhw_events_ds.lat), len(mhw_events_ds.lon), len(mhw_events_ds.event_index)
    print(f"Processing grid: {n_lats} x {n_lons} x {n_events} events")
    
    def compute_magnitude_for_cell(cell_starts, cell_ends, cell_ssta):
        """
        Compute magnitude for all events in a single grid cell
        """
        # Initialize output arrays
        n_events = len(cell_starts)
        avg_magnitude = np.full(n_events, np.nan)
        max_magnitude = np.full(n_events, np.nan)
        median_magnitude = np.full(n_events, np.nan)
        
        for evt_idx in range(n_events):
            start_time = cell_starts[evt_idx]
            end_time = cell_ends[evt_idx]
            
            # Skip invalid events
            if np.isnat(start_time) or np.isnat(end_time):
                continue
            
            # Find time indices for this event
            time_mask = (ssta_times >= start_time) & (ssta_times <= end_time)
            time_indices = np.where(time_mask)[0]
            
            if len(time_indices) > 0:
                # Extract SSTA values for this event
                event_ssta = cell_ssta[time_indices]
                
                if len(event_ssta) > 0 and not np.all(np.isnan(event_ssta)):
                    avg_magnitude[evt_idx] = np.nanmean(event_ssta)
                    max_magnitude[evt_idx] = np.nanmax(event_ssta)
                    median_magnitude[evt_idx] = np.nanmedian(event_ssta)
        
        return avg_magnitude, max_magnitude, median_magnitude
    
    # Use xr.apply_ufunc for vectorized computation
    print("Applying vectorized computation...")
    
    results = xr.apply_ufunc(
        compute_magnitude_for_cell,
        event_start_times,      # (lat, lon, event_index)
        event_end_times,        # (lat, lon, event_index)  
        ssta_data,              # (time, lat, lon)
        input_core_dims=[['event_index'], ['event_index'], [time_dim]],
        output_core_dims=[['event_index'], ['event_index'], ['event_index']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float],
        dask_gufunc_kwargs={'output_sizes': {'event_index': n_events}}
    )
    
    # Extract results
    avg_magnitude, max_magnitude, median_magnitude = results
    
    # Create output dataset
    magnitude_ds = xr.Dataset({
        'avg_magnitude': avg_magnitude,
        'max_magnitude': max_magnitude,
        'median_magnitude': median_magnitude
    })
    
    # Add attributes
    magnitude_ds.avg_magnitude.attrs = {
        'long_name': 'Average MHW magnitude',
        'units': 'degC',
        'description': 'Average SSTA during MHW events'
    }
    magnitude_ds.max_magnitude.attrs = {
        'long_name': 'Maximum MHW magnitude', 
        'units': 'degC',
        'description': 'Maximum SSTA during MHW events'
    }
    magnitude_ds.median_magnitude.attrs = {
        'long_name': 'Median MHW magnitude',
        'units': 'degC', 
        'description': 'Median SSTA during MHW events'
    }
    
    return magnitude_ds

# Even more optimized version using map_blocks
def compute_event_magnitude_map_blocks(mhw_events_ds, ssta_data, time_dim='time'):
    """
    Optimized version using Dask's map_blocks for better parallelization
    """
    print("Computing MHW magnitude (map_blocks optimized)...")
    
    # Ensure Dask chunks
    event_start_times = mhw_events_ds.event_start_times.chunk({'lat': 50, 'lon': 50, 'event_index': 50})
    event_end_times = mhw_events_ds.event_end_times.chunk({'lat': 50, 'lon': 50, 'event_index': 50})
    ssta_data_chunked = ssta_data.chunk({time_dim: -1, 'lat': 50, 'lon': 50})
    
    ssta_times = ssta_data[time_dim].values
    n_events = len(mhw_events_ds.event_index)
    
    def compute_magnitude_block(starts_block, ends_block, ssta_block):
        """
        Compute magnitude for a block of data
        """
        block_shape = starts_block.shape
        avg_result = np.full(block_shape, np.nan)
        max_result = np.full(block_shape, np.nan)
        median_result = np.full(block_shape, np.nan)
        
        # Iterate over the block
        for i in range(block_shape[0]):  # lat in block
            for j in range(block_shape[1]):  # lon in block
                # Get SSTA time series for this grid cell in the block
                cell_ssta = ssta_block[:, i, j]
                
                for k in range(block_shape[2]):  # event_index
                    start_time = starts_block[i, j, k]
                    end_time = ends_block[i, j, k]
                    
                    if np.isnat(start_time) or np.isnat(end_time):
                        continue
                    
                    # Find time indices for this event
                    time_mask = (ssta_times >= start_time) & (ssta_times <= end_time)
                    time_indices = np.where(time_mask)[0]
                    
                    if len(time_indices) > 0:
                        event_ssta = cell_ssta[time_indices]
                        
                        if len(event_ssta) > 0 and not np.all(np.isnan(event_ssta)):
                            avg_result[i, j, k] = np.nanmean(event_ssta)
                            max_result[i, j, k] = np.nanmax(event_ssta)
                            median_result[i, j, k] = np.nanmedian(event_ssta)
        
        return np.stack([avg_result, max_result, median_result], axis=-1)
    
    print("Processing with Dask map_blocks...")
    
    # Use map_blocks to compute all three metrics at once
    magnitude_blocks = da.map_blocks(
        compute_magnitude_block,
        event_start_times.data,
        event_end_times.data,
        ssta_data_chunked.data,
        dtype=float,
        chunks=event_start_times.chunks + (3,),  # Add extra dimension for 3 metrics
        new_axis=3  # Add new axis for the 3 metrics
    )
    
    # Split the results
    avg_magnitude_dask = magnitude_blocks[..., 0]
    max_magnitude_dask = magnitude_blocks[..., 1]
    median_magnitude_dask = magnitude_blocks[..., 2]
    
    # Convert to DataArrays
    avg_magnitude = xr.DataArray(
        avg_magnitude_dask,
        dims=event_start_times.dims,
        coords=event_start_times.coords,
        name='avg_magnitude'
    )
    
    max_magnitude = xr.DataArray(
        max_magnitude_dask,
        dims=event_start_times.dims,
        coords=event_start_times.coords,
        name='max_magnitude'
    )
    
    median_magnitude = xr.DataArray(
        median_magnitude_dask,
        dims=event_start_times.dims,
        coords=event_start_times.coords,
        name='median_magnitude'
    )
    
    # Create dataset
    magnitude_ds = xr.Dataset({
        'avg_magnitude': avg_magnitude,
        'max_magnitude': max_magnitude,
        'median_magnitude': median_magnitude
    })
    
    # Add attributes
    magnitude_ds.avg_magnitude.attrs = {
        'long_name': 'Average MHW magnitude',
        'units': 'degC',
        'description': 'Average SSTA during MHW events'
    }
    magnitude_ds.max_magnitude.attrs = {
        'long_name': 'Maximum MHW magnitude', 
        'units': 'degC',
        'description': 'Maximum SSTA during MHW events'
    }
    magnitude_ds.median_magnitude.attrs = {
        'long_name': 'Median MHW magnitude',
        'units': 'degC', 
        'description': 'Median SSTA during MHW events'
    }
    
    return magnitude_ds

# Test with a small subset first
def test_vectorized_small():
    """
    Test the vectorized approach on a small subset
    """
    print("Testing vectorized approach on small subset...")
    
    # Take a small subset
    small_lats = o_mhws.lat[::100]  # Every 100th latitude
    small_lons = o_mhws.lon[::100]  # Every 100th longitude
    small_events = o_mhws.event_index[:10]  # First 10 events
    
    small_mhws = o_mhws.sel(lat=small_lats, lon=small_lons, event_index=small_events)
    small_ssta = ossta.sel(lat=small_lats, lon=small_lons)
    
    print(f"Small subset: {len(small_lats)} x {len(small_lons)} x {len(small_events)}")
    
    # Test the vectorized approach
    small_magnitude = compute_event_magnitude_vectorized(small_mhws, small_ssta)
    
    # Check results
    valid_avg = np.sum(~np.isnan(small_magnitude.avg_magnitude.values))
    valid_max = np.sum(~np.isnan(small_magnitude.max_magnitude.values))
    
    print(f"Small test results: {valid_avg} valid average magnitudes, {valid_max} valid max magnitudes")
    
    return small_magnitude

# Progressive scaling
def compute_magnitude_progressive(mhw_events_ds, ssta_data, batch_size=100):
    """
    Process in batches to manage memory
    """
    print("Computing magnitude with progressive batching...")
    
    n_lats = len(mhw_events_ds.lat)
    results = []
    
    for lat_start in range(0, n_lats, batch_size):
        lat_end = min(lat_start + batch_size, n_lats)
        lat_slice = slice(lat_start, lat_end)
        
        print(f"Processing latitudes {lat_start} to {lat_end}...")
        
        # Process this batch
        batch_mhws = mhw_events_ds.isel(lat=lat_slice)
        batch_ssta = ssta_data.isel(lat=lat_slice)
        
        batch_magnitude = compute_event_magnitude_vectorized(batch_mhws, batch_ssta)
        results.append(batch_magnitude)
    
    # Combine results
    print("Combining results...")
    full_magnitude = xr.concat(results, dim='lat')
    
    return full_magnitude

# ==========================================================================================================================================================
## MAGNITUDE PLOTS (UPDATED - RENAMED FROM INTENSITY TO MAGNITUDE)
# ==========================================================================================================================================================

def plot_avg_magnitude_map(magnitude_ds, model_name="OSTIA", figsize=(12, 8), 
                          central_longitude=180, vmin=None, vmax=None, title=None):
    """
    Plot map of average magnitude (average of event averages per grid cell)
    """
    print("Plotting average magnitude map...")
    
    # Compute mean magnitude across events for each grid cell (ignore NaNs)
    avg_magnitude_2d = magnitude_ds.avg_magnitude.mean(dim='event_index', skipna=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, 
                          subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Plot the data with vmin/vmax
    im = avg_magnitude_2d.plot(ax=ax, transform=ccrs.PlateCarree(),
                              cmap='Reds', add_colorbar=True,
                              cbar_kwargs={'label': 'Average Magnitude (°C)',
                                          'shrink': 0.8},
                              vmin=vmin, vmax=vmax)
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Add title
    if title is None:
        title = f'MHW average magnitude of all average magnitudes - {model_name}'
    
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig, ax

def plot_avg_of_max_magnitude_map(magnitude_ds, model_name="OSTIA", figsize=(12, 8),
                                 central_longitude=180, vmin=None, vmax=None, title=None):
    """
    Plot map of average magnitude of maximum magnitudes per grid cell
    """
    print("Plotting average of maximum magnitudes map...")
    
    # Compute average of maximum magnitudes across events for each grid cell
    avg_of_max_magnitude_2d = magnitude_ds.max_magnitude.mean(dim='event_index', skipna=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize,
                          subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Plot the data with vmin/vmax
    im = avg_of_max_magnitude_2d.plot(ax=ax, transform=ccrs.PlateCarree(),
                                     cmap='OrRd', add_colorbar=True,
                                     cbar_kwargs={'label': 'Average of Max Magnitudes (°C)',
                                                 'shrink': 0.8},
                                     vmin=vmin, vmax=vmax)
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Add title
    if title is None:
        title = f'MHW average magnitude of all maximum magnitudes - {model_name}'
    
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig, ax

def plot_max_magnitude_map(magnitude_ds, model_name="OSTIA", figsize=(12, 8),
                          central_longitude=180, vmin=None, vmax=None, title=None):
    """
    Plot map of maximum magnitude (max of event maxima per grid cell)
    """
    print("Plotting maximum magnitude map...")
    
    # Compute maximum magnitude across events for each grid cell (ignore NaNs)
    max_magnitude_2d = magnitude_ds.max_magnitude.max(dim='event_index', skipna=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize,
                          subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Plot the data with vmin/vmax
    im = max_magnitude_2d.plot(ax=ax, transform=ccrs.PlateCarree(),
                              cmap='OrRd', add_colorbar=True,
                              cbar_kwargs={'label': 'Maximum Magnitude (°C)',
                                          'shrink': 0.8},
                              vmin=vmin, vmax=vmax)
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Add title
    if title is None:
        title = f'MHW max magnitude of all max magnitudes - {model_name}'
    
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig, ax

# Multi-model plotting functions
def plot_multi_model_magnitude_maps(magnitude_dict, plot_type='avg_magnitude', 
                                   figsize=(15, 10), central_longitude=180, 
                                   vmin=None, vmax=None, titles=None):
    """
    Plot magnitude maps for multiple models
    
    Parameters:
    -----------
    magnitude_dict : dict
        Dictionary with model names as keys and magnitude datasets as values
    plot_type : str
        Type of plot: 'avg_magnitude', 'avg_of_max_magnitude', or 'max_magnitude'
    figsize : tuple
        Figure size
    central_longitude : float
        Central longitude for map projection
    vmin, vmax : float
        Colorbar limits (same for all models)
    titles : dict
        Dictionary with model names as keys and custom titles as values
    """
    models = list(magnitude_dict.keys())
    n_models = len(models)
    
    # Calculate grid size
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})
    
    # Handle single subplot case
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each model
    for idx, model_name in enumerate(models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        magnitude_ds = magnitude_dict[model_name]
        
        # Get custom title if provided
        if titles and model_name in titles:
            title = titles[model_name]
        else:
            title = None
        
        # Select the appropriate plot type
        if plot_type == 'avg_magnitude':
            data_2d = magnitude_ds.avg_magnitude.mean(dim='event_index', skipna=True)
            cmap = 'OrRd'
            default_title = f'Avg Magnitude - {model_name}'
        elif plot_type == 'avg_of_max_magnitude':
            data_2d = magnitude_ds.max_magnitude.mean(dim='event_index', skipna=True)
            cmap = 'OrRd'
            default_title = f'Avg of Max Magnitude - {model_name}'
        elif plot_type == 'max_magnitude':
            data_2d = magnitude_ds.max_magnitude.max(dim='event_index', skipna=True)
            cmap = 'OrRd'
            default_title = f'Max Magnitude - {model_name}'
        else:
            raise ValueError("plot_type must be 'avg_magnitude', 'avg_of_max_magnitude', or 'max_magnitude'")
        
        # Plot with same vmin/vmax for all models
        im = data_2d.plot(ax=ax, transform=ccrs.PlateCarree(),
                         cmap=cmap, add_colorbar=False,
                         vmin=vmin, vmax=vmax)
        
        # Add map features
        ax.coastlines(linewidth=0.5, color='black')
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
        ax.set_global()
        ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
        
        # Set title
        if title is None:
            title = default_title
        ax.set_title(title, fontsize=10, pad=5)
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), 
                       orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Magnitude (°C)', fontsize=12)
    
    # Hide unused subplots
    for idx in range(len(models), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    return fig, axes

# 3. Scatter plots of duration vs magnitude metrics
def plot_duration_magnitude_scatter(mhw_events_ds, magnitude_ds, model_name="OSTIA", 
                                   max_points=10000, figsize=(15, 5)):
    """
    Plot scatter plots of duration vs magnitude metrics
    """
    print("Creating duration vs magnitude scatter plots...")
    
    # Extract duration and magnitude data
    durations = mhw_events_ds.event_durations
    avg_magnitude = magnitude_ds.avg_magnitude
    max_magnitude = magnitude_ds.max_magnitude
    median_magnitude = magnitude_ds.median_magnitude
    
    # Flatten the arrays and remove NaN values
    print("  Flattening data and removing NaNs...")
    
    # Convert to numpy arrays (compute if Dask)
    durations_flat = durations.values.flatten()
    avg_magnitude_flat = avg_magnitude.values.flatten()
    max_magnitude_flat = max_magnitude.values.flatten()
    median_magnitude_flat = median_magnitude.values.flatten()
    
    # Remove NaN values and invalid durations
    valid_mask = (~np.isnan(durations_flat)) & (durations_flat > 0) & \
                 (~np.isnan(avg_magnitude_flat)) & (~np.isnan(max_magnitude_flat))
    
    durations_valid = durations_flat[valid_mask]
    avg_magnitude_valid = avg_magnitude_flat[valid_mask]
    max_magnitude_valid = max_magnitude_flat[valid_mask]
    median_magnitude_valid = median_magnitude_flat[valid_mask]
    
    print(f"  Valid data points: {len(durations_valid)}")
    
    # Sample data if too many points for scatter plot
    if len(durations_valid) > max_points:
        print(f"  Sampling {max_points} points for clarity...")
        indices = np.random.choice(len(durations_valid), max_points, replace=False)
        durations_valid = durations_valid[indices]
        avg_magnitude_valid = avg_magnitude_valid[indices]
        max_magnitude_valid = max_magnitude_valid[indices]
        median_magnitude_valid = median_magnitude_valid[indices]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Duration vs Average Magnitude
    sc1 = axes[0].scatter(durations_valid, avg_magnitude_valid, alpha=0.6, s=1, c='blue')
    axes[0].set_xlabel('Duration (days)')
    axes[0].set_ylabel('Average Magnitude (°C)')
    axes[0].set_title('Duration vs Average Magnitude')
    axes[0].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_avg = np.corrcoef(durations_valid, avg_magnitude_valid)[0,1]
    axes[0].text(0.05, 0.95, f'Correlation: {corr_avg:.3f}', 
                transform=axes[0].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Duration vs Maximum Magnitude
    sc2 = axes[1].scatter(durations_valid, max_magnitude_valid, alpha=0.6, s=1, c='red')
    axes[1].set_xlabel('Duration (days)')
    axes[1].set_ylabel('Maximum Magnitude (°C)')
    axes[1].set_title('Duration vs Maximum Magnitude')
    axes[1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_max = np.corrcoef(durations_valid, max_magnitude_valid)[0,1]
    axes[1].text(0.05, 0.95, f'Correlation: {corr_max:.3f}', 
                transform=axes[1].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Duration vs Median Magnitude
    sc3 = axes[2].scatter(durations_valid, median_magnitude_valid, alpha=0.6, s=1, c='green')
    axes[2].set_xlabel('Duration (days)')
    axes[2].set_ylabel('Median Magnitude (°C)')
    axes[2].set_title('Duration vs Median Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_median = np.corrcoef(durations_valid, median_magnitude_valid)[0,1]
    axes[2].text(0.05, 0.95, f'Correlation: {corr_median:.3f}', 
                transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(f'MHW Duration vs Magnitude Relationships - {model_name}\n'
                f'{len(durations_valid)} events', fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig, axes

# All-in-one function to create all plots (updated for multiple models)
def create_all_magnitude_plots(mhw_events_dict, magnitude_dict, model_names=None,
                              vmin_avg_magnitude=None, vmax_avg_magnitude=None,
                              vmin_avg_of_max_magnitude=None, vmax_avg_of_max_magnitude=None,
                              vmin_max_magnitude=None, vmax_max_magnitude=None):
    """
    Create all magnitude visualization plots for multiple models
    
    Parameters:
    -----------
    mhw_events_dict : dict
        Dictionary with MHW event datasets
    magnitude_dict : dict
        Dictionary with magnitude datasets
    model_names : list, optional
        List of model names to plot
    vmin_*, vmax_* : float, optional
        Colorbar limits for different magnitude plots
    """
    if model_names is None:
        model_names = list(magnitude_dict.keys())
    
    for model_name in model_names:
        print(f"Creating all magnitude plots for {model_name}...")
        
        if model_name not in mhw_events_dict or model_name not in magnitude_dict:
            print(f"  Warning: Data not found for {model_name}, skipping...")
            continue
        
        # 1. Average magnitude map
        fig1, ax1 = plot_avg_magnitude_map(magnitude_dict[model_name], model_name,
                                          vmin=vmin_avg_magnitude, vmax=vmax_avg_magnitude)
        plt.show()
        
        # 2. Average of maximum magnitudes map
        fig2, ax2 = plot_avg_of_max_magnitude_map(magnitude_dict[model_name], model_name,
                                                 vmin=vmin_avg_of_max_magnitude, vmax=vmax_avg_of_max_magnitude)
        plt.show()
        
        # 3. Maximum magnitude map  
        fig3, ax3 = plot_max_magnitude_map(magnitude_dict[model_name], model_name,
                                          vmin=vmin_max_magnitude, vmax=vmax_max_magnitude)
        plt.show()
        
        # 4. Duration vs magnitude scatter plots
        fig4, axes4 = plot_duration_magnitude_scatter(mhw_events_dict[model_name], 
                                                     magnitude_dict[model_name], model_name)
        plt.show()






# ===================================================================================================================================================
# SST TIME SERIES PLOTTING FUNCTIONS (WITH MHW HIGHLIGHTING)
# ===================================================================================================================================================


def plot_sst_with_event_markers(sst, ssta, thresholds, mhw_events, lat, lon, time_slice=None):
    """
    Simple version: Just mark MHW event periods with vertical lines
    """
    # Select data
    sst_pt = sst.sel(lat=lat, lon=lon, method='nearest')
    ssta_pt = ssta.sel(lat=lat, lon=lon, method='nearest')
    thresh_pt = thresholds.sel(lat=lat, lon=lon, method='nearest')
    events_pt = mhw_events.sel(lat=lat, lon=lon, method='nearest')
    
    # Get actual coordinates
    actual_lat = float(sst_pt.lat)
    actual_lon = float(sst_pt.lon)
    
    # Apply time slice
    if time_slice is not None:
        sst_pt = sst_pt.sel(time=time_slice)
        ssta_pt = ssta_pt.sel(time=time_slice)
    
    # Calculate
    clim = sst_pt - ssta_pt
    doy = sst_pt.time.dt.dayofyear
    thresh_anom = thresh_pt.sel(dayofyear=doy, method='nearest')
    thresh_abs = thresh_anom + clim
    
    # Compute
    data = xr.Dataset({
        'sst': sst_pt,
        'clim': clim,
        'thresh': thresh_abs
    }).compute()
    
    events_data = events_pt.compute()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_dates = mdates.date2num(data['sst'].time.values.astype('datetime64[s]'))
    
    # Plot lines
    ax.plot(time_dates, data['sst'].values, 'k-', label='SST', linewidth=2)
    ax.plot(time_dates, data['clim'].values, 'b-', label='Climatology', linewidth=1.5, alpha=0.7)
    ax.plot(time_dates, data['thresh'].values, 'g-', label='Threshold', linewidth=1.5, alpha=0.7)
    
    # Fill exceedances
    exceed = data['sst'].values > data['thresh'].values
    ax.fill_between(time_dates, 
                    data['thresh'].values,
                    data['sst'].values,
                    where=exceed,
                    color='red', 
                    alpha=0.3)
    
    # ============================================
    # ADD SIMPLE EVENT MARKERS
    # ============================================
    event_count = int(events_data.event_count.values)
    
    if event_count > 0:
        event_starts = events_data.event_start_times[:event_count].values
        event_ends = events_data.event_end_times[:event_count].values
        
        plot_start = data['sst'].time.values[0]
        plot_end = data['sst'].time.values[-1]
        
        # Add vertical lines at event starts and ends
        for i in range(event_count):
            start = event_starts[i]
            end = event_ends[i]
            
            if (start <= plot_end) and (end >= plot_start):
                start_date = mdates.date2num(start.astype('datetime64[s]'))
                end_date = mdates.date2num(end.astype('datetime64[s]'))
                
                # Add vertical line at event start
                ax.axvline(start_date, color='purple', alpha=0.5, linestyle='--', linewidth=1)
                # Add vertical line at event end
                ax.axvline(end_date, color='purple', alpha=0.5, linestyle='--', linewidth=1)
        
        # Add a single purple patch to legend
        from matplotlib.lines import Line2D
        event_line = Line2D([0], [0], color='purple', alpha=0.5, linestyle='--', 
                           label=f'MHW Events ({event_count})')
    
    # ============================================
    # SIMPLIFIED DATE FORMATTING (NO COMPLEX LOGIC)
    # ============================================
    ax.xaxis_date()
    
    # Use matplotlib's AutoDateLocator which handles everything automatically
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    
    # Auto-format the x-axis dates
    fig.autofmt_xdate(rotation=30, ha='right')
    
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Date')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if event_count > 0:
        handles.append(event_line)
        labels.append(f'MHW Events ({event_count})')
    
    # Add custom entry for red fill
    from matplotlib.patches import Patch
    red_patch = Patch(facecolor='red', alpha=0.3, label='SST > Threshold')
    handles.append(red_patch)
    
    ax.legend(handles=handles)
    ax.grid(True, alpha=0.3)
    
    # Title
    if time_slice is not None:
        start_date = str(data['sst'].time.values[0])[:10]
        end_date = str(data['sst'].time.values[-1])[:10]
        title = f'lat={actual_lat:}, lon={actual_lon:} ({start_date} to {end_date})'
    else:
        title = f'lat={actual_lat:}, lon={actual_lon:}'
    
    if event_count > 0:
        title += f' - {event_count} MHW events'
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def plot_sst_trim_mhw_events(sst, ssta, thresholds, mhw_events, lat, lon, 
                             time_slice=None, trim_to_events=False, buffer_days=10):
    """
    Plot SST, climatology, threshold, fill exceedances, and mark MHW events
    
    Parameters:
    -----------
    sst : xarray.DataArray
        Sea surface temperature
    ssta : xarray.DataArray  
        Sea surface temperature anomaly
    thresholds : xarray.DataArray
        Threshold anomalies
    mhw_events : xarray.Dataset
        MHW events dataset with event_start_times, event_end_times, etc.
    lat, lon : float
        Coordinates (will find nearest grid point)
    time_slice : slice, str, or None
        Time range to plot
    trim_to_events : bool
        If True, show only time periods around MHW events (with buffer)
    buffer_days : int
        Number of days before/after each event to include (if trim_to_events=True)
    """
    
    # Select data at point - get the actual coordinates selected
    sst_pt = sst.sel(lat=lat, lon=lon, method='nearest')
    ssta_pt = ssta.sel(lat=lat, lon=lon, method='nearest')
    thresh_pt = thresholds.sel(lat=lat, lon=lon, method='nearest')
    events_pt = mhw_events.sel(lat=lat, lon=lon, method='nearest')
    
    # Get ACTUAL coordinates selected
    actual_lat = float(sst_pt.lat)
    actual_lon = float(sst_pt.lon)
    
    # Compute MHW events data first to know which events we have
    events_data = events_pt.compute()
    event_count = int(events_data.event_count.values)
    
    # ============================================
    # HANDLE TIME TRIMMING IF REQUESTED
    # ============================================
    if trim_to_events and event_count > 0:
        # Get event start and end times
        event_starts = events_data.event_start_times[:event_count].values
        event_ends = events_data.event_end_times[:event_count].values
        
        # Create a list of time ranges to include (event ± buffer)
        time_ranges = []
        for i in range(event_count):
            # Convert numpy datetime64 to python datetime for easier manipulation
            start_date = pd.Timestamp(event_starts[i]).to_pydatetime()
            end_date = pd.Timestamp(event_ends[i]).to_pydatetime()
            
            # Apply buffer
            buffer_start = start_date - timedelta(days=buffer_days)
            buffer_end = end_date + timedelta(days=buffer_days)
            
            time_ranges.append((buffer_start, buffer_end))
        
        # If we have multiple events, we need to create a mask for the time dimension
        # First, get the full time range from the original data
        if time_slice is not None:
            sst_pt_full = sst_pt.sel(time=time_slice)
        else:
            sst_pt_full = sst_pt
        
        # Also get ssta_pt for the same time range
        if time_slice is not None:
            ssta_pt_full = ssta_pt.sel(time=time_slice)
        else:
            ssta_pt_full = ssta_pt
        
        # Create a mask that's False everywhere
        time_mask = np.zeros(len(sst_pt_full.time), dtype=bool)
        
        # For each event range, mark True in the mask
        for buffer_start, buffer_end in time_ranges:
            # Convert to numpy datetime64 for comparison
            buffer_start_np = np.datetime64(buffer_start)
            buffer_end_np = np.datetime64(buffer_end)
            
            # Find indices within this range
            in_range = (sst_pt_full.time >= buffer_start_np) & (sst_pt_full.time <= buffer_end_np)
            time_mask = time_mask | in_range.values
        
        # Apply the mask to get only the event periods
        sst_pt = sst_pt_full.isel(time=time_mask)
        ssta_pt = ssta_pt_full.isel(time=time_mask)
        
        print(f"Trimmed to {np.sum(time_mask)} days around {event_count} MHW events")
        
    elif time_slice is not None:
        # Apply regular time slice
        sst_pt = sst_pt.sel(time=time_slice)
        ssta_pt = ssta_pt.sel(time=time_slice)
    
    # ============================================
    # CALCULATIONS AND PLOTTING
    # ============================================
    
    # Calculate climatology and thresholds
    clim = sst_pt - ssta_pt
    doy = sst_pt.time.dt.dayofyear
    thresh_anom = thresh_pt.sel(dayofyear=doy, method='nearest')
    thresh_abs = thresh_anom + clim
    
    # Compute once
    data = xr.Dataset({
        'sst': sst_pt,
        'clim': clim,
        'thresh': thresh_abs
    }).compute()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Time conversion
    time_vals = data['sst'].time.values
    time_dates = mdates.date2num(time_vals.astype('datetime64[s]'))
    
    # ============================================
    # NEW: Create continuous x-axis with breaks marked
    # ============================================
    if trim_to_events and event_count > 0 and len(time_vals) > 0:
        # Find gaps between event periods
        time_diffs = np.diff(time_vals).astype('timedelta64[D]').astype(int)
        
        # Find indices where gap > 1 day (indicates a break between events)
        break_indices = np.where(time_diffs > 1)[0]
        
        # Create continuous x-axis values (days from start)
        # This makes all segments appear back-to-back
        x_values = np.arange(len(time_vals))
        
        # Plot the continuous line
        ax.plot(x_values, data['sst'].values, 'k-', label='SST', linewidth=2)
        ax.plot(x_values, data['clim'].values, 'b-', label='Climatology', linewidth=1.5, alpha=0.7)
        ax.plot(x_values, data['thresh'].values, 'g-', label='Threshold', linewidth=1.5, alpha=0.7)
        
        # Fill where SST > threshold
        exceed = data['sst'].values > data['thresh'].values
        if np.any(exceed):
            ax.fill_between(x_values, 
                          data['thresh'].values,
                          data['sst'].values,
                          where=exceed,
                          color='red', 
                          alpha=0.3,
                          label='SST > Threshold')
        
        # ============================================
        # ADD VERTICAL DASHED BLACK LINES at break points
        # ============================================
        if len(break_indices) > 0:
            # Create custom x-axis ticks at segment starts
            segment_starts = [0] + [idx + 1 for idx in break_indices]
            
            # Get corresponding dates for each segment start
            segment_dates = [pd.Timestamp(time_vals[start]).strftime('%Y-%m-%d') 
                           for start in segment_starts]
            
            # Set x-ticks at segment starts
            ax.set_xticks([x_values[start] for start in segment_starts])
            ax.set_xticklabels(segment_dates, rotation=45, ha='right')
            
            # Add vertical dashed black lines at break points
            for break_idx in break_indices:
                # Position the line between segments
                # The break is between index break_idx and break_idx + 1
                break_x = x_values[break_idx] + 0.5  # Midpoint between the two points
                
                # Add vertical dashed black line
                ax.axvline(break_x, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
            
        else:
            # Single segment, just format normally
            ax.set_xticks([x_values[0], x_values[-1]])
            ax.set_xticklabels([
                pd.Timestamp(time_vals[0]).strftime('%Y-%m-%d'),
                pd.Timestamp(time_vals[-1]).strftime('%Y-%m-%d')
            ], rotation=45, ha='right')
        
        # Set x-axis label
        ax.set_xlabel('Event Timeline (vertical lines indicate time gaps)')
        
    else:
        # Original plotting for non-trimmed or no events case
        ax.plot(time_dates, data['sst'].values, 'k-', label='SST', linewidth=2)
        ax.plot(time_dates, data['clim'].values, 'b-', label='Climatology', linewidth=1.5, alpha=0.7)
        ax.plot(time_dates, data['thresh'].values, 'g-', label='Threshold', linewidth=1.5, alpha=0.7)
        
        # Fill where SST > threshold
        exceed = data['sst'].values > data['thresh'].values
        if np.any(exceed):
            ax.fill_between(time_dates, 
                          data['thresh'].values,
                          data['sst'].values,
                          where=exceed,
                          color='red', 
                          alpha=0.3,
                          label='SST > Threshold')
        
        # Format x-axis for normal time series
        ax.xaxis_date()
        n_dates = len(time_dates)
        if n_dates <= 90:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        elif n_dates <= 365:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_xlabel('Date')
    
    # ============================================
    # ADD MHW EVENT MARKERS (from pre-computed events)
    # ============================================
    events_in_plot = 0
    if event_count > 0:
        # Get events again
        event_starts = events_data.event_start_times[:event_count].values
        event_ends = events_data.event_end_times[:event_count].values
        event_durations = events_data.event_durations[:event_count].values
        
        # Filter events that fall within our current plot time range
        plot_start = time_vals[0] if len(time_vals) > 0 else None
        plot_end = time_vals[-1] if len(time_vals) > 0 else None
        
        if plot_start is not None and plot_end is not None:
            for i in range(event_count):
                start = event_starts[i]
                end = event_ends[i]
                
                # Check if event overlaps with our plot time range
                if (start <= plot_end) and (end >= plot_start):
                    events_in_plot += 1
                    
                    # Find where in our continuous timeline this event occurs
                    if trim_to_events:
                        # Find indices of this event in our time_vals
                        event_mask = (time_vals >= start) & (time_vals <= end)
                        event_indices = np.where(event_mask)[0]
                        
                        if len(event_indices) > 0:
                            # Convert to x_values positions
                            event_start_x = x_values[event_indices[0]]
                            event_end_x = x_values[event_indices[-1]]
                            
                            # Add vertical span for the event
                            ax.axvspan(event_start_x, event_end_x, 
                                      alpha=0.1,  # Very light shading
                                      color='purple',
                                      label='MHW Event' if i == 0 else "")
                    else:
                        # For non-trimmed plot, use time_dates
                        start_date = mdates.date2num(start.astype('datetime64[s]'))
                        end_date = mdates.date2num(end.astype('datetime64[s]'))
                        
                        ax.axvspan(start_date, end_date, 
                                  alpha=0.1,
                                  color='purple',
                                  label='MHW Event' if i == 0 else "")
        
        # Add a legend entry for MHW events
        if events_in_plot > 0:
            from matplotlib.patches import Patch
            mhw_patch = Patch(facecolor='purple', alpha=0.1, label=f'MHW Events ({events_in_plot} shown)')
    
    # ============================================
    # FORMATTING
    # ============================================
    
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True, alpha=0.3)
    
    # Handle legend - MOVE IT OUTSIDE THE GRAPH
    handles, labels = ax.get_legend_handles_labels()
    if event_count > 0 and events_in_plot > 0:
        handles.append(mhw_patch)
    
    # Remove duplicate labels
    unique = dict(zip(labels, handles))
    
    # Place legend outside the plot to the right
    ax.legend(unique.values(), unique.keys(), 
              loc='upper left', 
              bbox_to_anchor=(1.02, 1),  # Move to the right of the plot
              borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    plt.subplots_adjust(right=0.8)  # Make room for legend on the right
    
    # Title
    if len(time_vals) > 0:
        start_date = str(time_vals[0])[:10]
        end_date = str(time_vals[-1])[:10]
        
        if trim_to_events:
            title = f'lat={actual_lat:}, lon={actual_lon:} - MHW Events Only'
            if event_count > 0:
                title += f' ({events_in_plot} events, ±{buffer_days} days buffer)'
            if len(break_indices) > 0:
                title += f'\n{len(break_indices) + 1} segments shown back-to-back'
        else:
            title = f'lat={actual_lat:}, lon={actual_lon:} ({start_date} to {end_date})'
            if event_count > 0:
                title += f' - {events_in_plot} MHW events in range'
    else:
        title = f'lat={actual_lat:}, lon={actual_lon:} - No data in selected time range'
    
    ax.set_title(title)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    
    # Print summary
    if trim_to_events and event_count > 0 and len(time_vals) > 0:
        print(f"\nDisplaying {len(time_vals)} days around {events_in_plot} MHW events:")
        print("-" * 60)
        
        # Group events by segments
        if len(break_indices) > 0:
            segments = []
            start_idx = 0
            for break_idx in break_indices:
                end_idx = break_idx
                segments.append((start_idx, end_idx))
                start_idx = break_idx + 1
            segments.append((start_idx, len(time_vals) - 1))
            
            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                seg_start_date = pd.Timestamp(time_vals[seg_start]).strftime('%Y-%m-%d')
                seg_end_date = pd.Timestamp(time_vals[seg_end]).strftime('%Y-%m-%d')
                seg_length = seg_end - seg_start + 1
                
                print(f"Segment {seg_idx+1}: {seg_start_date} to {seg_end_date} ({seg_length} days)")
        
        print("\nMHW Events in plot:")
        for i in range(min(5, event_count)):
            start = event_starts[i]
            end = event_ends[i]
            duration = event_durations[i]
            start_py = pd.Timestamp(start).strftime('%Y-%m-%d')
            end_py = pd.Timestamp(end).strftime('%Y-%m-%d')
            
            # Check if this event is in our current plot
            in_plot = (start <= plot_end) and (end >= plot_start)
            marker = "✓" if in_plot else " "
            
            if in_plot:
                print(f"{marker} Event {i+1}: {start_py} to {end_py} ({duration} days)")
        
        if event_count > 5:
            print(f"... and {event_count - 5} more events")
        print("-" * 60)
    
    return fig, events_data



# ===================================================================================
# POINTS VISUALIZATION ON MASKS
# ===================================================================================

def plot_points_on_masks(masks_dict, model_name, points_of_interest=None, figsize=(12, 8), 
                        central_longitude=180, point_size=100, point_color='red', 
                        point_labels=True, label_fontsize=8, label_offset=(2, 1), 
                        label_alpha=0.7, title=None):
    """
    Plot points of interest on top of the combined regions mask
    
    Parameters:
    -----------
    masks_dict : dict
        Dictionary with model-specific masks (from create_model_specific_masks)
    model_name : str
        Name of model to plot
    points_of_interest : list of dicts, optional
        List of points with 'lat', 'lon', and optionally 'name' and 'color'
        Example: [{'lat': 0, 'lon': 160, 'name': 'Western_EQ_1'}, ...]
    figsize : tuple
        Figure size
    central_longitude : float
        Central longitude for map projection
    point_size : int or list
        Size of points (can be single value or list matching points_of_interest)
    point_color : str or list
        Color of points (can be single value or list matching points_of_interest)
    point_labels : bool
        Whether to show point labels
    label_fontsize : int
        Font size for point labels (default: 8)
    label_offset : tuple
        (x_offset, y_offset) in degrees for label positioning
    label_alpha : float
        Transparency of label background (0-1)
    title : str, optional
        Custom title for the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    if model_name not in masks_dict:
        raise ValueError(f"Model '{model_name}' not found in masks dictionary")
    
    # Create the base combined regions plot
    fig, ax = plot_combined_regions_mask(masks_dict, model_name, figsize=figsize, 
                                         central_longitude=central_longitude)
    
    # If no points provided, just return the base plot
    if points_of_interest is None or len(points_of_interest) == 0:
        return fig, ax
    
    # Prepare point data
    lats = []
    lons = []
    names = []
    colors = []
    
    for i, point in enumerate(points_of_interest):
        lats.append(point.get('lat'))
        lons.append(point.get('lon'))
        names.append(point.get('name', f'P{i+1}'))  # Shorter default names
        
        # Handle point colors
        if isinstance(point_color, list) and i < len(point_color):
            colors.append(point_color[i])
        elif 'color' in point:
            colors.append(point['color'])
        else:
            colors.append(point_color)
    
    # Handle point sizes
    if isinstance(point_size, list):
        sizes = point_size[:len(lats)]
    else:
        sizes = [point_size] * len(lats)
    
    # Plot the points FIRST (lower zorder)
    scatter = ax.scatter(lons, lats, transform=ccrs.PlateCarree(),
                        s=sizes, c=colors, edgecolor='black', 
                        linewidth=1.5, zorder=5, alpha=0.9)
    
    # Add point labels if requested
    if point_labels:
        offset_x, offset_y = label_offset
        
        for i, (lon, lat, name) in enumerate(zip(lons, lats, names)):
            # Alternate label positions to avoid overlap
            if i % 4 == 0:
                ha, va = 'right', 'bottom'  # Left of point
                x_offset = -offset_x
                y_offset = offset_y
            elif i % 4 == 1:
                ha, va = 'left', 'bottom'   # Right of point
                x_offset = offset_x
                y_offset = offset_y
            elif i % 4 == 2:
                ha, va = 'center', 'top'    # Above point
                x_offset = 0
                y_offset = offset_y
            else:
                ha, va = 'center', 'bottom' # Below point
                x_offset = 0
                y_offset = -offset_y
            
            # Add label with smaller font and more transparency
            ax.text(lon + x_offset, lat + y_offset, name,
                   transform=ccrs.PlateCarree(),
                   fontsize=label_fontsize, 
                   fontweight='normal',  # Not bold
                   bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor='white', 
                            alpha=label_alpha,  # More transparent
                            edgecolor='gray',  # Lighter edge
                            linewidth=0.5),
                   ha=ha, va=va, zorder=6)  # Slightly higher than points
    
    # Update title if provided
    if title is None:
        title = f'Extremes Regions with Points of Interest - {model_name}'
    
    ax.set_title(title, fontsize=14, pad=20)
    
    # Create a separate legend for points (simpler)
    if len(points_of_interest) <= 10:  # Only add legend if not too many points
        from matplotlib.lines import Line2D
        legend_elements = []
        
        for i, (point, color) in enumerate(zip(points_of_interest, colors)):
            name = point.get('name', f'Point {i+1}')
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markeredgecolor='black',
                      markersize=8, label=name)
            )
        
        # Add point legend to existing legend
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(legend_elements)
        
        ax.legend(handles=handles, 
                  loc='center left', 
                  bbox_to_anchor=(1.05, 0.5),
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=9)  # Smaller font in legend
    
    plt.tight_layout()
    
    return fig, ax

def plot_point_model_comparison_subplots(point, models_data_mapping, buffer_days=15, 
                                        time_slice=None, figsize=(18, 5)):
    """
    Plot a single point for all models in subplots (side-by-side comparison)
    FIXED: Each subplot now has its own independent y-axis for proper visualization
    
    Parameters:
    -----------
    point : dict
        Dictionary with 'name', 'lat', 'lon'
    models_data_mapping : dict
        Dictionary with model names as keys and model data as values
        Each model data should have: 'sst', 'ssta', 'thresholds', 'mhw_events'
    buffer_days : int
        Number of days before/after each event to include
    time_slice : slice, optional
        Time range to plot
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure with subplots
    axes : list
        List of axes objects
    events_data_list : list
        List of events data for each model
    """
    
    models = list(models_data_mapping.keys())
    n_models = len(models)
    
    # Create figure with subplots - DO NOT share y-axis
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=False)
    
    # Handle case when there's only one model
    if n_models == 1:
        axes = [axes]
    
    events_data_list = []
    
    # Plot each model in its own subplot
    for idx, (model_name, ax) in enumerate(zip(models, axes)):
        model_data = models_data_mapping[model_name]
        
        try:
            # Create individual plot for this model
            fig_single, events_data = plot_sst_trim_mhw_events(
                model_data['sst'],
                model_data['ssta'],
                model_data['thresholds'],
                model_data['mhw_events'],
                lat=point['lat'],
                lon=point['lon'],
                time_slice=time_slice,
                trim_to_events=True,
                buffer_days=buffer_days
            )
            
            # Get the single plot's axis
            ax_single = fig_single.axes[0]
            
            # Copy all artists from single plot to subplot
            # 1. Copy lines (SST, climatology, threshold)
            for line in ax_single.get_lines():
                ax.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(),
                       linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth() * 0.8,  # Slightly thinner for subplot
                       alpha=line.get_alpha(),
                       label=line.get_label())
            
            # 2. Copy filled areas (SST > threshold)
            for collection in ax_single.collections:
                if hasattr(collection, 'get_facecolor'):
                    # This is a filled area (SST > threshold)
                    paths = collection.get_paths()
                    for path in paths:
                        patch = mpatches.PathPatch(path,
                                                  facecolor=collection.get_facecolor()[0],
                                                  alpha=collection.get_alpha() or 0.3,
                                                  edgecolor='none')
                        ax.add_patch(patch)
            
            # 3. Copy vertical spans (MHW events)
            for patch in ax_single.patches:
                if isinstance(patch, mpatches.Rectangle):
                    # This is a vertical span for MHW events
                    new_patch = mpatches.Rectangle(
                        patch.get_xy(),
                        patch.get_width(),
                        patch.get_height(),
                        facecolor=patch.get_facecolor(),
                        alpha=patch.get_alpha() or 0.1,
                        edgecolor='none'
                    )
                    ax.add_patch(new_patch)
            
            # 4. Copy vertical lines (event starts/ends)
            for line in ax_single.lines:
                if line.get_linestyle() == '--':  # Event markers
                    ax.axvline(line.get_xdata()[0],
                              color=line.get_color(),
                              linestyle=line.get_linestyle(),
                              alpha=line.get_alpha(),
                              linewidth=line.get_linewidth())
            
            # 5. Copy x-axis formatting
            ax.xaxis_date()
            ax.xaxis.set_major_locator(ax_single.xaxis.get_major_locator())
            ax.xaxis.set_major_formatter(ax_single.xaxis.get_major_formatter())
            
            # Auto-format x-axis dates
            ax.figure.autofmt_xdate(rotation=90, ha='right')
            
            # Set subplot title with model name and event count
            event_count = int(events_data.event_count.values)
            ax.set_title(f"{model_name}\n{event_count} events", fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set y-label for each subplot (not just first one)
            ax.set_ylabel('Temperature (°C)', fontsize=11)
            
            # Add x-label for each subplot
            ax.set_xlabel('Date', fontsize=10)
            
            # CRITICAL FIX: Set y-limits for EACH subplot independently
            if ax_single.get_ylim():
                ylim = ax_single.get_ylim()
                # Add a small padding to y-axis for better visualization
                y_padding = (ylim[1] - ylim[0]) * 0.05
                ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)
            
            # Close the single figure to save memory
            plt.close(fig_single)
            
            events_data_list.append(events_data)
            
            print(f"  {model_name}: {event_count} events (temp range: {ylim[0]:.1f} to {ylim[1]:.1f}°C)")
            
        except Exception as e:
            # If there's an error, show error message in subplot
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}...",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=9, color='red', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_title(f"{model_name}\nERROR", fontsize=12, color='red')
            ax.grid(False)
            events_data_list.append(None)
            print(f"  {model_name}: Error - {str(e)[:50]}...")
    
    # Main title for the entire figure
    fig.suptitle(f"{point['name']} (lat={point['lat']:.1f}, lon={point['lon']:.1f})",
                fontsize=14, y=1.02)
    
    # Add a common legend (using handles/labels from first subplot)
    if axes[0].get_legend_handles_labels()[0]:
        # Get unique handles and labels across all subplots
        all_handles = []
        all_labels = []
        
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in all_labels:
                    all_handles.append(handle)
                    all_labels.append(label)
        
        # Add legend to the right of the figure
        fig.legend(all_handles, all_labels,
                  loc='upper left',
                  bbox_to_anchor=(1.02, 0.9),
                  fontsize=9,
                  title='Legend',
                  title_fontsize=10)
    
    plt.tight_layout()
    
    return fig, axes, events_data_list


import matplotlib.dates as mdates

def plot_all_points_model_comparisons(point, models_data_mapping, buffer_days=5):
    """
    Plots model comparisons in a 2x3 grid for better readability.
    """
    models = list(models_data_mapping.keys())
    n_models = len(models)
    
    # Define grid: max 3 columns, calculate rows needed
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Adjust figsize: wider for 3 columns, taller for multiple rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=False)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten() if n_models > 1 else [axes]
    
    for idx, model_name in enumerate(models):
        ax = axes_flat[idx]
        m = models_data_mapping[model_name]
        
        # Call your ORIGINAL working function
        fig_tmp, _ = plot_sst_trim_mhw_events(
            m['sst'], m['ssta'], m['thresholds'], m['mhw_events'],
            lat=point['lat'], lon=point['lon'], buffer_days=buffer_days
        )
        
        if fig_tmp:
            ax_src = fig_tmp.axes[0]
            
            # Transfer lines (SST, Climatology, Threshold)
            for line in ax_src.get_lines():
                ax.plot(line.get_xdata(), line.get_ydata(), 
                        color=line.get_color(), linestyle=line.get_linestyle(), 
                        label=line.get_label())
            
            # Transfer the shaded heatwave areas
            for poly in ax_src.collections:
                # We re-create the fill to ensure it scales correctly to the new axis
                ax.add_collection(poly)
            
            ax.set_title(f"Model: {model_name}", fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            
            # Improve date formatting on the X-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            plt.close(fig_tmp) # Close temporary figure to save memory

    # Hide empty subplots if the grid isn't full (e.g., 5 models in a 6-slot grid)
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    fig.suptitle(f"MHW Comparison: {point['name']} (Lat: {point['lat']}, Lon: {point['lon']})", 
                 fontsize=18, y=1.02)
    
    plt.tight_layout()
    return fig


def print_summary_table(all_summaries, models_data_mapping):
    """Print a summary table of events for all points and models"""
    
    models = list(models_data_mapping.keys())
    
    print("\n" + "="*80)
    print("SUMMARY OF MHW EVENTS")
    print("="*80)
    
    # Header
    header = f"{'Point':<20} {'Lat':<6} {'Lon':<7} "
    for model in models:
        header += f"{model:^30} "
    print(header)
    
    subheader = " " * 33
    for model in models:
        subheader += f"{'Events':<8} {'AvgDur':<6} {'MaxDur':<6} {'TempMin':<8} {'TempMax':<8} "
    print(subheader)
    print("-" * 80)
    
    # Data rows
    for summary in all_summaries:
        row = f"{summary['point']:<20} {summary['lat']:<6.1f} {summary['lon']:<7.1f} "
        
        for model in models:
            events = summary.get(f'{model}_events', 'N/A')
            avg_dur = summary.get(f'{model}_avg_duration', 'N/A')
            max_dur = summary.get(f'{model}_max_duration', 'N/A')
            temp_min = summary.get(f'{model}_temp_min', 'N/A')
            temp_max = summary.get(f'{model}_temp_max', 'N/A')
            
            if isinstance(avg_dur, (int, float)):
                avg_dur_str = f"{avg_dur:.1f}"
            else:
                avg_dur_str = str(avg_dur)
                
            if isinstance(max_dur, (int, float)):
                max_dur_str = f"{max_dur:.0f}"
            else:
                max_dur_str = str(max_dur)
            
            if isinstance(temp_min, (int, float)):
                temp_min_str = f"{temp_min:.1f}"
            else:
                temp_min_str = str(temp_min)
                
            if isinstance(temp_max, (int, float)):
                temp_max_str = f"{temp_max:.1f}"
            else:
                temp_max_str = str(temp_max)
            
            row += f"{events:<8} {avg_dur_str:<6} {max_dur_str:<6} {temp_min_str:<8} {temp_max_str:<8} "
        
        print(row)
    
    print("-" * 80)
    
    # Totals
    print("\nTOTALS:")
    for model in models:
        total_events = 0
        for summary in all_summaries:
            events = summary.get(f'{model}_events', 0)
            if isinstance(events, (int, float)):
                total_events += events
        
        print(f"  {model}: {total_events} total events across all points")