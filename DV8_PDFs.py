import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import dask.array as da
import geopandas as gpd
from pathlib import Path
from getpass import getuser

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

# ===================================================================================
# SELF-CONTAINED MODEL-SPECIFIC MASK FUNCTIONS (PDF-SPECIFIC)
# ===================================================================================

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

def create_pdf_specific_shapefile_mask(data_array, model_name=None, shapefile_path=None, mask_save_dir=None):
    """
    Create oceanic regions mask for PDF analysis with PDF-specific directory and naming
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
    
    # Create PDF-specific mask file path
    if mask_save_dir is None:
        mask_save_dir = Path('/scratch') / getuser()[0] / getuser() / 'mhws' / 'DV8' / 'pdf_model_masks'
    
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PDF-specific unique identifier
    grid_hash = f"pdf_model_{lats.shape[0]}x{lons.shape[0]}_{hash(str(lats[:5].tobytes()) + str(lons[:5].tobytes()))}"
    mask_file = mask_save_dir / f"{grid_hash}_region_masks.zarr"
    
    # Check if PDF masks already exist for this specific grid
    if mask_file.exists():
        print(f"Loading existing PDF masks for model grid {lats.shape} x {lons.shape}...")
        region_masks_ds = xr.open_zarr(str(mask_file))
        region_masks = {var: region_masks_ds[var] for var in region_masks_ds.data_vars}
        return region_masks
    
    print(f"Creating PDF-specific masks from shapefile for model grid {lats.shape} x {lons.shape}...")
    
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

    # Save PDF-specific masks to Zarr
    region_masks_ds = xr.Dataset(region_masks)
    region_masks_ds.to_zarr(str(mask_file))
    print(f"PDF-specific masks saved to {mask_file}")

    return region_masks

def create_model_specific_masks(models_dict, shapefile_path=None, mask_save_dir=None):
    """
    Create PDF-specific masks for each model based on their specific grid
    """
    print("Creating PDF-specific model masks...")
    masks_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Creating PDF masks for {model_name}...")
        
        data_array = extract_data_array(model_value)
        
        # Create PDF-specific masks for this model's grid
        masks_dict[model_name] = create_pdf_specific_shapefile_mask(
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
    
    fig.suptitle(f'PDF Oceanic Region Masks - {model_name}', fontsize=16, y=0.95)
    plt.tight_layout()
    
    return fig, axes

def plot_combined_regions_mask(masks_dict, model_name, figsize=(12, 8), central_longitude=180):
    """
    Plot a combined map showing all regions with different colors
    FIXED: Properly handle region assignment to prevent overwriting
    UPDATED: 'No region' now in white instead of lightblue
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
    # Start with all zeros (no region)
    first_mask = model_masks[regions[0]]
    combined_data = np.zeros(first_mask.shape, dtype=int)
    
    # Assign unique values to each region in priority order
    # This ensures higher priority regions don't get overwritten by lower priority ones
    priority_order = [
        'Southern_Ocean',
        'North_Pacific_SubTropics', 'North_Pacific_MiddleLats', 'South_Pacific_SubTropics',
        'North_Atlantic_SubTropics', 'North_Atlantic_MiddleLats', 'South_Atlantic_SubTropics', 
        'Indian_NorthSubTropics', 'Indian_SouthSubTropics',
        'Pacific_Equatorial', 'Atlantic_Equatorial', 'Indian_Equatorial',
        'Mediterranean_Sea'
    ]
    
    # Filter to only include regions that actually exist for this model
    available_regions = [r for r in priority_order if r in regions]
    
    for idx, region_name in enumerate(available_regions):
        mask = model_masks[region_name]
        # Only assign this region where no previous region has been assigned
        region_pixels = mask.values & (combined_data == 0)
        combined_data[region_pixels] = idx + 1
    
    # Create colormap for all regions
    colors = [region_colors.get(region, 'gray') for region in available_regions]
    # Change 'No region' from lightblue to white
    cmap = ListedColormap(['white'] + colors)
    
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
                      vmin=0, vmax=len(available_regions) + 0.5,
                      transform=ccrs.PlateCarree())
    
    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Create legend (include "No Region" for background)
    legend_patches = [mpatches.Patch(color='white', label='No Region')]  # Changed to white
    for region_name, color in zip(available_regions, colors):
        patch = mpatches.Patch(color=color, label=region_name.replace('_', ' ').title())
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, 
              loc='center left', 
              bbox_to_anchor=(1.05, 0.5),
              frameon=True,
              fancybox=True,
              shadow=True)
    
    ax.set_title(f'PDF Combined Oceanic Regions - {model_name}', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    return fig, ax



def quick_visualize_masks(masks_dict, model_name=None):
    """
    Quick visualization of masks for a model or all models
    """
    if model_name is None:
        model_name = list(masks_dict.keys())[0]
    
    print(f"Visualizing PDF masks for {model_name}")
    print("=" * 50)
    
    # 1. Combined regions plot
    fig1, ax1 = plot_combined_regions_mask(masks_dict, model_name)
    plt.show()
    
    # 2. Individual masks plot
    fig2, axes2 = plot_model_masks(masks_dict, model_name)
    plt.show()
    
    return fig1, fig2

# ===================================================================================
# GLOBAL PDF FUNCTIONS
# ===================================================================================

def compute_global_pdfs_classic(models_dict, bins=100, xlim=(-5, 5)):
    """
    Compute global PDFs for multiple models using classic histogram method
    """
    print("COMPUTING GLOBAL PDFS ")
    print("=" * 50)
    
    pdfs_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Processing {model_name}...")
        
        # Extract data array using helper function
        data_array = extract_data_array(model_value)
        
        # Get basic stats
        n_total = int(data_array.count().compute())
        mean_val = float(data_array.mean().compute())
        std_val = float(data_array.std().compute())
        
        print(f"  Total points: {n_total:,}")
        print(f"  Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        
        # Compute histogram with fixed range
        counts, bin_edges = da.histogram(data_array.data, bins=bins, range=xlim, density=True)
        counts = counts.compute()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        pdfs_dict[model_name] = {
            'counts': counts,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges,
            'stats': {
                'mean': mean_val,
                'std': std_val,
                'n_points': n_total
            }
        }
    
    return pdfs_dict

def plot_global_pdfs_classic(pdfs_dict, title="Global SST Anomaly PDFs"):
    """
    Plot global PDFs computed with classic histogram method
    """
    # Define colors for models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    plt.figure(figsize=(12, 8))
    
    for i, (model_name, pdf_data) in enumerate(pdfs_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(pdf_data['bin_centers'], pdf_data['counts'], 
                color=color, label=model_name, linewidth=2, alpha=0.8)
    
    plt.xlabel('SST Anomaly (°C)')
    plt.ylabel('Probability Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nGLOBAL STATISTICS:")
    print("=" * 60)
    print(f"{'Model':<15} | {'Mean':>8} | {'Std':>8} | {'Data Points':>12}")
    print("-" * 60)
    for model_name, pdf_data in pdfs_dict.items():
        stats = pdf_data['stats']
        print(f"{model_name:<15} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>12,}")

# ===================================================================================
# REGIONAL PDF FUNCTIONS WITH PDF-SPECIFIC MASKS
# ===================================================================================

def compute_regional_pdfs_classic_fast(models_dict, bins=100, xlim=(-5, 5), regions=None, 
                                      shapefile_path=None, mask_save_dir=None):
    """
    FAST VERSION: Compute regional PDFs using PDF-specific masks
    """
    print("COMPUTING REGIONAL PDFS (FAST WITH PDF-SPECIFIC MASKS)")
    print("=" * 65)
    
    # Create PDF-specific masks using the updated approach
    print("Creating PDF-specific masks...")
    masks_dict = create_model_specific_masks(
        models_dict,
        shapefile_path=shapefile_path,
        mask_save_dir=mask_save_dir
    )
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    regional_pdfs = {}
    
    for region_name in regions:
        print(f"\nProcessing region: {region_name}")
        regional_pdfs[region_name] = {}
        
        for model_name, model_value in models_dict.items():
            print(f"  {model_name}...")
            
            try:
                # Extract data array
                data_array = extract_data_array(model_value)
                
                # Get PDF-specific mask
                if region_name not in masks_dict[model_name]:
                    continue
                    
                region_mask = masks_dict[model_name][region_name]
                
                # Apply mask to create a new data array where masked values are NaN
                masked_data = data_array.where(region_mask)
                
                # Use the SAME approach as global PDFs but on the masked data
                n_total = int(masked_data.count().compute())
                mean_val = float(masked_data.mean().compute())
                std_val = float(masked_data.std().compute())
                
                # Compute histogram with fixed range using dask
                counts, bin_edges = da.histogram(masked_data.data, bins=bins, range=xlim, density=True)
                counts = counts.compute()
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                regional_pdfs[region_name][model_name] = {
                    'counts': counts,
                    'bin_centers': bin_centers,
                    'bin_edges': bin_edges,
                    'stats': {
                        'mean': mean_val,
                        'std': std_val,
                        'n_points': n_total
                    }
                }
                
                print(f"    {n_total:,} points, mean: {mean_val:.3f}")
                
            except Exception as e:
                print(f"    ERROR processing {model_name} in {region_name}: {str(e)}")
                continue
    
    return regional_pdfs, masks_dict

def compute_regional_pdfs_classic_ultrafast(models_dict, bins=100, xlim=(-5, 5), regions=None,
                                           shapefile_path=None, mask_save_dir=None):
    """
    ULTRA-FAST VERSION: Process all regions for one model at a time with PDF-specific masks
    """
    print("COMPUTING REGIONAL PDFS (ULTRA-FAST WITH PDF-SPECIFIC MASKS)")
    print("=" * 70)
    
    # Create PDF-specific masks using the updated approach
    print("Creating PDF-specific masks...")
    masks_dict = create_model_specific_masks(
        models_dict,
        shapefile_path=shapefile_path,
        mask_save_dir=mask_save_dir
    )
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    regional_pdfs = {region: {} for region in regions}
    
    # Process one model at a time to minimize memory usage
    for model_name, model_value in models_dict.items():
        print(f"\nProcessing model: {model_name}")
        
        try:
            # Extract data array once per model
            data_array = extract_data_array(model_value)
            
            # Get all masks for this model
            model_masks = masks_dict[model_name]
            
            for region_name in regions:
                if region_name not in model_masks:
                    continue
                    
                print(f"  Region: {region_name}")
                
                region_mask = model_masks[region_name]
                
                # Apply mask and compute histogram efficiently
                masked_data = data_array.where(region_mask)
                
                # Use dask's built-in histogram (fast!)
                n_total = int(masked_data.count().compute())
                mean_val = float(masked_data.mean().compute())
                std_val = float(masked_data.std().compute())
                
                counts, bin_edges = da.histogram(masked_data.data, bins=bins, range=xlim, density=True)
                counts = counts.compute()
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                regional_pdfs[region_name][model_name] = {
                    'counts': counts,
                    'bin_centers': bin_centers,
                    'bin_edges': bin_edges,
                    'stats': {
                        'mean': mean_val,
                        'std': std_val,
                        'n_points': n_total
                    }
                }
                
                print(f"    {n_total:,} points, mean: {mean_val:.3f}")
                
        except Exception as e:
            print(f"  ERROR processing {model_name}: {str(e)}")
            continue
    
    return regional_pdfs, masks_dict

def plot_regional_pdfs_classic(regional_pdfs, regions=None, n_cols=2):
    """
    Plot regional PDFs computed with classic histogram method
    """
    if regions is None:
        regions = list(regional_pdfs.keys())
    
    # Define colors for models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    n_regions = len(regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_regions == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    for region_idx, region_name in enumerate(regions):
        ax = axes[region_idx]
        
        if region_name in regional_pdfs:
            for model_idx, (model_name, pdf_data) in enumerate(regional_pdfs[region_name].items()):
                color = colors[model_idx % len(colors)]
                ax.plot(pdf_data['bin_centers'], pdf_data['counts'],
                       color=color, label=model_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('SST Anomaly (°C)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{region_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(regions), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # Print regional statistics
    print(f"\nREGIONAL STATISTICS:")
    print("=" * 80)
    for region_name in regions:
        if region_name in regional_pdfs:
            print(f"\n{region_name.replace('_', ' ').title()}:")
            print(f"{'Model':<15} | {'Mean':>8} | {'Std':>8} | {'Data Points':>12}")
            print("-" * 60)
            for model_name, pdf_data in regional_pdfs[region_name].items():
                stats = pdf_data['stats']
                print(f"{model_name:<15} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>12,}")

# ===================================================================================
# SEASONAL PDF FUNCTIONS (EQUATORIAL EXCLUDED ONLY FROM REGIONAL SEASONAL)
# ===================================================================================

def compute_global_seasonal_pdfs_classic(models_dict, bins=100, xlim=(-5, 5), by_hemisphere=False):
    """
    Compute global seasonal PDFs for multiple models using classic histogram method
    MODIFIED: Now accepts both (dataset, var_name) tuples and direct data arrays
    """
    print("COMPUTING GLOBAL SEASONAL PDFS")
    if by_hemisphere:
        print("(DIVIDED BY HEMISPHERE)")
    else:
        print("(INCLUDING EQUATORIAL REGIONS)")
    print("=" * 50)
    
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    
    seasonal_pdfs = {}
    
    for season_name, season_months in seasons.items():
        print(f"\nProcessing season: {season_name} (months {season_months})")
        seasonal_pdfs[season_name] = {}
        
        for model_name, model_value in models_dict.items():
            print(f"  {model_name}...")
            
            # Extract data array using helper function
            data_array = extract_data_array(model_value)
            
            # Select data for the season months
            if 'time' in data_array.dims:
                season_data = data_array.sel(time=data_array.time.dt.month.isin(season_months))
            else:
                print(f"    Warning: No time dimension in {model_name}, using all data")
                season_data = data_array
            
            if by_hemisphere:
                # Split by hemisphere
                if 'lat' in season_data.dims:
                    # Northern Hemisphere (lat >= 0)
                    nh_data = season_data.where(season_data.lat >= 0, drop=True)
                    # Southern Hemisphere (lat < 0)
                    sh_data = season_data.where(season_data.lat < 0, drop=True)
                    
                    # Initialize hemisphere structure
                    if 'NH' not in seasonal_pdfs[season_name]:
                        seasonal_pdfs[season_name]['NH'] = {}
                    if 'SH' not in seasonal_pdfs[season_name]:
                        seasonal_pdfs[season_name]['SH'] = {}
                    
                    # Process Northern Hemisphere
                    nh_total = int(nh_data.count().compute())
                    nh_mean = float(nh_data.mean().compute())
                    nh_std = float(nh_data.std().compute())
                    
                    nh_counts, nh_edges = da.histogram(nh_data.data, bins=bins, range=xlim, density=True)
                    nh_counts = nh_counts.compute()
                    nh_centers = (nh_edges[:-1] + nh_edges[1:]) / 2
                    
                    seasonal_pdfs[season_name]['NH'][model_name] = {
                        'counts': nh_counts,
                        'bin_centers': nh_centers,
                        'bin_edges': nh_edges,
                        'stats': {
                            'mean': nh_mean,
                            'std': nh_std,
                            'n_points': nh_total
                        }
                    }
                    
                    # Process Southern Hemisphere
                    sh_total = int(sh_data.count().compute())
                    sh_mean = float(sh_data.mean().compute())
                    sh_std = float(sh_data.std().compute())
                    
                    sh_counts, sh_edges = da.histogram(sh_data.data, bins=bins, range=xlim, density=True)
                    sh_counts = sh_counts.compute()
                    sh_centers = (sh_edges[:-1] + sh_edges[1:]) / 2
                    
                    seasonal_pdfs[season_name]['SH'][model_name] = {
                        'counts': sh_counts,
                        'bin_centers': sh_centers,
                        'bin_edges': sh_edges,
                        'stats': {
                            'mean': sh_mean,
                            'std': sh_std,
                            'n_points': sh_total
                        }
                    }
                    
                    print(f"    NH: {nh_total:,} points, mean: {nh_mean:.3f}")
                    print(f"    SH: {sh_total:,} points, mean: {sh_mean:.3f}")
                    
                else:
                    print(f"    Warning: No lat dimension in {model_name}, cannot split by hemisphere")
                    # Fall back to global without hemisphere split
                    by_hemisphere = False
            
            if not by_hemisphere:
                # Process as global (including equatorial)
                # Get basic stats
                n_total = int(season_data.count().compute())
                mean_val = float(season_data.mean().compute())
                std_val = float(season_data.std().compute())
                
                print(f"    Points: {n_total:,}, Mean: {mean_val:.3f}, Std: {std_val:.3f}")
                
                # Compute histogram with fixed range
                counts, bin_edges = da.histogram(season_data.data, bins=bins, range=xlim, density=True)
                counts = counts.compute()
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                seasonal_pdfs[season_name][model_name] = {
                    'counts': counts,
                    'bin_centers': bin_centers,
                    'bin_edges': bin_edges,
                    'stats': {
                        'mean': mean_val,
                        'std': std_val,
                        'n_points': n_total
                    }
                }
    
    return seasonal_pdfs

def plot_global_seasonal_pdfs_classic(seasonal_pdfs, by_hemisphere=False, n_cols=2):
    """
    Plot global seasonal PDFs computed with classic histogram method
    
    Parameters:
    - seasonal_pdfs: output from compute_global_seasonal_pdfs_classic
    - by_hemisphere: whether the PDFs are divided by hemisphere
    - n_cols: number of columns for subplots
    """
    # Define colors for models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    seasons = list(seasonal_pdfs.keys())
    
    if by_hemisphere:
        # Plot hemispheres separately
        hemispheres = ['NH', 'SH']
        hemisphere_names = {'NH': 'Northern Hemisphere', 'SH': 'Southern Hemisphere'}
        
        for hemisphere in hemispheres:
            print(f"\nPlotting {hemisphere_names[hemisphere]}...")
            
            n_seasons = len(seasons)
            n_rows = (n_seasons + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            if n_seasons == 1:
                axes = np.array([axes])
            axes = axes.ravel()
            
            for season_idx, season_name in enumerate(seasons):
                ax = axes[season_idx]
                
                if hemisphere in seasonal_pdfs[season_name]:
                    for model_idx, (model_name, pdf_data) in enumerate(seasonal_pdfs[season_name][hemisphere].items()):
                        color = colors[model_idx % len(colors)]
                        ax.plot(pdf_data['bin_centers'], pdf_data['counts'], 
                               color=color, label=model_name, linewidth=2, alpha=0.8)
                
                ax.set_xlabel('SST Anomaly (°C)')
                ax.set_ylabel('Probability Density')
                ax.set_title(f'{season_name} Season - {hemisphere_names[hemisphere]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(seasons), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.show()
        
        # Print hemisphere statistics
        print(f"\nGLOBAL SEASONAL STATISTICS BY HEMISPHERE:")
        print("=" * 80)
        for season_name in seasons:
            print(f"\n{season_name} Season:")
            for hemisphere in hemispheres:
                if hemisphere in seasonal_pdfs[season_name]:
                    print(f"\n  {hemisphere_names[hemisphere]}:")
                    print(f"  {'Model':<15} | {'Mean':>8} | {'Std':>8} | {'Data Points':>12}")
                    print("  " + "-" * 60)
                    for model_name, pdf_data in seasonal_pdfs[season_name][hemisphere].items():
                        stats = pdf_data['stats']
                        print(f"  {model_name:<15} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>12,}")
    
    else:
        # Plot all seasons together (original behavior)
        n_seasons = len(seasons)
        n_rows = (n_seasons + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_seasons == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        
        for season_idx, season_name in enumerate(seasons):
            ax = axes[season_idx]
            
            for model_idx, (model_name, pdf_data) in enumerate(seasonal_pdfs[season_name].items()):
                color = colors[model_idx % len(colors)]
                ax.plot(pdf_data['bin_centers'], pdf_data['counts'], 
                       color=color, label=model_name, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('SST Anomaly (°C)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{season_name} Season (Global)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(seasons), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # Print global statistics
        print(f"\nGLOBAL SEASONAL STATISTICS (INCLUDING EQUATORIAL REGIONS):")
        print("=" * 80)
        for season_name in seasons:
            print(f"\n{season_name} Season:")
            print(f"{'Model':<15} | {'Mean':>8} | {'Std':>8} | {'Data Points':>12}")
            print("-" * 60)
            for model_name, pdf_data in seasonal_pdfs[season_name].items():
                stats = pdf_data['stats']
                print(f"{model_name:<15} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>12,}")

def compute_regional_seasonal_pdfs_classic(models_dict, bins=100, xlim=(-5, 5), regions=None,
                                          shapefile_path=None, mask_save_dir=None):
    """
    Compute regional seasonal PDFs using PDF-specific masks
    EXCLUDES equatorial regions from seasonal analysis
    """
    print("COMPUTING REGIONAL SEASONAL PDFS (EXCLUDING EQUATORIAL REGIONS)")
    print("=" * 70)
    
    # Create PDF-specific masks using the updated approach
    print("Creating PDF-specific masks...")
    masks_dict = create_model_specific_masks(
        models_dict,
        shapefile_path=shapefile_path,
        mask_save_dir=mask_save_dir
    )
    
    if regions is None:
        # Exclude equatorial regions from seasonal analysis
        all_regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
        regions = [r for r in all_regions if 'Equatorial' not in r]
        print(f"Excluding equatorial regions. Using regions: {regions}")
    
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    
    regional_seasonal_pdfs = {}
    
    for region_name in regions:
        print(f"\nProcessing region: {region_name}")
        regional_seasonal_pdfs[region_name] = {}
        
        for season_name, season_months in seasons.items():
            print(f"  Season: {season_name}")
            regional_seasonal_pdfs[region_name][season_name] = {}
            
            for model_name, model_value in models_dict.items():
                print(f"    {model_name}...")
                
                try:
                    # Extract data array
                    data_array = extract_data_array(model_value)
                    
                    # Get PDF-specific mask
                    if region_name not in masks_dict[model_name]:
                        continue
                        
                    region_mask = masks_dict[model_name][region_name]
                    
                    # Select data for the season months
                    if 'time' in data_array.dims:
                        season_data = data_array.sel(time=data_array.time.dt.month.isin(season_months))
                    else:
                        print(f"      Warning: No time dimension in {model_name}, using all data")
                        season_data = data_array
                    
                    # Apply mask to create a new data array where masked values are NaN
                    masked_data = season_data.where(region_mask)
                    
                    # Compute statistics
                    n_total = int(masked_data.count().compute())
                    mean_val = float(masked_data.mean().compute())
                    std_val = float(masked_data.std().compute())
                    
                    # Compute histogram with fixed range using dask
                    counts, bin_edges = da.histogram(masked_data.data, bins=bins, range=xlim, density=True)
                    counts = counts.compute()
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    regional_seasonal_pdfs[region_name][season_name][model_name] = {
                        'counts': counts,
                        'bin_centers': bin_centers,
                        'bin_edges': bin_edges,
                        'stats': {
                            'mean': mean_val,
                            'std': std_val,
                            'n_points': n_total
                        }
                    }
                    
                    print(f"      {n_total:,} points, mean: {mean_val:.3f}")
                    
                except Exception as e:
                    print(f"      ERROR processing {model_name} in {region_name} {season_name}: {str(e)}")
                    continue
    
    return regional_seasonal_pdfs, masks_dict

def plot_regional_seasonal_pdfs_classic(regional_seasonal_pdfs, regions=None, n_cols=2):
    """
    Plot regional seasonal PDFs computed with classic histogram method
    """
    if regions is None:
        regions = list(regional_seasonal_pdfs.keys())
    
    # Define colors for models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    for region_name in regions:
        if region_name not in regional_seasonal_pdfs:
            continue
            
        print(f"\nPlotting seasonal PDFs for {region_name}")
        
        n_seasons = len(seasons)
        n_rows = (n_seasons + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_seasons == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        
        for season_idx, season_name in enumerate(seasons):
            ax = axes[season_idx]
            
            if season_name in regional_seasonal_pdfs[region_name]:
                for model_idx, (model_name, pdf_data) in enumerate(regional_seasonal_pdfs[region_name][season_name].items()):
                    color = colors[model_idx % len(colors)]
                    ax.plot(pdf_data['bin_centers'], pdf_data['counts'],
                           color=color, label=model_name, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('SST Anomaly (°C)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{region_name.replace("_", " ").title()} - {season_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(seasons), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # Print seasonal statistics for this region
        print(f"\n{region_name.replace('_', ' ').title()} - SEASONAL STATISTICS:")
        print("=" * 80)
        for season_name in seasons:
            if season_name in regional_seasonal_pdfs[region_name]:
                print(f"\n{season_name} Season:")
                print(f"{'Model':<15} | {'Mean':>8} | {'Std':>8} | {'Data Points':>12}")
                print("-" * 60)
                for model_name, pdf_data in regional_seasonal_pdfs[region_name][season_name].items():
                    stats = pdf_data['stats']
                    print(f"{model_name:<15} | {stats['mean']:>8.3f} | {stats['std']:>8.3f} | {stats['n_points']:>12,}")

# ===================================================================================
# QUICK ANALYSIS FUNCTIONS
# ===================================================================================

def quick_regional_analysis(models_dict, bins=100, xlim=(-5, 5), regions=None, 
                           method='fast', shapefile_path=None, mask_save_dir=None):
    """
    Quick regional PDF analysis with PDF-specific masks
    """
    if method == 'ultrafast':
        regional_pdfs, masks_dict = compute_regional_pdfs_classic_ultrafast(
            models_dict, bins=bins, xlim=xlim, regions=regions,
            shapefile_path=shapefile_path, mask_save_dir=mask_save_dir
        )
    elif method == 'fast':
        regional_pdfs, masks_dict = compute_regional_pdfs_classic_fast(
            models_dict, bins=bins, xlim=xlim, regions=regions,
            shapefile_path=shapefile_path, mask_save_dir=mask_save_dir
        )
    else:
        # Fallback to fast method
        regional_pdfs, masks_dict = compute_regional_pdfs_classic_fast(
            models_dict, bins=bins, xlim=xlim, regions=regions,
            shapefile_path=shapefile_path, mask_save_dir=mask_save_dir
        )
    
    plot_regional_pdfs_classic(regional_pdfs, regions=regions)
    return regional_pdfs, masks_dict

def quick_global_analysis(models_dict, bins=100, xlim=(-5, 5)):
    """Quick global PDF analysis"""
    pdfs = compute_global_pdfs_classic(models_dict, bins=bins, xlim=xlim)
    plot_global_pdfs_classic(pdfs)
    return pdfs

def quick_global_seasonal_analysis(models_dict, bins=100, xlim=(-5, 5), by_hemisphere=False):
    """Quick global seasonal PDF analysis"""
    seasonal_pdfs = compute_global_seasonal_pdfs_classic(
        models_dict, bins=bins, xlim=xlim, by_hemisphere=by_hemisphere
    )
    plot_global_seasonal_pdfs_classic(seasonal_pdfs, by_hemisphere=by_hemisphere)
    return seasonal_pdfs

def quick_regional_seasonal_analysis(models_dict, bins=100, xlim=(-5, 5), regions=None,
                                    shapefile_path=None, mask_save_dir=None):
    """Quick regional seasonal PDF analysis (excluding equatorial regions)"""
    regional_seasonal_pdfs, masks_dict = compute_regional_seasonal_pdfs_classic(
        models_dict, bins=bins, xlim=xlim, regions=regions,
        shapefile_path=shapefile_path, mask_save_dir=mask_save_dir
    )
    plot_regional_seasonal_pdfs_classic(regional_seasonal_pdfs, regions=regions)
    return regional_seasonal_pdfs, masks_dict