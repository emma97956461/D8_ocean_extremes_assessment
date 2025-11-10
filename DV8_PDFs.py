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




def extract_data_array(model_value):
    """
    Extract data array from either (dataset, variable_name) tuple or direct data array
    """
    if isinstance(model_value, tuple):
        ds, var_name = model_value
        return ds[var_name]
    else:
        return model_value










# PREVIOUS MASK
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

# MASK THAT USES SHAPEFILE COAST BOUNDARIES
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
        "MidSouth": (-40, -10), #South_SubTropics
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
            # Skip bands not relevant for this ocean
            if ocean == "Indian Ocean" and band_name == "Northern":
                continue
            if band_name == "Northern" and not (ocean.startswith("North") or ocean=="Indian Ocean"):
                continue
            if band_name == "MidSouth" and not (ocean.startswith("South") or ocean=="Indian Ocean"):
                continue
            if band_name == "MidNorth" and ocean.startswith("South"):
                continue  # remove South MidNorth masks

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
    
    # 1. Pacific Equatorial modification (combine Pacific, South China equatorial, and part of Indian)
    if ('Pacific_Equatorial' in region_masks and 
        'South_China_Eastern_Archipelagic_Seas' in region_masks and 
        'Indian_Equatorial' in region_masks):
        
        pacific = region_masks['Pacific_Equatorial']
        south_china = region_masks['South_China_Eastern_Archipelagic_Seas']
        indian = region_masks['Indian_Equatorial']

        # 1️⃣ Restrict Pacific & South China masks to -10 <= lat <= 10
        lat_mask = (pacific.lat >= -10) & (pacific.lat <= 10)
        pacific_eq = pacific.where(lat_mask, False)
        south_china_eq = south_china.where(lat_mask, False)

        # 2️⃣ Restrict Indian mask to 120 <= lon <= 142
        lon_mask = (indian.lon >= 120) & (indian.lon <= 142)
        indian_eq = indian.where(lon_mask, False)

        # 3️⃣ Combine all the masks into one
        combined_mask = pacific_eq | south_china_eq | indian_eq
        
        # 4️⃣ Remove the Indian equatorial points from 120-142°E and assign to Pacific
        region_masks['Pacific_Equatorial'] = combined_mask
        region_masks['Indian_Equatorial'] = indian.where(~lon_mask, False)

    # 2. North Pacific Subtropics modification (combine North Pacific and South China subtropical parts)
    if ('North Pacific Ocean_MidNorth' in region_masks and 
        'South_China_Eastern_Archipelagic_Seas' in region_masks):
        
        north_pacific = region_masks['North Pacific Ocean_MidNorth']
        south_china = region_masks['South_China_Eastern_Archipelagic_Seas']

        # Define latitude range mask
        lat_mask = (north_pacific.lat >= 10) & (north_pacific.lat <= 30)

        # Apply the lat restriction
        north_pacific_band = north_pacific.where(lat_mask, False)
        south_china_band = south_china.where(lat_mask, False)

        # Combine (logical OR)
        region_masks['North Pacific Ocean_MidNorth'] = north_pacific_band | south_china_band

    # 3. Indian South Subtropics modification (add specific box)
    if 'Indian Ocean_MidSouth' in region_masks:
        indian_mid_south = region_masks['Indian Ocean_MidSouth']

        # Define the lat/lon box
        lat_mask = (indian_mid_south.lat >= -11) & (indian_mid_south.lat <= -10)
        lon_mask = (indian_mid_south.lon >= 105) & (indian_mid_south.lon <= 130)

        # Make a new boolean array for the box
        box_mask = lat_mask & lon_mask

        # Expand the original mask to include the box
        region_masks['Indian Ocean_MidSouth'] = indian_mid_south | box_mask

    # 4. Remove Baltic Sea and South China Sea as separate regions
    regions_to_remove = ['Baltic_Sea', 'South_China_Eastern_Archipelagic_Seas']
    for region in regions_to_remove:
        if region in region_masks:
            del region_masks[region]
            print(f"Removed {region} from regions")

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
        # All others remain the same
    }
    
    # Apply renaming
    region_masks = {rename_map.get(k, k): v for k, v in region_masks.items()}

    # ----- NEW: Ensure masks are mutually exclusive -----
    region_masks = ensure_mutually_exclusive_masks(region_masks)

    # ----- Save masks to Zarr -----
    region_masks_ds = xr.Dataset(region_masks)
    region_masks_ds.to_zarr(str(mask_file))
    print(f"Masks saved to {mask_file}")

    return region_masks




def ensure_mutually_exclusive_masks(region_masks, priority_order=None):
    """
    Ensure that no latitude/longitude point belongs to more than one mask.
    Uses a priority system to resolve conflicts.
    
    Parameters:
    - region_masks: dictionary of region masks
    - priority_order: list of region names in priority order (first has highest priority)
    
    Returns:
    - unique_masks: dictionary of mutually exclusive region masks
    """
    print("Ensuring masks are mutually exclusive...")
    
    if priority_order is None:
        # Define default priority order (you can modify this)
        priority_order = [
            'Southern_Ocean',           # Highest priority
            'Pacific_Equatorial',       # Next priority
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
            'Mediterranean_Sea'         # Lowest priority
        ]
    
    # Convert all masks to numpy arrays for easier manipulation
    bool_masks = {}
    for region_name, mask in region_masks.items():
        bool_masks[region_name] = mask.values if hasattr(mask, 'values') else mask
    
    # Create a copy to modify
    unique_masks = bool_masks.copy()
    
    # Track conflicts
    total_conflicts = 0
    
    # Process regions in priority order
    for i, high_priority_region in enumerate(priority_order):
        if high_priority_region not in unique_masks:
            continue
            
        for j, low_priority_region in enumerate(priority_order[i+1:], i+1):
            if low_priority_region not in unique_masks:
                continue
                
            # Find overlapping points
            overlap = unique_masks[high_priority_region] & unique_masks[low_priority_region]
            conflict_count = np.sum(overlap)
            
            if conflict_count > 0:
                total_conflicts += conflict_count
                print(f"  Resolving {conflict_count} conflicts: {high_priority_region} over {low_priority_region}")
                
                # Remove overlapping points from lower priority region
                unique_masks[low_priority_region] = unique_masks[low_priority_region] & ~overlap
    
                print(f"Total conflicts resolved: {total_conflicts}")
    
    # Convert back to xarray DataArrays
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
    
    # Verify no overlaps remain
    verify_no_overlaps(result_masks)
    
    return result_masks



def verify_no_overlaps(region_masks):
    """
    Verify that no overlaps exist between masks by double checking the ensure_mutually_exclusive_masks function
    """
    print("Verifying no overlaps between masks...")
    
    regions = list(region_masks.keys())
    total_overlaps = 0
    
    for i in range(len(regions)):
        region1 = regions[i]
        mask1 = region_masks[region1].values if hasattr(region_masks[region1], 'values') else region_masks[region1]
        
        for j in range(i+1, len(regions)):
            region2 = regions[j]
            mask2 = region_masks[region2].values if hasattr(region_masks[region2], 'values') else region_masks[region2]
            
            overlap = mask1 & mask2
            overlap_count = np.sum(overlap)
            
            if overlap_count > 0:
                total_overlaps += overlap_count
                print(f"  ❌ OVERLAP FOUND: {region1} and {region2} share {overlap_count} points")
    
    if total_overlaps == 0:
        print("  ✅ SUCCESS: No overlaps found between any masks")
    else:
        print(f"  ⚠️  WARNING: {total_overlaps} overlapping points remain")
    
    return total_overlaps == 0






def get_region_colors_shapefile():
    """
    Get color mapping for shapefile-based regions with updated names
    MODIFIED: Remove Baltic Sea and South China Sea colors
    """
    return {
        # Southern Ocean
        'Southern_Ocean': 'purple',
        
        # Pacific Ocean regions
        'North_Pacific_SubTropics': 'lightblue',
        'North_Pacific_MiddleLats': 'blue',
        'South_Pacific_SubTropics': 'darkblue',
        'Pacific_Equatorial': 'lightgreen',
        
        # Atlantic Ocean regions
        'North_Atlantic_SubTropics': 'yellow',
        'North_Atlantic_MiddleLats': 'orange',
        'South_Atlantic_SubTropics': 'red',
        'Atlantic_Equatorial': 'green',
        
        # Indian Ocean regions
        'Indian_SouthSubTropics': 'pink',
        'Indian_NorthSubTropics': 'magenta',
        'Indian_Equatorial': 'darkgreen',
        
        # Small seas (only Mediterranean remains)
        'Mediterranean_Sea': 'cyan'
        # REMOVED: 'Baltic_Sea': 'teal',
        # REMOVED: 'South_China_Eastern_Archipelagic_Seas': 'brown'
    }


def plot_region_masks(lats, lons, masks, title="Oceanic Regions Mask"):
    """
    Plot region masks using the renamed shapefile-based regions.
    UPDATED: Remove colors for deleted regions (Baltic Sea, South China Sea)
    """
    # Create coordinate grids if not already 2D
    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons

    # Use the updated color mapping (without Baltic Sea and South China Sea)
    colors = {
        # Southern Ocean
        'Southern_Ocean': 'purple',
        
        # Pacific Ocean regions
        'North_Pacific_SubTropics': 'lightblue',
        'North_Pacific_MiddleLats': 'blue',
        'South_Pacific_SubTropics': 'darkblue',
        'Pacific_Equatorial': 'lightgreen',
        
        # Atlantic Ocean regions
        'North_Atlantic_SubTropics': 'yellow',
        'North_Atlantic_MiddleLats': 'orange',
        'South_Atlantic_SubTropics': 'red',
        'Atlantic_Equatorial': 'green',
        
        # Indian Ocean regions
        'Indian_SouthSubTropics': 'pink',
        'Indian_NorthSubTropics': 'magenta',
        'Indian_Equatorial': 'darkgreen',
        
        # Small seas (only Mediterranean remains)
        'Mediterranean_Sea': 'cyan'
        # REMOVED: 'Baltic_Sea': 'teal',
        # REMOVED: 'South_China_Eastern_Archipelagic_Seas': 'brown'
    }
    

    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

    for region, mask in masks.items():
        if region in colors:
            mask_data = mask.values if hasattr(mask, 'values') else mask
            # Only plot where mask is True
            masked_lat = lat_grid[mask_data]
            masked_lon = lon_grid[mask_data]
            ax.scatter(masked_lon, masked_lat, color=colors[region], s=1, transform=ccrs.PlateCarree())


    ax.coastlines(zorder=8)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=10)
    ax.set_global()

    # Legend
    legend_patches = [
        mpatches.Patch(color=color, label=r.replace('_',' ').title())
        for r, color in colors.items()
    ]
    ax.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    return fig, ax



def plot_shapefile_regions(lats, lons, shapefile_path=None, title="Oceanic Regions"):
    """
    Plot the shapefile-based regions directly
    """
    # Create masks using shapefile method
    masks = create_shapefile_oceanic_regions_mask(lats, lons, shapefile_path)
    
    # Plot using the general plotting function
    return plot_region_masks(lats, lons, masks, title)

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
        # Handle both numpy arrays and xarray DataArrays
        if hasattr(mask, 'values'):
            mask_data = mask.values
        else:
            mask_data = mask
            
        if data_array.ndim == 3:  # Multiple time steps (time, lat, lon)
            if time_idx is not None:
                # Extract data for specific time
                region_data[region] = data_array[time_idx, mask_data]
            else:
                # Extract all data across time and space
                # Reshape to (time * spatial_points)
                region_data[region] = data_array[:, mask_data].flatten()
        else:  # Single time step (lat, lon)
            region_data[region] = data_array[mask_data]
    
    return region_data




def create_oceanic_regions_mask(lats, lons, method='shapefile', shapefile_path=None):
    """
    MAIN FUNCTION - Create oceanic regions mask with choice of method
    Parameters:
    - lats: latitude array (1D or 2D)
    - lons: longitude array (1D or 2D)
    - method: 'shapefile' (new) or 'coordinate' (old)
    - shapefile_path: optional custom shapefile path
    
    Returns:
    - masks: dictionary of boolean masks for each region
    """
    if method == 'shapefile':
        return create_shapefile_oceanic_regions_mask(lats, lons, shapefile_path=shapefile_path)
    elif method == 'coordinate':
        return create_old_oceanic_regions_mask(lats, lons)
    else:
        raise ValueError("Method must be 'shapefile' or 'coordinate'")

# MODIFIED: Update create_model_specific_masks to handle both formats
def create_model_specific_masks(models_dict):
    """
    Create masks for each model based on their specific grid
    MODIFIED: Now accepts both (dataset, var_name) tuples and direct data arrays
    """
    print("Creating model-specific masks...")
    masks_dict = {}
    
    for model_name, model_value in models_dict.items():
        print(f"Creating masks for {model_name}...")
        
        # Extract data array using helper function
        data_array = extract_data_array(model_value)
        
        lats = data_array.lat.values
        lons = data_array.lon.values
        masks_dict[model_name] = create_oceanic_regions_mask(lats, lons)
        print(f"  {model_name} grid: {lats.shape} x {lons.shape}")
    
    return masks_dict

# ===================================================================================================================================================
# PDF FUNCTIONS USING CLASSIC HISTOGRAM METHOD
# ===================================================================================================================================================

def compute_global_pdfs_classic(models_dict, bins=100, xlim=(-5, 5)):
    """
    Compute global PDFs for multiple models using classic histogram method
    MODIFIED: Now accepts both (dataset, var_name) tuples and direct data arrays
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









# MODIFIED: Update compute_regional_pdfs_classic to handle both formats
def compute_regional_pdfs_classic(models_dict, bins=100, xlim=(-5, 5), regions=None):
    """
    Compute regional PDFs for multiple models using classic histogram method
    MODIFIED: Now accepts both (dataset, var_name) tuples and direct data arrays
    """
    print("COMPUTING REGIONAL PDFS")
    print("=" * 55)
    
    # Create model-specific masks
    masks_dict = create_model_specific_masks(models_dict)
    
    if regions is None:
        regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    
    regional_pdfs = {}
    
    for region_name in regions:
        print(f"\nProcessing region: {region_name}")
        regional_pdfs[region_name] = {}
        
        for model_name, model_value in models_dict.items():
            print(f"  {model_name}...")
            
            # Get model-specific mask
            mask = masks_dict[model_name][region_name]
            
            # Extract data array using helper function
            data_array = extract_data_array(model_value)
            
            regional_data = data_array.where(mask)
                
            # Flatten and remove NaNs
            values = regional_data.values.flatten()
            clean_data = values[~np.isnan(values)]
            
            if len(clean_data) == 0:
                print(f"    Warning: No data for {region_name} in {model_name}")
                continue
            
            # Compute histogram
            counts, bin_edges = np.histogram(clean_data, bins=bins, range=xlim, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            regional_pdfs[region_name][model_name] = {
                'counts': counts,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'stats': {
                    'mean': np.mean(clean_data),
                    'std': np.std(clean_data),
                    'n_points': len(clean_data)
                }
            }
            
            print(f"    {len(clean_data):,} points, mean: {np.mean(clean_data):.3f}")
    
    return regional_pdfs, masks_dict


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

# Simple wrapper functions for quick analysis
def quick_global_analysis(models_dict, bins=100, xlim=(-5, 5)):
    """Quick global PDF analysis"""
    pdfs = compute_global_pdfs_classic(models_dict, bins=bins, xlim=xlim)
    plot_global_pdfs_classic(pdfs)
    return pdfs

def quick_regional_analysis(models_dict, bins=100, xlim=(-5, 5), regions=None):
    """Quick regional PDF analysis"""
    regional_pdfs, masks_dict = compute_regional_pdfs_classic(models_dict, bins=bins, xlim=xlim, regions=regions)
    plot_regional_pdfs_classic(regional_pdfs, regions=regions)
    return regional_pdfs, masks_dict







# ========================================================================================================================================
# SEASONAL PDF FUNCTIONS (EQUATORIAL EXCLUDED ONLY FROM REGIONAL SEASONAL)
# ========================================================================================================================================

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



def compute_regional_seasonal_pdfs_classic(models_dict, bins=100, xlim=(-5, 5), regions=None):
    """
    Compute regional seasonal PDFs for multiple models using classic histogram method
    MODIFIED: Now accepts both (dataset, var_name) tuples and direct data arrays
    """
    print("COMPUTING REGIONAL SEASONAL PDFS (EXCLUDING EQUATORIAL REGIONS)")
    print("=" * 70)
    
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5], 
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    
    # Create model-specific masks
    masks_dict = create_model_specific_masks(models_dict)
    
    # Filter out equatorial regions
    all_regions = list(masks_dict[list(masks_dict.keys())[0]].keys())
    equatorial_keywords = ['equatorial', 'Equatorial']
    non_equatorial_regions = [
        region for region in all_regions 
        if not any(keyword in region for keyword in equatorial_keywords)
    ]
    
    if regions is None:
        regions = non_equatorial_regions
    else:
        # Ensure only non-equatorial regions are included
        regions = [region for region in regions if region in non_equatorial_regions]
    
    print(f"Non-equatorial regions for seasonal analysis: {regions}")
    
    regional_seasonal_pdfs = {}
    
    for season_name, season_months in seasons.items():
        print(f"\nProcessing season: {season_name}")
        regional_seasonal_pdfs[season_name] = {}
        
        for region_name in regions:
            print(f"  Region: {region_name}")
            regional_seasonal_pdfs[season_name][region_name] = {}
            
            for model_name, model_value in models_dict.items():
                print(f"    {model_name}...")
                
                # Get model-specific mask
                mask = masks_dict[model_name][region_name]
                
                # Extract data array using helper function
                data_array = extract_data_array(model_value)
                
                try:
                    # Select data for the season months
                    if 'time' in data_array.dims:
                        season_data = data_array.sel(time=data_array.time.dt.month.isin(season_months))
                    else:
                        print(f"      Warning: No time dimension in {model_name}, using all data")
                        season_data = data_array
                    
                    # Apply regional mask
                    regional_data = season_data.where(mask)
                    
                    # Flatten and remove NaNs
                    values = regional_data.values.flatten()
                    clean_data = values[~np.isnan(values)]
                    
                    if len(clean_data) == 0:
                        print(f"      Warning: No data for {region_name} in {model_name} season {season_name}")
                        continue
                    
                    # Compute histogram
                    counts, bin_edges = np.histogram(clean_data, bins=bins, range=xlim, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    regional_seasonal_pdfs[season_name][region_name][model_name] = {
                        'counts': counts,
                        'bin_centers': bin_centers,
                        'bin_edges': bin_edges,
                        'stats': {
                            'mean': np.mean(clean_data),
                            'std': np.std(clean_data),
                            'n_points': len(clean_data)
                        }
                    }
                    
                    print(f"      {len(clean_data):,} points, mean: {np.mean(clean_data):.3f}")
                    
                except Exception as e:
                    print(f"      Error processing {model_name} - {region_name} - {season_name}: {e}")
                    continue
    
    return regional_seasonal_pdfs, masks_dict



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

# Simple wrapper functions for quick seasonal analysis
def quick_global_seasonal_analysis(models_dict, bins=100, xlim=(-5, 5), by_hemisphere=False):
    """Quick global seasonal PDF analysis"""
    seasonal_pdfs = compute_global_seasonal_pdfs_classic(
        models_dict, bins=bins, xlim=xlim, by_hemisphere=by_hemisphere
    )
    plot_global_seasonal_pdfs_classic(seasonal_pdfs, by_hemisphere=by_hemisphere)
    return seasonal_pdfs

def quick_regional_seasonal_analysis(models_dict, bins=100, xlim=(-5, 5), regions=None):
    """Quick regional seasonal PDF analysis (excluding equatorial regions)"""
    regional_seasonal_pdfs, masks_dict = compute_regional_seasonal_pdfs_classic(
        models_dict, bins=bins, xlim=xlim, regions=regions
    )
    plot_regional_seasonal_pdfs_classic(regional_seasonal_pdfs, regions=regions)
    return regional_seasonal_pdfs, masks_dict