# @author: Marc Kevin Schneider
# @purpose: Processes the MAIAC .hdf files and converts them to GeoTIFF
# @date: November 2025

# import packages
import numpy as np
from pyhdf.SD import SD, SDC
import re
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import glob
import os

def tile_from_name(fname):
    """
    Helper function: Extracts MODIS tile (hXX vYY) from filename
    """
    base = os.path.basename(fname)
    tile = base.split(".")[1]   # e.g. h10v09
    h = int(tile[1:5])
    v = int(tile[5:9])
    #print("Extracted basename...")
    return h, v

# fill values for the modis files
FILL_INT16 = -28672
FILL_INT32 = -9999

# scaling factors
# (if you want to change the parameter and then the scaling factors are already there)
# for example rn we are using Optical_Depth_055 but if you want to use WV you can do that
scale_factor = {
    "Optical_Depth_047": 0.001,
    "Optical_Depth_055": 0.001,
    "AOD_Uncertainty": 0.001,
    "Column_WV": 0.001,
    "AngstromExp_470-780": 0.001,
    "FineModeFraction": 0.001,
    "Injection_Height": 1.0,
}

def apply_scaling(name, array):
    """
    Helper function for applying the scaling factor
    """
    if name in scale_factor:
        arr = array.astype(float)
        arr[arr == FILL_INT16] = np.nan
        return arr * scale_factor[name]
    else:
        return array.astype(float)
    
def extract_value(pattern, text):
    """
    Helper function for extracting the values from the modis metadata
    """
    match = re.search(pattern, text)
    return match.group(1)

def maiac_file_processing(file_path, out_tiff_path):
    """
    Purpose:
    Processes the MAIAC files and converts them to GeoTIFF

    Parameters:
    ----------------------------------

    file_path: str
        Path to file

    out_tiff_path: str
        Path to where the tiff file should get stored at

    
    Returns:
    ----------------------------------
    GeoTIFF of a specific MODIS layer
    
    """
    # read file
    hdf = SD(file_path, SDC.READ)
    # extract keys
    datasets = {}
    for name in hdf.datasets().keys():
        ds = hdf.select(name)
        datasets[name] = ds[:]

    # extracts AOD
    # (though you can choose any of the layers given by the file)
    aod_raw = datasets["Optical_Depth_055"]
    # applies scaling
    aod = apply_scaling("Optical_Depth_055", aod_raw)[0]

    # extracts metadata
    metadata = hdf.attributes()["StructMetadata.0"]

    # searches for the upper left point of the tile in the metadata
    m = re.search(r"UpperLeftPointMtrs=\(([-0-9\.]+),([-0-9\.]+)\)", metadata)
    # extracts the x and y values of the upper left point
    ULx, ULy = float(m.group(1)), float(m.group(2))

    # searches for the lower right point of the tile in the metadata
    m = re.search(r"LowerRightMtrs=\(([-0-9\.]+),([-0-9\.]+)\)", metadata)
    # extracts the x and y values of the lower right point
    LRx, LRy = float(m.group(1)), float(m.group(2))

    # searches the grid dimensions (should be 1200 for both in this case)
    Nx = int(re.search(r"XDim=([0-9]+)", metadata).group(1))
    Ny = int(re.search(r"YDim=([0-9]+)", metadata).group(1))

    # everything below can be uncommented if latitudes and longitudes are needed
    # right now they are not since we dont need them for the GeoTIFFs

    # dx = (LRx - ULx) / Nx
    # dy = (ULy - LRy) / Ny 

    # x = ULx + np.arange(Nx) * dx
    # y = ULy - np.arange(Ny) * dy  # flip for top->bottom

    # xx, yy = np.meshgrid(x, y)

    # R = 6371007.181
    # lat = np.arcsin(yy / R) * 180 / np.pi
    # lon = xx / (R * np.cos(np.deg2rad(lat))) * 180 / np.pi

    # create a transform 
    transform = from_bounds(ULx, LRy, LRx, ULy, Nx, Ny)  # bounds: left, bottom, right, top
    # crs is the MODIS sinussoidal 
    crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")

    # write the data to GeoTIFF 
    with rasterio.open(
        out_tiff_path,
        'w',
        driver='GTiff',
        height=Ny,
        width=Nx,
        count=1,
        dtype=aod.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(aod, 1)

    print(f"Saved GeoTIFF to {out_tiff_path}")

# input folder of your files
input_folder = "~/AOD_Temp/MCD19A2_061_AOD"

# all files with the correct name
files = sorted(glob.glob(os.path.join(input_folder, "MCD19A2*.hdf")))

# loop over all files
for file in files:
    # for the final name
    h, v = tile_from_name(file)
    # construct outpath
    outpath = f"~/MODIS_AOD/TIFF/MODIS_AOD_Year_{h}_JulianDay_{v}.tiff"
    # save everything
    maiac_file_processing(file, outpath)

