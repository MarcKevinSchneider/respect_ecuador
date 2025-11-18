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
    Extract MODIS tile (hXX vYY) from filename
    """
    base = os.path.basename(fname)
    tile = base.split(".")[1]   # e.g. h10v09
    h = int(tile[1:5])
    v = int(tile[5:9])
    #print("Extracted basename...")
    return h, v

FILL_INT16 = -28672
FILL_INT32 = -9999

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
    if name in scale_factor:
        arr = array.astype(float)
        arr[arr == FILL_INT16] = np.nan
        return arr * scale_factor[name]
    else:
        return array.astype(float)
    
def extract_value(pattern, text):
    match = re.search(pattern, text)
    return match.group(1)

def maiac_file_processing(file_path, out_tiff_path):
    hdf = SD(file_path, SDC.READ)
    datasets = {}
    for name in hdf.datasets().keys():
        ds = hdf.select(name)
        datasets[name] = ds[:]

    aod_raw = datasets["Optical_Depth_055"]
    aod = apply_scaling("Optical_Depth_055", aod_raw)[0]

    metadata = hdf.attributes()["StructMetadata.0"]

    m = re.search(r"UpperLeftPointMtrs=\(([-0-9\.]+),([-0-9\.]+)\)", metadata)
    ULx, ULy = float(m.group(1)), float(m.group(2))

    m = re.search(r"LowerRightMtrs=\(([-0-9\.]+),([-0-9\.]+)\)", metadata)
    LRx, LRy = float(m.group(1)), float(m.group(2))

    # Grid dimensions
    Nx = int(re.search(r"XDim=([0-9]+)", metadata).group(1))
    Ny = int(re.search(r"YDim=([0-9]+)", metadata).group(1))

    dx = (LRx - ULx) / Nx
    dy = (ULy - LRy) / Ny  # note ULy-LRy

    # x = ULx + np.arange(Nx) * dx
    # y = ULy - np.arange(Ny) * dy  # flip for top->bottom

    # xx, yy = np.meshgrid(x, y)

    # R = 6371007.181
    # lat = np.arcsin(yy / R) * 180 / np.pi
    # lon = xx / (R * np.cos(np.deg2rad(lat))) * 180 / np.pi

    transform = from_bounds(ULx, LRy, LRx, ULy, Nx, Ny)  # bounds: left, bottom, right, top
    crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")  # MODIS Sinusoidal

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

input_folder = "C:/Users/kevis/OneDrive/Desktop/HiWi-Job/AOD_Paulina/data/AOD_Temp/MCD19A2_061_AOD"

#input_folder = "C:/Users/kevis/OneDrive/Desktop/HiWi-Job/AOD_Paulina/data/AOD_Temp"

files = sorted(glob.glob(os.path.join(input_folder, "MCD19A2*.hdf")))

#files = files[:3]

for file in files:
    h, v = tile_from_name(file)
    outpath = f"D:/Universitaet/HiWi-Job/MODIS_AOD/TIFF/MODIS_AOD_Year_{h}_JulianDay_{v}.tiff"
    maiac_file_processing(file, outpath)
