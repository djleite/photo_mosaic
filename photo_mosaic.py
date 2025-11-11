"""
Photo Mosaic Generator - Streamlit Web Version
----------------------------------------------
- Upload target image and tiles (multiple files or ZIP)
- Supports all options: tile size, grid size, blend, grayscale, borders
- Robust handling of uploaded files and ZIPs
- Parallel processing for speed
"""

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from scipy.spatial import KDTree
import random
from io import BytesIO
from zipfile import ZipFile
from multiprocessing import Pool, cpu_count

# ------------------------------
# Utility Functions
# ------------------------------

def average_color(image: Image.Image):
    arr = np.array(image)
    if len(arr.shape) == 3:
        return arr[:, :, :3].mean(axis=(0, 1))
    else:
        return arr.mean()

def image_variance(image: Image.Image):
    arr = np.array(image)
    return arr.var() if arr.ndim == 2 else arr[:, :, :3].var()

def split_into_diverse_regions(img: Image.Image, x: int, tile_size: int, grayscale: bool):
    regions = []
    w, h = img.size
    step_x = max(tile_size // 2, 1)
    step_y = max(tile_size // 2, 1)
    candidates = []

    for y in range(0, h - tile_size + 1, step_y):
        for x_ in range(0, w - tile_size + 1, step_x):
            patch = img.crop((x_, y, x_ + tile_size, y + tile_size))
            var = image_variance(patch)
            candidates.append((var, patch))

    candidates.sort(key=lambda v: v[0], reverse=True)
    for _, patch in candidates[:x]:
        patch = patch.convert('L' if grayscale else 'RGB').resize((tile_size, tile_size))
        regions.append(patch)
    return regions

# ------------------------------
# Tile Loading
# ------------------------------

def load_uploaded_tiles(tile_files, tile_size, grayscale, subregions, border_px, border_color):
    all_tiles = []

    for file in tile_files:
        name = file.name.lower()
        try:
            if name.endswith(".zip"):
                zip_bytes = BytesIO(file.read())
                with ZipFile(zip_bytes) as zf:
                    for fname in zf.namelist():
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            try:
                                with zf.open(fname) as f:
                                    tile = Image.open(BytesIO(f.read())).convert('RGB')
                                    sub_imgs = split_into_diverse_regions(tile, subregions, tile_size, grayscale)
                                    for patch in sub_imgs:
                                        if border_px > 0:
                                            patch = ImageOps.expand(patch, border=border_px, fill=border_color)
                                        all_tiles.append(patch)
                            except UnidentifiedImageError:
                                print(f"Skipping invalid image in ZIP: {fname}")
            elif name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                tile = Image.open(BytesIO(file.read())).convert('RGB')
                sub_imgs = split_into_diverse_regions(tile, subregions, tile_size, grayscale)
                for patch in sub_imgs:
                    if border_px > 0:
                        patch = ImageOps.expand(patch, border=border_px, fill=border_color)
                    all_tiles.append(patch)
            else:
                print(f"Skipping unsupported file: {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    return all_tiles

# ------------------------------
# Mosaic Building
# ------------------------------

def find_best_tile(color, tree, tiles, used_counts, reuse_limit):
    idx = tree.query(color)[1]
    attempts = 0
    while used_counts[idx] >= reuse_limit and attempts < len(tiles):
        idx = random.randint(0, len(tiles) - 1)
        attempts += 1
    used_counts[idx] += 1
    return tiles[idx]

def process_block(args):
    x, y, block, tree, tiles, used_counts, blend, reuse_limit, grayscale = args
    avg = block.mean() if grayscale else block.mean(axis=(0, 1))
    tile = find_best_tile(avg, tree, tiles, used_counts, reuse_limit)

    if blend > 0:
        if grayscale:
            tint_val = int(avg)
            tint = Image.new('L', tile.size, tint_val)
        else:
            tint = Image.new('RGB', tile.size, tuple(avg.astype(int)))
        tile = Image.blend(tile, tint, blend)
    return (x, y, tile)

def build_mosaic(target_img, tiles, tile_size=50, grid_width=50, grid_height=50,
                 reuse_limit=10, blend=0.3, grayscale=False):
    
    if len(tiles) == 0:
        raise ValueError("No valid tiles found.")

    colors = np.array([average_color(t) for t in tiles])
    if grayscale and colors.ndim == 1:
        colors = colors[:, np.newaxis]

    tree = KDTree(colors)
    used_counts = np.zeros(len(tiles), dtype=int)

    target_img = target_img.convert('L' if grayscale else 'RGB')
    target_img = target_img.resize((grid_width * tile_size, grid_height * tile_size))
    target_array = np.array(target_img)
    mosaic = Image.new('L' if grayscale else 'RGB', target_img.size)
    tasks = []

    for y in range(0, target_img.size[1], tile_size):
        for x in range(0, target_img.size[0], tile_size):
            block = target_array[y:y + tile_size, x:x + tile_size]
            tasks.append((x, y, block, tree, tiles, used_counts, blend, reuse_limit, grayscale))

    for x, y, tile in Pool(cpu_count()).imap_unordered(process_block, tasks):
        mosaic.paste(tile, (x, y))

    return mosaic

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Photo Mosaic Generator")

# Upload target image
target_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png", "bmp", "webp"])

# Upload tiles (multiple files or zip)
tile_files = st.file_uploader(
    "Upload Tiles (Images or ZIP)", 
    type=["jpg", "jpeg", "png", "bmp", "webp", "zip"], 
    accept_multiple_files=True
)

# Mosaic options
tile_size = st.number_input("Tile Size (px)", min_value=10, max_value=500, value=50)
grid_width = st.number_input("Grid Width (tiles)", min_value=1, max_value=200, value=50)
grid_height = st.number_input("Grid Height (tiles)", min_value=1, max_value=200, value=50)
reuse_limit = st.number_input("Tile Reuse Limit", min_value=1, max_value=100, value=10)
blend = st.slider("Color Blend Ratio", 0.0, 1.0, 0.3, 0.05)
grayscale = st.checkbox("Grayscale Mode", value=False)
subregions = st.number_input("Diverse Subregions per Tile", min_value=1, max_value=20, value=3)
border_px = st.number_input("Tile Border Width", min_value=0, max_value=50, value=0)
border_color = st.color_picker("Tile Border Color", value="#FFFFFF")

if st.button("Generate Mosaic"):
    if not target_file or not tile_files:
        st.error("Please upload both target image and tile files.")
    else:
        # Load tiles
        with st.spinner("Loading tiles..."):
            tiles = load_uploaded_tiles(tile_files, tile_size, grayscale, subregions, border_px, border_color)
        
        st.write(f"Loaded {len(tiles)} tile patches")
        if len(tiles) == 0:
            st.error("No valid tiles were loaded! Check your files.")
        else:
            # Build mosaic
            with st.spinner("Building mosaic... this may take a few minutes ⏳"):
                target_img = Image.open(BytesIO(target_file.read()))
                mosaic = build_mosaic(target_img, tiles, tile_size, grid_width, grid_height,
                                      reuse_limit, blend, grayscale)
            
            st.success("Mosaic generated!")
            st.image(mosaic, caption="Generated Mosaic", use_column_width=True)

            # Download button
            buf = BytesIO()
            mosaic.save(buf, format="PNG")
            st.download_button("Download Mosaic", data=buf, file_name="mosaic.png", mime="image/png")
