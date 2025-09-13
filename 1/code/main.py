import cv2
import numpy as np
import glob
import os

def split_bgr(img):
    # w = img.shape[1]
    # margin = int(0.04 * w)
    # img = img[margin:-margin, margin:-margin]
    # cv2.imshow('Cropped Image', img)
    h = img.shape[0] // 3
    margin = int(0.12 * h)
    b = img[:h, :][margin:-margin, margin:-margin]
    g = img[h:2*h, :][margin:-margin, margin:-margin]
    r = img[2*h:3*h, :][margin:-margin, margin:-margin]
    # cv2.imshow('B Channel', cv2.resize(b, (b.shape[1]//4, b.shape[0]//4)))
    # cv2.imshow('G Channel', cv2.resize(g, (g.shape[1]//4, g.shape[0]//4)))
    # cv2.imshow('R Channel', cv2.resize(r, (r.shape[1]//4, r.shape[0]//4)))
    return b, g, r

def pyramid_align(ref, img, search_range=20, min_size=64):
    """Recursive pyramid alignment, suitable for large images."""
    if min(ref.shape) < min_size:
        return (0, 0), -np.inf
    # Downsample
    ref_small = cv2.pyrDown(ref)
    img_small = cv2.pyrDown(img)
    # Recursively aligning
    offset_small, _ = pyramid_align(ref_small, img_small, search_range, min_size)
    # Upscale to current level
    offset = (offset_small[0]*2, offset_small[1]*2)
    best_offset = offset
    best_score = -np.inf
    ref_crop = ref[search_range:-search_range, search_range:-search_range]
    ref_norm = (ref_crop - np.mean(ref_crop)) / (np.std(ref_crop) + 1e-9)
    for dy in range(offset[0]-2, offset[0]+3):
        for dx in range(offset[1]-2, offset[1]+3):
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            shifted_crop = shifted[search_range:-search_range, search_range:-search_range]
            shifted_norm = (shifted_crop - np.mean(shifted_crop)) / (np.std(shifted_crop) + 1e-9)
            score = np.sum(ref_norm * shifted_norm)
            if score > best_score:
                best_score = score
                best_offset = (dy, dx)
    return best_offset, best_score

def normalize_channel(channel, mean=130, std=70):
    """Normalize single channel image to specified mean and std"""
    channel = (channel - np.mean(channel)) / (np.std(channel) + 1e-9) * std + mean
    # channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
    return np.clip(channel, 0, 255).astype('uint8')

def process_file(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skip {filepath}: not a valid BGR stacked image.")
        return
    b, g, r = split_bgr(img)
    offset_g2b, score_g2b = pyramid_align(b, g)
    offset_r2b, score_r2b = pyramid_align(b, r)
    offset_r2g, score_g2r = pyramid_align(g, r)
    # Choose best offsets based on scores 
    if score_g2r > score_r2b:
        if score_g2b > score_r2b:
            offset_g = offset_g2b
            offset_r = (offset_g2b[0] + offset_r2g[0], offset_g2b[1] + offset_r2g[1])
        else:
            offset_r = offset_r2b
            offset_g = (offset_r2b[0] - offset_r2g[0], offset_r2b[1] - offset_r2g[1])
    else:
        offset_r = offset_r2b
        if score_g2b > score_r2b:
            offset_g = offset_g2b
        else:
            offset_g = (offset_r2b[0] - offset_r2g[0], offset_r2b[1] - offset_r2g[1])

    g_aligned = np.roll(np.roll(g, offset_g[0], axis=0), offset_g[1], axis=1)
    r_aligned = np.roll(np.roll(r, offset_r[0], axis=0), offset_r[1], axis=1)
    bgr = np.dstack([b, g_aligned, r_aligned])
    margin = int(0.05 * min(b.shape))
    bgr = bgr[margin:-margin, margin:-margin]
    # Make sure each channel has same value of lightness
    bgr[:,:,0] = normalize_channel(bgr[:,:,0])
    bgr[:,:,1] = normalize_channel(bgr[:,:,1])
    bgr[:,:,2] = normalize_channel(bgr[:,:,2])
    outname = './results/' + os.path.splitext(os.path.basename(filepath))[0] + '_result.jpg'
    cv2.imwrite(outname, bgr)
    print(f"{filepath} -> {outname}, G offset: {offset_g}, R offset: {offset_r}")

if __name__ == '__main__':
    # Process all files in current directory
    os.makedirs('./results', exist_ok=True)
    for ext in ("./data/*.jpg", "./data/*.tif"):
        print(f"Processing files with extension: {ext}")
        for file in glob.glob(ext):
            print(f"Processing file: {file}")
            process_file(file)