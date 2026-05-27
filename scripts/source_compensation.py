import blindspot as bs
import click
import numpy as np
import matplotlib.pyplot as plt

import cv2
from pathlib import Path

@click.command()
@click.option('--dataset', default='/Volumes/Charles/data/blindpoint/source')
@click.option('--index', default=4)
@click.option('--save_path', default='./compensation_test.png')
@click.option('--detection', default='gb', type=click.Choice(['gb', '3sigma', 'fit']), help='Blind pixel detection method.')
@click.option('--img_path', default=None, help='Path to an external image to use for compensation test.')
@click.option('--compensation', default='median', type=click.Choice(['median', 'pyramid', 'sobel']), help='Blind pixel compensation method.')
@click.option('--x1', default=0, help='Zoom region x1')
@click.option('--x2', default=1, help='Zoom region x2')
@click.option('--y1', default=0, help='Zoom region y1')
@click.option('--y2', default=1, help='Zoom region y2')
def main(dataset, index, save_path, detection, img_path, compensation, x1, x2, y1, y2):
    bs.set_base_path(dataset)
    info = bs.get_proj_info_by_index(index)
    
    # Ensure data is loaded
    bs.load_low_voltages(info)
    
    # Detect Blind Pixels first to get dimensions
    if detection == 'gb':
        mask = bs.dect_gb(info)
        title_suffix = "(GB)"
    elif detection == '3sigma':
        mask = bs.dect_three_sigma(info)
        title_suffix = "(3-Sigma)"
    elif detection == 'fit':
        mask = bs.dect_curved_surface_fitting(info)
        title_suffix = "(Surface Fitting)"
    else:
        mask = bs.dect_gb(info)
        title_suffix = "(GB)"
        
    # Load Image
    if img_path and Path(img_path).exists():
        # Load external image
        img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
             print(f"Error: Could not read image from {img_path}. Using default.")
             img = np.average(info['vol_l'], axis=0)
        else:
            # Resize to match mask dimensions
            rows, cols = mask.shape
            img = cv2.resize(img_raw, (cols, rows))
            print(f"Loaded external image from {img_path} and resized to ({rows}, {cols})")
    else:
        # Original Image (Low Temp Average)
        img = np.average(info['vol_l'], axis=0)
    
    # Apply Compensation
    img_compensated = None
    comp_title = ""
    edge_mask = None
    
    if compensation == 'median':
        img_compensated = bs.compensate_median(img, mask)
        comp_title = "Median Filter Compensation"
    elif compensation == 'pyramid':
        img_compensated = bs.compensate_pyramid(img, mask)
        comp_title = "Pyramid (Weighted Mean) Compensation"
    elif compensation == 'sobel':
        img_compensated, edge_mask = bs.compensate_adaptive(img, mask)
        comp_title = "Adaptive (Sobel+Mean) Compensation"
    
    # Determine Zoom Region if not provided or invalid
    rows, cols = img.shape
    if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
        # Try to find a region with blind pixels
        blind_indices = np.argwhere(mask)
        if len(blind_indices) > 0:
            # Pick the first one
            cy, cx = blind_indices[0]
            # defined a 50x50 window around it
            pad = 25
            x1 = max(0, cx - pad)
            x2 = min(cols, cx + pad)
            y1 = max(0, cy - pad)
            y2 = min(rows, cy + pad)
        else:
             # Center crop 100x100
             cy, cx = rows // 2, cols // 2
             pad = 50
             x1 = max(0, cx - pad)
             x2 = min(cols, cx + pad)
             y1 = max(0, cy - pad)
             y2 = min(rows, cy + pad)
    
    # Visualization
    # Calculate aspect ratios for width_ratios to keep height consistent
    h_img, w_img = img.shape
    h_zoom = y2 - y1
    w_zoom = x2 - x1
    
    aspect_img = w_img / h_img
    aspect_zoom = w_zoom / h_zoom
    
    # We want height to be same.
    # W_plot / H_plot = aspect
    # If H_plot is constant, W_plot proportional to aspect.
    width_ratios = [aspect_img, aspect_img, aspect_zoom, aspect_zoom]
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={'width_ratios': width_ratios})
    
    # 1. Original Image with Mask Overlay
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original Image")
    # Overlay red dots for blind pixels
    y, x = np.where(mask)
    axs[0].scatter(x, y, c='red', s=1, alpha=0.5)
    # Draw rectangle for zoom region
    rect0 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
    axs[0].add_patch(rect0)
    
    # 2. Compensated Image
    axs[1].imshow(img_compensated, cmap='gray')
    axs[1].set_title(comp_title)
    # Draw rectangle for zoom region
    rect1 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
    axs[1].add_patch(rect1)

    if edge_mask is not None:
        y_edge, x_edge = np.where(edge_mask)
        axs[1].scatter(x_edge, y_edge, c='blue', s=0.5, alpha=0.3)
    
    # 3. Zoomed Original Image
    zoom_img_orig = img[y1:y2, x1:x2]
    axs[2].imshow(zoom_img_orig, cmap='gray')
    axs[2].set_title("Zoomed Original")
    
    # 4. Zoomed Compensated Image
    zoom_img_comp = img_compensated[y1:y2, x1:x2]
    axs[3].imshow(zoom_img_comp, cmap='gray')
    axs[3].set_title("Zoomed Compensated")
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Save result
    # fig.savefig(save_path)

if __name__ == "__main__":
    main()