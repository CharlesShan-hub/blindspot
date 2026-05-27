import blindspot as bs
import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@click.command()
@click.option('--dataset', default='/Volumes/Charles/data/blindpoint/source')
@click.option('--index', default=30)
@click.option('--save_path', default='./curve_compare.png')
def main(dataset, index, save_path):
    bs.set_base_path(dataset)
    info = bs.get_proj_info_by_index(index)
    
    # Ensure data is loaded
    bs.load_low_voltages(info)
    
    # Original Image (Low Temp Average)
    img_l = np.average(info['vol_l'], axis=0)
    
    # Global Stats
    mean_val = np.mean(img_l)
    std_val = np.std(img_l)
    
    # Curved Surface Fitting
    S = bs.curved_surface_fitting(img_l)
    
    # Calculate Sigma
    sigma = np.sqrt(np.sum((img_l.ravel() - S.ravel())**2) / (img_l.size - 1))
    
    # Detect Blind Pixels with different k
    ks = [2, 3, 6]
    masks = []
    for k in ks:
        mask = np.abs(img_l - S) > k * sigma
        masks.append(mask)
    
    # Plotting
    fig = plt.figure(figsize=(20, 10))
    
    # Get global min/max for z-axis to keep consistent scale
    z_min, z_max = np.min(img_l), np.max(img_l)
    
    # Row 1: 3D Plots
    # 1. Original Image (Global 3-sigma)
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    S_global = np.full_like(img_l, mean_val)
    plot_surface_with_thresholds(ax1, img_l, S_global, 3*std_val, "Global Mean +/- 3$\sigma$", zlim=(z_min, z_max))
    
    # 2-4. Fitted Surface with Thresholds for each k
    for i, k in enumerate(ks):
        ax = fig.add_subplot(2, 4, i + 2, projection='3d')
        plot_surface_with_thresholds(ax, img_l, S, k*sigma, f"Image & S +/- {k}$\sigma$", zlim=(z_min, z_max))

    # Row 2: Blind Pixel Masks (2D)
    # 1. 3-Sigma Method (No Curve Fitting)
    mask_3sigma = np.abs(img_l - mean_val) > 3 * std_val
    
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(mask_3sigma, cmap='gray')
    ax5.set_title(f"3-Sigma (No Fit) Count:{mask_3sigma.sum()}")
    ax5.axis('off')
    
    # 2-4. Masks for each k
    for i, k in enumerate(ks):
        ax = fig.add_subplot(2, 4, i + 6)
        ax.imshow(masks[i], cmap='gray')
        ax.set_title(f"Blind Mask (k={k}) Count:{masks[i].sum()}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Saved result to {save_path}")

def plot_surface(ax, data, title, plane=None, zlim=None):
    rows, cols = data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, data, cmap='viridis', alpha=0.8)
    if plane is not None:
        # Draw a semi-transparent plane at height 'plane'
        plane_data = np.full_like(data, plane)
        ax.plot_surface(X, Y, plane_data, color='r', alpha=0.3)
    
    if zlim is not None:
        ax.set_zlim(zlim)
        
    ax.set_title(title)

def plot_surface_with_thresholds(ax, img, S, threshold, title, zlim=None):
    rows, cols = img.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Plot Original Image
    ax.plot_surface(X, Y, img, cmap='viridis', alpha=0.6)
    
    # Plot Upper Threshold Surface (S + threshold)
    ax.plot_surface(X, Y, S + threshold, color='r', alpha=0.2)
    
    # Plot Lower Threshold Surface (S - threshold)
    ax.plot_surface(X, Y, S - threshold, color='r', alpha=0.2)
    
    if zlim is not None:
        ax.set_zlim(zlim)
        
    ax.set_title(title)

if __name__ == "__main__":
    main()