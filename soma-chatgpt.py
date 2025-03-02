#!/usr/bin/env python3
"""
Fluorescent Soma Counting with Cellpose on a Directory of Images

This script processes all fluorescent microscopy images in a specified directory to detect, count,
and classify cell soma into:
    - EGFP (green)
    - tdTomato (red)
    - Coexpressed (yellow/orange)

It uses Cellpose (a deep learning segmentation tool) to generate accurate cell masks.
For each detected cell, the script extracts its centroid and determines the cell type based on the
original green and red channels. For each processed image, the script outputs a CSV file (with cell
details: ID, pixel and micron coordinates, cell type, and hex color at the centroid) and an annotated
overlay image.

Additional CLI arguments let you adjust the conversion factor (pixels per micron) and configure
Cellpose settings such as the model type, GPU usage, estimated cell diameter, flow threshold, cell
probability threshold, and channels.

Usage Example:
    python soma_counter_cellpose_dir.py input --cellpose_flow_threshold 0.4 \
         --cellpose_cellprob_threshold 0.0 --cellpose_channels 0,0
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
from cellpose import models  # Requires cellpose package
from scipy import ndimage

def run_cellpose_segmentation(composite, cellpose_model, diameter=None, channels=[0,0],
                              flow_threshold=0.4, cellprob_threshold=0.0):
    """
    Use Cellpose to segment cells from the composite image.
    
    Parameters:
        composite: A single-channel (grayscale) image (numpy array).
        cellpose_model: A preloaded Cellpose model.
        diameter: Estimated cell diameter. If None, Cellpose will estimate it.
        channels: List specifying which channels to use (e.g. [0,0]).
        flow_threshold: Flow threshold for Cellpose segmentation.
        cellprob_threshold: Cell probability threshold for Cellpose segmentation.
        
    Returns:
        masks: A labeled mask (numpy array) with unique labels for each cell.
    """
    # Normalize image to float values in range [0,1]
    composite_norm = composite.astype(np.float32) / 255.0
    # Run Cellpose segmentation with configurable channels
    masks, flows, styles, diams = cellpose_model.eval(
        composite_norm, 
        channels=channels,
        diameter=diameter, 
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold
    )
    return masks

def classify_cell(centroid, g_channel, r_channel, intensity_thresh=60):
    """
    Classify a cell based on the intensities at its centroid.
    
    Parameters:
        centroid: (x, y) coordinates of the cell centroid.
        g_channel: Original green channel image.
        r_channel: Original red channel image.
        intensity_thresh: Intensity threshold to consider signal present.
        
    Returns:
        A string indicating the cell type ("EGFP", "tdTomato", "coexpressed", or "unknown").
    """
    x, y = int(round(centroid[0])), int(round(centroid[1]))
    if y >= g_channel.shape[0] or x >= g_channel.shape[1]:
        return "unknown"
    green_val = g_channel[y, x]
    red_val = r_channel[y, x]
    green_present = green_val >= intensity_thresh
    red_present = red_val >= intensity_thresh

    if green_present and red_present:
        return "coexpressed"
    elif green_present:
        return "EGFP"
    elif red_present:
        return "tdTomato"
    else:
        return "unknown"

def hex_color_at(image, centroid):
    """
    Return the hex color code at the given centroid in the original image.
    Note: OpenCV loads images in BGR order.
    """
    x, y = int(round(centroid[0])), int(round(centroid[1]))
    b, g, r = image[y, x]
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)

def process_image(image_path, intensity_thresh=60, diameter=None, conversion_factor=1/1.6091,
                  cellpose_model=None, channels=[0,0], flow_threshold=0.4, cellprob_threshold=0.0):
    """
    Process the given image to detect, count, and classify soma using Cellpose.
    
    Parameters:
        image_path: Path to the input image.
        intensity_thresh: Intensity threshold for fluorescence classification.
        diameter: Estimated cell diameter for Cellpose segmentation (optional).
        conversion_factor: Factor to convert pixel coordinates to microns.
        cellpose_model: Pre-instantiated Cellpose model.
        channels: List specifying which channels to use for Cellpose.
        flow_threshold: Flow threshold for Cellpose segmentation.
        cellprob_threshold: Cell probability threshold for Cellpose segmentation.
        
    Returns:
        results: List of dictionaries with soma details.
        overlay: Annotated overlay image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image at {image_path}")
    
    # Create overlay image (convert to RGB for annotation)
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split channels (OpenCV uses BGR order)
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # Create composite grayscale image by averaging green and red channels
    composite = cv2.addWeighted(g_channel, 0.5, r_channel, 0.5, 0)
    
    # Run Cellpose segmentation
    masks = run_cellpose_segmentation(
        composite, 
        cellpose_model, 
        diameter=diameter, 
        channels=channels,
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold
    )
    
    # Extract centroids using connected components on the Cellpose masks
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        masks.astype(np.uint8), connectivity=8
    )
    
    results = []
    for label in range(1, num_labels):  # Skip background label 0
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 40:  # Skip very small regions
            continue
        centroid = centroids[label]
        cell_type = classify_cell(centroid, g_channel, r_channel, intensity_thresh)
        hex_val = hex_color_at(image, centroid)
        x_micron = centroid[0] * conversion_factor
        y_micron = centroid[1] * conversion_factor
        results.append({
            "ID": label,
            "X_pixel": centroid[0],
            "Y_pixel": centroid[1],
            "X_micron": x_micron,
            "Y_micron": y_micron,
            "Type": cell_type,
            "Hex": hex_val
        })
        # Annotate overlay image
        center_coords = (int(round(centroid[0])), int(round(centroid[1])))
        if cell_type == "EGFP":
            draw_color = (0, 255, 0)  # Green (RGB)
        elif cell_type == "tdTomato":
            draw_color = (255, 0, 0)  # Red (RGB)
        elif cell_type == "coexpressed":
            draw_color = (255, 255, 0)  # Yellow (RGB)
        else:
            draw_color = (255, 255, 255)  # White for unknown
        cv2.circle(overlay, center_coords, 5, draw_color, 2)
        cv2.putText(overlay, str(label), (center_coords[0] + 5, center_coords[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
    
    return results, overlay

def save_results(results, overlay, base_filename, output_dir, save_csv=True, save_overlay=True):
    """
    Save the results to a CSV file and the annotated overlay image.
    
    Parameters:
        results: List of dictionaries with cell data.
        overlay: Annotated overlay image.
        base_filename: Base name of the input image (without extension).
        output_dir: Directory to save the outputs.
        save_csv: Whether to save the CSV file.
        save_overlay: Whether to save the overlay image.
    """
    if save_csv:
        df = pd.DataFrame(results)
        csv_filename = os.path.join(output_dir, f"{base_filename}_soma.csv")
        df.to_csv(csv_filename, index=False)
        print(f"CSV file saved: {csv_filename}")
    if save_overlay:
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_filename = os.path.join(output_dir, f"{base_filename}_overlay.png")
        cv2.imwrite(overlay_filename, overlay_bgr)
        print(f"Overlay image saved: {overlay_filename}")

def process_directory(input_dir, intensity_thresh=60, diameter=None, conversion_factor=1/1.6091,
                      cellpose_model=None, channels=[0,0], flow_threshold=0.4, cellprob_threshold=0.0,
                      save_csv=True, save_overlay=True, output_dir="output"):
    """
    Process all images in the specified directory.
    
    Parameters:
        input_dir: Directory containing input images.
        intensity_thresh: Intensity threshold for fluorescence classification.
        diameter: Estimated cell diameter for Cellpose segmentation (optional).
        conversion_factor: Factor to convert pixel coordinates to microns.
        cellpose_model: Preloaded Cellpose model.
        channels: List specifying which channels to use for Cellpose.
        flow_threshold: Flow threshold for Cellpose segmentation.
        cellprob_threshold: Cell probability threshold for Cellpose segmentation.
        save_csv: Flag to save CSV outputs.
        save_overlay: Flag to save overlay images.
        output_dir: Directory to save output files (default: 'output').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Supported image extensions.
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    
    if not image_files:
        print("No valid image files found in the directory.")
        return
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(f"Processing {image_path}...")
        try:
            results, overlay = process_image(
                image_path, intensity_thresh, diameter, conversion_factor, cellpose_model,
                channels, flow_threshold, cellprob_threshold
            )
            base_filename, _ = os.path.splitext(image_file)
            save_results(results, overlay, base_filename, output_dir, save_csv, save_overlay)
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Automated Soma Counting in Fluorescent Microscopy Images using Cellpose on a Directory"
    )
    parser.add_argument("input_dir", nargs="?", default="input",
                        help="Path to the directory containing input images (default: 'input')")
    parser.add_argument("--intensity_thresh", type=int, default=60,
                        help="Intensity threshold for fluorescence classification (default: 60)")
    parser.add_argument("--diameter", type=float, default=None,
                        help="Estimated cell diameter for Cellpose segmentation (optional)")
    parser.add_argument("--pixels_per_micron", type=float, default=1.6091,
                        help="Number of pixels per micron (default: 1.6091)")
    parser.add_argument("--cellpose_model_type", type=str, default="cyto3",
                        help="Cellpose model type (default: 'cyto3')")
    parser.add_argument("--no_gpu", dest="cellpose_gpu", action="store_false",
                        help="Disable GPU for Cellpose segmentation")
    parser.set_defaults(cellpose_gpu=True)
    parser.add_argument("--cellpose_flow_threshold", type=float, default=0.4,
                        help="Flow threshold for Cellpose segmentation (default: 0.4)")
    parser.add_argument("--cellpose_cellprob_threshold", type=float, default=0.0,
                        help="Cell probability threshold for Cellpose segmentation (default: 0.0)")
    parser.add_argument("--cellpose_channels", type=str, default="0,0",
                        help="Channels for Cellpose segmentation as a comma-separated list (default: '0,0')")
    parser.add_argument("--no_csv", dest="save_csv", action="store_false",
                        help="Do not save CSV files")
    parser.set_defaults(save_csv=True)
    parser.add_argument("--no_overlay", dest="save_overlay", action="store_false",
                        help="Do not save overlay images")
    parser.set_defaults(save_overlay=True)
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files (default: 'output')")
    
    args = parser.parse_args()
    
    # Compute conversion factor (pixels -> microns)
    conversion_factor = 1 / args.pixels_per_micron
    
    # Parse cellpose channels from a comma-separated string into a list of ints.
    try:
        channels = [int(ch.strip()) for ch in args.cellpose_channels.split(',')]
    except Exception as e:
        raise ValueError("Error parsing cellpose_channels. It should be a comma-separated list of integers, e.g., '0,0'.")
    
    # Instantiate a Cellpose model using provided parameters.
    cellpose_model = models.Cellpose(
        model_type=args.cellpose_model_type,
        gpu=args.cellpose_gpu
    )
    
    process_directory(
        args.input_dir,
        intensity_thresh=args.intensity_thresh,
        diameter=args.diameter,
        conversion_factor=conversion_factor,
        cellpose_model=cellpose_model,
        channels=channels,
        flow_threshold=args.cellpose_flow_threshold,
        cellprob_threshold=args.cellpose_cellprob_threshold,
        save_csv=args.save_csv,
        save_overlay=args.save_overlay,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
