#!/usr/bin/env python3
"""
Fluorescent Soma Counting Algorithm

This script processes fluorescent microscopy images to detect, count, and classify cell soma into:
    - EGFP (green)
    - tdTomato (red)
    - Coexpressed (yellow/orange)

Outputs:
    - A CSV file with soma details (ID, X_pixel, Y_pixel, X_micron, Y_micron, Type, Hex)
    - An overlay image with annotated detections

Default parameter values have been tuned using the provided example image.
Usage example:
    python soma_counter.py path/to/image.png --save_csv --save_overlay --output_folder my_output
"""

import cv2
import numpy as np
import pandas as pd
import argparse
from skimage.feature import peak_local_max  # from scikit-image
from skimage.segmentation import watershed
from scipy import ndimage

# Conversion factor: 1 micron = 1.6091 pixels, so pixels -> microns is 1/1.6091
PIXEL_TO_MICRON = 1 / 1.6091

def preprocess_channel(channel, median_ksize=3, gaussian_sigma=1.0):
    """Apply median filtering and Gaussian blur to reduce noise."""
    filtered = cv2.medianBlur(channel, median_ksize)
    filtered = cv2.GaussianBlur(filtered, (0, 0), gaussian_sigma)
    return filtered

def threshold_channel(channel, thresh_val=60):
    """Threshold a channel to create a binary image."""
    _, binary = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY)
    return binary

def split_cells(binary_mask, min_distance=10):
    """
    Use watershed segmentation to separate merged cells.
    Parameters:
        binary_mask: Binary image from thresholding.
        min_distance: Minimum distance between local maxima.
    Returns:
        A labeled mask where each separated cell has a unique label.
    """
    # Compute the distance transform of the binary image.
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    # Identify local peaks in the distance map.
    # Remove the deprecated "indices" parameter.
    local_max_coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    # Convert coordinates to a boolean mask.
    local_max = np.zeros(distance.shape, dtype=bool)
    if local_max_coords.size:
        local_max[tuple(local_max_coords.T)] = True
    markers, _ = ndimage.label(local_max)
    labels = watershed(-distance, markers, mask=binary_mask)
    return labels

def detect_soma_centroids(binary_mask, min_area=40):
    """
    Detect connected components in a binary mask and return centroids for regions
    that meet the minimum area criteria.
    Returns:
        A list of tuples (label, centroid) and the labeled mask.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    soma_list = []
    for label in range(1, num_labels):  # Skip label 0 (background)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            soma_list.append((label, centroids[label]))
    return soma_list, labels

def classify_cell(centroid, green_mask, red_mask):
    """
    Classify the cell at the given centroid based on its presence in the green and red masks.
    """
    x, y = int(round(centroid[0])), int(round(centroid[1]))
    green_present = green_mask[y, x] > 0
    red_present = red_mask[y, x] > 0

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
    Return the hex code of the pixel at the given centroid in the original image.
    Note: OpenCV loads images in BGR order.
    """
    x, y = int(round(centroid[0])), int(round(centroid[1]))
    b, g, r = image[y, x]
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)

def process_image(image_path, green_thresh=60, red_thresh=60, median_ksize=3,
                  gaussian_sigma=1.0, min_area=40, min_distance=10):
    """
    Process the given image to detect, count, and classify soma.
    Returns:
        - results: List of dictionaries with soma details.
        - overlay: Annotated overlay image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image at {image_path}")
    
    # Create an overlay image (convert to RGB for annotation)
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split channels (OpenCV uses BGR order)
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # Preprocess channels
    g_proc = preprocess_channel(g_channel, median_ksize, gaussian_sigma)
    r_proc = preprocess_channel(r_channel, median_ksize, gaussian_sigma)
    
    # Threshold channels
    g_binary = threshold_channel(g_proc, green_thresh)
    r_binary = threshold_channel(r_proc, red_thresh)
    
    # Apply watershed segmentation to split merged cells
    g_labels = split_cells(g_binary, min_distance)
    r_labels = split_cells(r_binary, min_distance)
    
    # Convert watershed labels to binary masks (non-zero pixels are cells)
    g_watershed = (g_labels > 0).astype(np.uint8) * 255
    r_watershed = (r_labels > 0).astype(np.uint8) * 255
    
    # Create a combined mask (logical OR of green and red detections)
    combined_mask = cv2.bitwise_or(g_watershed, r_watershed)
    
    # Detect soma centroids in the combined mask
    soma_centroids, _ = detect_soma_centroids(combined_mask, min_area)
    
    results = []
    for label, centroid in soma_centroids:
        cell_type = classify_cell(centroid, g_watershed, r_watershed)
        hex_val = hex_color_at(image, centroid)
        x_micron = centroid[0] * PIXEL_TO_MICRON
        y_micron = centroid[1] * PIXEL_TO_MICRON
        results.append({
            "ID": label,
            "X_pixel": centroid[0],
            "Y_pixel": centroid[1],
            "X_micron": x_micron,
            "Y_micron": y_micron,
            "Type": cell_type,
            "Hex": hex_val
        })
        # Annotate overlay image with a circle and label
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

def save_results(results, overlay, output_folder="output", save_csv=True, save_overlay=True):
    """
    Save results to a CSV file and the annotated overlay image.
    """
    if save_csv:
        df = pd.DataFrame(results)
        csv_filename = f"{output_folder}/soma.csv"
        df.to_csv(csv_filename, index=False)
        print(f"CSV file saved: {csv_filename}")
    if save_overlay:
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_filename = f"{output_folder}/overlay.png"
        cv2.imwrite(overlay_filename, overlay_bgr)
        print(f"Overlay image saved: {overlay_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Automated Soma Counting in Fluorescent Microscopy Images"
    )
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--green_thresh", type=int, default=60,
                        help="Intensity threshold for green channel (default: 60)")
    parser.add_argument("--red_thresh", type=int, default=60,
                        help="Intensity threshold for red channel (default: 60)")
    parser.add_argument("--median_ksize", type=int, default=3,
                        help="Kernel size for median filtering (default: 3)")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0,
                        help="Sigma for Gaussian blur (default: 1.0)")
    parser.add_argument("--min_area", type=int, default=40,
                        help="Minimum area (in pixels) for a region to be considered a soma (default: 40)")
    parser.add_argument("--min_distance", type=int, default=10,
                        help="Minimum distance between peaks in watershed segmentation (default: 10)")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Folder for output files (default: 'output')")
    parser.add_argument("--save_csv", action="store_true",
                        help="Flag to save the CSV file")
    parser.add_argument("--save_overlay", action="store_true",
                        help="Flag to save the overlay image")
    
    args = parser.parse_args()
    
    results, overlay = process_image(
        args.image_path,
        green_thresh=args.green_thresh,
        red_thresh=args.red_thresh,
        median_ksize=args.median_ksize,
        gaussian_sigma=args.gaussian_sigma,
        min_area=args.min_area,
        min_distance=args.min_distance
    )
    
    save_results(results, overlay, output_folder=args.output_folder,
                 save_csv=args.save_csv, save_overlay=args.save_overlay)

if __name__ == "__main__":
    main()
