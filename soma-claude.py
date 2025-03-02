#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import cv2
import os
from skimage import measure, exposure, morphology, feature, segmentation
from cellpose import models
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Count and locate cell somas in fluorescent microscopy images.')
    
    # Input/output options
    parser.add_argument('--input_dir', type=str, default='input', help='Path to input directory with images (default: input)')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to output directory (default: output)')
    parser.add_argument('--input', type=str, help='Path to a single input image (overrides input_dir)')
    parser.add_argument('--output_csv', type=str, help='Path to output CSV file (for single image mode)')
    parser.add_argument('--output_image', type=str, help='Path to output annotated image (for single image mode)')
    
    # Hardware options
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    
    # Cellpose options
    parser.add_argument('--model_type', type=str, default='cyto3', choices=['cyto', 'nuclei', 'cyto2', 'cyto3'],
                        help='Cellpose model type (default: cyto3)')
    parser.add_argument('--channels', type=int, nargs=2, default=[1, 2],
                        help='Channel settings [0=grayscale, 1=red, 2=green, 3=blue] (default: [1, 2])')
    parser.add_argument('--diameter', type=float, default=30.0,
                        help='Expected cell diameter in pixels (default: 30.0)')
    
    # Cell classification options
    parser.add_argument('--classification_method', type=str, default='ratio', 
                        choices=['threshold', 'ratio'],
                        help='Method for classifying cells (default: ratio)')
    parser.add_argument('--red_threshold', type=float, default=0.3,
                        help='Threshold for red channel to classify as tdTomato cell (default: 0.3)')
    parser.add_argument('--green_threshold', type=float, default=0.3,
                        help='Threshold for green channel to classify as EGFP cell (default: 0.3)')
    parser.add_argument('--rg_ratio_threshold', type=float, default=1.5,
                        help='Ratio threshold for red/green to classify cell type (default: 1.5)')
    parser.add_argument('--coexpress_min_ratio', type=float, default=0.7,
                        help='Minimum ratio between channels for coexpression (default: 0.7)')
    parser.add_argument('--coexpress_max_ratio', type=float, default=1.3,
                        help='Maximum ratio between channels for coexpression (default: 1.3)')
    
    # Preprocessing options
    parser.add_argument('--contrast_enhancement', type=str, default='adapthist',
                        choices=['none', 'hist', 'adapthist'], 
                        help='Contrast enhancement method (default: adapthist)')
    parser.add_argument('--denoise', action='store_true',
                        help='Apply denoising before processing')
    parser.add_argument('--background_subtraction', action='store_true',
                        help='Apply background subtraction')
    parser.add_argument('--background_radius', type=int, default=50,
                        help='Radius for background subtraction (default: 50 pixels)')
    
    # Soma separation options
    parser.add_argument('--enable_soma_separation', action='store_true', default=True,
                        help='Enable separation of merged somas')
    parser.add_argument('--local_region_size', type=int, default=200,
                        help='Size of local region for calculating average soma size (default: 200 pixels)')
    parser.add_argument('--size_deviation_threshold', type=float, default=1.8,
                        help='Threshold for determining oversized somas (default: 1.8)')
    parser.add_argument('--min_distance_peaks', type=int, default=15,
                        help='Minimum distance between peaks for watershed separation (default: 15 pixels)')
    
    # Filtering options
    parser.add_argument('--min_soma_size', type=float, default=15.0,
                        help='Minimum soma size in pixels to include (default: 15.0)')
    parser.add_argument('--max_soma_size', type=float, default=1000.0,
                        help='Maximum soma size in pixels to include (default: 1000.0)')
    parser.add_argument('--circularity_threshold', type=float, default=0.3,
                        help='Minimum circularity to include (0-1, default: 0.3)')
    
    # Confidence settings
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Confidence threshold for cellpose (default: 0.0)')
    parser.add_argument('--flow_threshold', type=float, default=0.4,
                        help='Flow threshold for cellpose (default: 0.4)')
    
    # Pixel to micron conversion
    parser.add_argument('--pixels_per_micron', type=float, default=1.6091,
                        help='Pixels per micron conversion factor (default: 1.6091)')
    
    return parser.parse_args()

def preprocess_image(image_path, args):
    """Preprocess the image for better segmentation."""
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Make a copy of the original
    original_img = img.copy()
    
    # Convert to float and scale to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Split channels
    r, g, b = cv2.split(img)
    
    # Apply denoising if requested
    if args.denoise:
        r = cv2.GaussianBlur(r, (3, 3), 0)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        b = cv2.GaussianBlur(b, (3, 3), 0)
    
    # Apply background subtraction if requested
    if args.background_subtraction:
        r = subtract_background(r, args.background_radius)
        g = subtract_background(g, args.background_radius)
        b = subtract_background(b, args.background_radius)
    
    # Apply contrast enhancement
    if args.contrast_enhancement == 'hist':
        r = exposure.equalize_hist(r)
        g = exposure.equalize_hist(g)
        b = exposure.equalize_hist(b)
    elif args.contrast_enhancement == 'adapthist':
        r = exposure.equalize_adapthist(r)
        g = exposure.equalize_adapthist(g)
        b = exposure.equalize_adapthist(b)
    
    # Merge channels
    img_preprocessed = cv2.merge([r, g, b])
    
    return original_img, img_preprocessed

def subtract_background(img, radius):
    """Subtract background using a rolling ball algorithm."""
    # Create a structuring element (ball)
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    
    # Perform morphological opening (erosion followed by dilation)
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, selem)
    
    # Subtract the background from the original image
    result = img - background
    
    # Clip to [0, 1] range
    result = np.clip(result, 0, 1)
    
    return result

def segment_cells(img_preprocessed, args):
    """Segment cells using cellpose."""
    # Create cellpose model with GPU enabled by default unless --no_gpu is specified
    use_gpu = not args.no_gpu
    model = models.Cellpose(gpu=use_gpu, model_type=args.model_type)
    
    if use_gpu:
        print("Using GPU for cell segmentation")
    else:
        print("Using CPU for cell segmentation (GPU disabled)")
    
    # Run cellpose
    masks, flows, styles, diams = model.eval(img_preprocessed, 
                                           channels=args.channels,
                                           diameter=args.diameter,
                                           flow_threshold=args.flow_threshold,
                                           cellprob_threshold=args.confidence_threshold)
    
    return masks, flows, styles, diams

def calculate_local_average_size(masks, local_region_size):
    """Calculate average soma size in local regions."""
    h, w = masks.shape
    local_sizes = {}
    
    # Get properties of all regions
    props = measure.regionprops(masks)
    
    # Calculate sizes of all cells
    sizes = [p.area for p in props]
    
    # Calculate average size for each cell based on its neighborhood
    for prop in props:
        y, x = prop.centroid
        y, x = int(y), int(x)
        
        # Define local region
        y_min = max(0, y - local_region_size // 2)
        y_max = min(h, y + local_region_size // 2)
        x_min = max(0, x - local_region_size // 2)
        x_max = min(w, x + local_region_size // 2)
        
        # Get cells in local region
        local_mask = masks[y_min:y_max, x_min:x_max]
        local_labels = np.unique(local_mask)
        local_labels = local_labels[local_labels > 0]  # Remove background
        
        # Get sizes of local cells
        local_sizes_list = [sizes[label-1] for label in local_labels if label <= len(sizes)]
        
        if local_sizes_list:
            local_sizes[prop.label] = np.median(local_sizes_list)
        else:
            local_sizes[prop.label] = prop.area
    
    return local_sizes

def separate_merged_somas(masks, local_sizes, args):
    """Separate merged somas using watershed algorithm."""
    # Create a new mask for separated cells
    new_masks = masks.copy()
    max_label = np.max(masks)
    
    # Get properties of all regions
    props = measure.regionprops(masks)
    
    separation_count = 0
    
    for prop in props:
        label = prop.label
        area = prop.area
        
        # Skip if label not in local_sizes (this shouldn't happen normally)
        if label not in local_sizes:
            continue
            
        local_avg_size = local_sizes[label]
        
        # Check if this soma is significantly larger than the local average
        if area > args.size_deviation_threshold * local_avg_size:
            # Estimate number of somas in this region
            estimated_count = max(2, int(round(area / local_avg_size)))
            
            # Extract this cell
            cell_mask = (masks == label).astype(np.uint8)
            
            # Distance transform
            distance = cv2.distanceTransform(cell_mask, cv2.DIST_L2, 5)
            
            # Find local maxima
            coords = feature.peak_local_max(distance, min_distance=args.min_distance_peaks,
                                           num_peaks=estimated_count, exclude_border=False)
            
            # Skip if we couldn't find multiple peaks
            if len(coords) <= 1:
                continue
                
            # Create markers for watershed
            markers = np.zeros_like(cell_mask)
            for i, (x, y) in enumerate(coords):
                markers[x, y] = i + 1
            
            # Apply watershed
            separated = segmentation.watershed(-distance, markers, mask=cell_mask)
            
            # Replace the original cell with separated cells
            for i in range(1, np.max(separated) + 1):
                submask = (separated == i)
                if np.sum(submask) > 0:
                    max_label += 1
                    new_masks[submask] = max_label
            
            # Remove the original cell
            new_masks[cell_mask.astype(bool) & (new_masks == label)] = 0
            
            separation_count += 1
    
    print(f"Separated {separation_count} merged somas")
    
    # Relabel to ensure consecutive labels
    new_masks = measure.label(new_masks > 0)
    
    return new_masks

def filter_cells(masks, args):
    """Filter cells based on size and shape criteria."""
    # Get properties of all regions
    props = measure.regionprops(masks)
    
    # Create a new mask
    filtered_masks = np.zeros_like(masks)
    
    filtered_count = 0
    total_count = len(props)
    
    for prop in props:
        # Calculate circularity (4*pi*area/perimeter^2)
        perimeter = prop.perimeter
        area = prop.area
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        
        # Check if the cell meets the criteria
        if (args.min_soma_size <= area <= args.max_soma_size and 
            circularity >= args.circularity_threshold):
            # Keep this cell
            filtered_masks[masks == prop.label] = prop.label
        else:
            filtered_count += 1
    
    # Relabel to ensure consecutive labels
    filtered_masks = measure.label(filtered_masks > 0)
    
    print(f"Filtered out {filtered_count} of {total_count} cells based on size and shape")
    
    return filtered_masks

def classify_cells_threshold(img, masks, args):
    """Classify cells using simple thresholding."""
    # Get properties of all regions
    props = measure.regionprops(masks)
    
    # Extract RGB channels
    r, g, b = cv2.split(img)
    
    # Normalize to [0, 1] if needed
    if r.max() > 1:
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
    
    # Initialize empty list for cell data
    cells = []
    
    for prop in props:
        # Get mask for this cell
        cell_mask = masks == prop.label
        
        # Calculate mean intensities in each channel
        r_mean = np.mean(r[cell_mask])
        g_mean = np.mean(g[cell_mask])
        b_mean = np.mean(b[cell_mask])
        
        # Skip cells with very low intensity in both channels
        if r_mean < 0.05 and g_mean < 0.05:
            continue
        
        # Calculate center coordinates
        y, x = prop.centroid
        
        # Calculate area
        area_pixels = prop.area
        area_microns = area_pixels / (args.pixels_per_micron ** 2)
        
        # Get color at center
        y_int, x_int = int(y), int(x)
        if 0 <= y_int < img.shape[0] and 0 <= x_int < img.shape[1]:
            center_color = img[y_int, x_int]
            if center_color.max() <= 1.0:  # If normalized
                center_color = (center_color * 255).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(center_color[0]), int(center_color[1]), int(center_color[2]))
        else:
            if r_mean <= 1.0:  # If normalized
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(r_mean * 255), int(g_mean * 255), int(b_mean * 255))
            else:
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(r_mean), int(g_mean), int(b_mean))
        
        # Classify cell based on thresholds
        if r_mean > args.red_threshold and g_mean > args.green_threshold:
            # Check ratio for coexpression
            ratio = r_mean / g_mean if g_mean > 0 else float('inf')
            if args.coexpress_min_ratio <= ratio <= args.coexpress_max_ratio:
                cell_type = "coexpressed"
            else:
                cell_type = "tdTomato" if r_mean > g_mean else "EGFP"
        elif r_mean > args.red_threshold:
            cell_type = "tdTomato"
        elif g_mean > args.green_threshold:
            cell_type = "EGFP"
        else:
            # Not enough signal in either channel, could be noise
            continue
        
        cells.append({
            'id': prop.label,
            'type': cell_type,
            'center_x': x,
            'center_y': y,
            'center_color_hex': hex_color,
            'red_intensity': r_mean,
            'green_intensity': g_mean,
            'blue_intensity': b_mean,
            'area_pixels': area_pixels,
            'area_microns': area_microns,
            'perimeter': prop.perimeter,
            'circularity': 4 * np.pi * area_pixels / (prop.perimeter**2) if prop.perimeter > 0 else 0
        })
    
    return cells

def classify_cells_ratio(img, masks, args):
    """Classify cells based on red/green ratio."""
    # Get properties of all regions
    props = measure.regionprops(masks)
    
    # Extract RGB channels
    r, g, b = cv2.split(img)
    
    # Normalize to [0, 1] if needed
    if r.max() > 1:
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
    
    # Initialize empty list for cell data
    cells = []
    
    for prop in props:
        # Get mask for this cell
        cell_mask = masks == prop.label
        
        # Calculate mean intensities in each channel
        r_mean = np.mean(r[cell_mask])
        g_mean = np.mean(g[cell_mask])
        b_mean = np.mean(b[cell_mask])
        
        # Skip cells with very low intensity in both channels
        if r_mean < 0.05 and g_mean < 0.05:
            continue
        
        # Calculate ratio
        ratio = r_mean / g_mean if g_mean > 0 else float('inf')
        inverse_ratio = g_mean / r_mean if r_mean > 0 else float('inf')
        
        # Classify based on ratio
        if args.coexpress_min_ratio <= ratio <= args.coexpress_max_ratio or \
           args.coexpress_min_ratio <= inverse_ratio <= args.coexpress_max_ratio:
            cell_type = "coexpressed"
        elif ratio > args.rg_ratio_threshold:
            cell_type = "tdTomato"
        elif inverse_ratio > args.rg_ratio_threshold:
            cell_type = "EGFP"
        else:
            # Fallback to intensity-based classification
            if r_mean > g_mean:
                cell_type = "tdTomato"
            else:
                cell_type = "EGFP"
        
        # Calculate center coordinates and other properties
        y, x = prop.centroid
        area_pixels = prop.area
        area_microns = area_pixels / (args.pixels_per_micron ** 2)
        
        # Get color at center
        y_int, x_int = int(y), int(x)
        if 0 <= y_int < img.shape[0] and 0 <= x_int < img.shape[1]:
            center_color = img[y_int, x_int]
            if center_color.max() <= 1.0:  # If normalized
                center_color = (center_color * 255).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(center_color[0]), int(center_color[1]), int(center_color[2]))
        else:
            if r_mean <= 1.0:  # If normalized
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(r_mean * 255), int(g_mean * 255), int(b_mean * 255))
            else:
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(r_mean), int(g_mean), int(b_mean))
        
        cells.append({
            'id': prop.label,
            'type': cell_type,
            'center_x': x,
            'center_y': y,
            'center_color_hex': hex_color,
            'red_intensity': r_mean,
            'green_intensity': g_mean,
            'blue_intensity': b_mean,
            'red_green_ratio': ratio,
            'area_pixels': area_pixels,
            'area_microns': area_microns,
            'perimeter': prop.perimeter,
            'circularity': 4 * np.pi * area_pixels / (prop.perimeter**2) if prop.perimeter > 0 else 0
        })
    
    return cells

def classify_cells(img, masks, args):
    """Classify cells based on the selected method."""
    if args.classification_method == 'threshold':
        return classify_cells_threshold(img, masks, args)
    elif args.classification_method == 'ratio':
        return classify_cells_ratio(img, masks, args)
    else:
        print(f"Warning: Classification method {args.classification_method} not supported. Using ratio.")
        return classify_cells_ratio(img, masks, args)

def create_output_image(img, cells, masks):
    """Create an annotated output image."""
    # Create a copy of the original image for annotation
    if img.max() <= 1.0:  # If normalized to [0, 1]
        output_img = (img * 255).astype(np.uint8)
    else:
        output_img = img.copy().astype(np.uint8)
    
    # Define colors for different cell types (BGR for OpenCV)
    colors = {
        "EGFP": (0, 255, 0),        # Green
        "tdTomato": (255, 0, 0),    # Red
        "coexpressed": (255, 255, 0) # Yellow
    }
    
    # Create a mask to visualize cell types
    cell_type_mask = np.zeros_like(output_img)
    
    # Create a label overlay image
    label_overlay = img.copy()
    if label_overlay.max() <= 1.0:
        label_overlay = (label_overlay * 255).astype(np.uint8)
    
    # Generate a colormap for labels - using HSV for maximum differentiation
    num_cells = len(cells)
    label_colors = {}
    for i, cell in enumerate(cells):
        # Generate diverse colors using HSV color space
        # OpenCV uses hue values from 0-179 for uint8 images
        hue = int((i * 137.5) % 180)  # Golden angle, constrained to OpenCV's HSV range
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
        label_colors[cell['id']] = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))
    
    # Draw cell boundaries and centers
    for cell in cells:
        label = cell['id']
        cell_type = cell['type']
        center_x, center_y = int(cell['center_x']), int(cell['center_y'])
        
        # Get cell contours
        cell_mask = (masks == label)
        contours = measure.find_contours(cell_mask.astype(float), 0.5)
        
        # Draw contours
        for contour in contours:
            # Convert contour points to int and swap x, y for OpenCV
            contour = np.fliplr(contour).astype(np.int32)
            cv2.polylines(output_img, [contour], True, colors[cell_type], 2)
            
            # Fill the cell type mask
            cv2.fillPoly(cell_type_mask, [contour], colors[cell_type])
            
            # Fill the label overlay with unique colors for each cell
            cv2.fillPoly(label_overlay, [contour], label_colors[label])
        
        # Draw center marker
        cv2.circle(output_img, (center_x, center_y), 3, colors[cell_type], -1)
        
        # Draw cell ID
        cv2.putText(output_img, str(label), (center_x + 5, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw label in the center of each cell in the label overlay
        cv2.putText(label_overlay, str(label), (center_x - 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Create a blended image with partial transparency for cell type
    alpha = 0.3
    blended_img = cv2.addWeighted(output_img, 1 - alpha, cell_type_mask, alpha, 0)
    
    # Create a blended image for label overlay (semi-transparent)
    alpha_label = 0.4
    label_blended = cv2.addWeighted(img.astype(np.uint8) if img.max() <= 1.0 else img.copy(), 
                                    1 - alpha_label, label_overlay, alpha_label, 0)
    
    return output_img, blended_img, cell_type_mask, label_overlay, label_blended

def save_results(cells, args, masks, img, output_imgs, output_base=None):
    """Save results to CSV and/or image."""
    # Create a DataFrame from the cell data
    df = pd.DataFrame(cells)
    
    # Count cells by type
    count_types = ['EGFP', 'tdTomato', 'coexpressed']
    counts = {cell_type: 0 for cell_type in count_types}
    
    for cell in cells:
        counts[cell['type']] += 1
    
    print("\nCell counts:")
    for cell_type in count_types:
        print(f"{cell_type}: {counts[cell_type]}")
    print(f"Total: {len(cells)}")
    
    # Add columns for coordinates in microns
    if 'center_x' in df.columns and 'center_y' in df.columns:
        df['center_x_microns'] = df['center_x'] / args.pixels_per_micron
        df['center_y_microns'] = df['center_y'] / args.pixels_per_micron
    
    # Determine output paths
    csv_path = args.output_csv
    image_path = args.output_image
    
    # For batch processing mode, use output_base
    if output_base is not None:
        csv_path = f"{output_base}.csv"
        image_path = f"{output_base}.png"
    
    # Save CSV
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"\nSaved cell data to {csv_path}")
    
    # Save annotated image
    if image_path:
        output_img, blended_img, cell_type_mask, label_overlay, label_blended = output_imgs
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Plot original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot segmentation mask
        plt.subplot(2, 3, 2)
        plt.imshow(masks, cmap='viridis')
        plt.title('Cell Segmentation')
        plt.axis('off')
        
        # Plot annotated image
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
        plt.title('Annotated Cells')
        plt.axis('off')
        
        # Plot label overlay
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(label_blended, cv2.COLOR_BGR2RGB))
        plt.title('Label Overlay')
        plt.axis('off')
        
        # Plot cell type distribution
        plt.subplot(2, 3, 5)
        bars = plt.bar(count_types, [counts[t] for t in count_types], 
                      color=['green', 'red', 'yellow'])
        plt.title('Cell Type Distribution')
        plt.ylabel('Count')
        
        # Add count labels above bars
        for bar, count in zip(bars, [counts[t] for t in count_types]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(image_path, dpi=300)
        print(f"Saved annotated image to {image_path}")
        
        # Also save individual visualization components with suffix
        base, ext = os.path.splitext(image_path)
        cv2.imwrite(f"{base}_outlines{ext}", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{base}_filled{ext}", cv2.cvtColor(cell_type_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{base}_blended{ext}", cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{base}_labels{ext}", cv2.cvtColor(label_overlay, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"{base}_labels_blended{ext}", cv2.cvtColor(label_blended, cv2.COLOR_BGR2RGB))

def process_image(image_path, args, output_base=None):
    """Process a single image."""
    print(f"\nProcessing image: {image_path}")
    
    # Preprocess image
    img_original, img_preprocessed = preprocess_image(image_path, args)
    print("Image preprocessing complete")
    
    # Segment cells
    print("Running cellpose segmentation...")
    masks, flows, styles, diams = segment_cells(img_preprocessed, args)
    print(f"Initial segmentation found {len(np.unique(masks)) - 1} cells")
    
    # Filter masks based on size and shape
    masks = filter_cells(masks, args)
    print(f"After filtering: {len(np.unique(masks)) - 1} cells")
    
    # Separate merged somas if enabled
    if args.enable_soma_separation:
        print("Calculating local average sizes...")
        local_sizes = calculate_local_average_size(masks, args.local_region_size)
        
        print("Separating merged somas...")
        masks = separate_merged_somas(masks, local_sizes, args)
        print(f"After separation: {len(np.unique(masks)) - 1} cells")
    
    # Classify cells
    print(f"Classifying cells using {args.classification_method} method...")
    cells = classify_cells(img_original, masks, args)
    
    # Create output image
    output_imgs = create_output_image(img_original, cells, masks)
    
    # Save results
    save_results(cells, args, masks, img_original, output_imgs, output_base)
    
    return len(cells)

def main():
    """Main function for cell soma analysis."""
    # Parse arguments
    args = parse_arguments()
    
    print("\n=== Brain Cell Soma Analysis Tool ===")
    print(f"GPU Acceleration: {'Disabled' if args.no_gpu else 'Enabled'}")
    
    # Ensure the output directory exists
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Single image mode
    if args.input:
        process_image(args.input, args)
    
    # Directory mode
    else:
        # Ensure input directory exists
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' does not exist.")
            return
        
        # Get all image files in the input directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        image_files = [f for f in os.listdir(args.input_dir) 
                       if os.path.isfile(os.path.join(args.input_dir, f)) 
                       and f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No image files found in '{args.input_dir}'")
            return
        
        # Process each image
        print(f"Found {len(image_files)} image files in '{args.input_dir}'")
        total_cells = 0
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(args.input_dir, image_file)
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_base = os.path.join(args.output_dir, filename)
            
            print(f"\nProcessing file {i+1}/{len(image_files)}: {image_file}")
            cell_count = process_image(image_path, args, output_base)
            total_cells += cell_count
        
        print(f"\nAll processing complete! Total cells found: {total_cells}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
