#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from cellpose import models
import tifffile
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---------------- Preprocessing Functions ----------------

def apply_clahe(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def apply_denoise(image, sigma):
    return cv2.GaussianBlur(image, (0,0), sigma)

def apply_rolling_ball(image, radius):
    ksize = int(radius)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    subtracted = cv2.subtract(image, background)
    return cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def apply_hist_equalization(image):
    return cv2.equalizeHist(image)

def apply_gamma_correction(image, gamma):
    normalized = image / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)

def apply_morphological_closing(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def segment_dapi_adaptive(image, block_size, C):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, C)

# ---------------- Segmentation and Analysis Functions ----------------

def load_multichannel_tiff(path):
    img = tifffile.imread(path)
    print(f"Loaded multi-page TIFF: shape {img.shape}, dtype {img.dtype}")
    if img.shape[0] != 3:
        raise ValueError("Expected exactly 3 pages in the TIFF file.")
    return img

def create_composite(dapi, tdtomato, egfp):
    dapi = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tdtomato = cv2.normalize(tdtomato, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    egfp = cv2.normalize(egfp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge([dapi, egfp, tdtomato])

def segment_dapi_fixed(image, threshold):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def segment_channel(image, cellpose_model, diameter, flow_threshold, cellprob_threshold, augment, batch_size):
    if np.count_nonzero(image) == 0:
        print("Warning: image appears to be completely empty.")
        return np.zeros_like(image)
    print("Segmenting channel; mean normalized intensity:", np.mean(image))
    masks, _, _, _ = cellpose_model.eval(
        image,
        channels=[0, 0],
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        augment=augment,
        batch_size=batch_size
    )
    return masks

def find_contours(mask):
    binary = ((mask > 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def count_mask_contour_intersection(dapi_mask, contour):
    contour_mask = np.zeros_like(dapi_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    return np.count_nonzero(cv2.bitwise_and(dapi_mask, contour_mask))

def filter_contours_by_dapi_content(contours, dapi_mask, dapi_count_threshold):
    valid = []
    for contour in contours:
        if count_mask_contour_intersection(dapi_mask, contour) >= dapi_count_threshold:
            valid.append(contour)
    return valid

def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return (0, 0)

def combine_coexpressed_contours(red_contours, green_contours, overlap_threshold, pixels_per_micron, distance_threshold_microns = 50):
    coexpressed_contours = []
    red_used = [False] * len(red_contours)
    green_used = [False] * len(green_contours)
    distance_threshold_pixels = distance_threshold_microns * pixels_per_micron
    red_centroids = [compute_centroid(r) for r in red_contours]
    green_centroids = [compute_centroid(g) for g in green_contours]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for i, r in enumerate(red_contours):
        if red_used[i]:
            continue
        for j, g in enumerate(green_contours):
            if green_used[j]:
                continue

            c1 = red_centroids[i]
            c2 = green_centroids[j]
            centroid_distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            if centroid_distance > distance_threshold_pixels:
                continue

            all_points = np.vstack((r, g))
            x, y, w, h = cv2.boundingRect(all_points)
            mask_r_cpu = np.zeros((h, w), dtype=np.uint8)
            mask_g_cpu = np.zeros((h, w), dtype=np.uint8)
            r_shifted = r - np.array([x, y])
            g_shifted = g - np.array([x, y])
            cv2.fillPoly(mask_r_cpu, [r_shifted], 255)
            cv2.fillPoly(mask_g_cpu, [g_shifted], 255)
            mask_r = torch.from_numpy(mask_r_cpu).to(device)
            mask_g = torch.from_numpy(mask_g_cpu).to(device)
            intersection = torch.logical_and(mask_r, mask_g)
            intersection_area = int(torch.sum(intersection).item())
            area_r = cv2.contourArea(r)
            area_g = cv2.contourArea(g)
            union_area = area_r + area_g - intersection_area
            if union_area <= 0:
                continue
            iou = intersection_area / union_area
            
            if iou >= overlap_threshold:
                combined = cv2.convexHull(np.vstack((r, g)))
                coexpressed_contours.append(combined)
                red_used[i] = True
                green_used[j] = True
                break

    remaining_red = [c for idx, c in enumerate(red_contours) if not red_used[idx]]
    remaining_green = [c for idx, c in enumerate(green_contours) if not green_used[idx]]
    return remaining_red, remaining_green, coexpressed_contours

def draw_overlay(composite, dapi_mask, tdtomato_mask, egfp_mask, dapi_count_threshold, overlap_threshold, pixels_per_micron):
    overlay = composite.copy()
    red_contours = filter_contours_by_dapi_content(find_contours(tdtomato_mask), dapi_mask, dapi_count_threshold)
    green_contours = filter_contours_by_dapi_content(find_contours(egfp_mask), dapi_mask, dapi_count_threshold)
    red_contours, green_contours, coexpressed_contours = combine_coexpressed_contours(red_contours, green_contours, overlap_threshold, pixels_per_micron)
    print({
        "tdTomato": len(red_contours),
        "EGFP": len(green_contours),
        "Coexpressed": len(coexpressed_contours)
    })
    overlay[dapi_mask > 0] = (255, 255, 0)  # Cyan for DAPI.
    cv2.drawContours(overlay, red_contours, -1, (0,0,255), 2)
    cv2.drawContours(overlay, green_contours, -1, (0,255,0), 2)
    cv2.drawContours(overlay, coexpressed_contours, -1, (0,255,255), 2)
    return overlay, red_contours, green_contours, coexpressed_contours

def compute_cell_properties(contour, pixels_per_micron):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    equiv_diameter_pixels = 2 * np.sqrt(area / np.pi)
    if pixels_per_micron:
        equiv_diameter_microns = equiv_diameter_pixels / pixels_per_micron
    else:
        equiv_diameter_microns = None
    return {"centroid_x": cX, "centroid_y": cY, "area": area, "perimeter": perimeter,
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "equiv_diameter_microns": equiv_diameter_microns}

def export_cell_data(red_contours, green_contours, coexpressed_contours, out_folder, pixels_per_micron):
    cells = []
    cell_id = 1
    for contour, cat in zip(
        [*red_contours, *green_contours, *coexpressed_contours],
        ["tdTomato"] * len(red_contours) + ["EGFP"] * len(green_contours) + ["Coexpressed"] * len(coexpressed_contours)
    ):
        props = compute_cell_properties(contour, pixels_per_micron)
        props["cell_id"] = cell_id
        props["category"] = cat
        cells.append(props)
        cell_id += 1
    df = pd.DataFrame(cells)
    csv_path = os.path.join(out_folder, "cell_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Cell data saved to: {csv_path}")
    return df

def create_bar_chart(df, out_folder):
    counts = df["category"].value_counts()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Cell Counts by Category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    chart_path = os.path.join(out_folder, "cell_bar_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"Bar chart saved to: {chart_path}")

def create_cells_only_image(image_shape, red_contours, green_contours, coexpressed_contours, out_folder):
    cell_mask = np.zeros(image_shape, dtype=np.uint8)
    all_contours = red_contours + green_contours + coexpressed_contours
    cv2.drawContours(cell_mask, all_contours, -1, 255, thickness=cv2.FILLED)
    mask_path = os.path.join(out_folder, "cells_only.png")
    cv2.imwrite(mask_path, cell_mask)
    print(f"Cells-only image saved to: {mask_path}")
    return cell_mask

def create_cells_colored_image(image_shape, red_contours, green_contours, coexpressed_contours, out_folder):
    colored_cells = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    cv2.drawContours(colored_cells, red_contours, -1, (0,0,255), thickness=cv2.FILLED)   # Red for tdTomato.
    cv2.drawContours(colored_cells, green_contours, -1, (0,255,0), thickness=cv2.FILLED) # Green for EGFP.
    cv2.drawContours(colored_cells, coexpressed_contours, -1, (0,255,255), thickness=cv2.FILLED) # Yellow for Coexpressed.
    path = os.path.join(out_folder, "cells_colored.png")
    cv2.imwrite(path, colored_cells)
    print(f"Colored cells image saved to: {path}")
    return colored_cells

def create_scatter_plot(df, out_folder):
    plt.figure()
    categories = {"tdTomato": "red", "EGFP": "green", "Coexpressed": "blue"}
    for cat, color in categories.items():
        sub = df[df["category"] == cat]
        plt.scatter(sub["centroid_x"], sub["centroid_y"], c=color, label=cat, alpha=0.6)
    plt.title("Cell Centroids by Category")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    scatter_path = os.path.join(out_folder, "cell_centroids_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved to: {scatter_path}")

def create_area_histogram(df, out_folder, pixels_per_micron):
    plt.figure()
    if pixels_per_micron is not None:
        df["area_microns"] = df["area"] / (pixels_per_micron ** 2)
        plt.hist(df["area_microns"], alpha=0.5, label="Area (microns²)")
        plt.xlabel("Area (microns²)")
    else:
        plt.hist(df["area"], alpha=0.5, label="Area (pixels)")
        plt.xlabel("Area (pixels)")
    plt.title("Histogram of Cell Areas")
    plt.ylabel("Frequency")
    plt.legend()
    hist_path = os.path.join(out_folder, "cell_area_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"Area histogram saved to: {hist_path}")

def create_diameter_histogram(df, out_folder, pixels_per_micron):
    plt.figure()
    if pixels_per_micron is not None and "equiv_diameter_microns" in df.columns:
        diameters = df["equiv_diameter_microns"]
        plt.xlabel("Diameter (microns)")
    else:
        # Fallback: compute diameter in pixels from area.
        diameters = 2 * np.sqrt(df["area"] / np.pi)
        plt.xlabel("Diameter (pixels)")
    plt.hist(diameters, bins=30, alpha=0.7)
    plt.title("Histogram of Cell Diameters")
    plt.ylabel("Frequency")
    hist_path = os.path.join(out_folder, "cell_diameter_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"Diameter histogram saved to: {hist_path}")

def save_binary_mask(mask, path):
    binary_mask = ((mask > 0).astype(np.uint8)) * 255
    cv2.imwrite(path, binary_mask)
    print(f"Mask saved: {path}")

# ---------------- Output Generation Functions ----------------

def build_output(args, composite, dapi_mask, td_mask, egfp_mask):
    overlay, red_contours, green_contours, coexpressed_contours = draw_overlay(composite, dapi_mask, td_mask, egfp_mask, args.dapi_count_threshold, args.overlap_threshold, args.pixels_per_micron)
    overlay_path = os.path.join(args.out_subfolder, "composite_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"Overlay image saved: {overlay_path}")
    if args.save_masks:
        save_binary_mask(dapi_mask, os.path.join(args.out_subfolder, "dapi_mask.png"))
        save_binary_mask(td_mask, os.path.join(args.out_subfolder, "tdtomato_mask.png"))
        save_binary_mask(egfp_mask, os.path.join(args.out_subfolder, "egfp_mask.png"))
    
    df = export_cell_data(red_contours, green_contours, coexpressed_contours, args.out_subfolder, args.pixels_per_micron)
    create_bar_chart(df, args.out_subfolder)
    create_cells_only_image(dapi_mask.shape, red_contours, green_contours, coexpressed_contours, args.out_subfolder)
    create_cells_colored_image(dapi_mask.shape, red_contours, green_contours, coexpressed_contours, args.out_subfolder)
    create_scatter_plot(df, args.out_subfolder)
    create_area_histogram(df, args.out_subfolder, args.pixels_per_micron)
    create_diameter_histogram(df, args.out_subfolder, args.pixels_per_micron)

def handle_coexpression_testing(args):
    print("Coexpression testing mode: loading previously saved files.")
    composite_path = os.path.join(args.out_subfolder, "composite_overlay.png")
    dapi_mask_path = os.path.join(args.out_subfolder, "dapi_mask.png")
    tdtomato_mask_path = os.path.join(args.out_subfolder, "tdtomato_mask.png")
    egfp_mask_path = os.path.join(args.out_subfolder, "egfp_mask.png")
    composite = cv2.imread(composite_path)
    dapi_mask = cv2.imread(dapi_mask_path, cv2.IMREAD_GRAYSCALE)
    td_mask = cv2.imread(tdtomato_mask_path, cv2.IMREAD_GRAYSCALE)
    egfp_mask = cv2.imread(egfp_mask_path, cv2.IMREAD_GRAYSCALE)
    if composite is None or dapi_mask is None or td_mask is None or egfp_mask is None:
        raise FileNotFoundError("One or more required saved files for coexpression testing were not found.")
    build_output(args, composite, dapi_mask, td_mask, egfp_mask)

# ---------------- Main Processing Function ----------------

def process_file(file_path, args):
    print(f"Processing file: {file_path}")

    if args.coexpression_testing:
        handle_coexpression_testing(args)
        return

    multi_img = load_multichannel_tiff(file_path)
    dapi_page = multi_img[0]
    td_page = multi_img[1]
    egfp_page = multi_img[2]
    dapi_img = dapi_page[..., 1]
    td_img = td_page[..., 0]
    egfp_img = egfp_page[..., 1]
    print("DAPI image shape:", dapi_img.shape)
    print("tdtomato image shape:", td_img.shape)
    print("egfp image shape:", egfp_img.shape)
    if dapi_img.shape != td_img.shape or dapi_img.shape != egfp_img.shape:
        raise ValueError("The pages do not have matching dimensions.")
    
    # ---------------- Optional Preprocessing Steps ----------------
    if args.rolling_ball:
        dapi_img = apply_rolling_ball(dapi_img, args.rolling_ball_radius)
        td_img = apply_rolling_ball(td_img, args.rolling_ball_radius)
        egfp_img = apply_rolling_ball(egfp_img, args.rolling_ball_radius)
    if args.median_filter:
        dapi_img = apply_median_filter(dapi_img, args.median_kernel_size)
        td_img = apply_median_filter(td_img, args.median_kernel_size)
        egfp_img = apply_median_filter(egfp_img, args.median_kernel_size)
    if args.bilateral_filter:
        dapi_img = apply_bilateral_filter(dapi_img, args.bilateral_d, args.bilateral_sigmaColor, args.bilateral_sigmaSpace)
        td_img = apply_bilateral_filter(td_img, args.bilateral_d, args.bilateral_sigmaColor, args.bilateral_sigmaSpace)
        egfp_img = apply_bilateral_filter(egfp_img, args.bilateral_d, args.bilateral_sigmaColor, args.bilateral_sigmaSpace)
    if args.hist_eq:
        dapi_img = apply_hist_equalization(dapi_img)
        td_img = apply_hist_equalization(td_img)
        egfp_img = apply_hist_equalization(egfp_img)
    if args.gamma_correction:
        dapi_img = apply_gamma_correction(dapi_img, args.gamma)
        td_img = apply_gamma_correction(td_img, args.gamma)
        egfp_img = apply_gamma_correction(egfp_img, args.gamma)
    if args.morph_close:
        dapi_img = apply_morphological_closing(dapi_img, args.morph_close_kernel_size)
        td_img = apply_morphological_closing(td_img, args.morph_close_kernel_size)
        egfp_img = apply_morphological_closing(egfp_img, args.morph_close_kernel_size)
    if args.enhance_contrast:
        dapi_img = apply_clahe(dapi_img)
        td_img = apply_clahe(td_img)
        egfp_img = apply_clahe(egfp_img)
    if args.denoise:
        dapi_img = apply_denoise(dapi_img, args.denoise_sigma)
        td_img = apply_denoise(td_img, args.denoise_sigma)
        egfp_img = apply_denoise(egfp_img, args.denoise_sigma)
    
    composite = create_composite(dapi_img, td_img, egfp_img)
    
    # DAPI segmentation: adaptive threshold if enabled, otherwise fixed.
    if args.adaptive_threshold:
        dapi_mask = segment_dapi_adaptive(dapi_img, args.adaptive_block_size, args.adaptive_C)
    else:
        dapi_mask = segment_dapi_fixed(dapi_img, threshold=args.dapi_threshold)
    print("DAPI mask computed.")
    if args.save_masks:
        dapi_mask_path = os.path.join(args.out_subfolder, "dapi_mask.png")
        cv2.imwrite(dapi_mask_path, dapi_mask)
    if args.configure_dapi:
        return

    # Convert cell diameters (provided in microns) to pixels.
    red_diameter_pixels = args.red_cell_diameter * args.pixels_per_micron if args.red_cell_diameter is not None else None
    green_diameter_pixels = args.green_cell_diameter * args.pixels_per_micron if args.green_cell_diameter is not None else None

    cyto_model = models.Cellpose(model_type=args.cellpose_model, gpu=args.use_gpu)
    td_mask = segment_channel(td_img, cyto_model, diameter=red_diameter_pixels,
                                flow_threshold=args.red_flow_threshold,
                                cellprob_threshold=args.red_cellprob_threshold,
                                augment=args.cellpose_augment,
                                batch_size=args.cellpose_batch_size)
    egfp_mask = segment_channel(egfp_img, cyto_model, diameter=green_diameter_pixels,
                                  flow_threshold=args.green_flow_threshold,
                                  cellprob_threshold=args.green_cellprob_threshold,
                                  augment=args.cellpose_augment,
                                  batch_size=args.cellpose_batch_size)
    print("Channel masks computed.")
    build_output(args, composite, dapi_mask, td_mask, egfp_mask)

# ---------------- Main Entry Point and Argument Parsing ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Process multi-page TIFFs for cell segmentation and generate multiple outputs including overlays, CSV data, and visualizations."
    )
    # Basic I/O and conversion.
    parser.add_argument("input_dir", nargs="?", default="input",
                        help="Directory containing multi-page TIFF files (default: 'input')")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the output folders (default: 'output')")
    parser.add_argument("--pixels_per_micron", type=float, default=1.6091,
                        help="Number of pixels per micron (default: 1.6091)")
    # DAPI segmentation.
    parser.add_argument("--dapi_threshold", type=int, default=200,
                        help="Fixed threshold for DAPI segmentation (default: 200)")
    parser.add_argument("--adaptive_threshold", action="store_true", default=False,
                        help="Use adaptive thresholding for DAPI segmentation")
    parser.add_argument("--adaptive_block_size", type=int, default=11,
                        help="Block size for adaptive thresholding (must be odd, default: 11)")
    parser.add_argument("--adaptive_C", type=int, default=2,
                        help="Constant for adaptive thresholding (default: 2)")
    parser.add_argument("--dapi_count_threshold", type=int, default=5,
                        help="Minimum DAPI pixels for a valid nucleus (default: 5)")
    # Cytoplasmic segmentation.
    parser.add_argument("--cyto_flow_threshold", type=float, default=0.3,
                        help="Flow threshold for cytoplasmic segmentation (default: 0.3)")
    parser.add_argument("--cyto_cellprob_threshold", type=float, default=0.0,
                        help="Cell probability threshold for cytoplasmic segmentation (default: 0.0)")
    parser.add_argument("--overlap_threshold", type=float, default=0.5,
                        help="Overlap threshold for combining contours (default: 0.5)")
    parser.add_argument("--cellpose-model", type=str, default="cyto3",
                        help="Cellpose model type (cyto3, nuclei)")
    parser.add_argument("--dapi-use-nuclei-model", action="store_true",
                        help="Use nuclei model for DAPI segmentation (default uses threshold)")
    parser.add_argument("--use-gpu", type=bool, default=True,
                        help="Use GPU for Cellpose (default: True)")
    parser.add_argument("--cellpose-augment", action="store_true", default=True,
                        help="Use Cellpose augmentation (default: True)")
    parser.add_argument("--cellpose-batch-size", type=int, default=8,
                        help="Batch size for Cellpose (default: 8)")
    # Cell diameters (in microns); if None, auto-estimation is used.
    parser.add_argument("--default-cell-diameter", type=float, default=None,
                        help="Default cell diameter in microns")
    parser.add_argument("--dapi-cell-diameter", type=float, default=None,
                        help="DAPI cell diameter in microns")
    parser.add_argument("--red-cell-diameter", type=float, default=None,
                        help="Red cell diameter in microns")
    parser.add_argument("--green-cell-diameter", type=float, default=None,
                        help="Green cell diameter in microns")
    # Additional thresholds.
    parser.add_argument("--default-flow-threshold", type=float, default=0.4,
                        help="Default flow threshold for cell detection")
    parser.add_argument("--default-cellprob-threshold", type=float, default=0.0,
                        help="Default cell probability threshold")
    parser.add_argument("--dapi-flow-threshold", type=float, default=0.4,
                        help="Flow threshold for DAPI cell detection")
    parser.add_argument("--dapi-cellprob-threshold", type=float, default=0.0,
                        help="Cell probability threshold for DAPI")
    parser.add_argument("--red-flow-threshold", type=float, default=0.4,
                        help="Flow threshold for red cell detection")
    parser.add_argument("--red-cellprob-threshold", type=float, default=0.0,
                        help="Cell probability threshold for red channel")
    parser.add_argument("--green-flow-threshold", type=float, default=0.4,
                        help="Flow threshold for green cell detection")
    parser.add_argument("--green-cellprob-threshold", type=float, default=0.0,
                        help="Cell probability threshold for green channel")
    # Preprocessing options.
    parser.add_argument("--enhance-contrast", action="store_true", default=True,
                        help="Enhance contrast using CLAHE (default: True)")
    parser.add_argument("--denoise", action="store_true", default=True,
                        help="Apply Gaussian denoising (default: True)")
    parser.add_argument("--denoise-sigma", type=float, default=1.0,
                        help="Sigma for Gaussian denoising (default: 1.0)")
    parser.add_argument("--rolling-ball", action="store_true",
                        help="Apply rolling ball background subtraction")
    parser.add_argument("--rolling-ball-radius", type=float, default=50.0,
                        help="Radius for rolling ball subtraction in pixels (default: 50.0)")
    parser.add_argument("--median-filter", action="store_true", default=False,
                        help="Apply median filtering (default: False)")
    parser.add_argument("--median-kernel-size", type=int, default=3,
                        help="Kernel size for median filtering (must be odd, default: 3)")
    parser.add_argument("--bilateral-filter", action="store_true", default=False,
                        help="Apply bilateral filtering (default: False)")
    parser.add_argument("--bilateral-d", type=int, default=9,
                        help="Diameter for bilateral filter (default: 9)")
    parser.add_argument("--bilateral-sigmaColor", type=float, default=75.0,
                        help="SigmaColor for bilateral filter (default: 75.0)")
    parser.add_argument("--bilateral-sigmaSpace", type=float, default=75.0,
                        help="SigmaSpace for bilateral filter (default: 75.0)")
    parser.add_argument("--hist-eq", action="store_true", default=False,
                        help="Apply histogram equalization (default: False)")
    parser.add_argument("--gamma-correction", action="store_true", default=False,
                        help="Apply gamma correction (default: False)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma value (default: 1.0)")
    parser.add_argument("--morph-close", action="store_true", default=False,
                        help="Apply morphological closing (default: False)")
    parser.add_argument("--morph-close-kernel-size", type=int, default=5,
                        help="Kernel size for morphological closing (default: 5)")
    # Additional flags.
    parser.add_argument("--configure_dapi", action="store_true",
                        help="Configure DAPI segmentation only; do not proceed with cytoplasmic segmentation")
    parser.add_argument("--save_masks", action="store_true", default=True,
                        help="Also save segmentation masks (default: True)")
    parser.add_argument("--coexpression_testing", action="store_true",
                        help="Run coexpression testing by loading saved output files and re-running the overlay script")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    valid_extensions = ('.tif', '.tiff')
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(valid_extensions):
            input_filepath = os.path.join(args.input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            args.out_subfolder = os.path.join(args.output_dir, base_name)
            if not os.path.exists(args.out_subfolder):
                os.makedirs(args.out_subfolder)
            process_file(input_filepath, args)

if __name__ == "__main__":
    main()
