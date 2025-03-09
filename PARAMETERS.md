# Image Segmentation Pipeline

This pipeline processes multi-page TIFF files for cell segmentation and analysis. It uses Cellpose for segmentation and includes a suite of optional preprocessing steps to enhance image quality. The following sections detail each available option, how it works, and guidelines for optimal parameter tuning.

---

## 1. Basic I/O and Conversion

### Input & Output Directories
- **`input_dir`**  
  Directory containing the multi-page TIFF files.  
  **Details:**  
  - All files must have exactly 3 pages (channels).  
  **Tip:**  
  - Organize your files in a dedicated folder for consistency.

- **`--output_dir`**  
  Root directory for outputs. Each input file gets its own subfolder.  
  **Tip:**  
  - Use clear naming conventions to track different experiments.

### Pixel-to-Micron Conversion
- **`--pixels_per_micron`**  
  Conversion factor (default: `1.6091`).  
  **How it works:**  
  - Multiplies cell diameters (given in microns) by this factor to convert them into pixels.  
  **When to Adjust:**  
  - Use a calibrated value from your microscope; an incorrect factor affects diameter estimation and area calculations.

- **`--estimated_diameter`**  
  Estimated cell diameter in microns for cytoplasmic segmentation.  
  **When to Use:**  
  - If not provided, Cellpose auto-estimates cell size. A good estimate can improve segmentation accuracy.

---

## 2. DAPI (Nuclear) Segmentation Options

### Fixed Thresholding
- **`--dapi_threshold`**  
  Fixed intensity value (default: `200`) for creating a binary mask from the DAPI channel.  
  **How it works:**  
  - Uses OpenCV’s threshold function to binarize the image.  
  **When to Use:**  
  - Best for images with uniform illumination and minimal noise.  
  **Tips:**  
  - Increase if background noise is high; decrease if nuclei are faint.

### Adaptive Thresholding
- **`--adaptive_threshold`** (flag) with:
  - **`--adaptive_block_size`** (default: `11`, must be odd)  
  - **`--adaptive_C`** (default: `2`)  
  **How it works:**  
  - Computes a local threshold for each pixel based on the mean intensity of its neighborhood (block size) minus constant C.  
  **When to Use:**  
  - Ideal for images with non-uniform illumination.  
  **Tips:**  
  - Smaller block size captures local detail but is noise-sensitive; larger values smooth out local differences.  
  - Adjust C to fine-tune the threshold level.

### DAPI Model Option
- **`--dapi_use_nuclei_model`**  
  Uses the Cellpose nuclei model for DAPI segmentation instead of thresholding.  
  **How it works:**  
  - Optimized for nuclear boundaries, potentially improving segmentation in challenging conditions.  
  **Tradeoffs:**  
  - Slower than thresholding; may require tuning of additional parameters.

### DAPI Count Threshold
- **`--dapi_count_threshold`**  
  Minimum number of DAPI pixels for a contour to be valid (default: `5`).  
  **How it works:**  
  - Filters out small artifacts or noise.  
  **Tips:**  
  - Increase to remove false positives; decrease if small but real nuclei are being missed.

---

## 3. Cytoplasmic Segmentation Options (Red/Green Channels)

### CellPose Model Options
- **`--cellpose-model`**  
  Model selection (default: `cyto3`).  
  **Details:**  
  - "cyto3" is tuned for cytoplasmic segmentation; "nuclei" is for nuclear segmentation.  
  **Tip:**  
  - Choose based on the imaging type.

- **`--use-gpu`**  
  Enables GPU acceleration (default: enabled).  
  **Tip:**  
  - Use GPU for faster processing if available.

- **`--cellpose-augment`**  
  Enables augmentation during inference (default: enabled).  
  **Tradeoffs:**  
  - Improves robustness at the cost of increased processing time.

- **`--cellpose-batch-size`**  
  Batch size for Cellpose (default: `8`).  
  **Tip:**  
  - Adjust based on available GPU memory.

### Flow and Cell Probability Thresholds
- **`--cyto_flow_threshold`** and **`--cyto_cellprob_threshold`**  
  Control the sensitivity of cytoplasmic segmentation.  
  **Tradeoffs:**  
  - Lower thresholds may over-segment; higher thresholds yield more conservative cell boundaries.

### Cell Diameter (Microns)
- **`--red-cell-diameter`** and **`--green-cell-diameter`**  
  Expected diameters for the red and green channels, specified in microns.  
  **How it works:**  
  - Converted to pixels using `pixels_per_micron`.  
  **When to Use:**  
  - Provide if you have calibrated measurements; otherwise, leave as `None` for auto-estimation.

---

## 4. Preprocessing Options

Preprocessing improves image quality prior to segmentation. All steps are optional and can be enabled as needed.

### a. Contrast Enhancement and Denoising
- **`--enhance-contrast`**  
  Applies CLAHE for local contrast enhancement (default: enabled).  
  **How it works:**  
  - Equalizes histogram in small tiles while limiting noise amplification.  
  **Tip:**  
  - Useful for images with locally varying contrast.

- **`--denoise`** and **`--denoise-sigma`**  
  Applies Gaussian blur for noise reduction (default sigma: `1.0`).  
  **Tradeoffs:**  
  - Higher sigma reduces noise but may blur edges.

### b. Background Subtraction
- **`--rolling-ball`** and **`--rolling-ball-radius`**  
  Implements rolling ball background subtraction using morphological opening (default radius: `50`).  
  **How it works:**  
  - Removes uneven background illumination by subtracting a smoothed version of the image.  
  **When to Use:**  
  - Ideal for images with broad, uneven backgrounds.

### c. Additional Filtering
- **Median Filtering:**  
  - **`--median-filter`** and **`--median-kernel-size`** (default: `3`)  
    **How it works:**  
    - Replaces each pixel with the median of its neighbors to remove impulse noise.  
    **Tip:**  
    - Use when salt-and-pepper noise is present.
  
- **Bilateral Filtering:**  
  - **`--bilateral-filter`**, **`--bilateral-d`** (default: `9`), **`--bilateral-sigmaColor`** (default: `75.0`), **`--bilateral-sigmaSpace`** (default: `75.0`)  
    **How it works:**  
    - Smooths noise while preserving edges by combining spatial and intensity information.  
    **Tip:**  
    - Good for maintaining sharp boundaries in cell structures.

### d. Intensity Adjustments
- **Histogram Equalization:**  
  - **`--hist-eq`**  
    **How it works:**  
    - Globally equalizes the image histogram to boost contrast.  
    **Tradeoffs:**  
    - May oversaturate regions in images with a high dynamic range.
  
- **Gamma Correction:**  
  - **`--gamma-correction`** and **`--gamma`** (default: `1.0`)  
    **How it works:**  
    - Applies a nonlinear transformation to adjust brightness.  
    **Tips:**  
    - Gamma < 1 brightens the image; gamma > 1 darkens it. Experiment with values between 0.8 and 1.2.

### e. Morphological Operations
- **Morphological Closing:**  
  - **`--morph-close`** and **`--morph-close-kernel-size`** (default: `5`)  
    **How it works:**  
    - Performs dilation followed by erosion to fill small holes within cell regions.  
    **Tip:**  
    - Use to consolidate fragmented cell masks; avoid excessively large kernels to prevent merging distinct cells.

---

## 5. Tradeoffs and Optimization Strategies

- **Preprocessing Intensity:**  
  Combining multiple filters can lead to a very clean image but might also remove subtle features. Start with one or two steps and inspect intermediate results.

- **Thresholding Techniques:**  
  Fixed thresholding is simpler but may fail under variable lighting. Adaptive thresholding adjusts locally but may introduce noise. Choose based on your image characteristics.

- **Filter Selection:**  
  - **Median vs. Bilateral:**  
    Use median filtering for impulsive noise and bilateral filtering for edge-preserving smoothing.
  - **Rolling Ball:**  
    Adjust the radius relative to the expected cell size and background variations.

- **Cell Diameter Settings:**  
  Providing accurate cell diameters improves segmentation but rely on calibration. If unsure, let Cellpose auto-estimate.

- **Computational Considerations:**  
  More preprocessing steps can slow down processing. Use GPU acceleration and optimize batch sizes if processing large datasets.
