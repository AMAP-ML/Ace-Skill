---
name: ScientificInstrumentReading
description: |
  Extracts numerical values, labels, and measurements from scientific instruments via hybrid visual inspection and computational geometry. Supports rotational dials, precision linear scales (verniers/micrometers), fluid columns, and textual metadata across multi-object scenes, rotated views, and varying lighting conditions.
version: 3.3.0
---

# Scientific Instrument Reading SOP

## Overview & Scope
**Objective**: Visually extract readings, measurements, and labels from instruments requiring alignment between indicators and calibrated scales.
**Supported Geometries**:
- **Rotational**: Gauges, dials, meters, hierarchical sub-dials.
- **Linear**: Rulers, graduated rods, sliding scales (verniers/micrometers).
- **Volumetric**: Graduated cylinders, manometers, fluid columns.
- **Textual**: Model numbers, unit specifications, calibration codes.
**Context**: Handles isolated views, composite layouts, degraded lighting, mixed equipment scenes, rotated images, and precision requests.

## Operational Standards

### 1. Data Integrity & Input Handling
- **Source Standardization**: Convert all input sources to a unified array format immediately. Handle variable indexing dynamically (list vs. direct object).
- **Variable Safety**: Always use pre-loaded context variables (e.g., `[INPUT_IMAGE]`) instead of hardcoded file paths.
- **Coordinate Systems**: Account for image coordinates (Y-down) vs. Cartesian polar coordinates (Y-up). Invert Y-axis during angle/value calculations for vertical scales.
- **Orientation Verification**: Detect inverted text or illogical scale progression before processing. Apply rotation corrections if necessary.

### 2. Visual Interpretation & Enhancement
- **Indicator Definition**: Measure the **centerline** of the needle or interface boundary. For fluids, read the **meniscus interface**. Exclude centroids, hubs, or counterweights.
- **Scale Continuity**: Verify scale directionality (L-to-R, T-to-B) after rotation. Confirm increasing order for vertical scales.
- **Low-Light Text**: When text is illegible due to darkness or low contrast, apply adaptive histogram equalization on the luminance channel of the LAB color space before attempting OCR or visual reading.
- **Endpoint Calibration**: Use the outermost **tick marks** to define scale range, not text labels, as text often extends beyond physical markers and skews angles.

### 3. Validation Hierarchy
- **Truth Priority**: Prioritize **structural analysis** (counting subdivisions/alignment) over **geometric reconstruction**. Automated angle-to-value conversion often fails due to unknown start/end points.
- **Anchor Verification**: Do not assume the most prominent peak corresponds to a labeled number. Calibrate using at least two known landmarks to establish degrees-per-unit ratios.
- **Cross-Verification**: If a secondary scale exists (e.g., Vernier), convert the primary estimate and compare alignment to ensure consistency.
- **Quantitative Calibration**: For linear precision instruments, calculate `pixels_per_unit` from known distances before estimating fractional offsets. Avoid relying solely on visual estimation for scale factors.

## Execution Workflow

### Phase 1: Input & Preprocessing
1.  **Load Source**: Load image source safely using the correct context variable (e.g., `[INPUT_IMAGE]`).
2.  **Quality Assessment**: Determine if text or markings are obscured by lighting/contrast issues.
3.  **Target Isolation**:
    -   **Single Object**: Crop to ROI mask based on instrument face.
    -   **Multi-Object**: Calculate approximate coordinates based on spatial reasoning (e.g., relative position).
    -   **Enhancement**: If visibility is poor, apply adaptive histogram equalization on luminance channel (LAB color space).
4.  **ROI Refinement**: For complex scenes, crop to the target object first, then re-enhance the smaller region for higher detail.

### Phase 2: Geometry & Calibration
1.  **Landmark Identification**: Locate at least two numbered markers to define the scale range. **Crucial**: Ensure the crop includes the absolute '0' mark of the main scale to avoid offset errors.
2.  **Zero Error Check**: For precision tools, verify if jaws are closed and zero aligns; apply offset if necessary.
3.  **Pixel Calibration**: Calculate `pixels_per_unit` from known distances (e.g., gap between `[LANDMARK_A]` and `[LANDMARK_B]`). **Do not assume standard resolutions**; verify physical span matches expected units (e.g., mm vs cm).
4.  **Tick Interval Confirmation**: Count minor tick intervals between major numbers to determine resolution (e.g., 10 intervals = 0.5 units each).

### Phase 3: Extraction & Verification
1.  **Segmentation**:
    -   **Dials**: Detect center via geometric transform. Segment pointer using color masking. Find contour furthest from center.
    -   **Linear**: Use projection profiles to locate tick marks. Quantify alignment using pixel distances. Calculate fractional reading via `offset_pixels / pixels_per_unit`.
    -   **Fluid**: Threshold for liquid color/dark band. Locate liquid tip coordinates (meniscus interface).
2.  **Interpolation**: $Reading = Anchor_Lower + (Pixels_{indicator} - Pixels_{anchor}) / pixels\_per\_unit$.
3.  **Option Matching**: Compare calculated value against provided multiple-choice options. Select the closest match if visual estimation is ambiguous.

## Implementation Patterns

### Pattern: Robust Image Loading & Enhancement
```python
import cv2
import numpy as np

def load_and_enhance(input_source):
    try:
        image = np.array(input_source)
    except TypeError:
        image = np.array(input_source[0])
    
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=[CLAHE_CLIP], tileGridSize=[GRID_SIZE])
    enhanced_l = clahe.apply(l_channel)
    
    enhanced_lab = lab.copy()
    enhanced_lab[:,:,0] = enhanced_l
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
```

### Pattern: Scale Calibration & Alignment (Linear/Precision)
```python
def calibrate_linear_scale(image_path, ref_distance_mm):
    """
    Calculates pixels per mm by detecting edges of known scale markers.
    ref_distance_mm: Known distance between two detected markers (e.g., 10mm).
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Project vertically to find column peaks (ticks)
    col_sum = np.sum(binary > 0, axis=0)
    peaks = []
    for i in range(1, len(col_sum)-1):
        if col_sum[i] > col_sum[i-1] and col_sum[i] > col_sum[i+1]:
            peaks.append(i)
            
    # Filter peaks within ROI and map to known units
    # pixels_per_mm = ref_distance_mm / count_of_intervals_in_range
    # Return calibrated factor for subsequent offset calculation
    return {"pixels_per_mm": 1.5, "peaks": peaks} 
```

### Pattern: Dial Calibration & Angle Mapping
```python
def read_analog_scale(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_color = np.array([COLOR_LOWER_H, COLOR_LOWER_S, COLOR_LOWER_V])
    upper_color = np.array([COLOR_UPPER_H, COLOR_UPPER_S, COLOR_UPPER_V])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    needle_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(needle_contour)
    needle_cx = int(M["m10"] / M["m00"])
    needle_cy = int(M["m10"] / M["m00"])
    
    # Estimate Pivot
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, [THRESHOLD_VAL], 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    pivot_x, pivot_y = x + w/2, y + h/2
    
    vec_x = needle_cx - pivot_x
    vec_y = pivot_y - needle_cy
    angle = np.degrees(np.arctan2(vec_y, vec_x))
    if angle < 0: angle += 360
    
    return {"pivot": (pivot_x, pivot_y), "needle_angle": angle}
```

## Common Pitfalls

### Visual Artifacts
- **Preprocessing Noise**: Excessive contrast enhancement can introduce noise mimicking text. Balance `clipLimit`.
- **Morphological Noise**: Dilating thresholds often connects the hand to the outer ring. Use minimal dilation.
- **False Alignment**: Enhanced contrast can make non-aligned lines appear closer. Cross-reference with pixel distance calculations.

### Geometric Errors
- **Perspective Distortion**: Rely on arc length proportion rather than absolute angles if perspective distortion exists.
- **Pivot Detection**: Using `minEnclosingCircle` on tick centers often places the pivot too high. Anchor to the physical base.
- **Anchor Misidentification**: Do not assume the nearest peak to the needle is the labeled major tick. Verify spacing consistency across multiple ticks.
- **Radius Sensitivity**: Small errors in radius selection can shift detected tick angles significantly enough to cross a bin boundary.

### Interpretation Errors
- **Scale Confusion**: Instruments often have dual scales (e.g., inch/metric). Ensure reading corresponds to the requested unit.
- **Ambiguous Landmarks**: If OCR fails, use tick mark height (major ticks are longer) to identify numbers.
- **Precision Limits**: On sub-dials, do not round to the nearest labeled number if the hand points clearly to an intermediate integer tick.
- **Endpoint Skew**: Do not use text labels ('0', '10') to define scale angles; they often extend beyond the physical tick marks, causing systematic underestimation. Always target the last visible tick.
- **Unit Assumption**: Do not assume standard least counts (0.02mm, 0.05mm) or label meanings ('10' = 10mm) without verifying the physical span via pixel calibration. Visual hallucination of scale position is a common failure mode.