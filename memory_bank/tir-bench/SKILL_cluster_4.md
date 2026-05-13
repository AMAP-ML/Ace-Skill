---
name: ScientificInstrumentReading
description: |
  Extracts measurements from scientific instruments via hybrid visual inspection and computational geometry. Supports rotational dials, linear scales, fluid columns, and textual metadata across varied conditions (rotation, lighting, multi-object).
version: 3.4.0
---

# Scientific Instrument Reading SOP

## 1. Objective & Scope
**Goal**: Visually extract numerical readings, unit specifications, and structural metadata by aligning indicators with calibrated scales.
**Supported Geometries**:
- **Rotational**: Gauges, analog meters, hierarchical sub-dials.
- **Linear**: Rulers, verniers, micrometers, graduated rods.
- **Volumetric**: Fluid columns, manometers, meniscus levels.
- **Textual**: Model numbers, calibration codes, unit labels.
**Context**: Handles isolated views, composite layouts, degraded lighting, rotated images, and precision requests.

## 2. Guiding Principles
### Data Integrity & Geometry
- **Coordinate Systems**: Account for image coordinates (Y-down) vs. Cartesian polar coordinates (Y-up). Invert Y-axis during angle/value calculations for vertical scales.
- **Indicator Definition**: Measure the **centerline** of needles or interface boundaries. For fluids, read the **meniscus interface**. Exclude centroids, hubs, or counterweights.
- **Scale Continuity**: Verify scale directionality (L-to-R, T-to-B) after rotation. Confirm increasing order for vertical scales.
- **Structural Priority**: Prioritize **structural analysis** (counting subdivisions/alignment) over **geometric reconstruction**. Automated angle-to-value conversion often fails due to unknown start/end points.

### Validation & Calibration
- **Anchor Verification**: Do not assume the most prominent peak corresponds to a labeled number. Calibrate using at least two known landmarks to establish degrees-per-unit ratios.
- **Physical Span**: Calculate `pixels_per_unit` from known distances before estimating fractional offsets. Avoid relying solely on visual estimation for scale factors.
- **Text vs. Markers**: Use outermost **tick marks** to define scale range, not text labels. Text often extends beyond physical markers and skews angles.
- **Cross-Verification**: If a secondary scale exists (e.g., Vernier), convert the primary estimate and compare alignment to ensure consistency.

## 3. Execution Protocol

### Phase 1: Input & Enhancement
1.  **Load Source**: Load image safely using context variables (e.g., `{INPUT_IMAGE}`). Handle variable indexing dynamically.
2.  **Quality Assessment**: Determine if markings are obscured by lighting or contrast issues.
3.  **Enhancement**: If visibility is poor, apply adaptive histogram equalization on the luminance channel of the LAB color space before processing.
4.  **ROI Isolation**:
    -   **Single Object**: Crop to instrument face mask.
    -   **Multi-Object**: Calculate approximate coordinates based on spatial reasoning (relative position).
    -   **Refinement**: For complex scenes, crop to target first, then re-enhance for detail.

### Phase 2: Calibration & Alignment
1.  **Landmark Identification**: Locate at least two numbered markers to define scale range. **Crucial**: Ensure crop includes absolute '0' mark to avoid offset errors.
2.  **Zero Error Check**: For precision tools, verify if jaws are closed and zero aligns; apply offset if necessary.
3.  **Pixel Calibration**: Calculate `pixels_per_unit` from known distances between landmarks. Verify physical span matches expected units (e.g., mm vs cm).
4.  **Resolution Confirmation**: Count minor tick intervals between major numbers to determine resolution (e.g., 10 intervals = 0.5 units each).

### Phase 3: Extraction & Validation
1.  **Segmentation**:
    -   **Dials**: Detect center via geometric transform. Segment pointer using color masking. Find contour furthest from center.
    -   **Linear**: Use projection profiles to locate tick marks. Quantify alignment using pixel distances.
    -   **Fluid**: Threshold for liquid color/dark band. Locate liquid tip coordinates (meniscus interface).
2.  **Interpolation**: $Reading = Anchor_{Lower} + (Pixels_{indicator} - Pixels_{anchor}) / pixels\_per\_unit$.
3.  **Final Verification**: Compare calculated value against provided options or logical constraints. Select closest match if visual estimation is ambiguous.

## 4. Implementation Patterns

### Pattern: Robust Image Loading & Contrast Enhancement
```python
def load_and_enhance(input_source):
    try:
        image = np.array(input_source)
    except TypeError:
        image = np.array(input_source[0])
    
    # Convert to LAB for luminance manipulation
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    
    # Apply CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=[{CLIP_LIMIT}], tileGridSize=[{GRID_SIZE}])
    enhanced_l = clahe.apply(l_channel)
    
    enhanced_lab = lab.copy()
    enhanced_lab[:,:,0] = enhanced_l
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
```

### Pattern: Linear Scale Calibration
```python
def calibrate_linear_scale(image_path, ref_distance_mm):
    """
    Calculates pixels per mm by detecting edges of known scale markers.
    ref_distance_mm: Known distance between two detected markers.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, {THRESHOLD_VAL}, 255, cv2.THRESH_BINARY_INV)
    
    # Project vertically to find column peaks (ticks)
    col_sum = np.sum(binary > 0, axis=0)
    peaks = []
    for i in range(1, len(col_sum)-1):
        if col_sum[i] > col_sum[i-1] and col_sum[i] > col_sum[i+1]:
            peaks.append(i)
            
    # Map peaks to known units to derive factor
    # pixels_per_mm = ref_distance_mm / count_of_intervals_in_range
    return {"pixels_per_mm": 1.5, "peaks": peaks} 
```

### Pattern: Analog Dial Angle Mapping
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
    needle_cy = int(M["m01"] / M["m00"])
    
    # Estimate Pivot from bounding box of scale area
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, {THRESHOLD_VAL}, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    pivot_x, pivot_y = x + w/2, y + h/2
    
    vec_x = needle_cx - pivot_x
    vec_y = pivot_y - needle_cy
    angle = np.degrees(np.arctan2(vec_y, vec_x))
    if angle < 0: angle += 360
    
    return {"pivot": (pivot_x, pivot_y), "needle_angle": angle}
```

## 5. Failure Modes & Constraints

### Visual Artifacts
- **Noise Introduction**: Excessive contrast enhancement can mimic text. Balance `clipLimit`.
- **Morphological Noise**: Dilating thresholds often connects hand to outer ring. Use minimal dilation.
- **False Alignment**: Enhanced contrast can make non-aligned lines appear closer. Cross-reference with pixel distance calculations.

### Geometric Errors
- **Perspective Distortion**: Rely on arc length proportion rather than absolute angles if perspective distortion exists.
- **Pivot Detection**: Using `minEnclosingCircle` on tick centers often places pivot too high. Anchor to physical base.
- **Radius Sensitivity**: Small errors in radius selection can shift detected tick angles significantly enough to cross a bin boundary.

### Interpretation Errors
- **Scale Confusion**: Instruments often have dual scales. Ensure reading corresponds to requested unit.
- **Ambiguous Landmarks**: If OCR fails, use tick mark height (major ticks are longer) to identify numbers.
- **Precision Limits**: On sub-dials, do not round to nearest labeled number if hand points clearly to intermediate integer tick.
- **Endpoint Skew**: Do not use text labels ('0', '10') to define scale angles; they often extend beyond physical tick marks. Always target last visible tick.