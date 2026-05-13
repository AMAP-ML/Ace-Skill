---
name: Accurate_Visual_Proportion_Estimation
description: |
  Framework for calculating area ratios under ambiguity. Integrates geometric constraints, feature-based segmentation, spatial clustering, and structural footprint analysis. Includes environment verification protocols and bounding box upper-bound logic.
version: 33.0.0
---

# Accurate Visual Proportion Estimation

## Scope & Applicability
Use this framework for numerical proportion queries (e.g., "What percent of the image is `[TARGET]`?") where automated tools struggle with irregular shapes, occlusions, or subjective definitions.

**Apply when:**
- Queries require area ratios, percentages, or relative sizes.
- Subjects possess organic shapes, partial visibility, complex textures, or sparse structures.
- **Ambiguity**: Multiple similar objects exist; question implies distinct instance or summation.
- **Conflict**: Automated segmentation results conflict with visual intuition.
- **Occlusion**: Target object is partially covered (requires footprint estimation).

## Core Principles & Constraints
Adhere to these constraints to prevent calculation errors. These rules apply universally across estimation strategies.

| Constraint | Logic | Diagnostic Fix |
| :--- | :--- | :--- |
| **Geometric Upper Bound** | Target Area ≤ Bounding Box Area. Discard any option violating this immediately. | If Option > BB Ratio, eliminate. Select highest valid option below bound. |
| **Quantitative Trust** | Do not arbitrarily reduce tool outputs if they align closely with options. | If Tool_Output ≈ Option (within 2%), trust metric unless mask is visibly broken. |
| **Sanity Check** | Compare Segmented Ratio vs. Bounding Box Ratio. | If `Seg_Ratio` << `BB_Ratio` but object appears solid, assume under-segmentation. Adjust upwards. |
| **Tolerance Margin** | Manual bounding boxes have error margins. Do not treat calculated BB % as a hard ceiling if close to an option. | If `BB_Ratio ≈ Option` within tolerance, retain option. Measurement error likely exists. |
| **Footprint vs. Pixels** | Distinguish between "visible pixels" and "total object extent". Questions often imply physical footprint. | If segmentation < Options significantly, switch to geometric modeling (bounding box + fill factor). |
| **Context Separation** | Exclude attached accessories (chairs, laptops) unless specified. | If segmentation merges subject + background clutter, apply strict ROI masking. |

### Heuristic Adjustments
Apply multipliers based on object characteristics to calibrate raw segmentation data:
- **Sparse/Mesh**: Multiply mask area by density inverse.
- **Organic/Fur**: Apply Fill Factor 0.4–0.6.
- **Solid Geometric**: Apply Fill Factor 0.7–0.9.
- **Dominant Foreground**: Increase Fill Factor to 0.8+.
- **Complex Negative Space**: Reduce ratio based on void volume (limbs, wings).
- **Occluded/Hidden**: Apply Fill Factor 0.4–0.5 (consider full footprint).

## Operational Workflow

### 1. Preparation & Bounds
-   **Environment Validation:** Confirm image variable availability. Convert PIL to NumPy before processing. Handle type errors early.
-   **Target Definition:** Explicitly define scope. Distinguish between similar instances (standalone vs. integrated). **Crucial:** Exclude attached accessories (furniture, devices) unless specified.
-   **Baseline Calculation:** Estimate the tightest rectangle enclosing the target. Calculate `BB Ratio = BB Area / Total Image Area`. Any valid answer must be $\le$ `BB Ratio` (with tolerance).

### 2. Strategy Selection (Iterative)
Select approach based on query complexity and initial results:
-   **Initial Pass:** Start with simpler methods (e.g., Color Thresholding) to gauge complexity.
-   **Mask Validation:** **Crucial Step.** Inspect the generated mask image. Do not trust raw numbers if the mask visually misaligns with the target object.
-   **Refined Pass:** If initial pass shows noise (over-segmentation) or gaps (under-segmentation), switch to robust algorithms (e.g., GrabCut) initialized with the ROI from Step 1.
-   **Support Exclusion:** Manually exclude hands, perches, or frames if included in the mask during refinement.

### 3. Verification & Selection
-   **Cross-Method Comparison:** Compare raw outputs from different segmentation methods.
-   **Decision Rule:** Prioritize the method with the cleanest mask boundaries. If tool output diverges significantly from options, assume over-segmentation until proven otherwise.
-   **Visual Buffer:** If segmentation misses dark edges (common with thresholding), apply a slight upward adjustment to match visual intuition.
-   **High Estimate Validation:** Do not immediately discard high percentage estimates (e.g., >40%) as noise; verify if they capture the full object footprint including shadows/garnish.
-   **Avoid Over-Correction:** If a tool calculates a value close to an option (e.g., 58.6% vs 57%), trust the quantitative result unless the mask clearly includes large empty spaces.

## Implementation Patterns

### Universal Loader & Normalization
```python
def load_and_estimate(image_variable, bbox_norm_coords=None):
    """Load image safely and calculate proportion."""
    try:
        img_array = np.array(image_variable) 
    except TypeError:
        # Fallback for list/tuple inputs
        img_array = np.array(image_variable[0]) 
        
    h, w, _ = img_array.shape
    total_pixels = h * w
    
    if bbox_norm_coords:
        y_min, x_min, y_max, x_max = bbox_norm_coords
        x1, y1 = int(w * x_min), int(h * y_min)
        x2, y2 = int(w * x_max), int(h * y_max)
        
        roi_area = (x2 - x1) * (y2 - y1)
        base_ratio = roi_area / total_pixels
        
        return base_ratio, img_array, (x1, y1, x2, y2)
    
    return 0.0, img_array, None
```

### Robust Segmentation Template
```python
def segment_target_and_verify(image, method='hsv', roi_coords=None, target_params=None):
    """Isolate target using color or brightness thresholds with morphological cleanup."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    combined_mask = np.zeros_like(gray_image)
    
    if method == 'hsv':
        # Broad range to catch shadows/variations
        lower = np.array(target_params['lower']) if target_params else [0, 0, 0] 
        upper = np.array(target_params['upper']) if target_params else [180, 255, 255]
        mask = cv2.inRange(hsv_image, lower, upper)
    elif method == 'grabcut':
        bgd_model, fgd_model = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)
        mask = np.zeros(image.shape[:2],np.uint8)
        x1,y1,x2,x2 = roi_coords
        rect = (x1, y1, x2-x1, y2-y1)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    else:
        _, mask = cv2.threshold(gray_image, [THRESHOLD_VALUE], 255, cv2.THRESH_BINARY)
        
    # Morphological Closing to connect fragmented parts (e.g., crust/shadows)
    kernel = np.ones((5,5),np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = sum([cv2.contourArea(cnt) for cnt in contours])
    return max_area, cleaned_mask
```

### Ratio Calculator (Fill Factor Heuristic)
```python
def estimate_area_ratio(bb_width_pct, bb_height_pct, fill_factor=[FILL_FACTOR]):
    """Calculate estimated percentage using bounding box and fill factor."""
    bbox_ratio = bb_width_pct * bb_height_pct
    estimated_ratio = bbox_ratio * fill_factor
    return estimated_ratio * 100
```

## Risk Mitigation
-   **Color Variance:** Avoid relying solely on color thresholding (HSV ranges) as lighting changes can cause significant false positives/negatives.
-   **Noise Inclusion:** Always visualize the mask. High proportions (>60%) often indicate background noise was included (e.g., similar colored objects in the distance).
-   **Segmentation Gaps:** Algorithms may miss dark spots or edges within the target object. If the calculated value is close to an option boundary (e.g., 49.8% vs 51%), consider if the mask missed small areas that would push the value up.
-   **Context Merging:** Be wary of segmentation merging the subject with adjacent objects (e.g., person + chair). Use tight ROIs to isolate the semantic target.
-   **Over-Correction:** Do not apply arbitrary fill factors to segmentations that claim to measure area directly. If the tool returns a high ratio matching an option, trust it over manual downscaling assumptions.