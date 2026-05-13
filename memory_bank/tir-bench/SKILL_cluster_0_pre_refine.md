---
name: VisualAnalysisViaEnhancement
description: |
  Unified procedure for extracting precise counts or readable text from obscured, dense, distorted, or rotated visual data. Leverages programmatic intervention for ROI isolation, pixel-level enhancement, dynamic feature detection, and iterative orientation correction.
version: 5.1.0
---

# Visual Analysis via Enhancement & Systematic Scan

## Overview & Activation
**Objective**: Extract precise counts or readable text from complex visual inputs where standard perception fails. Never guess on degraded data; enhancement is a prerequisite.

**Activation Triggers**:
- **Quality Degradation**: Poor lighting, low contrast, blur, shadows, heavy noise, or extreme darkness.
- **Spatial Complexity**: Objects/text clustered densely, overlapping significantly, or occluded.
- **Geometric Ambiguity**: Targets requiring rotation (OCR) or scale adjustment (vertical/sideways text).
- **Semantic Confusion**: Targets blending with background textures, reflections, or uniform surfaces.
- **Physical Verification**: Distinguishing active participants from passive bystanders based on contact points.

**Constraint**: Do not assume external file paths exist; strictly use pre-loaded image variables.

## Operational Workflow

### Phase 1: Assessment & Planning
1.  **Define Target**: Identify the specific query target (`[TARGET]`) and distinguish from background artifacts.
2.  **Variable Check**: Verify the exact input variable name (e.g., `original_image`, `input_image`) before coding to prevent runtime errors.
3.  **Select Intervention**: Determine primary obstruction type:
    - *Crowding*: Requires spatial isolation (Cropping/ROI).
    - *Contrast/Light*: Requires pixel-level enhancement (**Priority: Apply before counting**).
    - *Alignment*: Requires geometric transformation (Rotation). **Use multi-angle grids.**
    - *Feature Detection*: For silhouettes/complex scenes, use edge/contour detection to locate candidate regions dynamically.
4.  **Set Constraints**: Define physical contact criteria to exclude proximity-based false positives. Note expected scale relative to image dimensions.

### Phase 2: Programmatic Intervention
1.  **Transform & Load**: Use `code_interpreter` to apply transformations:
    - **Rotate**: Generate a multi-angle grid (0°, 90°, 180°, 270°) to find upright orientation. Account for library-specific rotation directions (e.g., PIL rotates CCW by default).
    - **Crop**: Isolate dense regions using percentage-based coordinates (`[ROI_COORDS]`). Prefer dynamic feature detection over hardcoded ratios.
    - **Enhance**: Convert to LAB color space followed by CLAHE on the Luminance channel for superior contrast retention.
2.  **Visualize**: Explicitly render processed images. Do not assume tool execution displays results automatically.
3.  **Iterate**: Inspect enhanced outputs. Confirm distinct functional separators (joints, gaps) rather than relying on shadows or partial outlines.

### Phase 3: Synthesis & Validation
1.  **Enumerate**: Count items within ROIs sequentially (e.g., Left-to-Right). Explicitly describe positions to prevent double-counting.
2.  **Verify Structure**:
    - **Integrity**: Verify complete structures (closed loops, full bodies) over partial indicators (shadows, legs).
    - **Continuity**: For connected objects, check for continuous supports or shared components.
    - **Ambiguity Resolution**: If asked for "units," count individual seats even on multi-seat furniture unless clearly a sofa.
3.  **Contextual Check**: Validate results against broader scene context. Trust clear structural evidence over iterative refinement that changes the count arbitrarily.
4.  **Orientation Check**: For text, verify reading order (Left-to-Right vs Top-to-Bottom) in the final rotated view.

## Technical Toolkit

### Query Patterns
- **Enumeration**: `"How many [TARGET] are in [LOCATION]?"`
- **Extraction**: `"What is the [TEXT_TYPE] visible in the image?"`
- **Interaction**: `"Count only entities physically holding [OBJECT]."`
- **Dynamic ROI**: `"Enhance the top [percentage]% of the image to reveal objects on the skyline, then list coordinates of all vertical protrusions greater than [min_height] pixels."`

### Code Interpreter Template
A unified toolkit handling rotation, enhancement, and feature detection.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist

def enhance_visual_data(image_source=None, 
                        rotation_angles=[0, 90, 180, 270], 
                        roi_coords=None, 
                        clip_limit=0.03):
    """
    Unified processing pipeline: Rotation, Enhancement, Cropping, Feature Detection.
    
    Args:
        image_source: PIL Image object (Prefer pre-loaded variable like 'original_image')
        rotation_angles: List of degrees to rotate
        roi_coords: Tuple of (y_start%, y_end%, x_start%, x_end%)
        clip_limit: CLAHE clip limit (Default 0.03)
    """
    try:
        img = np.array(image_source) if image_source else np.array([IMAGE_SOURCE])
    except NameError:
        raise ValueError("Image variable not found. Check system prompts for correct variable name.")
    
    h, w, _ = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)
    results = []
    
    # 1. Handle Rotation Grid
    for angle in rotation_angles:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rot_img = cv2.warpAffine(img, M, (w, h))
        
        # 2. Crop ROI if specified
        if roi_coords:
            y1, y2 = int(h * roi_coords[0]), int(h * roi_coords[1])
            x1, x2 = int(w * roi_coords[2]), int(w * roi_coords[3])
            crop_img = rot_img[y1:y2, x1:x2]
        else:
            crop_img = rot_img
            
        # 3. Enhancement: LAB Color Space + CLAHE
        lab = cv2.cvtColor(crop_img, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        clahe = equalize_adapthist(l_channel.astype(float)/255, clip_limit=clip_limit)
        enhanced_l = (clahe * 255).astype(np.uint8)
        enhanced_lab = lab.copy()
        enhanced_lab[:,:,0] = enhanced_l
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        results.append({'angle': angle, 'image': enhanced})
        
    # Display Results
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1: axes = [axes]
    for i, res in enumerate(results):
        axes[i].imshow(res['image'])
        axes[i].set_title(f"{res['angle']}°")
        axes[i].axis('off')
    plt.tight_layout(); plt.show()
    
    return results[-1]['image']
```

## Risk Mitigation & Principles

| Category | Principle | Mitigation Strategy |
| :--- | :--- | :--- |
| **Visual Integrity** | Complete Structures | Prioritize closed loops/bodies over partial indicators (shadows, legs). |
| **Counting Logic** | Sequential Scanning | Scan systematically to prevent double-counting in crowded scenes. |
| **Orientation** | Multi-Angle Verification | Always generate candidate rotations to confirm natural reading direction. |
| **Context** | Artifact Exclusion | Exclude reflections, window figures, or deep shadows unless part of main scene. |
| **Topology** | Shape Differentiation | Differentiate objects based on topology (loops/strokes) and context clues. |
| **Interaction** | Contact Verification | Verify physical grip/contact points to exclude proximity-based false positives. |
| **Processing** | Conservative Enhancement | Start conservative with CLAHE limits to avoid noise amplification. |
| **Data Safety** | Variable Usage | Never assume file system access. Always use pre-loaded variables. |
| **Occlusion** | Visibility Threshold | Do not count objects <50% visible unless context strongly implies presence. |
| **Verification** | Cross-Check Evidence | Trust clear visual evidence from enhanced views over manual guessing. |
| **Coordinates** | Dynamic Detection | Prefer dynamic feature detection over fixed ratio coordinates. |
| **Classification** | Unit Definition | Clarify definitions (e.g., seats vs. benches) before counting. |
| **Coding** | Variable Names | Verify exact input variable names (e.g., `original_image`) to prevent `NameError`. |
| **Rotation** | Direction Awareness | Remember PIL rotates Counter-Clockwise; adjust signs accordingly (-90 vs 90). |
| **Sequence** | Vertical Order | For vertical text, confirm top-to-bottom vs bottom-to-top reading flow. |

## Output Standards
- **Final Output**: Wrap numerical answers in `<answer>...</answer>` tags.
- **Justification**: Include brief reasoning for the count when uncertainty exists.
- **Verification Statement**: Note whether cross-verification between enhanced and original images was performed.