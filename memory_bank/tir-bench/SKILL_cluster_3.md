---
name: ImageTextExtractionAndOrientationCorrection
description: |
  Standardized procedure for extracting text from images with ambiguous orientations or handwriting. Leverages programmatic manipulation to normalize geometry before transcription. Supports rotation (0°/90°/180°/270°), flipping, and combined transformations. Prevents hallucination by verifying orientation via code rather than visual estimation. Integrates heuristic assessment for common glyph ambiguities and case sensitivity. Accounts for tool execution errors where visual output may succeed despite console logs. Prioritizes semantic plausibility when resolving character ambiguities.
version: 4.0.0
tags: [ocr, image-processing, rotation, flip, code-interpreter, text-extraction, vertical-text, case-validation, handwriting, orientation-heuristic, error-handling, semantic-validation, variable-verification]
---

# Image Text Extraction and Orientation Correction

## Core Principle
Accurately transcribing text from non-standard orientations requires **programmatic normalization** rather than visual estimation. Human perception often fails to reconstruct glyphs correctly when inverted or rotated, leading to confident errors. Always transform the image to a horizontal baseline using the Code Interpreter before attempting transcription. When interpreting ambiguous characters (e.g., cursive 'b' vs '6', 'z' vs 'Z'), prioritize semantic plausibility over literal stroke matching.

## Operational Procedure

### 1. Pre-Processing & Assessment
*   **Variable Verification**: Identify the active image variable (e.g., `original_image`). **Critical**: Never use hardcoded file paths (e.g., `/tmp/image.png`) as they cause immediate `FileNotFoundError`. Ensure the variable name matches the environment context.
*   **Heuristic Analysis**: Analyze thumbnail direction to determine transformation needs:
    *   Tops Left → Rotate **+90°**. Tops Right → Rotate **-90°**. Tops Down → Rotate **180°**.
    *   Mirrored Appearance → Apply Horizontal Flip.
*   **Efficiency Strategy**: If orientation is uncertain, generate all four standard rotations (0°, 90°, 180°, 270°) in a single tool call to minimize token usage and latency compared to sequential attempts.

### 2. Transformation & Verification
*   **Tool Constraint**: Adhere to the single tool call limit per turn. If multiple rotations are needed, iterate through separate turns.
*   **Batch Generation**: Generate multiple rotation candidates in a single tool call. Include horizontal flips if text appears mirrored.
*   **Execution Safety**: Use `expand=True` in rotation functions to prevent cropping.
*   **Visual Confirmation**: Display transformed images to confirm:
    *   **Alignment**: Text is horizontally aligned and legible.
    *   **Completeness**: No content cropped at edges.
    *   **Direction**: Reading order is natural (left-to-right); ascenders/descenders align naturally for handwriting.
    *   **Error Handling**: Ignore `stderr` messages if the plotted image renders correctly; prioritize visual verification over console logs.

### 3. Extraction & Validation
*   **Glyph Resolution**: Distinguish similar glyphs (e.g., 'I' vs 'l', 'u' vs 'i') and handle artifacts (decorative lines/dots).
*   **Case Consistency**: Assume uniform casing unless context implies variation. Note that rotated text can blur lowercase letters (e.g., 'z' looking like 'Z'); cross-reference with ground truth style if available.
*   **Semantic Plausibility**: If characters are ambiguous (e.g., 'b' vs '6'), choose the interpretation that forms a meaningful word or phrase based on context.
*   **Formatting**: Briefly state the orientation issue found and the correction applied. Enclose final answer in `<answer>...</answer>` tags.

## Implementation Template

Use this pattern for transformation tasks. Replace placeholders as needed.

```python
from PIL import Image
import matplotlib.pyplot as plt

def orient_and_display(image, angles=[0, 90, 180, 270], flip=False):
    """
    Rotates an image by specified angles and optionally flips it for verification.
    
    Args:
        image: PIL Image object (verify variable name matches environment)
        angles: List of rotation angles in degrees (counter-clockwise)
        flip: Boolean, whether to apply horizontal flip to all versions
    """
    try:
        # CRITICAL: Use pre-loaded variables (e.g., original_image), not file paths.
        img = image 
        
        plt.figure(figsize=(15, 5))
        
        # Plot Original
        plt.subplot(1, len(angles) + 1, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis('off')
        
        # Plot Rotated Variants
        for i, angle in enumerate(angles):
            rotated_img = img.rotate(angle, expand=True)
            
            # Optional: Handle Mirrored Text
            if flip:
                rotated_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            plt.subplot(1, len(angles) + 1, i + 2)
            plt.imshow(rotated_img)
            plt.title(f"Rotated {angle}°{', Flip' if flip else ''}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error during rotation: {e}")

# Usage: orient_and_display(original_image, angles=[0, 90, 180, 270])
```

## Common Pitfalls & Mitigations

| Category | Issue | Mitigation Strategy |
| :--- | :--- | :--- |
| **Reference** | Hardcoded Paths (`/tmp/...`) | Always use the active image variable passed from the environment. |
| **Geometry** | Cropping, Wrong Direction | Set `expand=True`; Remember Positive=CCW; Always use original source. |
| **Efficiency** | Redundant Tool Calls | Stop iterating once text is legible. Trust the first successful transformation. |
| **Constraints** | Multi-Step Logic | Respect single tool call limit per turn. Iterate sequentially if needed. |
| **Interpretation** | Hallucination, Mirror Confusion | Fix orientation first; Do not guess words; Use flip for mirrors. |
| **Glyphs** | Serif Artifacts, Case Confusion | Check case consistency; Compare against known patterns; Re-verify edge characters. |
| **Handwriting** | Stroke Ambiguity | Rotate to align strokes; Don't rely on shape alone; Verify flow direction. |
| **Tooling** | Console Errors vs. Visual Output | Ignore stderr if the plotted image renders correctly; prioritize visual verification. |
| **Semantics** | Contextual Misinterpretation | If 'b' looks like '6', choose the real word (e.g., "beer" vs "6ce") based on likelihood. |

## Validation Checklist
- [ ] **Variable Check**: Correct image variable used (no hardcoded paths).
- [ ] **Orientation**: Transformed image displays text in standard left-to-right reading order.
- [ ] **Integrity**: All characters clearly visible without cropping or distortion.
- [ ] **Consistency**: Transcribed text matches expected format (word, sentence, or phrase).
- [ ] **Semantics**: Ambiguous characters resolved based on contextual plausibility.
- [ ] **Verification**: Side-by-side comparison confirms transformation intent.
- [ ] **Logging**: Visual output confirmed despite any console error logs.
- [ ] **Formatting**: Final answer enclosed in `<answer>...</answer>` tags with brief explanation.