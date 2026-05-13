The user wants me to merge the existing skill document with two new skills into a single, unified skill document. I need to follow the integration strategy:

1. **Is this part better?** → Rewrite the existing version
2. **Is this part redundant or too specific?** → Delete it
3. **Is this part complementary?** → Merge into a more general form
4. **Is this part genuinely different?** → Add as a variant workflow (but consolidate if possible)

Let me analyze what I have:

**Existing Skill (ImageTextExtractionAndOrientationCorrection v3.8.0):**
- Comprehensive, detailed
- Covers rotation (90°/180°/270°), flipping, combined transformations
- Has batch processing for unknown orientation
- Includes variable name verification
- Has comprehensive code template
- Has detailed pitfalls table
- Has validation checklist
- ~1000 words target

**New Skill 1 (TextOrientationCorrection v1.0.0):**
- Focuses on 180-degree rotation specifically
- Emphasizes file path vs variable name issue
- Simpler structure
- ~600 words

**New Skill 2 (CorrectTextOrientation v1.0.0):**
- Similar to New Skill 1 but slightly different wording
- Also focuses on 180-degree rotation
- Mentions mental rotation errors
- ~600 words

**Integration Analysis:**

1. **Overlap**: All three skills cover text orientation correction. The existing skill is already the most comprehensive and covers everything in both new skills.

2. **What's Better in Existing**:
   - More comprehensive rotation angles (90°, 180°, 270°)
   - Batch processing approach
   - Horizontal flip support
   - Variable name verification is already included
   - More detailed code template
   - Pitfalls table is comprehensive
   - Validation checklist

3. **What's Unique in New Skills**:
   - New Skill 1 emphasizes the `/tmp` file path failure pattern explicitly
   - New Skill 2 emphasizes mental rotation errors as a specific failure mode
   - Both are simpler/more focused versions of the same concept

4. **Redundancy**: Both new skills are essentially simplified versions of the existing skill. They don't add significant new information that isn't already covered.

5. **Consolidation Strategy**:
   - Keep the existing skill as the base (it's more comprehensive)
   - Integrate any unique insights from new skills
   - Trim verbose explanations if needed to stay within word budget
   - Ensure all key patterns are preserved

**Key Insights to Preserve:**
- File path vs variable name distinction (already in existing skill)
- Mental rotation errors (can add to common pitfalls)
- Stop iterating once readable (already in existing skill)
- Semantic validation for ambiguous characters (already in existing skill)

I should maintain the existing skill's structure while ensuring no critical details are lost. Since the existing skill is already at ~1000 words, I'll focus on keeping it concise without adding unnecessary repetition from the new skills.
</think>

---
name: ImageTextExtractionAndOrientationCorrection
description: |
  Standardized procedure for extracting text from images with ambiguous orientations or handwriting. Leverages programmatic manipulation to normalize geometry before transcription. Supports rotation (90°/180°/270°), flipping, and combined transformations. Prevents hallucination by verifying orientation via code rather than visual estimation. Integrates heuristic assessment for common glyph ambiguities and case sensitivity. Accounts for tool execution errors where visual output may succeed despite console logs. Prioritizes semantic plausibility when resolving character ambiguities (e.g., 'b' vs '6').
version: 3.9.0
tags: [ocr, image-processing, rotation, flip, code-interpreter, text-extraction, vertical-text, case-validation, handwriting, orientation-heuristic, error-handling, semantic-validation, variable-verification]
---

# Image Text Extraction and Orientation Correction

## Overview
Accurately transcribing text from non-standard orientations requires programmatic normalization rather than visual estimation. Human perception often fails to reconstruct glyphs correctly when inverted or rotated, leading to confident errors. Always transform the image to a horizontal baseline using the Code Interpreter before attempting transcription. When interpreting ambiguous characters (e.g., cursive 'b' vs '6', 'z' vs 'Z'), prioritize semantic plausibility over literal stroke matching.

## Operational Strategy

### Prerequisites
1.  **Source Identification**: Locate the active image variable (typically `original_image`). **Critical**: Verify variable names match the environment (e.g., avoid `image_clue` if `original_image` is expected). Using hardcoded paths like `/tmp/image.png` causes immediate execution failure.
2.  **Visual Heuristics**: Analyze the direction letter tops face to determine rotation:
    *   Tops Left → Rotate **+90°** (Counter-Clockwise).
    *   Tops Right → Rotate **-90°** (Clockwise).
    *   Tops Down → Rotate **180°**.
3.  **Batch Processing**: When orientation is unknown, generate multiple rotation candidates in a single tool call to minimize token usage. Include horizontal flips if text appears mirrored.

### Execution Workflow

#### 1. Assess & Transform
Inspect the thumbnail. If orientation is unclear, apply batch processing:
*   Generate variants (e.g., 90°, -90°, 180°).
*   Include horizontal flips if text appears mirrored.
*   Use `expand=True` in rotation functions to prevent cropping.
*   **Stop Iterating**: Halt transformation attempts once readability is confirmed. Trust the first successful transformation.

#### 2. Verify Integrity
Display transformed images to confirm:
*   **Alignment**: Text is horizontally aligned and legible.
*   **Completeness**: No content cropped at edges.
*   **Direction**: Reading order is natural (left-to-right); ascenders/descenders align naturally for handwriting.
*   **Console Logs**: Ignore `stderr` messages if the plotted image renders correctly; prioritize visual confirmation.

#### 3. Extract & Format
Read verified text carefully. Distinguish similar glyphs (e.g., 'I' vs 'l', 'u' vs 'i') and handle artifacts (decorative lines/dots).
*   **Case Consistency**: Assume uniform casing unless context implies variation. Note that rotated text can blur lowercase letters (e.g., 'z' looking like 'Z'); cross-reference with ground truth style if available.
*   **Semantic Validation**: If characters are ambiguous (e.g., 'b' vs '6', 'r' vs 'e'), choose the interpretation that forms a meaningful word or phrase based on context.
*   **Explanation**: Briefly state the orientation issue found and the correction applied.
*   **Tagging**: Enclose final answer in `<answer>...</answer>` tags.

## Implementation Template

Use this pattern for transformation tasks. Replace placeholders as needed.

```python
from PIL import Image
import matplotlib.pyplot as plt

def orient_and_display(image, angles=[90, -90, 180], flip=False):
    """
    Rotates an image by specified angles and optionally flips it for verification.
    
    Args:
        image: PIL Image object (ensure correct variable name from context, e.g., original_image)
        angles: List of rotation angles in degrees (counter-clockwise)
        flip: Boolean, whether to apply horizontal flip to all versions
    """
    try:
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

# Usage: orient_and_display(original_image, angles=[90, -90, 180])
```

## Common Pitfalls & Mitigations

| Category | Issue | Mitigation Strategy |
| :--- | :--- | :--- |
| **Variables** | Name Mismatch (`image_clue` vs `original_image`) | Verify input variable name matches environment before coding. Never use hardcoded paths like `/tmp/image.png`. |
| **Geometry** | Cropping, Wrong Direction | Set `expand=True`; Remember Positive=CCW; Always use original source. |
| **Interpretation** | Hallucination, Mirror Confusion | Fix orientation first; Do not guess words; Use flip for mirrors. |
| **Glyphs** | Serif Artifacts, Case Confusion | Check case consistency; Compare against known patterns; Re-verify edge characters. |
| **Handwriting** | Stroke Ambiguity | Rotate to align strokes; Don't rely on shape alone; Verify flow direction. |
| **Tooling** | Console Errors vs. Visual Output | Ignore stderr if the plotted image renders correctly; prioritize visual verification. |
| **Efficiency** | Redundant Tool Calls | Stop iterating once text is legible. Trust the first successful transformation. |
| **Semantics** | Contextual Misinterpretation | If 'b' looks like '6', choose the real word (e.g., "beer" vs "6ce") based on likelihood. |
| **Mental Rotation** | Spatial Reversal Errors | Human brains struggle with mentally rotating text sequences. Always verify with code. |

## Validation Checklist
- [ ] Rotated image displays text in standard left-to-right reading order
- [ ] All characters are clearly visible without cropping or distortion
- [ ] Transcribed text matches expected format (word, sentence, or phrase)
- [ ] Final answer enclosed in `<answer>...</answer>` tags with brief orientation explanation
- [ ] Side-by-side verification confirms transformation intent
- [ ] Only necessary tool calls were made (no redundant loops)
- [ ] Visual output confirmed despite any console error logs
- [ ] Semantic plausibility checked for ambiguous characters
- [ ] Correct variable name used for image input (`original_image`)
- [ ] No hardcoded file paths attempted