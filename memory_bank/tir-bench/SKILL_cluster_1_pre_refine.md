---
name: VisualGridSolver
description: |
  Converts visual grid inputs (mazes, matrices) into executable graph structures. Implements dynamic dimension estimation, robust cell sampling, dual-mode path validation (simulation + BFS), and colored marker detection. Handles coordinate mapping, occlusion, and connectivity verification.
version: 7.0.0
---

# Visual Grid Parser & Pathfinder

## Overview
Transforms visual representations (walls, paths, markers) into structured grid matrices to determine valid movement sequences or confirm unsolvability. Prioritizes perceptual accuracy through dynamic grid sizing, heuristic validation, and rigorous variable handling.

**Use Cases:**
- **Navigation:** Solving binary grids with start/end markers to find valid move sequences (R, L, U, D).
- **Validation:** Verifying if a given path string successfully navigates a visual grid structure.
- **Analysis:** Determining solution existence or extracting grid topology from images.

**Input Requirements:** Images containing defined boundaries, pathways, and distinct terminal markers. System-provided image variable (e.g., `[IMAGE_INPUT]`) must be accessible.

## Workflow

### 1. Preprocessing & Normalization
1.  **Color Space Conversion**: Convert input to grayscale or HSL. Apply adaptive thresholding (`THRESH_BINARY_INV + OTSU`) to isolate foreground elements.
2.  **Region of Interest (ROI)**: Crop to the minimal bounding box containing non-background pixels (`cv2.boundingRect`) to eliminate external padding artifacts.
3.  **Dimension Estimation**:
    *   **Primary**: Analyze middle row/column transitions (run-length analysis) to estimate median segment length. Calculate $N = \text{InnerDim} / \text{MedianSegment}$.
    *   **Fallback**: If connectivity fails, iterate candidate sizes minimizing intra-cell variance after resizing (`INTER_AREA`).

### 2. Matrix Construction & Feature Injection
1.  **Cell Sampling**: Sample cell centers using configurable margin tolerances to avoid boundary artifacts. Construct 2D matrix where `0=Walkable` and `1=Obstacle`.
2.  **Marker Detection**: Locate Start/End coordinates via color segmentation (HSV/RGB). Map pixel centroids to grid indices.
3.  **Occlusion Handling**: Explicitly force Start/End cells to be walkable regardless of sampled intensity to handle partial occlusion or wall overlaps.
4.  **Alignment Verification**: Validate extracted grid overlay against image structure. If alignment fails, visualize grid borders to check for offset issues (e.g., thick outer walls shifting indices).

### 3. Reasoning & Validation
1.  **First-Move Heuristic**: Before full simulation, check the first character of each option against the neighbors of the start cell. Discard any option starting with a move that leads immediately into a wall or boundary.
2.  **Path Simulation**: Step through provided option strings against the matrix. Abort simulation on wall hits or out-of-bounds errors.
3.  **Connectivity Check**: Run preliminary Breadth-First Search (BFS) from Start to End. If unreachable, flag "Grid Error" rather than immediate "No Answer".
4.  **Fallback Logic**: If all options fail strict simulation, compare options against the BFS shortest path using similarity metrics (e.g., Edit Distance). Flag potential alignment errors if significant prefix overlap exists.
5.  **Decision Logic**: Conclude "No Answer" only after verifying grid parameters and confirming BFS validity. Do not force a match based on partial overlaps.

## Reference Implementation

```python
import cv2
import numpy as np
from collections import deque

def extract_grid_structure(image_data):
    """
    Standardized pipeline for visual grid parsing and validation.
    
    Args:
        image_data: Raw image input (PIL/NumPy). Must use '[IMAGE_INPUT]' variable context.
        
    Returns:
        grid_matrix, start_pos, end_pos
    """
    # 1. Preprocess & Input Handling
    img = np.array(image_data)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Dynamic Grid Sizing
    h, w, _ = img.shape
    coords = cv2.findNonZero(thresh)
    if coords is None: return None, None, None
    x, y, w_maze, h_maze = cv2.boundingRect(coords)
    maze_crop = thresh[y:y+h_maze, x:x+w_maze]
    
    # Estimate cell size via median run length on center line
    mid_row = maze_crop[h_maze//2, :]
    runs = []
    curr, cnt = mid_row[0], 1
    for val in mid_row[1:]:
        if val == curr: cnt += 1
        else: runs.append(cnt); curr, cnt = val, 1
    runs.append(cnt)
    cell_size = int(np.median(runs)) if runs else 20
    
    grid_h, grid_w = int(h_maze / cell_size), int(w_maze / cell_size)
    
    # 3. Marker Enforcement (Color Agnostic)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define ranges dynamically based on prompt colors
    mask_start = cv2.inRange(hsv, [COLOR_START_LOW], [COLOR_START_HIGH])
    mask_end = cv2.inRange(hsv, [COLOR_END_LOW], [COLOR_END_HIGH])
    
    def get_centroid(mask, cell_h, cell_w):
        M = cv2.moments(mask)
        if M["m00"] == 0: return None
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        return (int(cy/cell_h), int(cx/cell_w))
        
    start_pos = get_centroid(mask_start, h/grid_h, w/grid_w)
    end_pos = get_centroid(mask_end, h/grid_h, w/grid_w)
    
    # 4. Build Grid
    grid = np.zeros((grid_h, grid_w), dtype=int)
    step_y, step_x = h_maze/grid_h, w_maze/grid_w
    for r in range(grid_h):
        for c in range(grid_w):
            cy, cx = int(y + (r+0.5)*step_y), int(x + (c+0.5)*step_x)
            if 0 <= cy < h and 0 <= cx < w:
                if thresh[cy-x, cx-y] > 128: grid[r,c] = 1 
            
    # Force markers walkable to handle occlusion
    if start_pos: grid[start_pos] = 0
    if end_pos: grid[end_pos] = 0
    
    return grid, start_pos, end_pos

def simulate_path(path_str, grid, start_pos, end_pos):
    """Validates a path string against the grid matrix."""
    if not start_pos: return False, "Missing Start"
    r, c = start_pos
    rows, cols = grid.shape
    
    for move in path_str:
        dr, dc = {'U':(-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}.get(move, (0,0))
        nr, nc = r + dr, c + dc
        
        if not (0 <= nr < rows and 0 <= nc < cols): 
            return False, "Out of Bounds"
        if grid[nr, nc] == 1: 
            return False, "Hit Wall"
            
        r, c = nr, nc
        
    return (r, c) == end_pos, "Success"
```

## Common Pitfalls & Mitigations

| Category | Symptom | Mitigation Strategy |
| :--- | :--- | :--- |
| **Variable Scope** | `NameError` on image input | Verify pre-loaded variable names. Use `[IMAGE_INPUT]` instead of guessing specific names (e.g., `original_image`). |
| **Spatial Consistency** | Misalignment / Index Shifts | Distinguish between pixel coordinates `(y,x)` and matrix indices `(row,col)`. Use center sampling with margins. |
| **Signal Integrity** | Binary Polarity Errors | Check threshold logic (`INV/BINARY`). Dark pixels may map to 255 in inverted masks. |
| **Occlusion Handling** | Markers Classified as Walls | Force Start/End grid cells to be walkable regardless of sampled intensity. |
| **Reasoning Logic** | Premature Conclusion | Run BFS fallback if all options fail. Check connectivity before simulation. Validate all options. |
| **Verification** | Hidden Offset Errors | If simulation fails consistently, plot the extracted grid over the original image to verify border thickness. |
| **Heuristic Failure** | Invalid First Moves | Check start neighbors before full simulation. Eliminate options with invalid initial directions immediately. |

## Interaction Templates

- **Solve Task**: `"Analyze the maze image to find the valid sequence from [START_MARKER] to [END_MARKER]. Options: [OPTION_LIST]"`
- **Validation Task**: `"Validate if the path [PATH_STRING] successfully navigates the provided grid structure."`
- **Existence Check**: `"Determine if a solution exists for the maze defined in [IMAGE_INPUT]."`
- **Extraction Task**: `"Parse the image into a [DIMENSIONS] grid matrix, locate [COLOR_A] start and [COLOR_B] end points, then simulate [OPTIONS_COUNT] path strings."`