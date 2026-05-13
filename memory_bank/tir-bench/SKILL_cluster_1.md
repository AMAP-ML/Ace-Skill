---
name: VisualGridSolver
description: |
  Converts visual grid inputs into executable graph structures. Implements dynamic dimension estimation, cell sampling, and dual-mode path validation (simulation + connectivity analysis). Handles coordinate mapping, occlusion, and topology verification. Integrates iterative refinement for ambiguous grid sizes.
version: 7.1.0
---

# Visual Grid Parser & Pathfinder

## Overview
Transforms visual representations (walls, paths, markers) into structured matrix graphs to determine valid movement sequences or verify solvability. Prioritizes perceptual accuracy through adaptive grid sizing, heuristic validation, and rigorous state tracking.

**Core Capabilities:**
- **Parsing:** Dynamic extraction of grid dimensions and obstacle topology from images.
- **Navigation:** Execution of move sequences (Cardinal Directions) against matrix constraints.
- **Verification:** Connectivity analysis (BFS) to confirm solution existence prior to simulation.
- **Robustness:** Iterative dimension checking to resolve ambiguity (e.g., 38x38 vs 39x39).

**Input Requirements:** Accessible image variable (e.g., `[IMAGE_INPUT]`) containing defined boundaries and terminal markers.

## Core Workflow

### 1. Perception Layer (Visual → Logical)
1.  **Normalization**: Convert to grayscale/HSL. Apply adaptive thresholding to isolate foreground elements from background noise.
2.  **ROI Extraction**: Crop to minimal bounding box containing non-background pixels to eliminate padding artifacts.
3.  **Dimension Estimation**:
    - **Primary**: Analyze run-length transitions on center lines to estimate median segment length. Calculate $N = \text{InnerDim} / \text{MedianSegment}$.
    - **Fallback (Iterative)**: If initial grid yields no valid path or alignment errors, test adjacent sizes ($\pm 1$ cell) to resolve ambiguity.
4.  **Feature Injection**:
    - Construct 2D matrix (`0=Walkable`, `1=Obstacle`) via center-point sampling.
    - Detect markers via color segmentation; map centroids to grid indices.
    - **Occlusion Override**: Force Start/End cells to `0` regardless of sampled intensity to handle partial overlaps.
5.  **Alignment Verification**: Validate extracted grid overlay against image structure. If alignment fails, visualize borders to detect offset issues (e.g., thick outer walls).

### 2. Reasoning Layer (Logical → Decision)
1.  **Heuristic Pruning**: Before full simulation, validate the first move of each option against Start neighbors. Discard options leading immediately to walls/boundaries.
2.  **Connectivity Check**: Run preliminary Breadth-First Search (BFS) from Start to End. If unreachable, flag "Unsolvable" before attempting path simulation.
3.  **Path Simulation**: Step through option strings against the matrix. Abort on wall collisions or out-of-bounds errors.
4.  **Fallback Analysis**: If strict simulation fails, compare options against BFS shortest path using similarity metrics (e.g., Edit Distance). Flag alignment errors if significant prefix overlap exists.
5.  **Decision Logic**: Conclude "No Answer" only after verifying grid parameters and confirming BFS validity. Avoid forcing matches based on partial overlaps.

## Implementation Reference

```python
import cv2
import numpy as np
from collections import deque

def extract_grid_structure(image_data):
    """Standardized pipeline for visual grid parsing."""
    img = np.array(image_data)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    coords = cv2.findNonZero(thresh)
    if coords is None: return None, None, None
    
    x, y, w_maze, h_maze = cv2.boundingRect(coords)
    maze_crop = thresh[y:y+h_maze, x:x+w_maze]
    
    # Dynamic Dimension Estimation
    mid_row = maze_crop[h_maze//2, :]
    runs = []
    curr, cnt = mid_row[0], 1
    for val in mid_row[1:]:
        if val == curr: cnt += 1
        else: runs.append(cnt); curr, cnt = val, 1
    runs.append(cnt)
    
    cell_size = int(np.median(runs)) if runs else [MIN_CELL_SIZE]
    grid_h, grid_w = int(h_maze / cell_size), int(w_maze / cell_size)
    
    # Marker Detection & Enforcement
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_start = cv2.inRange(hsv, [COLOR_START_LOW], [COLOR_START_HIGH])
    mask_end = cv2.inRange(hsv, [COLOR_END_LOW], [COLOR_END_HIGH])
    
    def get_centroid(mask, step_y, step_x):
        M = cv2.moments(mask)
        if M["m00"] == 0: return None
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        return (int(cy/cell_size), int(cx/cell_size))
        
    start_pos = get_centroid(mask_start, h_maze/grid_h, w_maze/grid_w)
    end_pos = get_centroid(mask_end, h_maze/grid_h, w_maze/grid_w)
    
    # Build Grid
    grid = np.zeros((grid_h, grid_w), dtype=int)
    step_y, step_x = h_maze/grid_h, w_maze/grid_w
    for r in range(grid_h):
        for c in range(grid_w):
            cy, cx = int(y + (r+0.5)*step_y), int(x + (c+0.5)*step_x)
            if 0 <= cy < h_maze and 0 <= cx < w_maze:
                if maze_crop[int(cy-y), int(cx-x)] > 128: grid[r,c] = 1 
            
    # Force markers walkable
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

## Failure Modes & Resolutions

| Category | Symptom | Resolution |
| :--- | :--- | :--- |
| **Variable Scope** | `NameError` on input | Verify pre-loaded variable names. Use consistent placeholders (e.g., `[IMAGE_INPUT]`). |
| **Spatial Consistency** | Misalignment / Index Shifts | Distinguish between pixel `(y,x)` and matrix `(row,col)`. Use center sampling with margins. |
| **Signal Integrity** | Binary Polarity Errors | Verify threshold logic (`INV/BINARY`). Dark pixels may map to 255 in inverted masks. |
| **Occlusion Handling** | Markers Classified as Walls | Force Start/End grid cells to `0` regardless of sampled intensity. |
| **Reasoning Logic** | Premature Conclusion | Run BFS fallback if all options fail. Check connectivity before simulation. |
| **Heuristic Failure** | Invalid First Moves | Check start neighbors before full simulation. Eliminate options with invalid initial directions immediately. |
| **Grid Ambiguity** | Incorrect Dimensions | If simulation fails, iterate grid size $\pm 1$ cell to resolve edge cases. |

## Usage Patterns

- **Solve Task**: `"Analyze the maze image to find the valid sequence from [START_MARKER] to [END_MARKER]. Options: [OPTION_LIST]"`
- **Validation Task**: `"Validate if the path [PATH_STRING] successfully navigates the provided grid structure."`
- **Existence Check**: `"Determine if a solution exists for the maze defined in [IMAGE_INPUT]."`
- **Extraction Task**: `"Parse the image into a [DIMENSIONS] grid matrix, locate [COLOR_A] start and [COLOR_B] end points, then simulate [OPTIONS_COUNT] path strings."`