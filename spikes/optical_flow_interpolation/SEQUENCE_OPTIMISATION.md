# Sequence Optimisation — Brief for Claude Code

## Context
See `CLAUDE.md` and `docs/decisions_log.md` for full project context.

The optical flow interpolation notebook (`notebooks/optical_flow_interpolation.ipynb`)
currently uses a manually specified set sequence. Transitions are jarring because face
position and orientation varies significantly between images, even within the same style set.

## Goal
Write and run a script that produces an optimised image sequence, ordered to minimise
face position jumps between consecutive frames. Output is a JSON file consumed by the
interpolation notebook instead of the manual `sets` config.

## Approach

### Keypoint extraction
- Use YOLOv8 pose model (`yolov8n-pose.pt`) — already used in `tools/prepare_dataset.py`
- For each image in `dataset/prepared/`, extract keypoints 0-4 (nose, left eye, right eye,
  left ear, right ear)
- Use nose + eyes only as the anchor (ears are often missing for sideways faces)
- Represent each image as a 2D point: centroid of visible nose+eye keypoints
- If no keypoints detected, flag the image and skip it

### Distance metric
- Euclidean distance between keypoint centroids of consecutive images
- Normalised to image dimensions (images are 512x512)

### Ordering algorithm
Within each set:
- Find the ordering that minimises total keypoint distance across consecutive images
- Nearest-neighbour greedy is fine for small sets (3-20 images)
- Start from the image closest to the centroid of the set

Across sets:
- Sets are specified manually in config
- For each pair of adjacent sets, try both forward and reversed ordering of the second set
- Pick whichever minimises the jump from the last image of set A to the first of set B

### Output format
`data/sequence.json`:
```json
{
  "sets": [
    {
      "name": "rhinestones",
      "images": ["path/to/image1.png", "path/to/image2.png"],
      "keypoints": [[cx, cy], [cx, cy]]
    }
  ],
  "total_images": 15,
  "undetected": ["path/to/failed.png"]
}
```

## Config
Accept CLI args:
- `--prepared_dir` — path to `dataset/prepared/` (local path)
- `--sets` — comma-separated list of set names in desired order
- `--output` — path to output JSON (default: `data/sequence.json`)

## After running
- Print summary: total images, undetected count, total keypoint distance before/after
- Show matplotlib visualisation of keypoint positions across the sequence so we can
  visually verify the ordering makes sense

## Integration with notebook
Update notebook cell 2 to read `sequence.json` instead of the manual `sets` list.
Iterate over `sequence["sets"]` using pre-optimised image order.

## Notes
- Prepared images are at:
  `/Users/jasonrobert/Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/.shortcut-targets-by-id/1wVX93UBmrYHkO8xjnKxh429ya-a0rqk3/Gh0st in the Loop/dataset/prepared/`
- No GPU needed — pose inference on 512x512 images is fast on CPU
- ultralytics may have Intel Mac dependency issues — check first, fall back to Colab if needed
