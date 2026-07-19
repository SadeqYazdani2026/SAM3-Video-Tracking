# Open-Vocabulary Video Target Tracking with SAM 3

**Student:** Sadeq Yazdani - ID: 4043614037

## Pipeline
1. **Hardware check** - verify GPU availability (`nvidia-smi`, `torch.cuda.is_available()`).
2. **Setup** - clone the official `facebookresearch/sam3` repo and install it (`pip install -e .`).
3. **Load weights** - mount Google Drive and load the checkpoint from `MyDrive/sam3.pt` (avoids hardcoding a Hugging Face token).
4. **Set video input** - point to the target video (`/content/data/ball.mp4`).
5. **Initialize model & prompt** - `build_sam3_video_predictor(...)`, start a session, add the text prompt (`"green ball"`) on frame 0.
6. **Propagate & compute centroids** - stream the mask across all frames (`propagate_in_video`); for each frame, clean the largest mask (morphological open + close) and compute its centroid:
   ```python
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   ```
7. **Render output video** - overlay the mask, bounding box, and motion trail onto each frame → `output_tracked.mp4`.
8. **Plot trajectory** - scatter plot of `[X, Y]` over time → `trajectory_plot.png`.

## How to Run (Google Colab)
1. Open the notebook and set `Runtime > Change runtime type > GPU`.
2. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3), download the checkpoint, and place it at `MyDrive/sam3.pt` in your Google Drive.
3. Upload your target video to `/content/data/ball.mp4` (or edit `video_path`).
4. Run all cells top to bottom; change `TEXT_PROMPT` for a different target if needed.

## Outputs
| File | Description |
|---|---|
| `ball.mp4` | Input video |
| `output_tracked.mp4` | Annotated output (mask, bounding box, motion trail) |
| `trajectory_plot.png` | `[X, Y]` trajectory scatter plot |
| `trajectory.csv` | Per-frame `frame_index, X, Y` values |
| `README.md` | This file |

## Reference
Official repo: <https://github.com/facebookresearch/sam3>
