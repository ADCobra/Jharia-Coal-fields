# Where OpenCV 5 does the work

Run `firewatch capabilities` to see what your install offers. On the reference build (opencv-contrib-python-headless 5.0.0.93, Python 3.13) every feature below reports `true`.

## Features new in OpenCV 5

| Feature | Call | Where | What it does here |
| --- | --- | --- | --- |
| ALIKED keypoints | `cv2.ALIKED_create(model, params)` with `params.engine = cv2.dnn.ENGINE_NEW` | `vision/registration.py` | Checks every alignment. After SIFT aligns a pass, ALIKED + LightGlue aligns it again on its own, and the distance between the two transforms goes into the pass record and the confidence score (more than 0.75 px apart costs 0.25). If SIFT fails outright, ALIKED's transform is used. |
| LightGlue matcher | `cv2.LightGlueMatcher_create(model, 0.0)`, `setPairInfo`, `match` | same | Matches ALIKED descriptors using keypoint positions and image sizes. |
| DISK keypoints | `cv2.DISK_create(model, 2048, 0.0)` | same, and `agent/tools.py` | The tie-breaker. When SIFT and ALIKED disagree, the agent's `refine_alignment` tool runs DISK on a 640 px window around the hotspot and checks which transform it agrees with. It is too slow to run on every pass (about 20 s per pair on two vCPUs), so it runs only then. |
| New DNN engine | `cv2.dnn.readNetFromONNX(path, cv2.dnn.ENGINE_NEW)` | `vision/cvcompat.py`, `vision/vlm.py` | Runs ALIKED, LightGlue and DISK, and the vision-language loop. The classic engine cannot import the ALIKED or the DISK graph (table below). |
| Approximate nearest neighbours | `cv2.ANNIndex_create(dim, cv2.ANNINDEX_DIST_EUCLIDEAN)` | `vision/similarity.py`, `pipeline/analogues.py`, `risk/scoring.py` | Precedent. Every earlier state of every track (area, mean and peak heat, growth, speed, length of history, distance to a road) is stored with whether that fire came within 400 m of a road over the next 64 days. A hotspot still more than 400 m from a road, with at least three analogues from two other tracks, gets its spread score raised to the share of analogues that did. |
| Image metadata | `cv2.imencodeWithMetadata(ext, img, [cv2.IMAGE_METADATA_XMP], [packet])`, `cv2.imdecodeWithMetadata` | `vision/media.py`, `pipeline/products.py` | Every WebP layer carries its provenance as XMP: run, scene, acquisition time, platform, alignment method, WGS84 bounds, value range, units and colour map. `GET /api/runs/<run>/layers/<name>/provenance` reads it back, so a layer saved from the dashboard still says where it came from. |
| Animated images | `cv2.Animation`, `cv2.imencodeanimation`, `cv2.imdecodeanimation` | `vision/media.py`, `pipeline/timelapse.py` | The season time-lapse: one frame per pass with the heat, hotspot outlines by risk, cones and closed roads, written as an animated WebP. `GET /api/timelapse` or `firewatch timelapse --out season.webp`. |
| Tokenizer | `cv2.dnn.Tokenizer.load(config)` | `vision/vlm.py` | Encodes the PaliGemma prompt and decodes the answer. `tests/test_vlm.py` builds a small Gemma-format tokenizer and runs the whole generation loop through it. |

## DNN engines on this build

| Network | Engine | Loads | Forward pass (s) | Why not |
| --- | --- | --- | --- | --- |
| ALIKED, 640 × 640 | `ENGINE_NEW` | yes | 1.11 | |
| ALIKED, 640 × 640 | `ENGINE_ORT` | yes | 1.05 | |
| ALIKED, 640 × 640 | `ENGINE_AUTO` | yes | 1.03 | |
| ALIKED, 640 × 640 | `ENGINE_CLASSIC` | no | | Unsqueeze node: multidimensional unsqueeze not implemented |
| DISK, 1024 × 1024 | `ENGINE_NEW` | yes | 10.0 | |
| DISK, 1024 × 1024 | `ENGINE_ORT` | yes | 11.4 | |
| DISK, 1024 × 1024 | `ENGINE_AUTO` | yes | 10.7 | |
| DISK, 1024 × 1024 | `ENGINE_CLASSIC` | no | | Shape node: dynamic shapes not supported |

From `firewatch eval bench` (`reports/bench.md`) on two x86 vCPUs. This wheel reports `ONNX Runtime: NO` and `DNN MLAS: YES` in `cv2.getBuildInformation()`, so `ENGINE_ORT` has nothing extra to call and runs at the speed of the new engine. The classic engine cannot import either network, which is why the code asks for `ENGINE_NEW` explicitly instead of trusting a default.

## Long-standing OpenCV that carries most of the load

| Step | Calls | Where |
| --- | --- | --- |
| Robust transform fit | `cv2.estimateAffine2D(..., method=cv2.USAC_ACCURATE)`; RANSAC and `USAC_MAGSAC` are compared in `reports/alignment.md` | `vision/registration.py` |
| Primary matcher | `cv2.SIFT_create`, `cv2.BFMatcher` with a ratio test | same |
| Contrast normalisation before matching | `cv2.createCLAHE` | same |
| Shift check before and after alignment | `cv2.phaseCorrelate` with `cv2.createHanningWindow` | same |
| Resampling onto the reference grid | `cv2.warpAffine` with a validity-weight trick so NaNs do not bleed | same |
| Matching the thermal footprint | `cv2.GaussianBlur` on masked sums, so albedo and NDVI are as soft as the 100 m thermal band before detrending | `vision/anomaly.py` |
| Leaving water out | `cv2.dilate` with an elliptical kernel around the `QA_PIXEL` water bit | same |
| Local background around each pixel | `cv2.boxFilter` over masked sums (61 × 61 px, cloud and hot pixels left out) | same |
| Hotspot regions | `cv2.connectedComponents` for hysteresis, `cv2.connectedComponentsWithStats`, `cv2.morphologyEx`; `cv2.dilate` to test overlap with last pass's candidates | `vision/hotspots.py` |
| Hotspot outlines and shape | `cv2.findContours`, `cv2.approxPolyDP`, `cv2.convexHull` (solidity), `cv2.minAreaRect` (rectangularity, elongation) | same |
| Cloud-edge test | `cv2.dilate` with an elliptical ring around each hotspot | same |
| Fire-front motion between passes | `cv2.DISOpticalFlow_create` (medium preset) on the anomaly maps | `vision/flow.py` |
| Spread projection | `cv2.Laplacian` inside an upwind advection–diffusion step, `cv2.findContours` for the cone outline | `physics/heat_spread.py`, `pipeline/projection.py` |
| Map overlays and time-lapse frames | `cv2.imencode('.webp', ...)`, `cv2.polylines`, `cv2.putText`, `cv2.addWeighted`, `cv2.resize` | `pipeline/products.py` |

## What changed from the original plan, and why

**SIFT is the primary matcher, not ALIKED + LightGlue.** The plan assumed the learned matcher would handle seasonal change better. The alignment evaluation (`reports/alignment.md`) says otherwise on our data. On the hardest condition, a dry-season scene against a monsoon reference with new dumps and pits, SIFT aligned all 5 pairs with a median error of 0.024 px; ALIKED + LightGlue aligned 1 of 5. SIFT was also the most accurate in the other four conditions (tied with ALIKED + LightGlue on the cross-sensor pair at 0.033 px) and about four times faster. ALIKED was trained on ground-level photographs, and 30 m satellite texture is a different domain.

**ALIKED checks every pass, and DISK breaks ties.** An alignment that silently drifts by a pixel would look like a fire front moving 30 m. So ALIKED + LightGlue, independent of SIFT and as accurate on the cross-sensor pair, now aligns every pass a second time, which adds about 3.4 s. When the two disagree by more than 0.75 px, the confidence drops and the agent asks DISK, a third independent method, around the hotspot. DISK aligned every pair in every condition, but takes about 20 s, so it is kept for that moment.

**USAC_ACCURATE, not MAGSAC++.** The first version fitted the transform with `USAC_MAGSAC`. On the same SIFT matches, `USAC_ACCURATE` was within about 0.001 px of plain RANSAC or better in every condition and clearly better on the cross-sensor pair (0.033 against 0.048 px), while MAGSAC++ was never better than RANSAC and worse in three of the five conditions. With 95 % of the matches wrong, RANSAC found the transform in 4 of 5 pairs, the two USAC estimators in all 5. `USAC_ACCURATE` costs a few milliseconds more than RANSAC on clean matches, nothing next to feature detection.

**DISK is matched with a ratio test, not LightGlue.** The published DISK LightGlue ONNX graph does not load in OpenCV 5.0.0 (its output count does not match what `LightGlueMatcher` expects), so DISK descriptors go through a brute-force matcher with Lowe's ratio test.

**No TRUCO contours.** The plan mentioned the newer contour-tracing algorithm. It is not exposed to Python in 5.0.0, so hotspot outlines come from `cv2.findContours`.

**The vision-language loop is tested, the 3-billion-parameter model is not.** `vision/vlm.py` follows OpenCV 5's PaliGemma 2 sample (`samples/dnn/vlm_inference.py`): a SigLIP vision encoder, an embedding graph and a Gemma 2 decoder, loaded with `readNetFromONNX` on the new engine, and a tokenizer config in OpenCV's format. The ONNX export the sample points to is `nklskyoy/paligemma2-3b-pt-224-onnx` on Hugging Face, which the build environment could not reach. `tests/test_vlm.py` therefore builds three tiny stand-in networks with the same input and output names (`pixel_values`, `input_ids`, `inputs_embeds`) and a small Gemma-format tokenizer, and runs the real loop through OpenCV: tokenise the prompt, encode the image, embed, decode token by token until end of sequence, detokenise. The answer comes back as "yes" or "no" as scripted. What remains unverified is the real model's accuracy and speed; a published walkthrough of the sample reports over a minute per caption on a strong CPU, which is why the agent asks it only when the cheaper industrial test is unsure, and why `vlm.enabled` is `false` by default. Without it, an unmapped plant goes to a person instead of being excluded (scenario C04 in `reports/agent.md`).

**The new engine runs on the CPU.** At release, `ENGINE_NEW` has no CUDA or OpenVINO target, which is one reason the deployment uses a Graviton CPU instance rather than a GPU.
