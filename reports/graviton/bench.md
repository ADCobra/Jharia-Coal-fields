# Run time

Linux aarch64, 2 CPUs, OpenCV 5.0.0 with 2 threads. Synthetic grid of 1364 x 857 px at 30 m, 4 passes with the agent after two archive passes. Alignment used SIFT.

Median time per pass: **41.95 s**.

| Stage | Median seconds |
| --- | --- |
| load | 2.421 |
| thermal | 0.066 |
| align | 33.297 |
| warp | 0.568 |
| anomaly | 0.958 |
| hotspots | 0.169 |
| tracking | 0.157 |
| assess | 0.504 |
| agent | 1.239 |
| persist | 2.853 |

## Image registration alone

| Method | Image (px) | Median seconds |
| --- | --- | --- |
| SIFT | 768 x 768 | 0.819 |
| ALIKED + LightGlue | 768 x 768 | 4.376 |
| ORB | 768 x 768 | 0.308 |


## OpenCV 5 DNN engines

The ALIKED and DISK networks loaded with `cv2.dnn.readNetFromONNX(path, engine)` and run on a random image. This wheel reports ONNX Runtime: NO and DNN MLAS: YES (SGEMM-only, vendored).

| Model | Engine | Input (px) | Loads | Load seconds | Forward seconds | Why not |
| --- | --- | --- | --- | --- | --- | --- |
| ALIKED | ENGINE_NEW | 640 x 640 | yes | 0.185 | 1.158 |  |
| ALIKED | ENGINE_ORT | 640 x 640 | yes | 0.155 | 1.252 |  |
| ALIKED | ENGINE_CLASSIC | 640 x 640 | no |  |  | Unsqueeze node: Multidimensional unsqueeze |
| ALIKED | ENGINE_AUTO | 640 x 640 | yes | 0.093 | 1.188 |  |

The classic engine cannot import either network, which is why the matchers ask for the new engine explicitly. Without ONNX Runtime in the build, ENGINE_ORT gives the same speed as the new engine.

Left out on this machine (`FIREWATCH_BENCH_SKIP`): disk.

Run `firewatch eval bench` on the Graviton instance to get the same tables for arm64.
