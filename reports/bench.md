# Run time

Linux x86_64, 2 CPUs, OpenCV 5.0.0 with 2 threads. Synthetic grid of 1364 x 857 px at 30 m, 4 passes with the agent after two archive passes. Alignment used SIFT.

Median time per pass: **7.25 s**.

| Stage | Median seconds |
| --- | --- |
| load | 0.233 |
| thermal | 0.059 |
| align | 4.361 |
| warp | 0.028 |
| anomaly | 0.328 |
| hotspots | 0.042 |
| tracking | 0.009 |
| assess | 0.342 |
| agent | 0.050 |
| persist | 1.510 |

## Image registration alone

| Method | Image (px) | Median seconds |
| --- | --- | --- |
| SIFT | 768 x 768 | 0.649 |
| ALIKED + LightGlue | 768 x 768 | 3.337 |
| DISK | 768 x 768 | 19.810 |
| ORB | 768 x 768 | 0.219 |


## OpenCV 5 DNN engines

The ALIKED and DISK networks loaded with `cv2.dnn.readNetFromONNX(path, engine)` and run on a random image. This wheel reports ONNX Runtime: NO and DNN MLAS: YES (SGEMM-only, vendored).

| Model | Engine | Input (px) | Loads | Load seconds | Forward seconds | Why not |
| --- | --- | --- | --- | --- | --- | --- |
| ALIKED | ENGINE_NEW | 640 x 640 | yes | 0.026 | 1.113 |  |
| ALIKED | ENGINE_ORT | 640 x 640 | yes | 0.094 | 1.053 |  |
| ALIKED | ENGINE_CLASSIC | 640 x 640 | no |  |  | Unsqueeze node: Multidimensional unsqueeze |
| ALIKED | ENGINE_AUTO | 640 x 640 | yes | 0.064 | 1.034 |  |
| DISK | ENGINE_NEW | 1024 x 1024 | yes | 0.080 | 10.041 |  |
| DISK | ENGINE_ORT | 1024 x 1024 | yes | 0.130 | 11.414 |  |
| DISK | ENGINE_CLASSIC | 1024 x 1024 | no |  |  | Shape node: !isDynamicShape |
| DISK | ENGINE_AUTO | 1024 x 1024 | yes | 0.087 | 10.678 |  |

The classic engine cannot import either network, which is why the matchers ask for the new engine explicitly. Without ONNX Runtime in the build, ENGINE_ORT gives the same speed as the new engine.

Run `firewatch eval bench` on the Graviton instance to get the same tables for arm64.
