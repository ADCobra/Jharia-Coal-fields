# Jharia Fire Watch: technical report

## 1. Summary

Jharia Fire Watch processes each Landsat 8 or 9 pass over the Jharia coalfield into a land surface temperature map, finds hot ground that the surface itself does not explain, tracks each fire from pass to pass, projects its spread for 48 days, and hands the result to an agent that can close threatened roads, ask a person, or propose reopening. Closing is the only consequential action the agent may take on its own; reopening always needs a person.

Tested on two synthetic coalfields built with Landsat's own calibration (section 3.3), one used for tuning and one held out, it gives these results:

| Check | Result | Report |
| --- | --- | --- |
| Temperature retrieval against the known surface | mean absolute error 0.10 K, bias −0.004 K | `reports/lst.md` |
| Hotspot detection, 16 passes of the tuned world | precision 1.00, recall 0.97; precision 0.16 and 19 false hotspots per pass without the albedo/NDVI detrending | `reports/detection.md` |
| Hotspot detection, 16 passes of the held-out world | precision 1.00, recall 0.86, after fixing the false alarms it exposed on its first run | `reports/detection.md` |
| Alignment, 25 synthetic pairs in 5 conditions | SIFT within 0.5 px on 25 of 25 and the most accurate or tied in every condition; DISK 25 of 25 at 22 s a pair; ALIKED + LightGlue 21 of 25 | `reports/alignment.md` |
| 48-day spread cone | covers 92 % of the ground a moving fire went on to burn, against 78 % for a circle of the same area (held-out world); reach-a-road Brier score 0.038 against 0.145 for the base rate | `reports/forecast.md` |
| Precedent from past fires (`cv2.ANNIndex`) | area under the ROC curve 0.97 for "will it come within 400 m of a road in 64 days", leaving one fire out at a time | `reports/forecast.md` |
| Agent, 36 scripted situations | 35 decisions match a safety officer's call, no unwarranted closures, no dismissed fires; 10 of 10 safety and memory checks; 14 one-at-a-time changes to its constants flip at most 3 decisions | `reports/agent.md` |
| Run time | median 7.3 s per pass on two x86 vCPUs, including a second, independent alignment; 42 s on the live Graviton t4g.small with 2 GB | `reports/bench.md`, `reports/graviton/bench.md` |

It has also processed 23 real Landsat 8 and 9 scenes of Jharia, from 4 November 2025 to 14 October 2026, end to end (section 5.7). Real scenes have no ground truth, so they show the system runs on real data and lands in the right range, not how accurate it is: on the 20 clear passes the hot ground has a median area of 2.90 km² (2.06 to 6.67), against NRSC's 1.89 km² for 2020.

## 2. The problem

Coal seams under Jharia, in Jharkhand, have been burning since 1916. The fire area mapped by NRSC from Landsat-8's thermal band fell from about 3.28 km² in 2017 to about 1.89 km² in 2020; the 2009 Master Plan had put it at 8.9 km² ([BCCL](https://www.bcclweb.in/masterplan/FireDrilling.pdf)). A fire hollows the ground above it, and the ground subsides. Roads, including the haul roads that dumpers use all day, cross that ground.

Thermal mapping of the fires is established: NRSC's figures come from Landsat band 10, and Biswal and Gorai (2019) describe the same digital-number-to-temperature workflow for Jharia. What we did not find is a system that runs on every pass, follows each fire over time, and turns a detection into a road decision with a record of why it was made. That is what this project builds.

## 3. Data

### 3.1 Landsat 8 and 9

Band 10 (10.6–11.19 µm) is recorded at 100 m and delivered at 30 m; the reflective bands give NDVI and albedo; `QA_PIXEL` gives cloud, shadow and water. With both satellites in orbit, each spot is imaged every 8 days. The pipeline reads two Collection 2 processing levels, which reach the temperature map differently:

| | Level-2 (L2SP) | Level-1 (L1TP) |
| --- | --- | --- |
| Used for | The 23 real scenes in section 5.7 (`data/real.py`, downloaded by `scripts/fetch_real_data.py`) | The live AWS mode (`data/landsat.py`), and the format the synthetic coalfields are written in |
| Source | Microsoft Planetary Computer, collection `landsat-c2-l2` | USGS STAC server, collection `landsat-c2l1`, pixels from `s3://usgs-landsat` (requester pays) |
| Temperature | `ST_B10`, the USGS surface temperature product, already corrected for the atmosphere and for emissivity; scaled as `DN × 0.00341802 + 149.0` K | Our own retrieval from band 10 digital numbers with each scene's `MTL.json` constants and an NDVI emissivity (section 4.1), with no atmospheric correction |
| Reflectance | Surface reflectance, `DN × 2.75e-5 − 0.2` | Top-of-atmosphere reflectance from the `MTL.json` rescaling and sun elevation |

Everything after the temperature map (alignment, detection, tracking, the agent) is the same code for both. Only the Level-2 path has run against real scenes so far; the Level-1 reader is tested on files written in the USGS layout (section 7). Details are in [data_sources.md](data_sources.md).

### 3.2 Sentinel-2 Level-2A

Bands 11 and 12 (20 m) from Earth Search, used by one agent tool to look for hot pixels.

### 3.3 The synthetic coalfields

Real Landsat scenes come without ground truth: nobody knows the exact outline of a subsurface fire on a given day. So the evaluation runs on synthetic coalfields (`data/synthetic.py`) on the real grid (1364 × 857 pixels at 30 m over 86.10–86.50° E, 23.63–23.86° N), with pits, dumps, water and villages that have their own reflectance, seasonal temperature and vegetation, random cloud with shadows, and misregistration on every pass. Each pass is encoded the way Landsat delivers it, with digital numbers through the real L8 constants, a bit-coded `QA_PIXEL` and an MTL dictionary, and read by the same code as real scenes.

The tuned world has five fires that move, grow, start and stop, a coke-oven battery hotter than any fire, and up to 1.6 pixels of misregistration. Every threshold was set on it. The held-out world was built afterwards to find out what that tuning had overfitted: a different seed and layout, seven irregular fires with up to three lobes (three of them heading for a road fast enough to reach it during the season), weaker heat, 1.7 times the sensor noise, cloud on half the passes, up to 2.4 pixels of misregistration, and a second coke plant that is not on the industrial map.

## 4. Method

### 4.1 Land surface temperature

Level-2 scenes arrive with surface temperature already computed by USGS, and that product is used as it is. For Level-1 scenes and the synthetic coalfields, the temperature is retrieved from band 10, with $M_L$, $A_L$, $K_1$, $K_2$ from the MTL:

$$L = M_L \cdot DN + A_L, \qquad T_B = \frac{K_2}{\ln(K_1 / L + 1)}, \qquad T_s = \frac{T_B}{1 + (\lambda T_B / \rho)\,\ln \varepsilon}$$

with $\lambda = 10.895$ µm and $\rho = hc/k_B = 14388$ µm·K. Emissivity follows the NDVI threshold method of Sobrino et al. (2004): soil (0.966) below NDVI 0.2, vegetation (0.986) above 0.5, and in between

$$\varepsilon = \varepsilon_v P_v + \varepsilon_s (1 - P_v) + (1 - \varepsilon_s)(1 - P_v) F \varepsilon_v, \qquad P_v = \left(\frac{NDVI - 0.2}{0.5 - 0.2}\right)^2, \quad F = 0.55.$$

Water takes 0.991. Broadband albedo uses Liang's (2001) coefficients, from surface reflectance on Level-2 scenes and top-of-atmosphere reflectance on Level-1. The Level-1 retrieval applies no atmospheric correction; section 7 explains why that matters less than it seems.

### 4.2 Alignment

A misregistration of one pixel between passes would look like a fire front moving 30 m, so each pass is aligned to a reference pass. The NIR band is scaled to 8 bits with CLAHE, SIFT keypoints (up to 4000) are matched with a 0.78 ratio test, and `cv2.estimateAffine2D` with `USAC_ACCURATE` fits the transform with a 1.5 px threshold. A fit needs 25 inliers and a scale within 2 % of 1. ALIKED + LightGlue, running on OpenCV 5's new DNN engine, then aligns the same pair on its own. The RMS distance between the two transforms over the image is stored with the pass; if it is above 0.75 px, confidence in every hotspot of that pass drops, and the agent can ask a third method, DISK, to align a 640 px window around the hotspot and say which of the two it agrees with. If SIFT fails outright and ALIKED does not, ALIKED's transform is used. All products are warped onto the reference grid with NaN-aware interpolation.

### 4.3 Anomalies and hotspots

Daytime temperature depends on the surface: dark coal and bare dumps heat up in the sun. A Huber-weighted regression (six iterations of reweighted least squares, on up to 200,000 clear pixels)

$$T_s = \beta_0 + \beta_1 \alpha + \beta_2\, NDVI + \beta_3\, NDVI^2 + r$$

removes what albedo $\alpha$ and NDVI explain. Band 10 is recorded at 100 m, so before the fit $\alpha$ and NDVI, which come from 30 m bands, are smoothed to the same footprint with a masked Gaussian ($\sigma$ = 1.4 px, a full width at half maximum of 100 m), as thermal sharpening methods do when they fit temperature against NDVI at the coarse resolution (Agam et al., 2007). Water pixels from `QA_PIXEL` and a 2-pixel ring around them are left out of the fit, the background and detection. Section 5.2 shows what went wrong without these two steps.

The residual $r$ is compared with a local background $b$, the mean over a 61 × 61 pixel window that ignores cloud and, after the first round, pixels more than $2\sigma$ above it. The anomaly is $a = r - b$ and $z = a / (1.4826\, \mathrm{MAD}(a))$.

Hotspots use hysteresis: a seed needs $z \ge 3$ and $a \ge 4$ K, and grows over neighbours with $z \ge 1.8$ and $a \ge 1.5$ K. Regions under 6 pixels are dropped. Persistence then keeps a weaker blob ($z \ge 2$, $a \ge 2.5$ K, at least 4 pixels) when it overlaps, within 2 pixels, a candidate or hotspot from the previous pass; a weak but steady fire is kept, a one-off flicker is not. For each region OpenCV gives the outline (`findContours`, `approxPolyDP`), solidity (area over `convexHull` area), rectangularity (area over `minAreaRect` area) and elongation; we add the cloud fraction of a 3-pixel ring around it and its overlap with mapped industrial sites.

### 4.4 Tracking and front motion

A hotspot joins a track when the track's position, extrapolated with its velocity, is within 450 m, or when the outlines overlap. Least-squares lines through the last eight points give area growth, velocity and the heat trend. DIS optical flow between consecutive anomaly maps, weighted by gradient magnitude around the hotspot, gives a second velocity; the two are blended 60/40 when the track has three or more points.

### 4.5 Spread projection

Each hotspot's anomaly field $u$ is pushed forward 48 days with

$$\frac{\partial u}{\partial t} = -\mathbf{v} \cdot \nabla u + D \nabla^2 u - k u$$

using upwind differences and a time step limited for stability. $D = \max(0, dA/dt) / 4\pi$, which is the diffusivity that grows the $e^{-1}$ contour of a Gaussian at the observed area growth rate $dA/dt$, capped at 900 m²/day; $k$ comes from a cooling trend. 24 runs draw the speed with a 35 % standard deviation, the heading with 15° and the diffusivity log-normally. The cone is the area covered in at least 20 % of runs; the share of runs touching a road is the reach probability.

### 4.6 Risk, confidence and precedent

$$\text{risk} = 0.35\, \hat a + 0.30\, \hat s + 0.35\, e^{-d/400\,\text{m}}$$

where $\hat a = \min(1, 0.5\,\bar a/12\,\text{K} + 0.5\, a_{max}/24\,\text{K})$, $\hat s = \min(1, \max(g/0.5,\; v_{\perp}/3\,\text{m/day}))$ with $g$ the area growth per 30 days and $v_\perp$ the speed toward the nearest road, and $d$ the smaller of today's distance and the projected cone's distance to a road. High risk starts at 0.62, medium at 0.40.

Precedent can raise $\hat s$. Every earlier state of every track goes into a `cv2.ANNIndex` as a 7-number signature (log area, mean and peak heat, growth, speed, length of history, distance to a road) with what happened next: did that fire come within 400 m of a road in the following 64 days? A hotspot that is still more than 400 m from a road and has at least three analogues from at least two other tracks gets $\hat s = \max(\hat s, p)$, where $p$ is the share of its five nearest analogues that did.

Confidence starts at 1 and is reduced for cloud edges, failed alignment, disagreeing matchers, first sightings, plant overlap and cloudy scenes, then moved by evidence (section 4.7). The weights and thresholds are set by hand; there is no labelled set of road collapses to fit them to. Section 5.5 measures how much the decisions depend on them.

### 4.7 The agent

A LangGraph state graph processes the hotspots in order of risk: observe, plan a tool, run it, rescore, plan again, then decide and act. Five tools gather evidence: earlier passes, a third alignment method around the hotspot, an industrial check (map overlap, footprint shape, peak heat, optionally a vision-language model), Sentinel-2 hot pixels, and a drone request. A planner (fixed policy, or Amazon Bedrock choosing among the offered tools) picks the next tool; it cannot pick an action. The decision rules close roads only for high-risk, high-confidence fires within 250 m of a road, escalate projection-only threats and anything uncertain to a person, exclude industrial heat and log the rest. A guard re-checks every action, refuses anything outside the agent's list, and makes reopening human-only. Memory across passes (track labels, one review per track, closures, reopen proposals) keeps alerts to a handful per season. The full rules and a real trace are in [agent.md](agent.md).

### 4.8 Products and deployment

Each pass writes WebP map layers whose provenance travels inside the file as XMP, written with OpenCV 5's `imencodeWithMetadata`: run, scene, acquisition time, platform, alignment method, WGS84 bounds, value range and units. Each pass also draws a time-lapse frame with OpenCV, and `cv2.imencodeanimation` turns the season into one animated WebP for the dashboard.

On AWS, the USGS Landsat notification topic feeds an SQS queue that a worker on a Graviton instance reads; results go to S3 and DynamoDB, alerts to SNS, agent traces to CloudWatch Logs, and the planner and alert text to Bedrock. One CloudFormation template creates everything ([architecture.md](architecture.md)).

## 5. Evaluation

Sections 5.1 to 5.6 are on the synthetic coalfields, where the truth is known; section 5.7 is on real Landsat scenes, where it is not. All of it is produced by `firewatch eval all` and stored in `reports/`.

### 5.1 Temperature retrieval

Over seven passes, the retrieved temperature differs from the known surface temperature (blurred by the sensor's response) by 0.10 K on average, with a 95th percentile of 0.24 K and a bias of −0.004 K. Against the synthetic Level-2 product, which carries its own noise, the mean error is 0.22 K. This mainly shows that the radiometry inverts cleanly; real Level-1 scenes would add an atmospheric bias of a few kelvin. The real scenes in section 5.7 use the USGS Level-2 temperature instead, which is corrected for the atmosphere.

### 5.2 Hotspot detection

| World | Variant | Precision | Fires found | False hotspots per pass |
| --- | --- | --- | --- | --- |
| Tuned | detrended | 1.00 | 57 of 61 | 0 |
| Tuned | detrended + persistence | 1.00 | 59 of 61 | 0 |
| Tuned | raw temperature | 0.16 | 61 of 61 | 19.4 |
| Held-out | first version: detrended, sharp inputs, water kept | 0.68 | 69 of 92 | 2.0 |
| Held-out | detrended | 1.00 | 69 of 92 | 0 |
| Held-out | detrended + persistence | 1.00 | 79 of 92 | 0 |
| Held-out | raw temperature | 0.18 | 75 of 92 | 20.9 |

A fire counts as present on a pass when at least 6 pixels of its true footprint are cloud-free. The coke plants are detected on every pass and counted separately, since excluding them is the agent's job. Without the albedo/NDVI detrending, the same detector finds nearly every fire but also about 20 false hotspots per pass on sun-heated coal and dumps.

The held-out world did what it was built for. Its first run produced false hotspots on every pass, always in one place: the shore of a pond next to a coal pit. The third row reproduces that version of the detector. Two things were wrong. The albedo and NDVI going into the detrending were sharp 30 m values while the temperature is a 100 m measurement, so every sharp edge between dark coal and a brighter surface left a warm fringe; and open water, about 20 K colder than the land around it, dragged the local background down. Matching the thermal footprint and masking water (section 4.3) removed every false hotspot in both worlds without losing a fire. After that fix the held-out world is no longer untouched, so the next unseen test has to be real scenes.

Persistence adds two fire sightings in the tuned world and ten in the held-out one, with no false ones. Recall on the held-out world is lower (0.86) because its fires are weaker: the misses are early passes where a young fire's peak contrast after the sensor blur is 1.8 to 3.5 K (section 6). Pixel overlap with the true footprints is 0.54 in the tuned world and 0.38 in the held-out one, whose fires have irregular lobes.

### 5.3 Alignment

Five conditions with five synthetic pairs each, shifted up to 2 px and rotated up to 0.15°:

| Condition | ORB | SIFT | DISK | ALIKED + LightGlue |
| --- | --- | --- | --- | --- |
| Geolocation only | 0.125 | **0.010** | 0.019 | 0.016 |
| Haze and low sun | 0.138 | **0.015** | 0.032 | 0.027 |
| Greener vegetation | 0.205 | **0.016** | 0.030 | 0.074 |
| Cross-sensor | 0.251 | **0.033** | 0.070 | **0.033** |
| Dry season, new dumps | failed 5/5 | **0.024** | 0.055 | 1 of 5 aligned |

Median error in pixels. Seconds per pair on two vCPUs: ORB 0.2, SIFT 0.7, ALIKED + LightGlue 3.2, DISK 22.

We expected the learned matchers to win on seasonal change. They did not: in the dry-season pair, where vegetation darkens and soil brightens in the near infrared, ALIKED + LightGlue aligned one pair in five and ORB none, while SIFT aligned all five. ALIKED was trained on ground-level photographs. SIFT is therefore the primary matcher. ALIKED + LightGlue, which is independent of SIFT and as accurate on the cross-sensor pair, checks every pass, and DISK, which aligned every pair in every condition but takes 30 times as long as SIFT, is kept for settling disagreements.

The robust estimator was compared on the same SIFT matches. `USAC_ACCURATE` was within about 0.001 px of plain RANSAC or better in every condition, and better on the cross-sensor pair (0.033 against 0.048 px); MAGSAC++ was never better than RANSAC and worse in three of the five conditions (0.052 px cross-sensor). With 200 true matches buried in random false ones, RANSAC found the transform in 4 of 5 pairs at 95 % outliers, while `USAC_ACCURATE` and MAGSAC++ found all 5. `USAC_ACCURATE` replaced MAGSAC++ as the default.

Over the 19 passes of the full demo, SIFT left 0.06 to 0.12 px of residual on the misregistered passes and cut the shift measured by phase correlation from as much as 2.0 px to under 0.1 px.

### 5.4 Spread projection and precedent

Every pass of a full season in both worlds goes through the pipeline, and each projection is compared with where the matching true fire actually burned over the next 48 days. Two baselines get the same test: today's outline, and a circle grown around it to the same area as the cone. "New ground" is the part of the future footprint that is not burning today.

| World | Fires | Projections | Cone covers | Same-area circle covers | New ground: cone vs circle |
| --- | --- | --- | --- | --- | --- |
| Tuned | moving | 21 | 92 % | 91 % | 87 % vs 85 % |
| Tuned | static | 17 | 100 % | 97 % | none |
| Held-out | moving | 46 | 94 % | 84 % | 92 % vs 78 % |
| Held-out | static | 23 | 87 % | 86 % | 64 % vs 59 % |

The cone earns its shape for fires that move: at equal area, it covers more of the new ground than a circle. It is also too wide. In the held-out world it is 2.5 times the area the fire actually reached, so 56 % of it burns; in the tuned world 1.3 times and 81 %.

On the reach probability, the held-out world's 51 projections that did not already touch a road include 9 fires that did reach one within 48 days. The Brier score is 0.038, against 0.145 for always forecasting the observed rate. In the tuned world none of the 11 such cases reached a road, because fire A is put out first; the cone's false alarms there give a Brier score of 0.093.

The precedent index was tested on 185 track states from both worlds, leaving out one fire at a time. Of the 94 states more than 400 m from a road, 13 came within 400 m in the following 64 days. The precedent gate was met for 62 of them, and on those the share of analogues that reached a road separates the two outcomes with an area under the ROC curve of 0.97 (Brier score 0.063 against 0.171 for the base rate). The worlds were built so that moving fires head for roads, so much of this is learning that motion near a road predicts arrival; real precedent needs past seasons of real tracks.

### 5.5 Agent decisions

36 situations, each run through the real graph, policy, guard and action executor with scripted tool results. The expected decision for each was written from the situation (what a careful safety officer would do), not from the code.

| Measure | Value |
| --- | --- |
| Decisions matching the expected one | 35 of 36 |
| Closures not called for | 0 |
| Fires dismissed that needed a person or a closure | 0 |
| Sent to a person when another decision was expected | 1 |
| Matching without any tool (first measurements only) | 30 of 36 |
| Decisions changed by gathered evidence | 5, all corrections |
| Tool calls per case | 1.64 |
| Safety and memory checks | 10 of 10 |

The corrections are another alignment method restoring confidence (A07: escalate to close; B08: escalate to log) or taking it away (A09, where DISK sides with ALIKED and nothing else confirms the fire: close to escalate), and the industrial check removing a plant (C01, C02: escalate to exclude). The safety checks cover the reopening flow end to end, duplicate proposals, the guard refusing medium-risk and low-confidence closures, and a planner naming a tool that does not exist.

The suite was then run again with one risk constant moved at a time: each weight by 20 %, each threshold by 0.05 and the 400 m proximity scale by 25 %. Thirteen of the 14 changes flip at most one decision. Lowering the confidence needed to close from 0.70 to 0.65 turns three escalations (A06, A09, E03) into closures; those cases sit at a confidence of exactly 0.65 by construction, which is the line the bar is meant to hold. Lowering the high-risk threshold by 0.05, or the spread weight by 20 %, makes the agent close the road for B09, a strong fire 60 m away with no measured spread, where a person was expected to decide. Raising the medium threshold by 0.05, or shrinking the proximity scale to 300 m, sends B03, a slow fire 300 m from a road, to the log instead of to a person. That is the most fragile decision in the set, and the only one where a small change would let a fire past a person.

### 5.6 Run time

Median 7.3 s per full-size pass on two x86 vCPUs: alignment 4.4 s, writing the map layers and the time-lapse frame 1.5 s, assessment and projection 0.34 s, anomaly 0.33 s. The second alignment with ALIKED + LightGlue is about 3.4 s of that, the price of never trusting one matcher. On this build ONNX Runtime is not compiled in, so `ENGINE_ORT` runs at the speed of the new engine, and the classic engine cannot import either learned feature network. `infra/ops.sh bench` runs the same benchmark on the deployed instance (`reports/graviton/bench.md`): a Graviton2 t4g.small with 2 vCPUs and 2 GB of memory, the largest Graviton type an AWS Free plan account may launch.

| | x86, 2 vCPUs | Graviton2 t4g.small |
| --- | --- | --- |
| SIFT alignment, 768 px | 0.65 s | 0.82 s |
| ALIKED + LightGlue alignment, 768 px | 3.34 s | 4.38 s |
| ALIKED forward pass, new engine, 640 px | 1.11 s | 1.16 s |
| Whole pass, median | 7.3 s | 42.0 s |
| of which alignment | 4.4 s | 33.3 s |
| of which loading the scene | 0.23 s | 2.42 s |

The isolated matchers run within 1.3 to 1.4 times the x86 time, and the network itself at the same speed. The whole pass does not: it is six times slower, and the slowdown sits in the stages that hold full-size products in memory. The instance was already using swap before the benchmark started, so memory, not the processor, is the likely cause; a 4 GB or larger instance should close most of the gap, but that has not been measured. DISK had to be left out (`FIREWATCH_BENCH_SKIP=disk`): its 1024 px forward pass took the process to 1.7 GB and the kernel stopped it.

`infra/cool_bench.sh` compares COOL, OpenCV's Graviton-optimised build (the Graviton4 edition, built on OpenCV 5), with the PyPI wheel on one c8g instance: the core OpenCV work of a pass in alternating rounds, with a checksum per kernel to confirm both builds compute the same result, and then the whole pipeline under each. The results go to `reports/cool.md`. This comparison has not been run: the AWS account the project is deployed from is on the Free plan, which does not allow c8g instances.

### 5.7 Real Landsat scenes

23 Landsat 8 and 9 Collection 2 Level-2 scenes of path 140, row 44, from 4 November 2025 to 14 October 2026, were downloaded with `scripts/fetch_real_data.py` (the coalfield window only, about 10 to 13 MB a scene) and processed in date order with `firewatch -c config/real.yaml real`, the first three without the agent to build history. Roads, mines, plants and towns come from OpenStreetMap and become 400 m road sections, truck routes and an industrial mask. Every scene processed end to end.

There is no ground truth for a subsurface fire on a given day, so the check is against the area NRSC mapped. `firewatch eval nrsc` sums the non-industrial hotspot area on each clear pass (cloud at most 20 %):

| Reference | NRSC area | Clear passes | Our median | Range | Ratio |
| --- | --- | --- | --- | --- | --- |
| NRSC 2020, against Nov 2025 to Oct 2026 | 1.89 km² | 20 | 2.90 km² | 2.06–6.67 km² | 1.54 |

A ratio between about 0.5 and 2 means the detector is in the right range; it is not a pixel-level validation, the methods and thresholds differ, and the reference is five years older. Most hotspots sit within 500 m of a mapped mine. The largest area (6.67 km² on 13 April 2026) falls in the pre-monsoon heat; without ground truth there is no telling whether that is more fire or sun-heated ground the detrending did not remove.

Real data needed different settings from the synthetic worlds (`config/real.yaml`): a stricter seed and grow contrast (8 K and 4 K, against 4 K and 1.5 K), a closure distance of 100 m instead of 250 m because roads are packed around the mines, and a 90 m position-noise allowance in tracking, because real outlines change shape between passes and their centres jitter. On real data the agent closes only mine haul roads by itself; public roads always go to a person. Over the season it closed 3 roads and raised 123 decisions for a person, which nobody worked through, so they pile up; a real deployment needs someone assigned to review them.

`firewatch eval firms` checks our hotspots against NASA FIRMS VIIRS and MODIS detections within a day of each pass. It has not been run, because it needs a FIRMS archive download.

## 6. Failure cases

1. **Small and young fires are missed.** In the tuned world, fire A's first two passes cover 8 to 9 pixels of true footprint, and after the sensor blur its peak contrast is 3.6 to 3.8 K, under the 4 K seed threshold. In the held-out world, 13 early sightings, spread over all seven fires, are missed with a peak contrast between 1.8 and 3.5 K. Persistence only helps once a fire has been seen above 2.5 K on the previous pass.
2. **The held-out world found false alarms the tuned world never showed** (section 5.2). They are fixed, but they are a reminder that every threshold here was set on synthetic surfaces.
3. **The learned matcher fails on dry-season change.** ALIKED + LightGlue aligns one dry-season pair in five, ORB none. If SIFT also failed on a real scene, the agent would see failed alignment, lose confidence, and send near-road cases to a person rather than act.
4. **The cone is too wide.** It covers the ground a fire reaches, but in the held-out world 44 % of it never burns. Closures depend on the fire itself being within 250 m, so a wide cone produces reviews, not closures.
5. **Unmapped plants become false alarms.** Without a vision-language model, an industrial site missing from the map goes to a person instead of being excluded (scenario C04).
6. **Found and fixed by the scenario suite.** The first version closed a road for a fire 500 m away on the strength of its projection alone (A04), and ignored a real fire straddling the edge of a coke plant because 30 % overlap was enough to call it industrial (C05). Both rules were changed (section 4.7); the scenarios now pass.
7. **Outlines are approximate.** Pixel overlap with the true footprints is 0.38 to 0.54, so areas from single passes should be read as indicative.

## 7. Limitations

- **Accuracy is measured on synthetic data only.** Sections 5.1 to 5.6 come from the synthetic coalfields. The real scenes in section 5.7 have no ground truth, so on real data we can show the pipeline runs and lands in the range NRSC reported, not its precision or recall.
- **The Level-1 live path has not run against the live service.** The real scenes came from the Level-2 collection on Planetary Computer. The Level-1 reader used by the AWS live mode is tested on files written in the USGS layout (a synthetic pass exported as GeoTIFFs plus `MTL.json` reads back identically through `LandsatSource`), and its STAC items are parsed from fixtures, but the USGS STAC search and requester-pays S3 reads have not been run.
- **No atmospheric correction on Level-1.** Temperatures retrieved from Level-1 scenes will be biased by a few kelvin on humid days; Level-2 scenes use the corrected USGS product. Detection depends on contrast against a local background 1.8 km across, which a smooth atmospheric bias barely changes.
- **Resolution and timing.** Band 10 is 100 m native. Fires under about a hectare are hard to see (failure 1), and the 10:30 overpass mixes in solar heating that the detrending only partly removes.
- **Surface heat is not subsurface extent.** Collapse risk depends on seam depth, mining history and the voids left behind, none of which are modelled. The system flags heat near roads; it does not predict collapses.
- **Hand-set risk model.** Weights and thresholds are judgment, not fitted; section 5.5 shows which decisions depend on them. The projection uses at most eight points of history and a simple transport model, and precedent is only as good as the archive of past tracks behind it.
- **Untested integrations.** The vision-language loop runs in the tests on stand-in networks, but the PaliGemma 2 model itself (several gigabytes) has not been run; the Bedrock planner has been tested only with scripted replies and failure injection, not against the live service; drone requests are records, not flights.
- **One instance.** The deployment is a single Graviton instance without failover.

## 8. Users

Fire Watch is built for three groups. None of them uses it in production yet; this is who it is designed for and what each one does with it.

| Who | What they need | What they do in Fire Watch |
| --- | --- | --- |
| **Mine-safety officers** at the coal company (BCCL's area safety teams) | To know which haul roads and work areas sit over active fire before trucks and people use them, without reading thermal images themselves | Sign in to the dashboard, review each fire with its history and the agent's reasoning, approve or refuse the closures and reviews the agent proposes, and reopen roads once a fire has gone quiet. Only they can reopen a road. |
| **District and road authorities** | Early warning when a fire nears a public road, a railway or homes | Receive alerts, and decide on public roads: on real data the agent never closes a public road by itself, it sends the case to a person. |
| **People who live and work on the coalfield**: residents, truck drivers, coal workers | To know whether the road they are about to use is closed, in plain words | Use the public homepage with no sign-in: the latest photo's date, the list of closed roads, and "Check a place" for the nearest fire to a town, mine or road. |

Researchers and judges are a fourth audience: the help guide's guided tour, the reports and this document are written for them.

The dashboard assumes users who know the coalfield but not remote sensing: every fire is summarised in one plain sentence, the technical numbers are folded away, and decisions waiting for a person are listed first.

## 9. Responsible use

Fire Watch can close roads that people depend on for work and travel, and it can fail to warn about a fire. These are the rules it is built around and the limits anyone deploying it should keep.

**A person stays in charge.**
- The agent may close a road by itself only when the fire is high risk, it is at least 70 % confident, and the fire itself is close to the road; a projection alone never closes a road. On real data it closes only mine haul roads by itself.
- Everything uncertain goes to a person with a recommendation: fires half under cloud, threats seen only in the projection, disagreeing alignments, and plants not on the map.
- Only a person can reopen a road. A guard re-checks every action and refuses anything outside the agent's list (section 4.7), and section 5.5 tests these rules, including what happens when the constants move.

**Every decision can be checked.** Each step the agent takes, the evidence it gathered and the reason for its decision are stored with the pass and shown in the dashboard. Approvals record the dashboard user who made them. Every map layer carries its source scene, time and processing in its own metadata.

**It is not an official notice, and it can be wrong.**
- It sees heat at the surface at about 10:30 every 8 days, not the fire itself. Deep fires, small fires under about a hectare and fires under cloud can be missed (section 6). The absence of a warning does not mean the ground is safe.
- Sun-heated coal and dumps can look like fire, so false alarms reach a person rather than closing roads.
- It flags heat near roads; it does not predict where the ground will collapse.
- The public pages say this in plain words and tell people to follow BCCL and district instructions on the ground.

**Closures have costs.** A closed haul road stops trucks and work, and on the coalfield it can stop income for the people who carry coal. That is why automatic closures are limited to clear cases, why a person reviews the rest, and why the agent proposes reopening as soon as a fire has been quiet for two passes.

**Data and privacy.** Every input is open data: Landsat, Sentinel-2 and OpenStreetMap. At 30 to 100 m per pixel, the imagery cannot identify individuals. Fire Watch stores no personal data about visitors to its public pages (the homepage loads fonts and photos from other sites, which see a visitor's IP address as with any web request); the dashboard stores only the user name that made each decision. Fire Watch must not be repurposed to watch people, for example to find or police informal coal collection; its outputs are about ground temperature near roads, and they should be used only for safety.

**Before a real deployment:**
- Turn off the filled-in demo sign-in (`api.show_demo_login: false`), and replace the shared demo login with one account per person, so that every approval is attributable to a named reviewer.
- Restrict the dashboard to known networks (`ALLOWED_CIDR`).
- Assign someone to work through the review queue. On real data it grew to 123 decisions in a season.
- Re-tune and re-test the thresholds with the safety teams who will use it, since every threshold here was set on synthetic data or by hand.

## 10. Reproducing

```bash
pip install -r requirements-dev.txt && pip install --no-deps -e .
firewatch download-models
pytest
firewatch demo --fresh
firewatch eval all --out reports
```

On real scenes:

```bash
bash get_real_data.sh
firewatch -c config/real.yaml real
firewatch -c config/real.yaml eval nrsc
```

## References

- Agam, N., Kustas, W. P., Anderson, M. C., Li, F., Neale, C. M. U. (2007). A vegetation index based technique for spatial sharpening of thermal imagery. Remote Sensing of Environment 107(4), 545–558.
- Barath, D., Matas, J. (2018). Graph-Cut RANSAC. CVPR.
- Barath, D., Noskova, J., Ivashechkin, M., Matas, J. (2020). MAGSAC++, a fast, reliable and accurate robust estimator. CVPR.
- Bharat Coking Coal Limited. Fire-related status report citing the NRSC studies of 2017, 2020 and October 2021. https://www.bcclweb.in/masterplan/FireDrilling.pdf
- Biswal, S. S., Gorai, A. K. (2019). Detection and delineation of subsurface coal mine fire from spaceborne thermal infrared data in Jharia Coalfield, Dhanbad, India. ISPRS Archives XLII-3/W8, 65–69.
- Kroeger, T., Timofte, R., Dai, D., Van Gool, L. (2016). Fast optical flow using dense inverse search. ECCV.
- Liang, S. (2001). Narrowband to broadband conversions of land surface albedo I: Algorithms. Remote Sensing of Environment 76(2), 213–238.
- Lindenberger, P., Sarlin, P.-E., Pollefeys, M. (2023). LightGlue: Local feature matching at light speed. ICCV.
- Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV 60(2), 91–110.
- Marchese, F., et al. (2019). A multi-channel algorithm for mapping volcanic thermal anomalies by means of Sentinel-2 MSI and Landsat-8 OLI data. Remote Sensing 11(23), 2876.
- Raguram, R., Chum, O., Pollefeys, M., Matas, J., Frahm, J.-M. (2013). USAC: A universal framework for random sample consensus. IEEE TPAMI 35(8), 2022–2038.
- Sobrino, J. A., Jiménez-Muñoz, J. C., Paolini, L. (2004). Land surface temperature retrieval from LANDSAT TM 5. Remote Sensing of Environment 90(4), 434–440.
- Tyszkiewicz, M., Fua, P., Trulls, E. (2020). DISK: Learning local features with policy gradient. NeurIPS.
- U.S. Geological Survey. Landsat 8-9 Collection 2 Level 1 and Level 2 product guides (band 10 rescaling, thermal constants, QA_PIXEL bits, surface temperature scaling).
- Zhao, X., et al. (2023). ALIKED: A lighter keypoint and descriptor extraction network via deformable transformation. IEEE Transactions on Instrumentation and Measurement.
