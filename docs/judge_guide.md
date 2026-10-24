# Guide for judges

Ten minutes from clone to a working dashboard, then where each competition requirement is met.

## Run it

Python 3.12 or newer (3.13 is what CI uses). No GPU, no AWS account needed. For real Landsat data instead of the synthetic season, run `bash get_real_data.sh` and then `bash run_real.sh` (see the README).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && pip install --no-deps -e .
firewatch download-models          # ALIKED, LightGlue, DISK ONNX files, checked by SHA-1 (about 56 MB)
pytest                             # 115 tests, about two minutes
firewatch demo --fresh             # 6 archive passes, then 10 with the agent, about two minutes
firewatch serve                    # http://localhost:8000 (homepage); dashboard at /dashboard, user demo, password demo
```

The deployed instance is at the URL given in the submission form, behind the same login.

## What to look at in the dashboard

The homepage at `/` needs no login; "I'm judging or exploring the project" opens the guided tour in `/help`, and "Open dashboard" goes to `/dashboard` (user demo, password demo).

1. The sentence under the date says what happened on that photo. The four steps below it are the pipeline and a summary at once: satellite photo, hot ground found, spread predicted, roads protected. Click "Hot ground found" to see the heat layer, and step through the dates with the arrows, the arrow keys, the dots along the bottom or Play.
2. The map is an aerial view of the coalfield: simulated for the synthetic data (terraced pits, dumps, villages, the coke-oven battery), and a real Landsat true-colour photo for real data. Coal trucks run each mine's route to the washery. When the agent closes a road, barriers go up and the trucks on that route take a detour or wait at the mine. Roads and trucks are drawn as they stood on the selected date.
3. The side panel lists decisions waiting for a person first, then every fire on that date in one sentence. Click a fire for a plain summary, its size and heat, a chart of its size over time and each step the agent took; "Technical details" holds the risk score, confidence, spread probability and the agent's own reasoning. Image quality for the photo (alignment method, residual, second matcher) is under "Photo quality".
4. On **5 November 2025** open Fire 6, beside haul road H2. It is high risk only because of its 48-day spread (the fire is 436 m away, the projection 247 m), so the agent asked a person. If you step through the season with "Process next photo" instead (start with `firewatch demo --fresh --passes 1`), the decision appears at the top of the panel on that date and you can approve the closure yourself.
5. On **23 December 2025** the same fire is 218 m from the road. The agent closed H2-03 to H2-05 itself; the barriers and the waiting trucks show it, and "Roads and trucks" lists the closures and affected routes.
6. On **4 October 2025** open "Factory heat". Its steps show the industrial check running before any Sentinel-2 check, because of its rectangular OpenCV contour and its overlap with the mapped plant.
7. After the 25 February 2026 photo, the agent proposes reopening H2. Approve it in the decision card; the road turns back to open and the trucks return to their normal route. Approving is the only way it reopens.

Every number on screen comes from `GET /api/...`, for example `/api/runs/<run>/trace?hotspot=<id>` for the raw trace, or `/api/runs/<run>/layers/lst/provenance` for the XMP metadata inside a map layer.

## Where each requirement is met

| Requirement | Where |
| --- | --- |
| OpenCV 5 does substantive analysis | After the temperature retrieval, OpenCV does the image work in every stage: CLAHE, SIFT, ALIKED + LightGlue cross-checking every pass, DISK as the tie-breaker, `USAC_ACCURATE`, phase correlation, masked Gaussian and box filters, connected components, contours, convex hull, min-area rectangles, DIS optical flow, the ANN index for precedent, XMP metadata on every layer and the animated season time-lapse. [opencv5.md](opencv5.md) maps each call to its file and benchmarks the DNN engines. |
| A meaningful AWS component | S3, DynamoDB, SNS, SQS fed by the USGS Landsat topic, Bedrock for planning and alert text, CloudWatch Logs and Metrics, Secrets Manager, ECR, Graviton EC2. [infra/cloudformation.yaml](../infra/cloudformation.yaml), [architecture.md](architecture.md). |
| Perception → decision → action loop | [agent/graph.py](../src/firewatch/agent/graph.py), [agent.md](agent.md), diagram [img/agent_loop.svg](img/agent_loop.svg) |
| A trace where OpenCV output changes the next tool call | The coke-oven trace in [agent.md](agent.md#traces); the A07 trace in [reports/agent.md](../reports/agent.md), where alignment confidence drives each step; A08 and A09, where two matchers disagree and a third settles it |
| Architecture diagram | [img/architecture.svg](img/architecture.svg) |
| Technical report | [technical_report.md](technical_report.md) |
| Pinned dependencies, build, deploy and test instructions | `requirements.txt`, `requirements-dev.txt`, `Dockerfile`, `Makefile`, `infra/deploy.sh`, `.github/workflows/ci.yml`, [README](../README.md) |
| Evaluation evidence, including failure cases | [reports/](../reports/README.md): temperature retrieval; detection on a tuned and a held-out world with ablations; alignment against four matchers and three robust estimators; the spread cone and precedent against where fires actually went; 38 agent scenarios, 10 safety checks and a sensitivity sweep of the risk constants; run time and DNN engines. Each report lists its failures. |

## What is not proven yet

The real-data path now runs end to end on 23 Landsat scenes (4 Nov 2025 to 14 Oct 2026). There is no ground truth for those fires; the only outside check is NRSC's mapped area, and Fire Watch's median of 2.9 km² against 1.89 km² for 2020 is in range rather than validated ([reports/nrsc.md](../reports/nrsc.md)). FIRMS has not been compared, and Sentinel-2 checks could not reach Earth Search from the build machine. The vision-language loop is tested with stand-in networks, but the PaliGemma 2 model itself has not been run ([opencv5.md](opencv5.md)).
