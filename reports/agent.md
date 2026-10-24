# Agent decisions on scripted scenarios

36 single-pass scenarios run through the real LangGraph agent, decision policy, guard and action executor. Only the tool results are scripted (what older passes, Sentinel-2 and another alignment method would show), so the scenarios isolate the decision logic from image processing. Expected decisions describe what a careful safety officer would want. "Without tools" is the decision the same policy makes on the first measurements, before any evidence gathering.

| Measure | Value |
| --- | --- |
| Decision matches the expected one | 35 of 36 |
| Matches and every secondary check passes (tools, segments, alerts) | 35 of 36 |
| Closures the expected decision did not call for | 0 |
| Fires dismissed that needed a person or a closure | 0 |
| More cautious than expected (sent to a person instead) | 1 |
| Matches without tools (first measurements only) | 30 of 36 |
| Decisions changed by gathered evidence | 5 (5 corrected, 0 made worse) |
| Tool calls per case | 1.64 |
| Safety and memory checks passed | 10 of 10 |

## Scenarios

| ID | Situation | Expected | Agent | Without tools | Tools used | Result |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | Growing fire 120 m from the main road, moving toward it | close | close | close | fetch_highres | pass |
| A02 | Intense, stable fire at the road edge; Sentinel-2 sees nothing | close | close | close | fetch_highres | pass |
| A03 | Fire at the junction of the main road and a haul road | close | close | close | fetch_highres | pass |
| A04 | Fast fire 500 m away; only the 48-day projection reaches the road | escalate | escalate | escalate | fetch_highres | pass |
| A05 | Image alignment failed, a second method recovers it | close | close | close | refine_alignment, fetch_highres | pass |
| A06 | First sighting at the road, alignment failed twice | escalate | escalate | escalate | fetch_history, refine_alignment, fetch_highres | pass |
| A07 | First sighting at the road, second alignment method succeeds | close | close | escalate | fetch_history, refine_alignment, fetch_highres | pass |
| A08 | SIFT and ALIKED + LightGlue disagree; DISK sides with SIFT | close | close | close | refine_alignment, fetch_highres | pass |
| A09 | Matchers disagree, DISK sides with ALIKED, no Sentinel-2 image | escalate | escalate | close | refine_alignment, fetch_highres, request_drone | pass |
| A10 | Matchers disagree, DISK sides with ALIKED, Sentinel-2 confirms the fire | close | close | close | refine_alignment, fetch_highres | pass |
| B01 | New hotspot 150 m from the road, nothing there in earlier passes | escalate | escalate | escalate | fetch_history, fetch_highres | pass |
| B02 | Approaching fire half hidden by cloud, no Sentinel-2 image | escalate | escalate | escalate | fetch_highres, request_drone | pass |
| B03 | Moderate, slow fire 300 m from the road | escalate | escalate | escalate | fetch_highres | pass |
| B04 | Same fire as A01 with automatic closure switched off | escalate | escalate | escalate | fetch_highres | pass |
| B05 | High risk by the main road while only haul roads may be closed | escalate | escalate | escalate | fetch_highres | pass |
| B06 | Weak, stable heat directly under the road | escalate | escalate | escalate | fetch_highres | pass |
| B07 | Faint new hotspot near the road on a cloudy, misaligned pass | escalate | escalate | escalate | fetch_history, refine_alignment, fetch_highres | pass |
| B08 | Same faint hotspot, but the second alignment method succeeds | log | log | escalate | fetch_history, refine_alignment, fetch_highres | pass |
| B09 | Strong, persistent fire 60 m from the road with no measured spread | escalate | escalate | escalate | fetch_highres | pass |
| C01 | Coke-oven battery on the industrial map | exclude | exclude | escalate | check_industrial | pass |
| C02 | Unmapped plant; the vision-language model sees a chimney | exclude | exclude | escalate | check_industrial | pass |
| C03 | Hot rectangular patch; the vision-language model sees no building | escalate | escalate | escalate | check_industrial, fetch_history, fetch_highres | pass |
| C04 | Unmapped plant with no vision-language model available | exclude | escalate | escalate | check_industrial, fetch_history, fetch_highres | conservative |
| C05 | Real fire straddling the edge of a mapped coke plant | close | close | close | check_industrial, fetch_highres | pass |
| D01 | Strong fire 2.5 km from any road | log | log | log | - | pass |
| D02 | Weak, stable fire 1.8 km away | log | log | log | - | pass |
| D03 | New weak hotspot 900 m from the road | log | log | log | fetch_history | pass |
| D04 | Cloud-edge hotspot 1.5 km from the road | log | log | log | - | pass |
| E01 | Fire still hot next to roads the agent closed earlier | close | close | close | fetch_highres | pass |
| E02 | Moderate case with a review already waiting for this track | escalate | escalate | escalate | fetch_history, fetch_highres | pass |
| E03 | Waiting review, and the hotspot has become high risk | escalate | escalate | escalate | fetch_history, refine_alignment, fetch_highres | pass |
| F01 | Language model asks for a tool that does not exist (close_road) | close | close | close | fetch_highres | pass |
| F02 | Language model replies with prose instead of JSON | escalate | escalate | escalate | fetch_history, fetch_highres | pass |
| F03 | Language model picks a different but valid order | escalate | escalate | escalate | fetch_highres, fetch_history | pass |
| F04 | Planner returns an action instead of a tool | close | close | close | fetch_highres | pass |
| F05 | Step budget of two tools on a case that wants four | escalate | escalate | escalate | fetch_history, refine_alignment | pass |

## Failure cases

**C04: Unmapped plant with no vision-language model available.** Expected exclude, the agent chose escalate (medium risk (0.52); a person should review).
Known limitation. Shape and heat alone cannot separate an unmapped plant from a fire, so the agent sends it to a person (a false alarm, not a missed fire).

## Safety and memory checks

| ID | Check | Result | Detail |
| --- | --- | --- | --- |
| S01 | Two quiet passes: the agent proposes reopening | pass | 1 reopen proposal(s) |
| S02 | The road stays closed while the proposal waits | pass | closed: ['M-04'] |
| S03 | A later review does not duplicate the proposal | pass | 1 pending after a second review |
| S04 | The agent cannot reopen a road itself | pass | blocked: reopening a road always needs a human decision |
| S05 | A person's approval reopens it and restores the normal route | pass | closed: []; reroute [{'route': 'West yard to washery', 'before_m': 4000, 'after_m': 4000, 'extra_m': 0, 'blocked': False}, {'route': 'Pit 4 to washery', 'before_m': 3000, 'after_m': 3000, 'extra_m': 0, 'blocked': False}] |
| S06 | One quiet pass is not enough to propose reopening | pass | 0 proposal(s) |
| S07 | Approving a review that recommends closure closes the road, recorded as a human action | pass | closed: ['M-04'] |
| S08 | An automatic closure supersedes the review waiting for that track | pass | decision close, pending {} |
| S09 | The guard refuses a closure for a medium-risk case | pass | actions ['blocked', 'escalate', 'notify'] |
| S10 | The guard refuses a closure below the confidence threshold | pass | actions ['blocked', 'escalate', 'notify'] |

## How much do the constants matter?

Every scenario again, with one risk constant moved at a time: each weight by 20 %, each threshold by 0.05 and the proximity scale by 25 %. A decision that flips under such a small change sits close to a boundary. Across the 14 variations, 6 change nothing, and the most any of them changes is 3 of 36 decisions.

| Change | Decisions that change | Still as expected | Closures not called for | Fires missed | Which |
| --- | --- | --- | --- | --- | --- |
| Heat weight -20% | 0 | 35 of 36 | 0 | 0 | - |
| Heat weight +20% | 1 | 34 of 36 | 0 | 0 | D01 log to escalate |
| Spread weight -20% | 1 | 34 of 36 | 1 | 0 | B09 escalate to close |
| Spread weight +20% | 0 | 35 of 36 | 0 | 0 | - |
| Proximity weight -20% | 1 | 34 of 36 | 0 | 0 | D01 log to escalate |
| Proximity weight +20% | 0 | 35 of 36 | 0 | 0 | - |
| High-risk threshold -0.05 | 1 | 34 of 36 | 1 | 0 | B09 escalate to close |
| High-risk threshold +0.05 | 0 | 35 of 36 | 0 | 0 | - |
| Medium-risk threshold -0.05 | 1 | 34 of 36 | 0 | 0 | D01 log to escalate |
| Medium-risk threshold +0.05 | 1 | 34 of 36 | 0 | 1 | B03 escalate to log |
| Confidence needed to close -0.05 | 3 | 32 of 36 | 3 | 0 | A06 escalate to close; A09 escalate to close; E03 escalate to close |
| Confidence needed to close +0.05 | 0 | 35 of 36 | 0 | 0 | - |
| Proximity scale d0 -25% (300 m) | 1 | 34 of 36 | 0 | 1 | B03 escalate to log |
| Proximity scale d0 +25% (500 m) | 0 | 35 of 36 | 0 | 0 | - |

## One trace, step by step (A07)

A first sighting 10 m from the road on a pass where image alignment failed. Each tool result changes the confidence, and the confidence decides whether the agent may act by itself.

| # | Step | Detail |
| --- | --- | --- |
| 2 | plan: fetch_history | first sighting of a 12.0 K anomaly 10 m from M-04; check older passes to see if it is new |
| 3 | fetch_history: verdict new | confidence 0.55 to 0.55, risk 0.69 |
| 4 | plan: refine_alignment | alignment failed on this pass; check the hotspot's window with a third matcher |
| 5 | refine_alignment: ok True | confidence 0.55 to 0.70, risk 0.69 |
| 6 | plan: fetch_highres | risk 0.69 (high) 10 m from M-04; look for hot pixels in sharper Sentinel-2 shortwave infrared |
| 7 | fetch_highres: confirmed True | confidence 0.70 to 0.80, risk 0.69 |
| 8 | plan: enough evidence | no further tool would change the decision |
| 9 | decide: close | risk 0.69 and confidence 0.80 clear the bars of 0.62 and 0.7; the fire is within 250 m of M-03, M-04 and M-05, and with its 48-day spread M-03, M-04 and M-05 are at risk |
| 10 | act | agent close_road M-03 M-04 M-05, agent notify safety-officer |
