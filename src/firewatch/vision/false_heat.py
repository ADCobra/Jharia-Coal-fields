from __future__ import annotations

from firewatch.models import Hotspot


def industrial_assessment(hotspot: Hotspot, persistence: float | None = None) -> dict:
    reasons: list[str] = []
    score = 0.0
    overlap = hotspot.industrial_overlap
    if overlap >= 0.7:
        score += 0.6
        reasons.append(f"{overlap:.0%} of the hotspot lies inside a mapped industrial site")
    elif overlap >= 0.3:
        score += 0.4
        reasons.append(f"{overlap:.0%} of the hotspot overlaps a mapped industrial site")
    regular = hotspot.rectangularity >= 0.8 and hotspot.solidity >= 0.9
    if regular:
        score += 0.2
        reasons.append("sharp, rectangular footprint typical of a building")
    if hotspot.max_anomaly_k >= 20.0:
        score += 0.15
        reasons.append(f"peak {hotspot.max_anomaly_k:.0f} K above its surroundings, typical of a furnace or coke oven")
    if persistence is not None and persistence >= 0.99 and regular:
        score += 0.1
        reasons.append("present in every past pass with the same shape")
    score = min(1.0, score)
    verdict = "industrial" if score >= 0.6 else ("uncertain" if score >= 0.3 else "fire")
    return {"score": round(score, 2), "verdict": verdict, "reasons": reasons}
