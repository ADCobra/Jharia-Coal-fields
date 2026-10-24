from __future__ import annotations

import math
from dataclasses import dataclass, field

from firewatch.config import AlignmentConfig, HotspotConfig, RiskConfig
from firewatch.models import AlignmentResult, Hotspot, Kinematics, Projection, RiskAssessment
from firewatch.risk.roads import RoadNetwork, hotspot_geometry
from firewatch.util import clamp


@dataclass
class ConfidenceInputs:
    cloud_ring_fraction: float
    alignment_ok: bool
    alignment_residual_px: float | None
    track_points: int
    industrial_overlap: float
    scene_cloud_fraction: float
    evidence: dict = field(default_factory=dict)
    alignment_agreement_px: float | None = None


def confidence(inputs: ConfidenceInputs, hot: HotspotConfig, align: AlignmentConfig) -> tuple[float, list[str]]:
    score = 1.0
    notes: list[str] = []
    if inputs.cloud_ring_fraction >= hot.cloud_edge_fraction:
        score -= 0.35
        notes.append(f"partly cloud-covered ({inputs.cloud_ring_fraction:.0%} of its edge): extent unknown")
    if not inputs.alignment_ok:
        score -= 0.3
        notes.append("image alignment failed for this pass")
    elif inputs.alignment_residual_px is not None and inputs.alignment_residual_px > align.max_residual_px:
        score -= 0.15
        notes.append(f"alignment residual {inputs.alignment_residual_px:.2f} px is above {align.max_residual_px} px")
    if inputs.alignment_ok and inputs.alignment_agreement_px is not None and inputs.alignment_agreement_px > align.max_disagreement_px:
        score -= 0.25
        notes.append(f"the two matchers disagree by {inputs.alignment_agreement_px:.2f} px")
    if inputs.track_points <= 1:
        score -= 0.15
        notes.append("first sighting, no history yet")
    if inputs.industrial_overlap >= 0.3:
        score -= 0.3
        notes.append("overlaps a mapped industrial site")
    if inputs.scene_cloud_fraction > 0.5:
        score -= 0.1
        notes.append(f"scene is {inputs.scene_cloud_fraction:.0%} cloud")
    ev = inputs.evidence
    if ev.get("history", {}).get("persistent"):
        score += 0.15
        notes.append("seen in earlier passes")
    if ev.get("highres", {}).get("confirmed"):
        score += 0.1
        notes.append("Sentinel-2 shortwave infrared shows hot pixels")
    industrial = ev.get("industrial", {})
    if industrial.get("verdict") == "fire":
        score += 0.15 if inputs.industrial_overlap >= 0.3 else 0.05
        notes.append("industrial check says fire, not a plant")
    realign = ev.get("realign", {})
    if realign.get("ok"):
        score += 0.15
        notes.append(f"a third matcher ({realign.get('method', 'another method')}) agrees with the alignment used")
    elif realign.get("agreement_px") is not None:
        score -= 0.1
        notes.append(f"a third matcher ({realign.get('method', 'another method')}) disagrees with the alignment used by {realign['agreement_px']:.2f} px")
    return clamp(score), notes


def assess(
    hotspot: Hotspot,
    kin: Kinematics | None,
    projection: Projection | None,
    roads: RoadNetwork,
    cfg: RiskConfig,
    conf: float,
    notes: list[str],
    precedent: dict | None = None,
) -> RiskAssessment:
    geom = hotspot_geometry(hotspot.polygon, hotspot.centroid)
    segment, distance = roads.nearest(geom)
    projected = projection.projected_distance_m if projection else None
    effective = distance if projected is None else min(distance if distance is not None else projected, projected)
    approach = 0.0
    if segment is not None and kin is not None and kin.n_points >= 2:
        ux, uy = roads.approach_direction(segment, hotspot.centroid)
        approach = kin.vx_m_per_day * ux + kin.vy_m_per_day * uy
    ref = cfg.anomaly_ref_k
    a_hat = clamp(0.5 * hotspot.mean_anomaly_k / ref + 0.5 * hotspot.max_anomaly_k / (2 * ref))
    growth = kin.growth_frac_per_30d if kin and kin.n_points >= 2 else 0.0
    s_hat = clamp(max(growth / cfg.growth_ref_frac_per_30d, approach / cfg.approach_ref_m_per_day, 0.0))
    p_hat = math.exp(-effective / cfg.d0_m) if effective is not None else 0.0
    components = {"anomaly": round(a_hat, 4), "spread": round(s_hat, 4), "proximity": round(p_hat, 4)}
    share = precedent_share(precedent, distance, cfg)
    if share is not None:
        components["precedent"] = round(share, 4)
        components["spread"] = round(max(s_hat, share), 4)
    total_w = sum(cfg.weights.values()) or 1.0
    score = sum(cfg.weights.get(k, 0.0) * v for k, v in components.items()) / total_w
    level = "high" if score >= cfg.high else "medium" if score >= cfg.medium else "low"
    return RiskAssessment(
        hotspot_id=hotspot.hotspot_id,
        track_id=hotspot.track_id,
        score=round(score, 4),
        level=level,
        components=components,
        distance_m=None if distance is None else round(distance, 1),
        projected_distance_m=None if projected is None else round(projected, 1),
        nearest_segment=segment,
        approach_m_per_day=round(approach, 3),
        confidence=round(conf, 3),
        confidence_notes=list(notes),
    )


def precedent_share(precedent: dict | None, distance_m: float | None, cfg: RiskConfig) -> float | None:
    if not precedent or cfg.precedent_min_analogues <= 0:
        return None
    if precedent.get("n", 0) < cfg.precedent_min_analogues or precedent.get("tracks", 0) < cfg.precedent_min_tracks or precedent.get("reached_road_share") is None:
        return None
    if distance_m is None or distance_m <= cfg.d0_m:
        return None
    return float(precedent["reached_road_share"])


def alignment_inputs(result: AlignmentResult) -> tuple[bool, float | None]:
    return result.ok, result.residual_px
