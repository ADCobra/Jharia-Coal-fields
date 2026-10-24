from __future__ import annotations

from dataclasses import dataclass, field

from firewatch.config import Settings
from firewatch.models import ActionRecord, Hotspot, Kinematics, Projection, RiskAssessment, SpreadEstimate, Track
from firewatch.risk.scoring import ConfidenceInputs


@dataclass
class Case:
    hotspot: Hotspot
    track: Track | None
    kin: Kinematics | None
    spread: SpreadEstimate | None
    projection: Projection | None
    risk: RiskAssessment
    conf_inputs: ConfidenceInputs
    evidence: dict = field(default_factory=dict)
    tried: list[str] = field(default_factory=list)
    steps: int = 0
    decision: str | None = None
    decision_reason: str = ""
    segments: list[str] = field(default_factory=list)
    excluded: bool = False
    actions: list[ActionRecord] = field(default_factory=list)

    @property
    def case_id(self) -> str:
        return self.hotspot.hotspot_id

    def flags(self, settings: Settings) -> dict[str, bool]:
        r = self.risk
        d0 = settings.risk.d0_m
        near = (r.distance_m is not None and r.distance_m <= 2 * d0) or (r.projected_distance_m is not None and r.projected_distance_m <= d0)
        history = self.evidence.get("history")
        new = (self.kin is None or self.kin.n_points <= 1) and not (history and history.get("persistent"))
        return {
            "new": bool(new and history is None),
            "cloud_edge": self.conf_inputs.cloud_ring_fraction >= settings.hotspots.cloud_edge_fraction,
            "low_alignment": (not self.conf_inputs.alignment_ok) or (self.conf_inputs.alignment_residual_px or 0.0) > settings.alignment.max_residual_px or (self.conf_inputs.alignment_agreement_px or 0.0) > settings.alignment.max_disagreement_px,
            "industrial_candidate": self.hotspot.industrial_overlap >= 0.3 or (self.hotspot.rectangularity >= 0.8 and self.hotspot.solidity >= 0.9 and self.hotspot.max_anomaly_k >= 20.0),
            "near_road": bool(near),
            "approaching": r.approach_m_per_day > 0.5,
        }

    def summary(self, settings: Settings) -> dict:
        h, r = self.hotspot, self.risk
        out = {
            "hotspot_id": h.hotspot_id,
            "track_id": h.track_id,
            "area_m2": round(h.area_m2),
            "mean_anomaly_k": round(h.mean_anomaly_k, 2),
            "max_anomaly_k": round(h.max_anomaly_k, 2),
            "risk": r.score,
            "level": r.level,
            "components": r.components,
            "distance_m": r.distance_m,
            "projected_distance_m": r.projected_distance_m,
            "nearest_segment": r.nearest_segment,
            "approach_m_per_day": r.approach_m_per_day,
            "confidence": r.confidence,
            "confidence_notes": r.confidence_notes,
            "flags": self.flags(settings),
            "history_points": self.kin.n_points if self.kin else 1,
        }
        if self.kin and self.kin.n_points >= 2:
            out["growth_frac_per_30d"] = round(self.kin.growth_frac_per_30d, 3)
            out["speed_m_per_day"] = round(self.kin.speed_m_per_day, 2)
            out["bearing_deg"] = round(self.kin.bearing_deg)
        if self.projection:
            out["reach_probability"] = self.projection.reach_probability
            out["median_days_to_road"] = self.projection.median_days_to_road
        if self.evidence:
            out["evidence"] = dict(self.evidence)
        return out
