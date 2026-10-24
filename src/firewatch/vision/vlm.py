from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from firewatch.config import VLMConfig
from firewatch.vision.cvcompat import dnn_engine

log = logging.getLogger(__name__)


@dataclass
class VLMAnswer:
    available: bool
    answer: str
    verdict: str
    seconds: float
    model: str
    note: str = ""


class VisionLanguageChecker(Protocol):
    available: bool
    model: str

    def ask(self, image_bgr: np.ndarray, question: str) -> VLMAnswer: ...


class NullVLM:
    available = False
    model = "none"

    def __init__(self, note: str = "vision-language model not configured"):
        self.note = note

    def ask(self, image_bgr: np.ndarray, question: str) -> VLMAnswer:
        return VLMAnswer(False, "", "unknown", 0.0, self.model, self.note)


def _verdict(answer: str) -> str:
    text = answer.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "unknown"


class PaliGemmaVLM:
    eos_id = 1

    def __init__(self, tokenizer, siglip, embed, gemma, max_new_tokens: int = 12, model: str = "paligemma2-3b-pt-224 (OpenCV DNN)"):
        self.tokenizer = tokenizer
        self.siglip = siglip
        self.embed = embed
        self.gemma = gemma
        self.max_new_tokens = max_new_tokens
        self.model = model
        self.available = True

    @classmethod
    def from_config(cls, cfg: VLMConfig, model: str = "paligemma2-3b-pt-224 (OpenCV DNN)") -> "PaliGemmaVLM":
        engine = dnn_engine("new")
        return cls(
            cv2.dnn.Tokenizer.load(cfg.tokenizer_config),
            cv2.dnn.readNetFromONNX(cfg.siglip_model, engine),
            cv2.dnn.readNetFromONNX(cfg.embedding_model, engine),
            cv2.dnn.readNetFromONNX(cfg.gemma_model, engine),
            cfg.max_new_tokens,
            model,
        )

    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose(2, 0, 1)[np.newaxis]

    def generate(self, pixel_values: np.ndarray, prompt: str) -> list[int]:
        tokens = list(self.tokenizer.encode(prompt))
        self.siglip.setInput(pixel_values, "pixel_values")
        image_features = self.siglip.forward()
        self.embed.setInput(np.array([tokens], dtype=np.int64), "input_ids")
        text_embeds = self.embed.forward()
        inputs = np.concatenate([image_features, text_embeds], axis=1)
        generated: list[int] = []
        for _ in range(self.max_new_tokens):
            self.gemma.setInput(inputs, "inputs_embeds")
            logits = self.gemma.forward()
            new_id = int(np.argmax(logits[0, -1, :]))
            if new_id == self.eos_id:
                break
            generated.append(new_id)
            self.embed.setInput(np.array([[new_id]], dtype=np.int64), "input_ids")
            inputs = np.concatenate([inputs, self.embed.forward()], axis=1)
        return generated

    def ask(self, image_bgr: np.ndarray, question: str) -> VLMAnswer:
        start = time.perf_counter()
        generated = self.generate(self.preprocess(image_bgr), f"answer en {question}\n")
        answer = self.tokenizer.decode(generated).strip()
        return VLMAnswer(True, answer, _verdict(answer), time.perf_counter() - start, self.model)


def build_vlm(cfg: VLMConfig) -> VisionLanguageChecker:
    if not cfg.enabled:
        return NullVLM()
    paths = [cfg.siglip_model, cfg.embedding_model, cfg.gemma_model, cfg.tokenizer_config]
    missing = [p for p in paths if not p or not Path(p).exists()]
    if missing:
        return NullVLM(f"vision-language model files missing: {missing}")
    try:
        return PaliGemmaVLM.from_config(cfg)
    except cv2.error as exc:
        log.warning("could not load the vision-language model: %s", exc)
        return NullVLM(f"vision-language model failed to load: {exc}"[:300])
