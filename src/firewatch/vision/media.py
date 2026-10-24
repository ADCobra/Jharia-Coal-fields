from __future__ import annotations

import html
import re

import cv2
import numpy as np

NAMESPACE = "urn:jharia-fire-watch:provenance:1"
_FIELD = re.compile(r'fw:(\w+)="([^"]*)"')


def xmp_packet(fields: dict) -> bytes:
    attrs = " ".join(f'fw:{key}="{html.escape(str(value), quote=True)}"' for key, value in fields.items() if value is not None)
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:fw="{NAMESPACE}" {attrs}/>'
        '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    ).encode("utf-8")


def parse_xmp(packet: bytes) -> dict:
    text = packet.decode("utf-8", "replace")
    return {key: html.unescape(value) for key, value in _FIELD.findall(text)}


def encode_image(ext: str, image: np.ndarray, params: list[int] | None = None, provenance: dict | None = None) -> bytes:
    params = list(params or [])
    if provenance and hasattr(cv2, "imencodeWithMetadata"):
        ok, buf = cv2.imencodeWithMetadata(ext, image, [cv2.IMAGE_METADATA_XMP], [np.frombuffer(xmp_packet(provenance), np.uint8)], params)
    else:
        ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"{ext} encoding failed")
    return np.asarray(buf).tobytes()


def read_provenance(data: bytes) -> dict:
    if not data or not hasattr(cv2, "imdecodeWithMetadata"):
        return {}
    _, kinds, payloads = cv2.imdecodeWithMetadata(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    for kind, payload in zip(np.asarray(kinds).ravel().tolist(), payloads):
        if int(kind) == cv2.IMAGE_METADATA_XMP:
            return parse_xmp(np.asarray(payload, np.uint8).tobytes())
    return {}


def encode_animation(frames: list[np.ndarray], durations_ms: list[int], ext: str = ".webp", quality: int = 82, loop: int = 0) -> bytes:
    if not frames:
        raise ValueError("no frames")
    if not hasattr(cv2, "imencodeanimation"):
        raise RuntimeError("this OpenCV build cannot encode animations")
    animation = cv2.Animation()
    animation.frames = [np.ascontiguousarray(f) for f in frames]
    animation.durations = [int(d) for d in durations_ms]
    animation.loop_count = loop
    params = [cv2.IMWRITE_WEBP_QUALITY, quality] if ext == ".webp" else []
    ok, buf = cv2.imencodeanimation(ext, animation, params)
    if not ok:
        raise RuntimeError(f"{ext} animation encoding failed")
    return np.asarray(buf).tobytes()


def decode_animation(data: bytes) -> list[np.ndarray]:
    ok, animation = cv2.imdecodeanimation(np.frombuffer(data, np.uint8))
    return list(animation.frames) if ok else []
