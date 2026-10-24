import json
import string

import cv2
import numpy as np
import pytest

from firewatch.config import VLMConfig
from firewatch.vision.vlm import NullVLM, PaliGemmaVLM, build_vlm

onnx = pytest.importorskip("onnx")
helper, numpy_helper, TensorProto = onnx.helper, onnx.numpy_helper, onnx.TensorProto

pytestmark = pytest.mark.skipif(not hasattr(cv2.dnn, "Tokenizer"), reason="this OpenCV build has no cv2.dnn.Tokenizer")

DIM = 8


def _tokenizer(folder):
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for ch in "▁" + string.ascii_lowercase + string.ascii_uppercase + "?.,'":
        vocab[ch] = len(vocab)
    for piece in ("ye", "yes", "no"):
        vocab[piece] = len(vocab)
    model = {"type": "BPE", "byte_fallback": True, "fuse_unk": True, "unk_token": "<unk>", "vocab": vocab, "merges": ["y e", "ye s", "n o"]}
    spec = {"version": "1.0", "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, "pre_tokenizer": None, "model": model,
            "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]}}
    (folder / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False))
    (folder / "config.json").write_text(json.dumps({"method": "Gemma"}))
    return folder / "config.json", vocab


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _networks(folder, vocab, answer):
    v = len(vocab)
    newline, reply = vocab["<0x0A>"], vocab[answer]
    table = np.zeros((v, DIM), np.float32)
    table[newline, 0] = 1.0
    table[reply, 1] = 1.0
    head = np.zeros((DIM, v), np.float32)
    head[0, reply] = 1.0
    head[1, vocab["<eos>"]] = 1.0
    conv = numpy_helper.from_array(np.zeros((DIM, 3, 56, 56), np.float32), "w")
    shape = numpy_helper.from_array(np.array([1, DIM, 16], np.int64), "shape")
    siglip = helper.make_graph(
        [helper.make_node("Conv", ["pixel_values", "w"], ["c"], strides=[56, 56]), helper.make_node("Reshape", ["c", "shape"], ["r"]), helper.make_node("Transpose", ["r"], ["image_features"], perm=[0, 2, 1])],
        "siglip_stand_in",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, [1, 16, DIM])],
        [conv, shape],
    )
    embed = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["inputs_embeds"], axis=0)],
        "embedding_stand_in",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "tokens"])],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, "tokens", DIM])],
        [numpy_helper.from_array(table, "table")],
    )
    gemma = helper.make_graph(
        [helper.make_node("MatMul", ["inputs_embeds", "head"], ["logits"])],
        "gemma_stand_in",
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, "length", DIM])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "length", v])],
        [numpy_helper.from_array(head, "head")],
    )
    return _save(siglip, folder / "siglip.onnx"), _save(embed, folder / "embedding.onnx"), _save(gemma, folder / "gemma.onnx")


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_paligemma_loop_runs_on_the_new_engine_with_stand_in_networks(tmp_path, answer):
    tokenizer, vocab = _tokenizer(tmp_path)
    siglip, embedding, gemma = _networks(tmp_path, vocab, answer)
    cfg = VLMConfig(enabled=True, siglip_model=siglip, embedding_model=embedding, gemma_model=gemma, tokenizer_config=str(tokenizer), max_new_tokens=6)
    vlm = build_vlm(cfg)
    assert isinstance(vlm, PaliGemmaVLM) and vlm.available
    image = np.full((180, 240, 3), 90, np.uint8)
    assert vlm.generate(vlm.preprocess(image), "answer en Is this a coal fire?\n") == [vocab[answer]]
    reply = vlm.ask(image, "Is this a coal fire?")
    assert reply.available and reply.answer == answer and reply.verdict == answer


def test_missing_model_files_fall_back_to_no_model(tmp_path):
    cfg = VLMConfig(enabled=True, siglip_model=str(tmp_path / "missing.onnx"), embedding_model=None, gemma_model=None, tokenizer_config=None, max_new_tokens=6)
    vlm = build_vlm(cfg)
    assert isinstance(vlm, NullVLM) and not vlm.ask(np.zeros((8, 8, 3), np.uint8), "?").available
