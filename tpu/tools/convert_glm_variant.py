"""Prepare the Huihui GLM-5.3-Flash weight overlay on a CPU.

The base files are from rahim3/glm53-flash-serve-v3. Install numpy and gguf
for this offline step. The resulting files keep the exact array layout used
by the existing TPU exports. Only the matching weight ranges are downloaded.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import zipfile

from gguf import GGMLQuantizationType, dequantize
import numpy as np

from apply_glm_overlay import LAYERS, WEIGHTS, fingerprint, read_overlay

sys.path.insert(0, str(Path(__file__).parent / "upstream/glm53-flash/engine/glm53"))
from gguf_reader import GGUFShard


BASE_GGUF = "unsloth/GLM-5.3-Flash-GGUF/resolve/a38483c"
VARIANT_GGUF = "huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF/resolve/e823ec1"
TENSORS = {"attn_output": "attn/o", "ffn_down_shexp": "mlp/shared/down"}


def fetch_bytes(url, offset, length):
    request = Request(
        f"{url}?download=true&span={offset}",
        headers={"Range": f"bytes={offset}-{offset + length - 1}"},
    )
    with urlopen(request, timeout=90) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("O servidor não devolveu o trecho solicitado.")
        payload = response.read(length + 1)
    if len(payload) != length:
        raise ValueError("O download do tensor veio incompleto.")
    return payload


class GGUFRange:
    # The non-initial split files have short headers. The first file only has metadata.
    size = 2 << 20

    def __init__(self, repo, part):
        self.url = f"https://huggingface.co/{repo}/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-{part:05d}-of-00004.gguf"

    def pread(self, offset, length):
        return fetch_bytes(self.url, offset, length)


def catalog(repo):
    result = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for shard in pool.map(lambda part: GGUFShard(GGUFRange(repo, part)), (2, 3, 4)):
            for name in shard.tensors:
                if name in result:
                    raise ValueError(f"Tensor repetido no checkpoint: {name}")
                result[name] = shard
    return result


def matrix(shard, name, start, rows):
    info = shard.tensors[name]
    width, height = info["dims"]
    if info["type"] != "Q6_K" or width % 256 or start + rows > height:
        raise ValueError(f"Formato inesperado: {name}")
    row_bytes = width // 256 * 210
    raw = shard.read_tensor(name, start * row_bytes, rows * row_bytes)
    return dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_bytes), GGMLQuantizationType.Q6_K).T


def convert_matrix(archive, destination, name, original, variant, pool):
    q, s = archive[destination + "/q"], archive[destination + "/s"]
    info = original.tensors[name]
    other = variant.tensors[name]
    if (info["dims"], info["type"]) != (other["dims"], other["type"]):
        raise ValueError(f"As duas fontes discordam sobre o formato de {name}.")
    if tuple(info["dims"]) != q.shape or q.dtype != np.int8 or s.dtype != np.float32:
        raise ValueError(f"A forma ou o tipo em {destination} difere da base pré-compilada.")
    if s.shape not in ((q.shape[0], 1), (1, q.shape[1])) or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError(f"Escalas incompatíveis para {destination}.")

    def paired(start, count):
        first = pool.submit(matrix, original, name, start, count)
        second = pool.submit(matrix, variant, name, start, count)
        return first.result(), second.result()

    reference, modified = paired(0, q.shape[1])
    # Check the layout and the role of the original GGUF before applying any delta.
    original_sample = reference[:, :32]
    scale = s if s.shape[1] == 1 else s[:, :32]
    base_sample = q[:, :32].astype(np.float32) * scale
    relative = np.linalg.norm(original_sample - base_sample) / np.linalg.norm(base_sample)
    if not np.isfinite(relative) or relative > 0.12:
        raise ValueError(f"A matriz {name} não corresponde à base (erro relativo {relative:.3f}).")

    modified -= reference
    del reference, original_sample
    if not np.isfinite(modified).all() or not np.any(modified):
        raise ValueError(f"Delta inválido ou vazio: {name}")
    modified += q * s
    axis = 1 if s.shape[1] == 1 else 0
    scales = np.maximum(np.max(np.abs(modified), axis=axis, keepdims=True), np.float32(1e-12)) / np.float32(127)
    np.divide(modified, scales, out=modified)
    np.rint(modified, out=modified)
    q_new = np.clip(modified, -127, 127).astype(np.int8)
    assert q_new.shape == q.shape and scales.shape == s.shape and scales.dtype == s.dtype
    print(f"{name}: {q_new.shape}, erro relativo da base {relative:.4f}", flush=True)
    return {destination + "/q": q_new, destination + "/s": scales}


def prepare(base_dir, output_dir, layers):
    for layer in layers:
        if not (base_dir / f"layer_{layer:02d}.npz").is_file():
            raise FileNotFoundError(f"Falta layer_{layer:02d}.npz de rahim3/glm53-flash-serve-v3.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        base_index = pool.submit(catalog, BASE_GGUF)
        variant_index = pool.submit(catalog, VARIANT_GGUF)
        original, variant = base_index.result(), variant_index.result()
        for layer in layers:
            name = f"layer_{layer:02d}.npz"
            converted = {}
            with np.load(base_dir / name, allow_pickle=False) as archive:
                original_weights = {key: archive[key] for key in WEIGHTS}
                for tensor, destination in TENSORS.items():
                    key = f"blk.{layer}.{tensor}.weight"
                    converted.update(convert_matrix(original_weights, destination, key, original[key], variant[key], pool))
                converted["__base__"] = np.frombuffer(fingerprint(original_weights), dtype=np.uint8)
            temporary = output_dir / (name + ".part")
            try:
                with temporary.open("wb") as file:
                    np.savez(file, **converted)
                read_overlay(original_weights, temporary)
                temporary.replace(output_dir / name)
            finally:
                temporary.unlink(missing_ok=True)


def bundle(output_dir, archive_file):
    names = [f"layer_{layer:02d}.npz" for layer in LAYERS]
    metadata = {
        "variant": "glm53-flash-abliterated",
        "base": "rahim3/glm53-flash-serve-v3/1",
        "original_gguf": BASE_GGUF,
        "variant_gguf": VARIANT_GGUF,
        "layers": LAYERS,
    }
    temporary = archive_file.with_suffix(".part")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in names:
                archive.write(output_dir / name, name)
            archive.writestr("weights.json", json.dumps(metadata, indent=2))
            license_file = Path(__file__).parent / "upstream/LICENSE"
            archive.write(license_file, "LICENSE")
        if temporary.stat().st_size >= 2 * 1024**3:
            raise ValueError("O pacote excede o limite de um arquivo na GitHub Release.")
        temporary.replace(archive_file)
    finally:
        temporary.unlink(missing_ok=True)
    with archive_file.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    archive_file.with_suffix(".sha256").write_text(checksum + "\n", encoding="ascii")
    print("Peso variante pronto:", checksum, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, choices=LAYERS, default=LAYERS)
    parser.add_argument("--bundle", type=Path)
    options = parser.parse_args()
    prepare(options.base_dir, options.output_dir, tuple(dict.fromkeys(options.layers)))
    if options.bundle:
        bundle(options.output_dir, options.bundle)
