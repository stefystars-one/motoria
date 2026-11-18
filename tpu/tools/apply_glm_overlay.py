"""Apply an offline weight overlay to the original GLM TPU base store."""

import hashlib
import numpy as np


LAYERS = tuple(range(15, 36))
PROJECTIONS = ("attn/o", "mlp/shared/down")
WEIGHTS = tuple(f"{name}/{part}" for name in PROJECTIONS for part in ("q", "s"))


def fingerprint(arrays):
    digest = hashlib.sha256()
    for name in WEIGHTS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}:{array.shape}:{array.dtype}\n".encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.digest()


def read_overlay(original, path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*WEIGHTS, "__base__"}:
            raise ValueError(f"Pesos incompletos em {path.name}.")
        if bytes(archive["__base__"]) != fingerprint(original):
            raise ValueError(f"A base GLM do Kaggle mudou em {path.name}.")
        weights = {name: archive[name] for name in WEIGHTS}
    for name, array in weights.items():
        if array.shape != original[name].shape or array.dtype != original[name].dtype:
            raise ValueError(f"Peso com formato incompatível em {path.name}: {name}")
        if name.endswith("/s") and (not np.isfinite(array).all() or (array <= 0).any()):
            raise ValueError(f"Escalas inválidas em {path.name}: {name}")
    return weights


def apply_overlay(store, directory):
    for index in LAYERS:
        layer = store["layers"][index]
        projections = {
            "attn/o": layer["attn"]["o"],
            "mlp/shared/down": layer["mlp"]["shared"]["down"],
        }
        original = {f"{name}/{part}": params[part] for name, params in projections.items() for part in ("q", "s")}
        weights = read_overlay(original, directory / f"layer_{index:02d}.npz")
        for name, params in projections.items():
            params.update({part: weights[f"{name}/{part}"] for part in ("q", "s")})
