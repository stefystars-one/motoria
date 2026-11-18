"""Restore a Hub TPU release asset into a Kaggle notebook."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile


def download_verified(url, sha256, name, work):
    if not re.fullmatch(r"https://github\.com/stefystars-one/motoria/releases/download/[\w.+-]+/[\w.+-]+\.zip", url):
        raise ValueError("Este modelo precisa de um pacote compilado publicado na Release do Motoria.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ValueError("Informe o SHA-256 do arquivo da Release.")
    archive_path = work / f"{name}-{sha256.lower()}.zip"
    if not archive_path.exists():
        temp = archive_path.with_suffix(".part")
        digest = hashlib.sha256()
        try:
            with urlopen(url, timeout=120) as source, temp.open("wb") as output:
                received = 0
                while chunk := source.read(1024 * 1024):
                    received += len(chunk)
                    if received > 2 * 1024**3:
                        raise ValueError("O arquivo da Release ultrapassou 2 GiB.")
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != sha256.lower():
                raise ValueError("O hash do motor baixado diverge da Release.")
            temp.replace(archive_path)
        finally:
            temp.unlink(missing_ok=True)
    else:
        with archive_path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != sha256.lower():
                raise ValueError("O arquivo local diverge da Release.")
    return archive_path


def restore_glm_overlay(url, sha256, work):
    archive_path = download_verified(url, sha256, "glm53-flash-abliterated-weights", work)
    expected = {f"layer_{layer:02d}.npz" for layer in range(15, 36)} | {"weights.json", "LICENSE"}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if ({member.filename for member in members} != expected or len(members) != len(expected)
                or sum(member.file_size for member in members) > 2 * 1024**3
                or any(stat.S_ISLNK(member.external_attr >> 16) for member in members)):
            raise ValueError("O pacote dos pesos GLM tem arquivos ausentes ou inválidos.")
        manifest = json.loads(archive.read("weights.json"))
        if manifest.get("variant") != "glm53-flash-abliterated" or manifest.get("layers") != list(range(15, 36)):
            raise ValueError("O pacote de pesos não corresponde à variante GLM.")
        destination = Path(tempfile.mkdtemp(prefix="glm53-weights-", dir=work))
        archive.extractall(destination)
    return str(destination)


def restore_bundle(url, sha256, model, work, config):
    archive_path = download_verified(url, sha256, model, work)
    roots = {"hub-engine.json", "jax_cache", "exported"} if model == "glm53-flash" else {
        "hub-engine.json", "manifest.json", "xla_cache.tar"}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if sum(member.file_size for member in members) > 4 * 1024**3:
            raise ValueError("Pacote descompactado maior que 4 GiB.")
        names = set()
        for member in members:
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or not path.parts or path.parts[0] not in roots or ".." in path.parts
                    or "\\" in member.filename or ":" in member.filename or member.is_dir()
                    or stat.S_ISLNK(member.external_attr >> 16) or path.as_posix() != member.filename
                    or member.filename in names):
                raise ValueError(f"Entrada inválida no pacote: {member.filename}")
            names.add(member.filename)
        if {name.split("/")[0] for name in names} != roots:
            raise ValueError("O pacote do motor está incompleto.")
        manifest = json.loads(archive.read("hub-engine.json"))
        if manifest.get("model") != model or manifest.get("accelerator") != "Kaggle TPU v5e-8":
            raise ValueError("O pacote pertence a outro modelo ou acelerador.")
        if model != "glm53-flash":
            runtime = json.loads(archive.read("manifest.json"))
            if runtime.get("python") != f"{sys.version_info.major}.{sys.version_info.minor}":
                raise ValueError("A versão do Python precisa coincidir com a do cache TPU.")
            profile = [config.get("max_model_len", 262144), config.get("max_num_seqs", 4),
                       config.get("mtp_tokens", 3), config.get("text_only", False)]
            if profile not in runtime.get("configs", []):
                raise ValueError("A configuração selecionada não está coberta pelo cache.")
            if model.endswith("-coder") and (
                runtime.get("checkpoint") != config.get("hf_model_id")
                or runtime.get("revision") != config.get("hf_revision")
            ):
                raise ValueError("O cache do Coder usa outra revisão do checkpoint.")
        destination = Path(tempfile.mkdtemp(prefix=f"{model}-cache-", dir=work))
        archive.extractall(destination)
    if model == "glm53-flash":
        config["strict_cache_dir"] = str(destination)
    else:
        with tarfile.open(destination / "xla_cache.tar", "r:") as entries:
            for entry in entries:
                path = PurePosixPath(entry.name)
                if (not path.parts or path.parts[0] != "xla_cache" or ".." in path.parts
                        or not (entry.isfile() or entry.isdir())):
                    raise ValueError("Caminho inválido no arquivo de cache XLA.")
        config["local_bundle"] = str(destination)
    config["require_precompiled"] = True
    config["build_bundle"] = False
    print(f"Cache da Release restaurado para {model}.")
