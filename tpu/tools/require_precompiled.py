"""Refuse a new XLA executable on TPU, while allowing the persistent-cache loader."""
import functools
import os


def install():
    import jax
    from jax._src import compiler

    jax.config.update("jax_raise_persistent_cache_errors", True)

    def guard(compile_function):
        @functools.wraps(compile_function)
        def blocked(backend, *args, **kwargs):
            if getattr(backend, "platform", None) == "tpu":
                raise RuntimeError("MOTORIA_CACHE_MISS: o pacote publicado não tem este executável TPU.")
            return compile_function(backend, *args, **kwargs)
        return blocked

    names = [name for name in ("backend_compile_and_load", "backend_compile") if callable(getattr(compiler, name, None))]
    if not names:
        raise RuntimeError("Runtime JAX incompatível com o modo de cache obrigatório.")
    for name in names:
        setattr(compiler, name, guard(getattr(compiler, name)))


# sitecustomize.py invokes this for the vLLM server and its spawned worker interpreters.
def install_in_worker():
    if os.environ.get("MOTORIA_REQUIRE_PRECOMPILED") == "1":
        try:
            install()
        except Exception as error:
            # Python otherwise reports a sitecustomize exception and continues booting.
            raise SystemExit(str(error)) from error
