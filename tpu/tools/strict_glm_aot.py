"""Require published AOT exports without changing the source used in cache keys."""


def require_exports(aot):
    program = aot._Program
    resolve_export = program._resolve

    def resolve(self, args):
        result = resolve_export(self, args)
        if self.origin != "loaded":
            label = aot._label(self._key)
            raise RuntimeError(f"MOTORIA_CACHE_MISS: export {label} ausente no motor GLM publicado.")
        return result

    def no_trace(self):
        label = aot._label(self._key)
        raise RuntimeError(f"MOTORIA_CACHE_MISS: assinatura {label} não suportada pelo motor GLM publicado.")

    program._resolve = resolve
    program._plain = no_trace
