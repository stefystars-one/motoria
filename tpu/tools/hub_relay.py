"""Move model streams to the Hub through short HTTP requests."""
import hmac
import json
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def start_relay(port, key):
    lock = threading.Lock()
    jobs = {}

    def generate(job, payload):
        try:
            request = Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urlopen(request, timeout=360) as answer:
                with job["changed"]:
                    if job["cancelled"]:
                        return
                for line in answer:
                    with job["changed"]:
                        if job["cancelled"] or time.monotonic() - job["seen"] > 90:
                            break
                        if not line.startswith(b"data:"):
                            continue
                        job["lines"].append(line[5:].strip().decode("utf-8"))
                        job["bytes"] += len(line)
                        if job["bytes"] > 16 * 1024 * 1024:
                            job["error"] = "Resposta TPU acima de 16 MiB."
                            break
                        job["changed"].notify_all()
        except HTTPError as error:
            with job["changed"]:
                job["error"] = f"O motor TPU respondeu HTTP {error.code}."
        except (OSError, ValueError):
            with job["changed"]:
                job["error"] = "A conexão com o motor TPU foi interrompida."
        finally:
            with job["changed"]:
                job["done"] = True
                job["changed"].notify_all()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            if hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {key}".encode()):
                return True
            self.reply(HTTPStatus.UNAUTHORIZED, {"error": "Chave da sessão inválida."})
            return False

        def do_POST(self):
            if not self.allowed():
                return
            if self.path != "/hub/generate":
                self.reply(404, {"error": "Rota indisponível."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= 32 * 1024 * 1024:
                self.reply(413, {"error": "Requisição TPU acima do limite."})
                return
            try:
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError()
                body["stream"] = True
                payload = json.dumps(body).encode()
            except ValueError:
                self.reply(400, {"error": "Requisição TPU inválida."})
                return
            with lock:
                for job_id, job in list(jobs.items()):
                    if time.monotonic() - job["seen"] > 90:
                        jobs.pop(job_id)
                if len(jobs) >= 12:
                    self.reply(429, {"error": "A sessão TPU está ocupada."})
                    return
                job_id = uuid4().hex
                job = {"changed": threading.Condition(), "lines": deque(), "bytes": 0,
                       "seen": time.monotonic(), "cancelled": False, "done": False}
                jobs[job_id] = job
            threading.Thread(target=generate, args=(job, payload), daemon=True).start()
            self.reply(202, {"id": job_id})

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "available"})
                return
            if not self.allowed():
                return
            if self.path == "/v1/models":
                try:
                    request = Request(f"http://127.0.0.1:{port}/v1/models",
                                      headers={"Authorization": f"Bearer {key}"})
                    with urlopen(request, timeout=5) as response:
                        data = json.load(response)
                    self.reply(200, {**data, "hub_relay": 1})
                except (OSError, ValueError):
                    self.reply(503, {"error": "Carregando o modelo TPU."})
                return
            job_id = self.path.removeprefix("/hub/generate/")
            with lock:
                job = jobs.get(job_id)
            if job is None:
                self.reply(404, {"error": "Execução TPU desconhecida."})
                return
            with job["changed"]:
                job["seen"] = time.monotonic()
                job["changed"].wait_for(lambda: job["lines"] or job["done"], 8)
                lines = [job["lines"].popleft() for _ in range(min(128, len(job["lines"])))]
                finished = job["done"] and not job["lines"]
                error = job.get("error")
            if finished:
                with lock:
                    jobs.pop(job_id, None)
            self.reply(200, {"lines": lines, "done": finished, "error": error if finished else None})

        def do_DELETE(self):
            if not self.allowed():
                return
            job_id = self.path.removeprefix("/hub/generate/")
            with lock:
                job = jobs.pop(job_id, None)
            if job:
                with job["changed"]:
                    job["cancelled"] = True
            self.reply(200, {"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", port + 1), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
