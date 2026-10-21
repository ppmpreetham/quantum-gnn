"""Local HTTP adapter for the repository's QUBO -> GNN -> outcome pipeline."""
from __future__ import annotations

import argparse
import traceback
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fleet_session import (  # noqa: E402
    SessionStore, available_solvers, validate_comparison, validate_faults,
    validate_params, validate_positions, validate_tasks,
)

MAX_BODY = 32768
STORE = SessionStore()
# a reload can leave an aborted solve running; spare slots stop it from blocking the new page
SOLVER_SLOTS = threading.BoundedSemaphore(4)


def parse_step(data):
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    params = validate_params(data.get("params"))
    positions = validate_positions(data.get("positions"), params["roverCount"])
    faults = validate_faults(data.get("faults"), params["roverCount"])
    comparison = validate_comparison(data.get("comparison", "none"))
    tasks = validate_tasks(data.get("tasks"), params["roverCount"])
    session_id = data.get("sessionId")
    if session_id is not None and (not isinstance(session_id, str) or len(session_id) > 64):
        raise ValueError("Invalid session identifier")
    return session_id, params, positions, faults, comparison, tasks


class Handler(BaseHTTPRequestHandler):
    server_version = "FleetConsole/2.0"

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/step"):
            super().log_message(fmt, *args)

    def respond(self, code, body):
        payload = json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/api/health":
            self.respond(200, {"status": "ok", "solvers": available_solvers(),
                               "engine": "EdgeSelectionQUBO + ProjectedEdgeAnnealing + TrustGNN"})
        else:
            self.respond(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        if self.path != "/api/step":
            self.respond(404, {"error": "Unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError(f"Request must be between 1 and {MAX_BODY} bytes")
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise ValueError("Content-Type must be application/json")
            session_id, params, positions, faults, comparison, tasks = parse_step(
                json.loads(self.rfile.read(length)))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self.respond(400, {"error": str(error)})
            return
        if not SOLVER_SLOTS.acquire(blocking=False):
            self.respond(503, {"error": "Planner is busy; retrying"})
            return
        try:
            session_id, session = STORE.acquire(session_id, params, lambda: uuid.uuid4().hex)
            with session.lock:
                result = session.step(positions, faults, comparison, tasks)
            self.respond(200, {"sessionId": session_id, **result})
        except OverflowError as error:
            self.respond(429, {"error": str(error)})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self.respond(500, {"error": "Planner failed; see the API terminal for the traceback"})
        finally:
            SOLVER_SLOTS.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Fleet API listening on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
