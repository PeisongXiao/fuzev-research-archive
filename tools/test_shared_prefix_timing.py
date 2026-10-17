import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from shared_prefix_timing import decision_payload, main


def test_structured_questions_use_systemone_text_encoding():
    payload = decision_payload("model", "shared", {"type": "choice",
        "instructions": {"task": ["choose", "one"]},
        "criteria": {"a": {"reason": ["one", "two"]}, "b": None}})
    assert payload["question"] == "task:\n  - choose\n  - one"
    assert payload["candidates"] == [
        {"id": "a", "text": "a: reason:\n  - one\n  - two"}, {"id": "b", "text": "b"}]
    score = decision_payload("model", "shared", {"type": "score", "criteria": [None, ["clear"]]})
    assert score["candidates"] == [{"id": "0", "text": ""}, {"id": "1", "text": "- clear"}]


def test_http_timing_records_usage_health_and_decision_counters(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond({"status": "ok", "versions": {"runtime": "1.1.0"}})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            if self.path == "/v1/systemone":
                self.respond({"usage": {"input_tokens": 2048, "output_tokens": 4}})
            else:
                self.respond({"execution": {"input_tokens": 2055, "submitted_tokens": 2063}})

        def respond(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    context = tmp_path / "context.txt"
    context.write_text("Shared text")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({str(i): {"type": "choice", "instructions": f"Question {i}",
        "criteria": {"a": "First", "b": "Second"}} for i in range(4)}))
    output = tmp_path / "timing.json"
    try:
        result = main(["--url", f"http://127.0.0.1:{server.server_port}", "--model", "model",
                       "--context", str(context), "--questions", str(questions),
                       "--samples", "2", "--output", str(output)])
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert json.loads(output.read_text()) == result
    assert result["question_count"] == 4
    grouped = [payload for path, payload in calls if path == "/v1/systemone" and len(payload["questions"]) == 4]
    assert grouped and all(set(payload["questions"]) == {"0", "1", "2", "3"} for payload in grouped)
    for sample in result["samples"]:
        assert sample["health"]["versions"] == {"runtime": "1.1.0"}
        assert sample["grouped"]["usage"]["input_tokens"] == 2048
        assert len(sample["concurrent"]["requests"]) == 4
        assert all(row["execution"]["submitted_tokens"] == 2063
                   for row in sample["decision_probes"].values())
    assert all(payload.get("state", "Shared text") == "Shared text" for _, payload in calls)
