from __future__ import annotations

"""
本地聊天 HTTP 服务。

这是一个最小部署模拟：

    GET  /health
    POST /chat {"message": "...", "history": ""}

它使用 Python 标准库实现，避免为教学项目额外引入 Web 框架依赖。
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_models.llm.chat import generate_chat_reply, load_chat_model
from foundation_models.llm.utils import choose_device


class ChatServer(ThreadingHTTPServer):
    model: Any
    tokenizer: Any
    device: Any
    defaults: dict[str, Any]


class ChatHandler(BaseHTTPRequestHandler):
    server: ChatServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {format % args}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/chat":
            self.send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            self.send_json(400, {"error": "invalid json"})
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self.send_json(400, {"error": "message is required"})
            return

        defaults = self.server.defaults
        reply, history = generate_chat_reply(
            model=self.server.model,
            tokenizer=self.server.tokenizer,
            user_text=message,
            history=str(payload.get("history", "")),
            device=self.server.device,
            max_new_tokens=int(payload.get("max_new_tokens", defaults["max_new_tokens"])),
            temperature=float(payload.get("temperature", defaults["temperature"])),
            top_k=int(payload.get("top_k", defaults["top_k"])),
        )

        self.send_json(
            200,
            {
                "reply": reply,
                "history": history,
            },
        )


def load_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {model_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an exported educational LLM chat model.")
    parser.add_argument("--model-dir", required=True, help="Directory created by export_model.py.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    manifest = load_manifest(model_dir)
    checkpoint_path = model_dir / manifest["model_file"]
    tokenizer_path = model_dir / manifest["tokenizer_file"]

    device = choose_device(args.device)
    model, tokenizer, _ = load_chat_model(checkpoint_path, device, tokenizer_path=tokenizer_path)

    server = ChatServer((args.host, args.port), ChatHandler)
    server.model = model
    server.tokenizer = tokenizer
    server.device = device
    server.defaults = manifest.get(
        "generation_defaults",
        {"max_new_tokens": 80, "temperature": 0.3, "top_k": 20},
    )

    print(f"Serving {manifest.get('name', 'model')} on http://{args.host}:{args.port}")
    print("GET  /health")
    print('POST /chat {"message": "你好"}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
