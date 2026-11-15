"""Exercise the real translator's attach, warm-up and repeated close paths."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch

from auto_translate.config import TranslateConfig
from auto_translate.glossary import Glossary
from auto_translate.translate.plamo import PlamoTranslator, _server_state


class ModelEndpoint(BaseHTTPRequestHandler):
    def reply(self, body):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        self.reply({"data": [{"id": "external"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.reply({"choices": [{"message": {"content": "Hello"}}]})

    def log_message(self, *args):
        pass


class ExternalServerTest(unittest.TestCase):
    def test_close_preserves_connected_server(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), ModelEndpoint) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/v1"
                cfg = TranslateConfig(plamo_base_url=url, plamo_model="external")
                with patch("auto_translate.translate.plamo.owned_popen") as spawn:
                    translator = PlamoTranslator(cfg, Glossary({}))
                    self.assertIsNone(translator.process)
                    translator.close()
                    translator.close()
                    self.assertTrue(translator.client.is_closed)
                    self.assertEqual(_server_state(url, "external"), "ready")
                    spawn.assert_not_called()
            finally:
                server.shutdown()
                thread.join(3)
