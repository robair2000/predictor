import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import quant_worker  # Importeer de main() logica uit je quant worker

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/run-worker':
            print("🚀 Cron-job verzoek ontvangen op /run-worker. Start quant_worker op de achtergrond...")
            # Start de training in een achtergrondthread om time-outs te voorkomen
            threading.Thread(target=quant_worker.main, daemon=True).start()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Worker Started Successfully')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')

    def log_message(self, format, *args):
        pass  # Houdt de logs lekker schoon van overbodige HTTP GET meldingen

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"🌐 Server gestart op poort {port}...")
    server.serve_forever()

if __name__ == '__main__':
    run_server()
