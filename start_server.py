"""Start the Email Triage Environment server."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn
uvicorn.run("email_triage_env.server.app:app", host="127.0.0.1", port=8000, timeout_keep_alive=5)
