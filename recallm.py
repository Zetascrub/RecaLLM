import os
import pty
import requests
import sys
import select
from datetime import datetime
from collections import deque
from typing import Deque

# ───── Config ─────
VERSION = "0.1.0"
HISTORY_FILE = "terminal_history.log"
CONTEXT_LINES = 100
LLM_TRIGGER_PREFIX = "--"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.8.10:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
history_buffer = deque(maxlen=500)

# ───── Prompt Banner ─────
def print_banner() -> None:
    """Display the program banner."""
    print(rf"""

__________                     .__  .__           
\______   \ ____   ____ _____  |  | |  |   _____  
 |       _// __ \_/ ___\\__  \ |  | |  |  /     \ 
 |    |   \  ___/\  \___ / __ \|  |_|  |_|  Y Y  \
 |____|_  /\___  >\___  >____  /____/____/__|_|  /
        \/     \/     \/     \/                \/ 

       Terminal Context Companion v{VERSION}
    """)

# ───── Logging ─────
def log_line(line: str) -> None:
    """Persist a line of terminal output with a timestamp."""
    timestamp = datetime.now().isoformat()
    clean = line.strip()
    history_buffer.append(f"{timestamp} {clean}")
    with open(HISTORY_FILE, "a") as f:
        f.write(f"{timestamp} {clean}\n")

# ───── LLM Call ─────
def ask_llm(query: str, context: str) -> str:
    """Send the query and context to the LLM and return its response."""
    system_prompt = (
        "You are Recallm, a command-line assistant. Your job is to review the terminal context "
        "and give precise, practical commands the user can run next.\n"
        "- If the user asks what to run, respond only with the command (no explanation).\n"
        "- Use the terminal context (interface names, IPs, commands run) to tailor your response.\n"
        "- Prefer clarity and brevity.\n"
        "- If unsure, ask for clarification instead of guessing.\n"
    )

    full_prompt = f"{system_prompt}\nContext:\n{context}\n\nUser query:\n{query}\n"

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=60
        )
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"⚠️ Error contacting Ollama: {e}"


# ───── Main Shell Loop ─────
def run_shell() -> None:
    """Spawn a bash shell and handle LLM queries."""
    pid, fd = pty.fork()

    if pid == 0:
        os.execvp("bash", ["bash"])
    else:
        try:
            while True:
                rlist, _, _ = select.select([fd, sys.stdin], [], [])
                if fd in rlist:
                    try:
                        output = os.read(fd, 1024).decode()
                        if output:
                            sys.stdout.write(output)
                            sys.stdout.flush()
                            log_line(output)
                    except OSError:
                        break  # Shell exited
                if sys.stdin in rlist:
                    user_input = sys.stdin.readline()
                    if user_input.startswith(LLM_TRIGGER_PREFIX):
                        query = user_input[len(LLM_TRIGGER_PREFIX):].strip()
                        context = "\n".join(list(history_buffer)[-CONTEXT_LINES:])
                        print("\n🧠 Recallm is thinking...\n")
                        response = ask_llm(query, context)
                        print(f"🤖 {response}\n")
                        os.write(fd, b"\n")  # Send a newline to the shell to re-show the prompt

                    else:
                        os.write(fd, user_input.encode())
        except KeyboardInterrupt:
            print("\n📴 Exiting Recallm.")
        except EOFError:
            pass

# ───── Entrypoint ─────
def main() -> None:
    """Entry point executed when running the script directly."""
    if "--help" in sys.argv:
        print_banner()
        print(
            "Usage:\n  recallm.py        Start Recallm\n  --help            Show this message\n  --version         Show version"
        )
        return
    if "--version" in sys.argv:
        print(f"Recallm version {VERSION}")
        return

    print_banner()
    print("💬 Type shell commands normally. Prefix questions with `--` to ask the AI.")
    run_shell()


if __name__ == "__main__":
    main()
