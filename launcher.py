"""Launch, log and reap the application as one process tree."""
import os
import subprocess
import sys
from pathlib import Path

# The packaged embeddable Python takes sys.path from python313._pth, and in that
# mode the running script's own directory is NOT added automatically.  Without
# this, `from lifecycle import ...` fails with ModuleNotFoundError.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lifecycle import terminate_tree, watch_launcher  # noqa: E402


def main():
    root = _ROOT
    watch_launcher()
    environment = os.environ.copy()
    environment["ANIMEFLOW_PARENT_PID"] = str(os.getpid())
    environment["PYTHONUTF8"] = "1"
    (root / "work").mkdir(exist_ok=True)
    child = None
    try:
        with (root / "work" / "startup.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, "-u", str(root / "app.py")], cwd=root,
                                     env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            exit_code = child.wait()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        if child is not None and child.poll() is None:
            terminate_tree(child.pid)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
