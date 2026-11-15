"""Model-free launch gate. Only the frontend holds the session job open."""
import os
import subprocess
import sys

from .windows_job import join_session


def main():
    session = os.environ["VIDEO_TRANSLATE_SESSION"]
    join_session(session)
    print(f"session={session} name=launcher pid={os.getpid()} parent={os.getppid()} owned=true", flush=True)
    with subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL,
                          creationflags=subprocess.CREATE_NO_WINDOW) as backend:
        print(f"session={session} name=backend pid={backend.pid} parent={os.getpid()} owned=true", flush=True)
        return backend.wait()


if __name__ == "__main__":
    sys.exit(main())
