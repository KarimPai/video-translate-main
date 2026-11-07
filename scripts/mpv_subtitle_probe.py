"""Exercise actual MPV IPC and subtitle rendering without ASR or translation models."""
import argparse
import logging
import os
import sys
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from auto_translate.config import MpvConfig
from auto_translate.output.mpv import MpvIpc, MpvOutput, find_mpv
from auto_translate.types import Subtitle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpv', default='', help='Path to mpv.exe if absent from PATH')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = root / 'probe.ppm'
        image.write_bytes(b'P6\n640 360\n255\n' + b'\x20\x30\x40' * (640 * 360))
        endpoint = (
            rf'\\.\pipe\auto-translate-probe-{os.getpid()}'
            if sys.platform == 'win32' else str(root / 'ipc.sock')
        )
        proc = subprocess.Popen([
            find_mpv(args.mpv), '--no-config', '--idle=yes', '--force-window=yes',
            '--image-display-duration=30', '--pause',
            f'--input-ipc-server={endpoint}',
        ])
        ipc = None
        try:
            ipc = MpvIpc(endpoint)
            out = object.__new__(MpvOutput)
            out.ipc = ipc
            out.cfg = MpvConfig()
            out._lock = threading.Lock()
            out._lines = []
            out._shown = ''
            out._lags = []
            out._logged_displayed_subtitles = set()
            out._session_start = 0
            out._now = None
            out._logged_subtitle_generations = set()
            ipc.command('loadfile', str(image), 'replace')
            if not out._wait_for_event('file-loaded', 10):
                raise RuntimeError('MPV did not load probe image')
            out.show(Subtitle(0, 30, 'テスト', 'TRANSLATED SUBTITLE PROBE'))
            ipc.command('set_property', 'pause', False)
            deadline = time.monotonic() + 10
            positions = []
            while time.monotonic() < deadline:
                pos = ipc.get('time-pos')
                if pos is not None:
                    positions.append(float(pos))
                    out._render(float(pos))
                time.sleep(0.1)
            if not positions or max(positions) <= min(positions):
                raise RuntimeError('MPV playback clock did not advance')
            if not out._shown:
                raise RuntimeError('Subtitle overlay was not acknowledged')
            print('PASS: IPC replies, advancing clock, and subtitle overlay acknowledged.')
            print('Visual check: TRANSLATED SUBTITLE PROBE should have appeared for 10 seconds.')
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
            if ipc is not None:
                ipc.close()


if __name__ == '__main__':
    main()
