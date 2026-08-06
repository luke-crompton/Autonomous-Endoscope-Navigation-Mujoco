r"""Relaunch the calling script under the Python 3.11 interpreter.

The perception stack -- CUDA torch, depth_anything_3, opencv -- exists only in
the 3.11 environment (README.md interpreter matrix). The IDE's Run button uses
whatever the project interpreter happens to be, and for this project that is
usually C:\Python314, because that is what navigation/ needs. Importing this
module first makes the camera scripts work from either, so the green Run arrow
does the right thing with no arguments and no per-file run configuration.

Import it BEFORE numpy / torch / cv2. Those are exactly the imports that fail
under the wrong interpreter, and a traceback beats the relaunch to it if they
come first.

Single-machine by design: the path below is hard-coded to this workstation.
Set MUJUCO_V3_PY311 in the environment to override it.
"""

import os
import subprocess
import sys

PY311 = os.environ.get(
    'MUJUCO_V3_PY311',
    r'C:\Users\lukec\AppData\Local\Programs\Python\Python311\python.exe',
)

# Set in the child's environment so a 3.11 exe that turns out not to be 3.11
# fails loudly instead of forking forever.
_GUARD = 'MUJUCO_V3_PY311_RELAUNCHED'


def _relaunch():
    if sys.version_info[:2] == (3, 11):
        return

    script = os.path.abspath(sys.argv[0])
    running = sys.version.split()[0]
    hint = f'  "{PY311}" "{script}" ' + ' '.join(sys.argv[1:])

    if os.environ.get(_GUARD):
        sys.exit(
            f'Relaunch loop: {PY311}\nreports Python {running}, not 3.11. '
            f'Fix the path in {os.path.basename(__file__)} or set '
            f'MUJUCO_V3_PY311.'
        )

    if not os.path.exists(PY311):
        sys.exit(
            f'This needs Python 3.11 (CUDA torch / opencv live only there) but '
            f'is running on {running}, and no 3.11 interpreter was found at:\n'
            f'  {PY311}\n'
            f'Set MUJUCO_V3_PY311 to the right path, or run it yourself:\n{hint}'
        )

    # execv/subprocess would drop the debugger, and a silently un-debuggable
    # session is worse than a clear refusal.
    if sys.gettrace() is not None or 'pydevd' in sys.modules:
        sys.exit(
            f'Debugger attached under Python {running}; not relaunching, '
            f'because the debugger would lose the process.\n'
            f'Point this run configuration at Python 3.11:\n  {PY311}\n'
            f'or run:\n{hint}'
        )

    print(f'[_py311] running under Python {running} -- relaunching under 3.11',
          flush=True)
    env = dict(os.environ)
    env[_GUARD] = '1'
    # subprocess, not os.execv: on Windows execv detaches, so the IDE reports
    # "process finished" while the real work is still printing.
    raise SystemExit(
        subprocess.run([PY311, script, *sys.argv[1:]], env=env).returncode
    )


_relaunch()
