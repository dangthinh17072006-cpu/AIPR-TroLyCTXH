"""Lệnh Windows/macOS/Linux. python run.py setup-demo hoặc python run.py eval."""
import os
import subprocess
import sys
import venv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY=ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')

def run(*args):
    subprocess.run([str(a) for a in args],cwd=ROOT,check=True)

if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'setup-demo'
    if sys.version_info<(3,11):raise SystemExit('Cần Python 3.11 trở lên')
    if action in ('setup','setup-demo'):
        if not PY.exists():venv.EnvBuilder(with_pip=True).create(ROOT/'.venv')
        run(PY,'-m','pip','install','-r','requirements.txt')
        if action=='setup':raise SystemExit(0)
        action='demo'
    if not PY.exists():raise SystemExit('Chạy python run.py setup-demo trước.')
    if action=='test':run(PY,'-m','unittest','discover','-s','tests','-v')
    elif action=='check':run(PY,'-m','src.check')
    elif action in ('demo','chat','eval'):run(PY,'-m','src.main',action,*sys.argv[2:])
    else:raise SystemExit('Dùng setup-demo, setup, demo, chat, eval, test hoặc check')
