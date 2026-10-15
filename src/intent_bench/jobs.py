"""Persistent background invocations of the same CLI used interactively."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid


def start(arguments):
    root=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'intent-bench/runs'/uuid.uuid4().hex
    root.mkdir(parents=True)
    arguments=[*arguments,'--output',str(root/'results')]
    (root/'request.json').write_text(json.dumps({'arguments':arguments,'cwd':os.getcwd()},indent=2)+'\n')
    with (root/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','intent_bench.jobs',str(root)],
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'job.json').write_text(json.dumps({'pid':process.pid,'state':'running'})+'\n')
    return {'run':str(root),'state':'running','results':str(root/'results'),'log':str(root/'worker.log')}


def status(directory):
    root=Path(directory).expanduser()
    if (root/'completion.json').exists():
        value=json.loads((root/'completion.json').read_text())
    else:
        value=json.loads((root/'job.json').read_text())
    value.update(run=str(root),log=str(root/'worker.log'))
    path=root/'results/results.json'
    if path.exists():
        from .results import read
        value['results']=read(path)
        value['report']=str(root/'results/report.html')
    return value


def main(directory):
    root=Path(directory)
    request=json.loads((root/'request.json').read_text())
    os.chdir(request['cwd'])
    try:
        from .cli import main as cli
        code=cli(request['arguments'])
    except SystemExit as error:
        code=error.code or 0
    except Exception:
        traceback.print_exc()
        code=1
    temporary=root/'completion.tmp'
    temporary.write_text(json.dumps({'state':'completed','exit_code':code})+'\n')
    temporary.replace(root/'completion.json')
    return code


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))
