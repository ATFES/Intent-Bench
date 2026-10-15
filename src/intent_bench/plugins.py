"""Export an installable plugin bound to the Python environment running the tool."""
import json
from pathlib import Path
import shutil
import sys


def export_plugin(destination, intent_root=None):
    destination=Path(destination).expanduser().resolve()
    shutil.copytree(Path(__file__).parent/'plugin',destination)
    for filename in ('mcp.json','.mcp.json'):
        path=destination/filename
        data=json.loads(path.read_text())
        for name,server in data['mcpServers'].items():
            server['command']=sys.executable
            server['args']=['-m','intent_bench.mcp_server',*server['args']]
            if name!='intent_bench' and intent_root is not None:
                server['args'].extend(['--intent-root',str(Path(intent_root).expanduser().resolve())])
        path.write_text(json.dumps(data,indent=2)+'\n')
    archive=shutil.make_archive(str(destination),'zip',root_dir=destination)
    return {'plugin':str(destination),'archive':archive,'python':sys.executable}
