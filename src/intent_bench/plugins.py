"""Export an installable plugin bound to the Python environment running the tool."""
import json
from pathlib import Path
import shutil
import sys


def export_plugin(destination):
    from importlib.util import find_spec
    if find_spec("intent") is None or find_spec("mcp") is None:
        raise RuntimeError("install intentdsl[manual] and intentdsl-bench[mcp] in the plugin interpreter before export")
    destination=Path(destination).expanduser().resolve()
    shutil.copytree(Path(__file__).parent/'plugin',destination)
    for filename in ('mcp.json','.mcp.json'):
        path=destination/filename
        data=json.loads(path.read_text())
        for name,server in data['mcpServers'].items():
            server['command']=sys.executable
            module = {'intent_bench':'intent_bench.mcp_server', 'intent_manual':'intent.mcp.manual',
                      'intent_compiler':'intent.mcp.compiler'}[name]
            server['args']=['-m',module]
        path.write_text(json.dumps(data,indent=2)+'\n')
    archive=shutil.make_archive(str(destination),'zip',root_dir=destination)
    return {'plugin':str(destination),'archive':archive,'python':sys.executable}
