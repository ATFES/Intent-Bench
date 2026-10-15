import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from intent_bench import agent
from intent_bench.loading import load
from intent_bench.corpus.agent_tritonbench.policy import validate_program


def test_generation_is_one_command_without_evaluation(tmp_path, monkeypatch):
    invocation=SimpleNamespace(metadata=lambda:{'x':{'shape':[4]}},call=lambda fn:fn())
    tasks=SimpleNamespace(invocation=lambda *a,**k:invocation,reference=lambda *a:lambda:None,
                          description=lambda *a:'Compute identity.',return_contract=lambda x:{'type':'tensor'})
    suite={'agent_seconds':10,'timing':'cuda_event','tolerances':{'default':{'atol':0}}}
    task={'input_index':0,'tolerance':'default'};row={'entry':'identity'}
    monkeypatch.setattr(agent,'task_data',lambda args:(tasks,suite,task,row))
    monkeypatch.setattr(agent,'benchmark',lambda *a,**k:pytest.fail('generation must not evaluate candidates'))
    script=tmp_path/'generator.py'
    script.write_text('from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text("def build(context):\\n    return lambda x: x\\n")\n')
    args=SimpleNamespace(reference=tmp_path,output=tmp_path/'submission',codex=None,language='triton',task='identity',
                         command=json.dumps([sys.executable,str(script),'{output}']))
    assert agent.generate(args)==0
    result=json.loads((args.output/'generation.json').read_text())
    assert result['status']=='submitted' and Path(result['program']).is_file()
    assert not list(args.output.rglob('measurement.json'))


def test_experimental_import_policy_does_not_apply_to_user_workloads(tmp_path):
    path=tmp_path/'bench.py';path.write_text('from pathlib import Path\nbench=Path("allowed")\n')
    assert load(str(path)+':bench')==Path('allowed')
    with pytest.raises(ValueError,match='allowlist'):
        validate_program(path,language='triton')
