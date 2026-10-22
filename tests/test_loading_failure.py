from intent_bench.cli import main
from intent_bench.results import read


def test_missing_optional_dependency_has_a_result(tmp_path):
    module=tmp_path/'custom.py';module.write_text('import missing_benchmark_sdk_987654321\n')
    output=tmp_path/'results'
    assert main(['run',str(module)+':bench','--impl','one,two','--output',str(output),
                 '--output-root',str(tmp_path)])==1
    rows=read(output)
    assert [row['implementation'] for row in rows]==['one','two']
    assert all(row['status']=='environment_missing' and row['stage']=='load' for row in rows)
