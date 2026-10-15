from types import SimpleNamespace
from intent_bench.model import Result
from intent_bench.results import read,save
from intent_bench import suites


def test_failed_rerun_does_not_reuse_a_previous_success(tmp_path,monkeypatch):
    item=dict(name='corpus/mojo/one',provider='mojo',kernel='one',family='custom',case='small')
    monkeypatch.setattr(suites,'catalog',lambda:[item])
    monkeypatch.setattr(suites.subprocess,'Popen',lambda *a,**k:SimpleNamespace(wait=lambda timeout:1))
    save([Result(item['name'],'custom','small',{},'intent:mojo',status='passed')],tmp_path/'cases/mojo/one')
    args=SimpleNamespace(suite='cpu',output=tmp_path,samples=1,warmup=0,threads=1,cache='flush',seed=0,device='cpu',
        timing='auto',scope='prepared_call',intent_root=None,compiler=None,baseline=None,throughput=None,
        check_only=True,impl=None,option=[],resume=False,timeout=10,jobs=1)
    assert suites.run_suite(args)==1
    assert read(tmp_path)[0]['status']=='worker_failed'
