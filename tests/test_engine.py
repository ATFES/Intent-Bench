import json
import sys
import numpy as np
from intent_bench import (Case, Environment, ExpressionGap, Implementation, Measurement,
                         PreparedCall, Workload, benchmark, callable_implementation)
from intent_bench.loading import load
from intent_bench.results import compare, read, save
from intent_bench.trees import clone


def test_custom_mutation_and_repeated_measurements(tmp_path):
    bench=load("examples/custom.py:bench")
    rows=benchmark(bench,measurement=Measurement(samples=4,warmup=2))
    assert all(r.correctness=="passed" and len(r.samples_ms)==4 for r in rows)
    path=save(rows,tmp_path)
    assert len(read(path))==2
    assert "mutable_normalize" in (path/"report.html").read_text()


def test_aliases_survive_input_copy():
    owner=np.arange(32,dtype=np.float32).reshape(4,8)
    a,b,c=clone((owner,owner[:,::2],owner[:,::-1]))
    b[1,2]=101
    assert a[1,4]==101 and c[1,3]==101
    assert owner[1,4]!=101


def test_n_way_oracle_is_not_baseline():
    impls={name:callable_implementation(name,fn) for name,fn in {
        "good":lambda x:x+1,"also_good":lambda x:1+x,"wrong":lambda x:x+2}.items()}
    bench=Workload("add",[Case("one")],lambda c,s:(np.ones(4),),impls,oracle=lambda c,a:a[0]+1)
    rows=benchmark(bench,baseline="good",measurement=Measurement(samples=2,warmup=0))
    assert rows[0].speedup==1 and rows[1].speedup>0
    assert rows[2].status=="numerical_failed" and rows[2].speedup is None and rows[2].median_ms>0


def test_expression_gap_does_not_remove_workload():
    def missing(*args):raise ExpressionGap("no packed-value expression yet")
    bench=Workload("packed",[Case("q4")],lambda c,s:(),{"intent":Implementation("intent",missing)})
    row,=benchmark(bench)
    assert row.status=="expression_gap" and "packed-value" in row.error


def test_cross_scope_comparison_not_merged(tmp_path):
    bench=load("examples/custom.py:bench")
    rows=benchmark(bench,measurement=Measurement(samples=1,warmup=0))
    left=read(save(rows,tmp_path/"a"))
    right=json.loads(json.dumps(left));right[0]["scope"]="host_call"
    changes=compare(left,right)
    assert sum(r["change"]=="removed" for r in changes)==1
    assert sum(r["change"]=="added" for r in changes)==1


def test_gpu_graph_refuses_to_change_reset_contract():
    from intent_bench.measurement import measure
    call=PreparedCall(lambda:None,lambda:None,reset=lambda:None,device="cuda:0")
    import pytest
    with pytest.raises(ValueError,match="restore mutable"):
        measure(call,Measurement("cuda_graph",1,0))


def test_check_only_does_not_call_a_timer():
    def prepare(case,args,env):
        def timer(settings):raise AssertionError('timer must not run')
        return PreparedCall(lambda:None,lambda:np.ones(1),native_measure=timer)
    bench=Workload('check',[Case('one')],lambda c,s:(),{'native':Implementation('native',prepare)},oracle=lambda c,a:np.ones(1))
    row,=benchmark(bench,measurement=Measurement(enabled=False))
    assert row.status=='passed' and row.median_ms is None and row.samples_ms==[]


def test_reference_failure_does_not_prevent_independent_candidate_measurement():
    def oracle(case,args):raise RuntimeError('reference dependency unavailable')
    bench=Workload('independent',[Case('one')],lambda c,s:(np.ones(1),),
                   {'candidate':callable_implementation('candidate',lambda x:x+1)},oracle=oracle)
    row,=benchmark(bench,measurement=Measurement(samples=1,warmup=0))
    assert row.status=='reference_failed' and row.median_ms>0 and row.speedup is None


def test_host_scope_uses_the_host_entry():
    calls=[]
    def prepare(case,args,env):
        return PreparedCall(lambda:calls.append('prepared'),lambda:np.ones(1),
                            host_run=lambda:calls.append('host'))
    bench=Workload('host',[Case('one')],lambda c,s:(),{'impl':Implementation('impl',prepare)},oracle=lambda c,a:np.ones(1))
    row,=benchmark(bench,measurement=Measurement(samples=2,warmup=0,scope='host_call'))
    assert row.status=='passed' and row.method=='host' and calls==['host']*3
