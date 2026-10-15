from intent_bench.results import read


def test_legacy_microseconds_and_ratio_are_explicit(tmp_path):
    path=tmp_path/'old.csv'
    path.write_text('task,reference_us,intent_triton_us,intent_triton_status,intent_triton_over_reference\nsoftmax,100,50,pass,0.5\n')
    rows=read(path)
    candidate=next(r for r in rows if r['implementation']=='intent:triton')
    assert candidate['median_ms']==.05 and candidate['speedup']==2
    assert candidate['status']=='historical_pass' and candidate['correctness']=='historical'
    assert candidate['metadata']['original_row']['intent_triton_over_reference']=='0.5'
