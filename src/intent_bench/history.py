"""Read known historical table formats without rewriting their original columns."""
import csv
from pathlib import Path
from .model import Result


def _value(row,name,scale=1):
    value=row.get(name)
    return None if value in (None,'','—','None','nan','NaN') else float(value)*scale


def import_csv(path):
    path=Path(path);results=[]
    with path.open() as stream:
        reader=csv.DictReader(stream)
        fields=set(reader.fieldnames or ())
        if 'task' in fields:
            units='us' if 'intent_triton_us' in fields else 'ms'
            factor=.001 if units=='us' else 1
            implementations=[('reference','reference'),('agent_triton','agent:triton'),('intent_triton','intent:triton'),('intent_cutile','intent:cutile')]
            if 'agent_triton_ms' not in fields and 'agent_triton_us' not in fields and 'agent_triton_status' in fields:
                implementations[1]=('agent_triton','agent:triton')
            for row in reader:
                for prefix,name in implementations:
                    column=prefix+'_'+units
                    if column not in row:continue
                    milliseconds=_value(row,column,factor)
                    status=row.get(prefix+'_status') or ('recorded' if milliseconds is not None else 'unavailable')
                    result=Result('agent/'+row['task'],'historical',str(row.get('profile','recorded')),{},name,
                        status='historical_'+status,correctness='historical',device='historical',method=row.get('timing') or 'unrecorded',
                        scope='historical_record',median_ms=milliseconds,baseline='reference',
                        metadata={'historical':True,'table':str(path),'original_row':row,'input_unit':units})
                    baseline=_value(row,'reference_'+units,factor)
                    if baseline and milliseconds:result.speedup=baseline/milliseconds
                    results.append(result)
        elif {'kernel','case','generated_p50_ms','source_p50_ms'}<=fields:
            for row in reader:
                for name,column in [('generated','generated_p50_ms'),('source','source_p50_ms')]:
                    results.append(Result(row['kernel'],'historical',row['case'],{},name,status='historical_'+row['status'],
                        correctness='historical',device='historical',method='unrecorded',scope='historical_record',
                        median_ms=_value(row,column),metadata={'historical':True,'table':str(path),'original_row':row,'input_unit':'ms'}))
        else:raise ValueError('unsupported historical CSV columns; supply an explicit importer for its units and ratio definition')
    return results
