"""Discovery uses metadata only, never imports an implementation's SDK."""
import json
from pathlib import Path

ADVANCED_OPERATIONS=('scaled_fp8_gemm','fused_linear_cross_entropy','flash_attention_bias','flash_varlen_attention',
    'chunk_retention','flash_attention_inline','flash_attention_select','flash_attention_full_causal_stream',
    'mamba3_siso_step','mamba3_siso_forward','paged_gqa_f16_reduction')


def catalog():
    path=Path(__file__).with_name("catalog.json")
    rows=json.loads(path.read_text()) if path.exists() else []
    for row in rows:
        provider=row['provider']
        if provider=='mojo':
            source='mojo' if '/baselines/mojo/' in (row.get('source_runtime') or '') else 'torch'
            row['implementations']=[source,'intent:mojo']
            if row['kernel'] in ADVANCED_OPERATIONS and 'mojo' not in row['implementations']:row['implementations'].append('mojo')
        elif provider=='weft':row['implementations']=['weft','intent:weft']
        elif provider=='bangc':
            source='cnnl' if row['kernel'] in {'relu','fused_softmax','dense_gemm'} else 'reference:mojo' if row['kernel'] in {'causal_attention_f32','causal_linear_attention_f32'} else 'reference:cuda'
            row['implementations']=[source,'intent:bangc']
    return rows


def get_workload(name):
    from .corpus_adapter import build
    return build(name)
