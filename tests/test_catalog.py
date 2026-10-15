import subprocess
import sys

from intent_bench.catalog import catalog


def test_every_original_registration_has_a_destination():
    from collections import Counter
    rows=catalog()
    assert len({r['name'] for r in rows})==309
    assert Counter(r['provider'] for r in rows)=={'triton':54,'cutile':37,'mojo':199,'weft':6,'bangc':13}


def test_discovery_does_not_import_optional_sdks():
    script="""
import sys
from intent_bench.cli import main
main(['list', '--json'])
assert not set(('intent','torch','triton','cuda')) & sys.modules.keys()
"""
    subprocess.run([sys.executable,'-c',script],check=True,capture_output=True,text=True)
