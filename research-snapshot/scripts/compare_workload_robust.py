"""Import-path bootstrap for the unchanged, completed curriculum worker.

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(os.environ['FUZEV_ROOT']).expanduser().resolve()
TREE = Path(os.environ['FUZEV_WORKLOAD_SOURCE']).expanduser().resolve()
ORIGINAL = TREE / 'scripts/run_workload_robust.py'
ORIGINAL_SHA = 'ed398e1283ff1909e0e63899e1d20a57411a726f017677bd182c43be3a45fcd3'
OLD = ROOT / 'reports/2026-10-09-shared-head-generalization'
GENERAL = ROOT / 'reports/2026-10-06-generalization'
RESULTS = ROOT / 'reports/2026-10-10-workload-robust/trial-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--bootstrap-sha256', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_robust_original_worker', ORIGINAL)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    shared = worker.helper()
    pins = {Path(__file__).resolve(): args.bootstrap_sha256, ORIGINAL: ORIGINAL_SHA}
    shared.require(not (RESULTS / 'comparison.json').exists(), 'Never overwrite a completed comparison')
    # Legacy compare imports its siblings by name; do not alter the live worker.
    sys.path.insert(0, str(OLD))
    sys.path.insert(0, str(GENERAL))
    worker.phase(SimpleNamespace(phase='compare', lane=None, arm=None, worker_sha256=ORIGINAL_SHA))
    shared.write(RESULTS / 'comparison-bootstrap.json', {
        'kind': 'curriculum-comparison-import-bootstrap-v1', 'completed': True,
        'bootstrap_sha256': args.bootstrap_sha256, 'original_training_worker_sha256': ORIGINAL_SHA,
        'observed_input_sha256': {str(path): shared.sha(path) for path in pins},
        'comparison_sha256': shared.sha(RESULTS / 'comparison.json'),
        'change': 'Legacy comparator sibling import directories only; original barrier/statistics/gates unchanged',
        'head_executed': False, 'fitting_performed': False, 'main_quality_evaluated': False,
        'qualified': False, 'servable': False})


if __name__ == '__main__':
    main()
