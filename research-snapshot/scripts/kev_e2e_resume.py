"""Locked, read-only recovery of an interrupted E2E journal into a new attempt."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def frozen_inputs(pins):
    # Only the offline runner may change; model/runtime provenance is checked separately.
    runners = [path for path in pins if Path(path).name == 'kev_e2e_client.py']
    require(len(runners) == 1, 'Expected exactly one pinned canonical runner')
    helpers = [path for path in pins if Path(path).name in ('kev_e2e_resume.py', 'kev_vllm_e2e_client.py')]
    require(len({Path(path).name for path in helpers}) == len(helpers), 'Duplicate offline helper pins')
    runners.extend(helpers)
    frozen = {}
    for path, value in pins.items():
        if path in runners:
            continue
        name = Path(path).name
        key = ('offline-script:' + name if Path(path).parent.name == 'scripts'
               and name in ('kev_requests_adapter.py', 'prepare_kev_e2e_panel.py') else path)
        require(key not in frozen or frozen[key] == value, 'Conflicting offline adapter/preparation pins')
        frozen[key] = value
    return frozen


@contextmanager
def run_lock(directory):
    with os.fdopen(os.open(directory / 'runner.lock', os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        yield


@contextmanager
def resume_source(directory, expected, verify):
    """Never resume a live writer or mix different frozen inputs/cells."""
    if directory is None:
        yield None
        return
    directory = Path(directory)
    binding = json.loads((directory / 'binding.json').read_bytes())
    require(binding.get('resume_format') in (None, 1), 'Unsupported resume format')
    old = {key: value for key, value in binding.items() if key not in ('input_sha256', 'resume_format')}
    new = {key: value for key, value in expected.items() if key not in ('input_sha256', 'resume_format')}
    require(old == new and frozen_inputs(binding['input_sha256']) == frozen_inputs(expected['input_sha256']),
            'Resume model/head/panel/policy or frozen input pins differ')
    verify(binding['input_sha256'])
    lock_path = directory / 'runner.lock'
    if lock_path.exists():
        lock = lock_path.open('rb')
    else:
        # Old runners have no lease; only an explicit terminal failure admits them.
        require(binding.get('resume_format') is None and (directory / 'failure.json').is_file(),
                'Legacy run has no terminal failure/lock; do not snapshot a possibly live writer')
        lock = None
    try:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ['binding.json', 'health-provenance.json', 'http.jsonl']
        names.extend(name for name in ('failure.json', 'summary.json', 'resume.json') if (directory / name).is_file())
        hashes = {name: digest(directory / name) for name in names}
        source = JournalSource(directory, hashes)
        yield source
        source.verify()
    finally:
        if lock is not None:
            lock.close()


class JournalSource:
    def __init__(self, directory, hashes):
        self.directory, self.hashes = directory, hashes
        self.provenance = json.loads((directory / 'health-provenance.json').read_bytes())
        self.initial_health = None
        self.processed = 0
        self.incomplete_tail = None

    def verify(self):
        require(all(digest(self.directory / name) == value for name, value in self.hashes.items()),
                'Resume evidence changed while being recovered')

    def decisions(self):
        """Yield a contiguous prefix; failures and a torn final line are not results."""
        ended = False
        with (self.directory / 'http.jsonl').open('rb') as handle:
            for line_number, line in enumerate(handle):
                if not line.endswith(b'\n'):
                    self.incomplete_tail = {'bytes': len(line), 'sha256': hashlib.sha256(line).hexdigest()}
                    break
                entry = json.loads(line)
                if line_number == 0:
                    require(entry.get('type') == 'health' and entry.get('status') == 200
                            and json.loads(entry['response_body']).get('provenance') == self.provenance,
                            'Resume journal lacks a matching initial healthy provenance')
                    self.initial_health = entry
                    continue
                require(not ended, 'Unexpected records after a terminal journal entry')
                if entry.get('type') == 'health':
                    ended = True
                    continue
                require(entry.get('type') == 'decision' and type(entry.get('index')) is int
                        and entry['index'] == self.processed, 'Resume decisions are duplicated, missing or reordered')
                if 'error' in entry:
                    require(isinstance(entry['error'], str), 'Malformed terminal failure')
                    ended = True
                    continue
                require(entry.get('status') == 200, 'Unvalidated non-200 result without a terminal failure')
                yield entry
                self.processed += 1
        require(self.initial_health is not None, 'Empty or torn initial health is not resumable')
        failure = self.directory / 'failure.json'
        if failure.is_file():
            value = json.loads(failure.read_bytes())
            require(value.get('status') == 'failed' and type(value.get('processed')) is int
                    and 0 <= self.processed - value['processed'] <= 1,
                    'Terminal failure and validated prefix disagree')
        self.verify()

    def receipt(self):
        return {'format': 1, 'source_directory': str(self.directory), 'source_sha256': self.hashes,
                'reused_requests': self.processed, 'ignored_incomplete_tail': self.incomplete_tail,
                'policy': 'Exact frozen inputs and loaded provenance; every carried response revalidated; no retries'}
