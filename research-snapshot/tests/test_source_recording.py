"""Portable provenance recording without engines or jobs."""

import subprocess

import pytest

from fuzev_research import paths


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")])
def test_source_revision_uses_installed_version_without_git(tmp_path, monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(paths.subprocess, "check_output", unavailable)
    monkeypatch.setattr(paths, "version", lambda name: "1.0.0")
    assert paths.source_revision(tmp_path, "fuzev-research") == "1.0.0"
    assert paths.source_revision(tmp_path) is None
