"""One Python version, named in four places, checked here.

The committed artefacts were generated on Python 3.14. Until this test existed
the repository disagreed with itself about that -- requirements.txt asked for
>= 3.10, the Dockerfile pinned 3.11, CI matrixed 3.11 and 3.12 -- so a reviewer
reproducing the results had three defensible choices and none of them was the
one the numbers came from. The four files below must keep naming 3.14.

pyproject.toml is deliberately NOT in the list: it has no [project] table (the
numbered directories are not importable packages), so it has no requires-python
to pin.
"""

import os
import re

import pytest

from conftest import PROJECT_ROOT

PINNED = '3.14'

REQUIREMENTS = os.path.join(PROJECT_ROOT, 'requirements.txt')
REQUIREMENTS_DEV = os.path.join(PROJECT_ROOT, 'requirements-dev.txt')
DOCKERFILE = os.path.join(PROJECT_ROOT, 'Dockerfile')
WORKFLOW = os.path.join(PROJECT_ROOT, '.github', 'workflows', 'ci.yml')


def _read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


@pytest.mark.parametrize('path', [REQUIREMENTS, REQUIREMENTS_DEV, DOCKERFILE, WORKFLOW])
def test_file_names_the_pinned_version(path):
    assert PINNED in _read(path), f'{os.path.basename(path)} no longer names Python {PINNED}'


def test_requirements_states_the_pin_in_prose():
    """A bare '3.14' could be a library pin; the header must say Python."""
    header = [line for line in _read(REQUIREMENTS).splitlines() if line.startswith('#')]
    assert any(PINNED in line and 'Python' in line for line in header), header


def test_requirements_dev_states_the_pin_in_prose():
    header = [line for line in _read(REQUIREMENTS_DEV).splitlines() if line.startswith('#')]
    assert any(PINNED in line and 'Python' in line for line in header), header


def test_dockerfile_base_image_is_the_pinned_version():
    from_lines = [line for line in _read(DOCKERFILE).splitlines()
                  if line.strip().upper().startswith('FROM ')]
    assert from_lines, 'Dockerfile has no FROM line'
    for line in from_lines:
        assert f'python:{PINNED}' in line, line


def test_every_ci_python_version_is_the_pinned_version():
    """No matrix, no stragglers: every python-version in CI is the same one."""
    versions = re.findall(r'python-version:\s*[\'"]?([0-9.]+)', _read(WORKFLOW))
    assert versions, 'ci.yml sets no python-version'
    assert set(versions) == {PINNED}, versions
