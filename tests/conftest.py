import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
sys.path.insert(0, ROOT)


@pytest.fixture
def fixture_dir():
    return FIXTURES


@pytest.fixture
def cosis():
    from common import read_cosi_list
    return read_cosi_list(os.path.join(FIXTURES, "cosis_fixture.txt"))
