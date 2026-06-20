import pytest

from tests.perf_utils import assert_perf_geomean, perf_cases_for, run_perf_case


_PERF_CASES = perf_cases_for("rad2deg")


@pytest.mark.parametrize("case", _PERF_CASES, ids=lambda case: case.case_name)
def test_rad2deg_perf(case):
    run_perf_case(case)


def test_rad2deg_perf_geomean():
    assert_perf_geomean("rad2deg")
