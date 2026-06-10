import pytest

from tests.t1_1_9_performance_utils import perf_cases_for, run_perf_case


_PERF_CASES = perf_cases_for("scatter_add")


@pytest.mark.parametrize("case", _PERF_CASES, ids=lambda case: case.case_name)
def test_scatter_add_performance(case):
    run_perf_case(case)
