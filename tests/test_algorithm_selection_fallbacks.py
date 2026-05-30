from types import SimpleNamespace
import sys
import types

from causal_discovery.context.algos.utils.json2txt import (
    create_filtered_benchmarking_results,
    create_filtered_benchmarking_results_ts,
)

sys.modules.setdefault("llm", types.SimpleNamespace(LLMClient=object))

from causal_discovery.filter import Filter


def test_filter_falls_back_when_llm_returns_no_algorithm_candidates():
    filter_obj = object.__new__(Filter)
    global_state = SimpleNamespace(statistics=SimpleNamespace(time_series=False))

    candidates = filter_obj.ensure_algorithm_candidates({}, global_state)

    assert list(candidates) == ["PC", "GES"]
    assert "LLM filter returned no algorithm candidates" in candidates["PC"]["justification"]


def test_benchmarking_filter_returns_text_for_empty_algorithm_list():
    result = create_filtered_benchmarking_results({"Linear Function": {"PC": {}}}, [])

    assert isinstance(result, str)
    assert "No algorithms specified" in result


def test_time_series_benchmarking_filter_returns_text_for_empty_algorithm_list():
    result = create_filtered_benchmarking_results_ts({"Linear Function": {"PCMCI": {}}}, [])

    assert isinstance(result, str)
    assert "No algorithms specified" in result
