"""Report selection and serialization shared by the CLI and Python callers."""

from .json import (
    SCHEMA_VERSION,
    check_result_to_json,
    search_result_to_json,
    to_json_text,
    unit_to_dict,
)
from .selection import (
    ACTIONABLE_TIERS,
    WITHHELD_TIERS,
    FailOnPolicy,
    FileSearchResult,
    ReportPolicy,
    ReportSelection,
    assign_unit_ids,
    collect_units,
    group_file_results,
    run_should_fail,
    select_findings,
    withheld_only_failure,
)

__all__ = [
    "ACTIONABLE_TIERS",
    "SCHEMA_VERSION",
    "WITHHELD_TIERS",
    "FailOnPolicy",
    "FileSearchResult",
    "ReportPolicy",
    "ReportSelection",
    "assign_unit_ids",
    "check_result_to_json",
    "collect_units",
    "group_file_results",
    "run_should_fail",
    "search_result_to_json",
    "select_findings",
    "to_json_text",
    "unit_to_dict",
    "withheld_only_failure",
]
