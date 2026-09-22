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
    DEFAULT_MAX_DUPLICATES,
    WITHHELD_TIERS,
    FailOnPolicy,
    FileSearchResult,
    HiddenGroup,
    ReportPolicy,
    ReportSelection,
    actionable_pairs,
    assign_unit_ids,
    collect_units,
    group_file_results,
    hidden_only_failure,
    run_should_fail,
    select_findings,
)

__all__ = [
    "ACTIONABLE_TIERS",
    "DEFAULT_MAX_DUPLICATES",
    "SCHEMA_VERSION",
    "WITHHELD_TIERS",
    "FailOnPolicy",
    "FileSearchResult",
    "HiddenGroup",
    "ReportPolicy",
    "ReportSelection",
    "actionable_pairs",
    "assign_unit_ids",
    "check_result_to_json",
    "collect_units",
    "group_file_results",
    "hidden_only_failure",
    "run_should_fail",
    "search_result_to_json",
    "select_findings",
    "to_json_text",
    "unit_to_dict",
]
