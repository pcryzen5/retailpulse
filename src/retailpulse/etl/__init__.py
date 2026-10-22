"""ETL cleaning, transformation, and aggregation pipelines."""

from retailpulse.etl.calendar import (
    build_calendar_records,
    days_to_christmas,
    detect_closure_windows,
    ensure_dim_date,
    ensure_dim_store,
    evaluate_dataset_last_day_completeness,
    is_in_closure_window,
    is_partial_last_day,
    mark_dim_date_completeness,
    project_closure_windows,
)

__all__ = [
    "build_calendar_records",
    "days_to_christmas",
    "detect_closure_windows",
    "ensure_dim_date",
    "ensure_dim_store",
    "evaluate_dataset_last_day_completeness",
    "is_in_closure_window",
    "is_partial_last_day",
    "mark_dim_date_completeness",
    "project_closure_windows",
]
