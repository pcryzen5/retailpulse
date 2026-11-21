"""Command-line interface for RetailPulse."""

from __future__ import annotations

import datetime
import json
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import sqlalchemy as sa
import typer

import retailpulse
from retailpulse.db.models.app import AppUser
from retailpulse.db.session import check_database, session_scope
from retailpulse.errors import DatabaseError, IngestionError, RetailPulseError
from retailpulse.etl.aggregate import AggregationResult, run_aggregation_pipeline
from retailpulse.etl.calendar import (
    ensure_dim_date,
    ensure_dim_store,
    evaluate_dataset_last_day_completeness,
    mark_dim_date_completeness,
)
from retailpulse.etl.clean import CleanedDataset, clean_sales_lines
from retailpulse.ingestion.adapters.generic_csv import GenericCSVAdapter
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter
from retailpulse.ingestion.loaders import LoadResult, load_sales_dataset
from retailpulse.logging import configure_logging
from retailpulse.security.auth import ensure_not_last_admin
from retailpulse.security.password import hash_password, validate_password
from retailpulse.timeutil import utc_now

app = typer.Typer(
    name="retailpulse",
    help="RetailPulse: AI-powered retail analytics and demand forecasting platform.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print version and exit when --version is supplied."""
    if value:
        typer.echo(f"RetailPulse version: {retailpulse.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[  # noqa: ARG001
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show application version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """RetailPulse CLI entrypoint."""
    configure_logging(level="INFO", json_format=False, stream=sys.stderr)


@app.command()
def version() -> None:
    """Display the RetailPulse application version."""
    typer.echo(f"RetailPulse version: {retailpulse.__version__}")


db_app = typer.Typer(
    name="db",
    help="Database inspection, migration, and maintenance commands.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")


@db_app.command("check")
def db_check(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output check result in JSON format.",
    ),
) -> None:
    """Verify database connectivity and migration head status."""
    try:
        status = check_database()
    except DatabaseError as exc:
        if json_output:
            typer.echo(json.dumps({"connected": False, "error": str(exc), "is_up_to_date": False}))
        else:
            typer.secho(f"Database check failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(status))
    else:
        typer.secho("Database connection: OK", fg=typer.colors.GREEN)
        typer.echo(f"Current revision: {status['current_revision'] or 'None'}")
        typer.echo(f"Head revision:    {status['head_revision'] or 'None'}")
        if status["is_up_to_date"]:
            typer.secho("Schema status:       UP TO DATE", fg=typer.colors.GREEN)
        else:
            typer.secho(
                "Schema status:       OUT OF DATE / UNAPPLIED MIGRATIONS",
                fg=typer.colors.YELLOW,
            )

    if not status["is_up_to_date"]:
        raise typer.Exit(code=1)


@db_app.command("seed-calendar")
def db_seed_calendar(
    start_year: int = typer.Option(2009, help="Start year for dim_date population."),
    end_year: int = typer.Option(2013, help="End year for dim_date population."),
) -> None:
    """Populate core.dim_date (2009-2013) and seed default store core.dim_store."""
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    typer.echo(f"Seeding core.dim_date from {start_date} to {end_date}...")
    try:
        with session_scope() as session:
            ensure_dim_store(session)
            inserted = ensure_dim_date(session, start_date=start_date, end_date=end_date)
    except Exception as exc:
        typer.secho(f"Calendar seed failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"Calendar seed complete. Inserted {inserted} new date rows.",
        fg=typer.colors.GREEN,
    )


etl_app = typer.Typer(
    name="etl",
    help="Data extraction, cleaning, loading, and aggregation commands.",
    no_args_is_help=True,
)
app.add_typer(etl_app, name="etl")


def _execute_etl_db_load(
    cleaned: CleanedDataset,
    source_lower: str,
    path: Path,
    chunk_size: int,
    *,
    skip_aggregation: bool,
    json_output: bool,
    last_date: datetime.date | None,
    is_complete: bool,
) -> tuple[LoadResult, AggregationResult | None]:
    """Execute database loading and aggregation in a transaction scope."""
    with session_scope() as session:
        ensure_dim_store(session)
        if not is_complete and last_date is not None:
            ensure_dim_date(session, last_date, last_date)
            mark_dim_date_completeness(session, last_date, is_complete=False)

        load_res = load_sales_dataset(
            session,
            cleaned,
            source_name=source_lower,
            source_file=str(path),
            chunk_size=chunk_size,
        )

        if not is_complete and last_date is not None:
            mark_dim_date_completeness(session, last_date, is_complete=False)

        agg_res: AggregationResult | None = None
        if not skip_aggregation and load_res.status == "succeeded":
            if not json_output:
                typer.echo("Running daily sales aggregation and ABC/XYZ classification...")
            agg_res = run_aggregation_pipeline(session, batch_id=load_res.batch_id)

        return load_res, agg_res


def _emit_etl_summary(
    load_res: LoadResult,
    cleaned: CleanedDataset,
    agg_res: AggregationResult | None,
    source_lower: str,
    path: Path,
    *,
    json_output: bool,
) -> None:
    """Format and display ETL execution results as JSON or human-readable console text."""
    if json_output:
        payload = {
            "status": load_res.status,
            "batch_id": str(load_res.batch_id),
            "source": source_lower,
            "path": str(path),
            "rows_in": load_res.rows_in,
            "rows_loaded": load_res.rows_loaded,
            "rows_skipped": load_res.rows_skipped,
            "rows_rejected": load_res.rows_rejected,
            "products_upserted": load_res.products_upserted,
            "customers_upserted": load_res.customers_upserted,
            "clean_counts": cleaned.row_counts,
            "daily_records_written": agg_res.daily_records_written if agg_res else 0,
            "abc_counts": agg_res.abc_counts if agg_res else {},
            "xyz_counts": agg_res.xyz_counts if agg_res else {},
            "data_as_of": str(agg_res.data_as_of) if agg_res and agg_res.data_as_of else None,
        }
        typer.echo(json.dumps(payload))
    else:
        typer.secho(
            f"ETL completed successfully! Batch: {load_res.batch_id}",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"  Raw rows read:         {load_res.rows_in:,}")
        typer.echo(f"  Sales lines loaded:    {load_res.rows_loaded:,}")
        typer.echo(f"  Sales lines skipped:   {load_res.rows_skipped:,}")
        typer.echo(f"  Products upserted:     {load_res.products_upserted:,}")
        typer.echo(f"  Customers upserted:    {load_res.customers_upserted:,}")
        if agg_res:
            typer.echo(f"  Daily records written: {agg_res.daily_records_written:,}")
            typer.echo(
                f"  ABC breakdown:         A={agg_res.abc_counts.get('A', 0)}, "
                f"B={agg_res.abc_counts.get('B', 0)}, C={agg_res.abc_counts.get('C', 0)}"
            )
            typer.echo(
                f"  XYZ breakdown:         X={agg_res.xyz_counts.get('X', 0)}, "
                f"Y={agg_res.xyz_counts.get('Y', 0)}, Z={agg_res.xyz_counts.get('Z', 0)}"
            )
            typer.echo(f"  Data as-of date:       {agg_res.data_as_of}")


def _get_adapter(source_lower: str) -> OnlineRetailIIAdapter | GenericCSVAdapter:
    """Instantiate adapter or raise IngestionError if unsupported."""
    if source_lower == "online_retail_ii":
        return OnlineRetailIIAdapter()
    if source_lower == "generic_csv":
        return GenericCSVAdapter()
    msg = f"Unsupported source '{source_lower}'. Must be 'online_retail_ii' or 'generic_csv'."
    raise IngestionError(msg, details={"source": source_lower})


def _read_and_clean(
    source_lower: str,
    path: Path,
    salt: str | None,
    dedupe: bool,
    *,
    json_output: bool,
) -> tuple[CleanedDataset, datetime.date | None, bool]:
    """Read raw records using the adapter and execute cleaning pipeline."""
    try:
        adapter = _get_adapter(source_lower)
    except IngestionError as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if not json_output:
        typer.echo(f"Reading dataset from {path} (adapter: {source_lower})...")

    try:
        raw_df = adapter.read(path)
    except IngestionError as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Failed to read dataset: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if not json_output:
        typer.echo(f"Read {len(raw_df):,} raw records. Cleaning and validating...")

    try:
        cleaned = clean_sales_lines(raw_df, salt=salt, dedupe_exact_rows=dedupe)
    except RetailPulseError as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Data cleaning failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    last_date: datetime.date | None = None
    is_complete: bool = True
    if not cleaned.df.empty:
        last_date, is_complete = evaluate_dataset_last_day_completeness(cleaned.df["invoice_ts"])

    return cleaned, last_date, is_complete


@etl_app.command("run")
def etl_run(  # noqa: PLR0917
    source: Annotated[
        str,
        typer.Option(
            "--source",
            "-s",
            help="Adapter name: 'online_retail_ii' or 'generic_csv'.",
        ),
    ] = "online_retail_ii",
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Path to source dataset file (CSV, XLSX, or Parquet).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = ...,  # type: ignore[assignment]
    dedupe: Annotated[
        bool,
        typer.Option(
            "--dedupe/--no-dedupe",
            help="Whether to drop exact duplicate rows within sheets (Rule 2b).",
        ),
    ] = True,
    salt: Annotated[
        str | None,
        typer.Option(
            "--salt",
            help="HMAC salt secret for customer pseudonymization (defaults to Settings).",
        ),
    ] = None,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Chunk size for sales lines insertion.",
        ),
    ] = 50000,
    skip_aggregation: Annotated[
        bool,
        typer.Option(
            "--skip-aggregation",
            help="Skip post-load daily sales aggregation and ABC/XYZ classification.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Execute complete ingestion, validation, loading, and aggregation pipeline."""
    if json_output:
        configure_logging(level="WARNING", json_format=False, stream=sys.stderr)
    else:
        configure_logging(level="INFO", json_format=False, stream=sys.stderr)

    source_lower = source.lower().strip()
    cleaned, last_date, is_complete = _read_and_clean(
        source_lower, path, salt, dedupe, json_output=json_output
    )

    if not json_output:
        typer.echo(f"Cleaned dataset: {len(cleaned.df):,} rows. Loading into database...")

    try:
        load_res, agg_res = _execute_etl_db_load(
            cleaned,
            source_lower,
            path,
            chunk_size,
            skip_aggregation=skip_aggregation,
            json_output=json_output,
            last_date=last_date,
            is_complete=is_complete,
        )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Database operation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _emit_etl_summary(
        load_res,
        cleaned,
        agg_res,
        source_lower,
        path,
        json_output=json_output,
    )


train_app = typer.Typer(
    name="train",
    help="Model training and champion selection workflows.",
    no_args_is_help=True,
)
app.add_typer(train_app, name="train")


def _emit_train_forecast_summary(result: Any, *, json_output: bool) -> None:
    """Format and print training pipeline leaderboard and gate verdict."""
    if json_output:
        out = {
            "status": "succeeded",
            "data_as_of": str(result.data_as_of),
            "n_series": result.total_series,
            "n_folds": result.n_folds,
            "horizon": result.horizon,
            "champion_model": result.champion_model,
            "champion_version": result.champion_version,
            "gate": {
                "passed": result.gate_result.passed_gate,
                "candidate": result.gate_result.candidate_model,
                "candidate_wape": result.gate_result.candidate_wape,
                "baseline_wape": result.gate_result.baseline_wape,
                "candidate_bias": result.gate_result.candidate_bias,
                "alert": result.gate_result.alert,
                "rejection_reasons": result.gate_result.rejection_reasons,
            },
            "models": {
                name: {
                    "global_wape": b.global_wape,
                    "global_bias": b.global_bias,
                    "global_smape": b.global_smape,
                    "registered_version": result.registered_versions.get(name),
                }
                for name, b in result.backtest_results.items()
            },
        }
        typer.echo(json.dumps(out, indent=2))
        return

    typer.secho(
        "\n================ Forecasting Backtest Leaderboard ================",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Cutoff Date (as_of): {result.data_as_of} | "
        f"Series: {result.total_series} | Folds: {result.n_folds} | "
        f"Horizon: {result.horizon}d\n"
    )
    typer.echo(f"{'Model':<18} {'WAPE':<10} {'Bias':<10} {'sMAPE':<10} {'Registered Ver':<15}")
    typer.echo("-" * 65)
    for name, b in result.backtest_results.items():
        ver = result.registered_versions.get(name, "-")
        typer.echo(
            f"{name:<18} {b.global_wape:<10.4f} {b.global_bias:<10.4f} "
            f"{b.global_smape:<10.4f} {ver:<15}"
        )

    typer.secho("\n---------------- Champion Gate Verdict ----------------", fg=typer.colors.CYAN)
    if result.gate_result.passed_gate:
        typer.secho(
            f"Gate: PASSED (Winner: {result.champion_model})",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.echo(
            f"Candidate WAPE: {result.gate_result.candidate_wape:.4f} < "
            f"Baseline WAPE: {result.gate_result.baseline_wape:.4f}"
        )
    else:
        typer.secho(
            f"Gate: FAILED (Fallback to Baseline: {result.champion_model})",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        for r in result.gate_result.rejection_reasons:
            typer.echo(f"  - {r}")
        if result.gate_result.alert:
            typer.secho(f"Alert: {result.gate_result.alert}", fg=typer.colors.RED)

    ver_label = result.champion_version or "unregistered"
    typer.echo(f"Champion: {result.champion_model} (Version: {ver_label})\n")


@train_app.command("forecast")
def train_forecast_command(  # noqa: PLR0917
    data_as_of: Annotated[
        str | None,
        typer.Option(
            "--data-as-of",
            help="Cutoff date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    n_folds: Annotated[
        int | None,
        typer.Option(
            "--n-folds",
            help="Number of backtest folds (defaults to config: 5).",
        ),
    ] = None,
    horizon: Annotated[
        int | None,
        typer.Option(
            "--horizon",
            help="Forecast horizon in days (defaults to config: 28).",
        ),
    ] = None,
    top_n: Annotated[
        int | None,
        typer.Option(
            "--top-n",
            help="Number of top SKUs to forecast (defaults to config: 300).",
        ),
    ] = None,
    models: Annotated[
        str,
        typer.Option(
            "--models",
            help="Comma-separated model names to evaluate (e.g. 'seasonal_naive,naive,lightgbm').",
        ),
    ] = "seasonal_naive,naive,lightgbm",
    auto_promote: Annotated[
        bool | None,
        typer.Option(
            "--auto-promote/--no-auto-promote",
            help="Automatically promote winning candidate to champion alias.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for deterministic execution.",
        ),
    ] = 42,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Execute demand forecasting training pipeline, backtesting, and champion gate."""
    from retailpulse.ml.forecasting.train import train_forecast_pipeline  # noqa: PLC0415

    parsed_as_of: datetime.date | None = None
    if data_as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(data_as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{data_as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    models_list = [m.strip().lower() for m in models.split(",") if m.strip()]

    if not json_output:
        typer.secho("Starting demand forecasting training pipeline...", fg=typer.colors.CYAN)

    try:
        with session_scope() as session:
            result = train_forecast_pipeline(
                data_as_of=parsed_as_of,
                session=session,
                models_to_run=models_list,
                n_folds=n_folds,
                horizon=horizon,
                top_n=top_n,
                auto_promote=auto_promote,
                seed=seed,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Training pipeline failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _emit_train_forecast_summary(result, json_output=json_output)


@train_app.command("segmentation")
def train_segmentation_command(  # noqa: PLR0917
    data_as_of: Annotated[
        str | None,
        typer.Option(
            "--data-as-of",
            help="Cutoff date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    k: Annotated[
        int | None,
        typer.Option(
            "--k",
            help="Explicit number of clusters (defaults to optimal search in 3..8).",
        ),
    ] = None,
    auto_promote: Annotated[
        bool,
        typer.Option(
            "--auto-promote",
            help="Promote newly trained model to active champion.",
        ),
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for clustering initialization (defaults to config: 42).",
        ),
    ] = None,
    rfm_window_days: Annotated[
        int | None,
        typer.Option(
            "--rfm-window-days",
            help="Lookback window for RFM features in days (defaults to config: 730).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Execute customer segmentation training, k selection, and profile generation."""
    from retailpulse.ml.segmentation.train import train_segmentation_pipeline  # noqa: PLC0415

    parsed_as_of: datetime.date | None = None
    if data_as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(data_as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{data_as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho("Starting customer segmentation training pipeline...", fg=typer.colors.CYAN)

    try:
        with session_scope() as session:
            result = train_segmentation_pipeline(
                session=session,
                data_as_of=parsed_as_of,
                k=k,
                auto_promote=auto_promote,
                seed=seed,
                rfm_window_days=rfm_window_days,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Segmentation training failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        out = {
            "status": "succeeded",
            "model_run_id": str(result.model_run_id),
            "data_as_of": str(result.data_as_of),
            "chosen_k": result.chosen_k,
            "silhouette": result.silhouette,
            "davies_bouldin": result.davies_bouldin,
            "mean_ari_stability": result.mean_ari_stability,
            "n_customers": result.n_customers,
            "is_champion": result.is_champion,
            "version": result.version,
            "profiles": result.profiles_df.to_dict(orient="records"),
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n================ Customer Segmentation Training Completed ================",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Model Run ID: {result.model_run_id}")
        typer.echo(
            f"Chosen k: {result.chosen_k} | Silhouette: {result.silhouette:.4f} | "
            f"DB: {result.davies_bouldin:.4f} | ARI Stability: {result.mean_ari_stability:.4f}"
        )
        typer.echo(
            f"Customers Clustered: {result.n_customers:,} | "
            f"Champion Promoted: {result.is_champion} (Version: {result.version or 'N/A'})\n"
        )
        typer.secho("Segment Profiles (100% customer coverage):", bold=True)
        for _, row in result.profiles_df.iterrows():
            pct = row["n_customers"] / result.n_customers
            typer.echo(
                f"  [{row['segment_id']}] {row['segment_label']:<16} | "
                f"Count: {int(row['n_customers']):>5} ({pct:>5.1%}) | "
                f"Rev: {float(row['revenue_share']):>5.1%} | "
                f"R: {row['avg_recency_days']:>5.1f}d | "
                f"F: {row['avg_frequency']:>4.1f} | "
                f"M: ${row['avg_monetary']:>9.2f}"
            )
        typer.echo()


@train_app.command("churn")
def train_churn_command(
    data_as_of: Annotated[
        str | None,
        typer.Option(
            "--data-as-of",
            help="Cutoff date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    churn_window_days: Annotated[
        int | None,
        typer.Option(
            "--churn-window-days",
            help="Outcome observation window in days (defaults to config: 90).",
        ),
    ] = None,
    auto_promote: Annotated[
        bool,
        typer.Option(
            "--auto-promote",
            help="Promote newly trained model to active champion if gate passes.",
        ),
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for model training (defaults to config: 42).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Execute customer churn model training, calibration, and champion gating."""
    from retailpulse.ml.churn.train import train_churn_pipeline  # noqa: PLC0415

    parsed_as_of: datetime.date | None = None
    if data_as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(data_as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{data_as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho("Starting customer churn training pipeline...", fg=typer.colors.CYAN)

    try:
        with session_scope() as session:
            result = train_churn_pipeline(
                session=session,
                data_as_of=parsed_as_of,
                churn_window_days=churn_window_days,
                auto_promote=auto_promote,
                seed=seed,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Churn training failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        out = {
            "status": "succeeded",
            "model_run_id": str(result.model_run_id),
            "data_as_of": str(result.data_as_of),
            "selected_algorithm": result.selected_algorithm,
            "calibration_method": result.calibration_method,
            "passed_gate": result.passed_gate,
            "candidate_pr_auc": result.candidate_pr_auc,
            "baseline_pr_auc": result.baseline_pr_auc,
            "candidate_metrics": result.candidate_metrics,
            "baseline_metrics": result.baseline_metrics,
            "is_champion": result.is_champion,
            "version": result.version,
            "train_cutoffs": [str(c) for c in result.train_cutoffs],
            "val_cutoffs": [str(c) for c in result.val_cutoffs],
            "test_cutoffs": [str(c) for c in result.test_cutoffs],
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        status_color = typer.colors.GREEN if result.passed_gate else typer.colors.YELLOW
        typer.secho(
            "\n================ Customer Churn Training Completed ================",
            fg=status_color,
        )
        typer.echo(f"Model Run ID: {result.model_run_id}")
        typer.echo(
            f"Selected Algorithm: {result.selected_algorithm} | "
            f"Calibration: {result.calibration_method or 'None'} | "
            f"Champion Gate Passed: {result.passed_gate}"
        )
        typer.echo(
            f"Candidate PR-AUC: {result.candidate_pr_auc:.4f} | "
            f"Baseline PR-AUC: {result.baseline_pr_auc:.4f} | "
            f"ROC-AUC: {result.candidate_metrics['roc_auc']:.4f}"
        )
        typer.echo(
            f"Brier Score: {result.candidate_metrics['brier_score']:.4f} "
            f"(Base Rate: {result.candidate_metrics['brier_score_base_rate']:.4f}) | "
            f"Prec@10%: {result.candidate_metrics['precision_top_10']:.4f} | "
            f"Lift@10%: {result.candidate_metrics['lift_top_10']:.2f}x"
        )
        typer.echo(
            f"Champion Promoted: {result.is_champion} (Version: {result.version or 'N/A'})\n"
        )
        typer.echo()


score_app = typer.Typer(
    name="score",
    help="Batch scoring and active output generation commands.",
    no_args_is_help=True,
)
app.add_typer(score_app, name="score")


@score_app.command("forecast")
def score_forecast_command(  # noqa: PLR0917
    data_as_of: Annotated[
        str | None,
        typer.Option(
            "--data-as-of",
            help="Origin cutoff date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    horizon: Annotated[
        int | None,
        typer.Option(
            "--horizon",
            help="Forecast horizon in trading days (defaults to config: 28).",
        ),
    ] = None,
    top_n: Annotated[
        int | None,
        typer.Option(
            "--top-n",
            help="Number of top SKUs to forecast (defaults to config: 300).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model algorithm to use (defaults to active champion or best baseline).",
        ),
    ] = None,
    interval_level: Annotated[
        float | None,
        typer.Option(
            "--interval-level",
            help="Nominal prediction interval coverage (defaults to 0.80).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Execute batch forecasting, generate intervals, and update ml.active_output."""
    from retailpulse.ml.forecasting.predict import predict_batch_forecast  # noqa: PLC0415

    parsed_as_of: datetime.date | None = None
    if data_as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(data_as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{data_as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho("Executing batch forecasting...", fg=typer.colors.CYAN)

    try:
        with session_scope() as session:
            result = predict_batch_forecast(
                session=session,
                data_as_of=parsed_as_of,
                model_name=model,
                horizon=horizon,
                top_n=top_n,
                interval_level=interval_level,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Batch forecasting failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        out = {
            "status": "succeeded",
            "model_name": result.model_name,
            "model_run_id": str(result.model_run_id),
            "origin_date": str(result.origin_date),
            "horizon": result.horizon,
            "n_series": result.n_series,
            "n_forecast_rows": result.n_forecast_rows,
            "interval_level": result.interval_level,
            "active_output_updated": result.active_output_updated,
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n================ Batch Forecast Completed ================",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Model: {result.model_name} (Run ID: {result.model_run_id})")
        typer.echo(
            f"Origin Date: {result.origin_date} | Horizon: {result.horizon}d | "
            f"Interval Level: {result.interval_level:.0%}"
        )
        typer.echo(
            f"Series Forecasted: {result.n_series:,} | "
            f"Forecast Rows Written: {result.n_forecast_rows:,}"
        )
        typer.secho(
            "Active Output Pointer: UPDATED (ml.active_output)\n",
            fg=typer.colors.GREEN,
            bold=True,
        )


@score_app.command("segmentation")
def score_segmentation_command(
    data_as_of: Annotated[
        str | None,
        typer.Option(
            "--data-as-of",
            help="Cutoff date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    model_run_id: Annotated[
        str | None,
        typer.Option(
            "--model-run-id",
            help="Explicit ModelRun UUID to score with (defaults to active champion).",
        ),
    ] = None,
    rfm_window_days: Annotated[
        int | None,
        typer.Option(
            "--rfm-window-days",
            help="Lookback window for RFM features in days (defaults to config: 730).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Score customers into segments and atomically update ml.active_output."""
    from retailpulse.ml.segmentation.score import score_segmentation_pipeline  # noqa: PLC0415

    parsed_as_of: datetime.date | None = None
    if data_as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(data_as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{data_as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    parsed_run_id: uuid.UUID | None = None
    if model_run_id:
        try:
            parsed_run_id = uuid.UUID(model_run_id)
        except ValueError as exc:
            msg = f"Invalid UUID format '{model_run_id}'"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho("Executing customer segmentation scoring...", fg=typer.colors.CYAN)

    try:
        with session_scope() as session:
            result = score_segmentation_pipeline(
                session=session,
                data_as_of=parsed_as_of,
                model_run_id=parsed_run_id,
                rfm_window_days=rfm_window_days,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Segmentation scoring failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        out = {
            "status": "succeeded",
            "model_run_id": str(result.model_run_id),
            "data_as_of": str(result.data_as_of),
            "n_customers": result.n_customers,
            "active_output_updated": result.active_output_updated,
            "profiles": result.profiles_df.to_dict(orient="records"),
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n================ Customer Segmentation Scoring Completed ================",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Model Run ID: {result.model_run_id}")
        typer.echo(f"Data As Of: {result.data_as_of} | Customers Scored: {result.n_customers:,}")
        typer.secho(
            "Active Output Pointer: UPDATED (ml.active_output)\n",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.secho("Segment Profiles (100% customer coverage):", bold=True)
        for _, row in result.profiles_df.iterrows():
            pct = row["n_customers"] / result.n_customers
            typer.echo(
                f"  [{row['segment_id']}] {row['segment_label']:<16} | "
                f"Count: {int(row['n_customers']):>5} ({pct:>5.1%}) | "
                f"Rev: {float(row['revenue_share']):>5.1%} | "
                f"R: {row['avg_recency_days']:>5.1f}d | "
                f"F: {row['avg_frequency']:>4.1f} | "
                f"M: ${row['avg_monetary']:>9.2f}"
            )
        typer.echo()


@score_app.command("churn")
def score_churn_command(
    data_as_of: Annotated[
        str | None,
        typer.Option(
            "--data-as-of",
            help="Cutoff date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    model_run_id: Annotated[
        str | None,
        typer.Option(
            "--model-run-id",
            help="Explicit ModelRun UUID to score with (defaults to active champion).",
        ),
    ] = None,
    lookback_days: Annotated[
        int,
        typer.Option(
            "--lookback-days",
            help="Trailing window for active customers in days (default: 365).",
        ),
    ] = 365,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Score active customers for churn risk and atomically update ml.active_output."""
    from retailpulse.ml.churn.predict import score_churn_pipeline  # noqa: PLC0415

    parsed_as_of: datetime.date | None = None
    if data_as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(data_as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{data_as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    parsed_run_id: uuid.UUID | None = None
    if model_run_id:
        try:
            parsed_run_id = uuid.UUID(model_run_id)
        except ValueError as exc:
            msg = f"Invalid UUID format '{model_run_id}'"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho("Executing customer churn batch scoring...", fg=typer.colors.CYAN)

    try:
        with session_scope() as session:
            result = score_churn_pipeline(
                session=session,
                data_as_of=parsed_as_of,
                model_run_id=parsed_run_id,
                lookback_days=lookback_days,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Churn scoring failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        out = {
            "status": "succeeded",
            "model_run_id": str(result.model_run_id),
            "algorithm": result.algorithm,
            "data_as_of": str(result.data_as_of),
            "n_customers": result.n_customers,
            "active_output_updated": result.active_output_updated,
            "band_counts": result.band_counts,
            "mean_churn_probability": round(result.mean_churn_probability, 4),
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n================ Customer Churn Scoring Completed ================",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Model Run ID: {result.model_run_id} ({result.algorithm})")
        typer.echo(
            f"Data As Of: {result.data_as_of} | Customers Scored: {result.n_customers:,} | "
            f"Mean Churn Prob: {result.mean_churn_probability:.2%}"
        )
        typer.secho(
            "Active Output Pointer: UPDATED (ml.active_output)\n",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.secho("Risk Band Distribution:", bold=True)
        total = max(1, result.n_customers)
        for band in ["high", "medium", "low"]:
            count = result.band_counts.get(band, 0)
            pct = count / total
            color = (
                typer.colors.RED
                if band == "high"
                else (typer.colors.YELLOW if band == "medium" else typer.colors.GREEN)
            )
            typer.secho(
                f"  {band.upper():<8} | Count: {count:>5} ({pct:>5.1%})",
                fg=color,
            )
        typer.echo()


@score_app.command("inventory")
def score_inventory_command(
    store_id: Annotated[
        int,
        typer.Option(
            "--store-id",
            help="Store ID to generate recommendations for (defaults to 1).",
        ),
    ] = 1,
    data_as_of: Annotated[
        str | None,
        typer.Option(
            "--data-as-of",
            help="Origin cutoff date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    forecast_run_id: Annotated[
        str | None,
        typer.Option(
            "--forecast-run-id",
            help="Optional explicit forecast model_run_id UUID (defaults to active champion).",
        ),
    ] = None,
    overstock_days: Annotated[
        float | None,
        typer.Option(
            "--overstock-days",
            help="Days of cover threshold for overstock status (defaults to config: 90.0).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output execution summary in JSON format.",
        ),
    ] = False,
) -> None:
    """Generate replenishment recommendations and update ml.active_output."""
    from retailpulse.ml.inventory.recommend import (  # noqa: PLC0415
        generate_inventory_recommendations,
    )

    parsed_as_of: datetime.date | None = None
    if data_as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(data_as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{data_as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    parsed_run_id: uuid.UUID | None = None
    if forecast_run_id:
        try:
            parsed_run_id = uuid.UUID(forecast_run_id)
        except ValueError as exc:
            msg = f"Invalid UUID format '{forecast_run_id}'"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho(
            "Generating inventory replenishment recommendations...",
            fg=typer.colors.CYAN,
        )

    try:
        with session_scope() as session:
            result = generate_inventory_recommendations(
                session=session,
                store_id=store_id,
                data_as_of=parsed_as_of,
                forecast_run_id=parsed_run_id,
                overstock_days=overstock_days,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Inventory recommendation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        out = {
            "status": "succeeded",
            "forecast_run_id": str(result.forecast_run_id),
            "store_id": result.store_id,
            "data_as_of": str(result.data_as_of),
            "n_recommendations": result.n_recommendations,
            "status_counts": result.status_counts,
            "mean_stockout_risk": round(result.mean_stockout_risk, 4),
            "total_recommended_order_qty": result.total_recommended_order_qty,
            "active_output_updated": result.active_output_updated,
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n================ Inventory Recommendations Generated ================",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Forecast Run ID: {result.forecast_run_id} | Store ID: {result.store_id}")
        typer.echo(
            f"Data As Of: {result.data_as_of} | SKUs Evaluated: {result.n_recommendations:,} | "
            f"Mean Stockout Risk: {result.mean_stockout_risk:.2%}"
        )
        typer.echo(f"Total Recommended Order Qty: {result.total_recommended_order_qty:,} units")
        typer.secho(
            "Active Output Pointer: UPDATED (ml.active_output)\n",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.secho("Inventory Status Distribution:", bold=True)
        total = max(1, result.n_recommendations)
        for st_name in ["reorder_now", "overstock", "ok", "no_data"]:
            count = result.status_counts.get(st_name, 0)
            pct = count / total
            color = (
                typer.colors.RED
                if st_name == "reorder_now"
                else (
                    typer.colors.YELLOW
                    if st_name == "overstock"
                    else (typer.colors.GREEN if st_name == "ok" else typer.colors.WHITE)
                )
            )
            typer.secho(
                f"  {st_name.upper():<12} | Count: {count:>5} ({pct:>5.1%})",
                fg=color,
            )
        typer.echo()


# =====================================================================
# Inventory Optimization Subcommands
# =====================================================================

inventory_app = typer.Typer(
    name="inventory",
    help="Inventory optimization and parameter management commands.",
    no_args_is_help=True,
)
app.add_typer(inventory_app, name="inventory")


@inventory_app.command("seed-params")
def seed_params_command(
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            help="Snapshot date in YYYY-MM-DD format (defaults to latest complete date).",
        ),
    ] = None,
    store_id: Annotated[
        int,
        typer.Option(
            "--store-id",
            help="Store ID to seed parameters and snapshots for.",
        ),
    ] = 1,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for deterministic generation (defaults to config: 42).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results as JSON.",
        ),
    ] = False,
) -> None:
    """Seed deterministic synthetic inventory parameters and initial snapshots."""
    from retailpulse.ml.inventory.params import seed_inventory_params  # noqa: PLC0415

    parsed_as_of: datetime.date | None = None
    if as_of:
        try:
            parsed_as_of = datetime.date.fromisoformat(as_of)
        except ValueError as exc:
            msg = f"Invalid date format '{as_of}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho(
            "Seeding synthetic inventory parameters and snapshots...",
            fg=typer.colors.CYAN,
        )

    try:
        with session_scope() as session:
            result = seed_inventory_params(
                session=session,
                store_id=store_id,
                as_of=parsed_as_of,
                seed=seed,
            )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Inventory seeding failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        out = {
            "status": "succeeded",
            "store_id": result.store_id,
            "data_as_of": str(result.data_as_of),
            "n_params_seeded": result.n_params_seeded,
            "n_snapshots_seeded": result.n_snapshots_seeded,
            "seed": result.seed,
            "source": result.source,
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n================ Synthetic Inventory Parameters Seeded ================",
            fg=typer.colors.GREEN,
        )
        typer.echo(
            f"Store ID: {result.store_id} | Snapshot Date: {result.data_as_of} | "
            f"Seed: {result.seed} | Source: {result.source}"
        )
        typer.echo(
            f"Parameters Seeded: {result.n_params_seeded:,} | "
            f"Snapshots Seeded: {result.n_snapshots_seeded:,}"
        )
        typer.secho(
            "Notice: Parameters and snapshots are synthetic for simulation purposes.",
            fg=typer.colors.YELLOW,
        )
        typer.echo()


@inventory_app.command("simulate")
def simulate_inventory_command(  # noqa: PLR0917, PLR0915
    store_id: Annotated[
        int,
        typer.Option("--store-id", help="Store identifier to simulate."),
    ] = 1,
    window_start: Annotated[
        str | None,
        typer.Option("--window-start", help="Simulation start date in YYYY-MM-DD format."),
    ] = None,
    window_end: Annotated[
        str | None,
        typer.Option("--window-end", help="Simulation end date in YYYY-MM-DD format."),
    ] = None,
    abc_class: Annotated[
        str | None,
        typer.Option("--abc-class", help="Scope filter: 'A', 'B', 'C', or 'all'."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of SKUs to simulate (max 200)."),
    ] = 100,
    service_level: Annotated[
        float | None,
        typer.Option("--service-level", help="Target cycle service level for RetailPulse policy."),
    ] = None,
    cover_days_extra: Annotated[
        int,
        typer.Option(
            "--cover-days-extra",
            help="Extra cover days for baseline policy (default 14).",
        ),
    ] = 14,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Deterministic random seed (defaults to config: 42)."),
    ] = None,
    report: Annotated[
        bool,
        typer.Option("--report/--no-report", help="Generate markdown benchmark report."),
    ] = True,
    output: Annotated[
        Path,
        typer.Option("--output", help="Output path for simulation markdown report."),
    ] = Path("reports/inventory_simulation.md"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results in JSON format."),
    ] = False,
) -> None:
    """Run historical lost-sales inventory policy simulation and benchmark against baseline."""
    from retailpulse.ml.inventory.report import generate_simulation_report  # noqa: PLC0415
    from retailpulse.ml.inventory.simulate import (  # noqa: PLC0415
        SimulationConfig,
        run_policy_simulation,
    )

    parsed_start: datetime.date | None = None
    if window_start:
        try:
            parsed_start = datetime.date.fromisoformat(window_start)
        except ValueError as exc:
            msg = f"Invalid window-start format '{window_start}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    parsed_end: datetime.date | None = None
    if window_end:
        try:
            parsed_end = datetime.date.fromisoformat(window_end)
        except ValueError as exc:
            msg = f"Invalid window-end format '{window_end}': must be YYYY-MM-DD"
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if not json_output:
        typer.secho("Executing inventory policy simulation replay...", fg=typer.colors.CYAN)

    cfg = SimulationConfig(
        store_id=store_id,
        window_start=parsed_start,
        window_end=parsed_end,
        abc_class=abc_class,
        limit=limit,
        service_level=service_level,
        cover_days_extra=cover_days_extra,
        seed=seed,
        persist=True,
    )

    try:
        with session_scope() as session:
            result = run_policy_simulation(session=session, config=cfg)
            report_generated_path: Path | None = None
            if report:
                report_generated_path = generate_simulation_report(
                    result=result,
                    output_path=output,
                    session=session,
                )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            typer.secho(f"Simulation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    rp = result.results.get("retailpulse", {})
    base = result.results.get("baseline", {})
    delta = result.results.get("delta", {})

    if json_output:
        out = {
            "status": "succeeded",
            "simulation_id": str(result.simulation_id),
            "window": {
                "start": str(result.window_start),
                "end": str(result.window_end),
            },
            "uses_synthetic_params": result.uses_synthetic_params,
            "results": result.results,
            "report_path": str(report_generated_path) if report_generated_path else None,
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.secho(
            "\n================ Inventory Policy Simulation Completed ================",
            fg=typer.colors.GREEN,
        )
        typer.echo(
            f"Simulation ID: {result.simulation_id} | "
            f"Window: {result.window_start} to {result.window_end}"
        )
        typer.secho(
            "Notice: Policy simulation evaluates synthetic inventory parameters.",
            fg=typer.colors.YELLOW,
        )
        typer.echo("\nPolicy Benchmark Comparison:")
        fill_rp = rp.get("fill_rate", 0.0) * 100.0
        fill_base = base.get("fill_rate", 0.0) * 100.0
        fill_diff = delta.get("fill_rate_diff", 0.0) * 100.0
        typer.echo(
            f"  - Demand Fill Rate:         RP {fill_rp:.2f}% vs Base {fill_base:.2f}% "
            f"({fill_diff:+.2f} pp)"
        )

        stk_rp = rp.get("stockout_day_rate", 0.0) * 100.0
        stk_base = base.get("stockout_day_rate", 0.0) * 100.0
        stk_diff = delta.get("stockout_day_rate_diff", 0.0) * 100.0
        typer.echo(
            f"  - Stockout-Day Frequency:   RP {stk_rp:.2f}% vs Base {stk_base:.2f}% "
            f"({stk_diff:+.2f} pp)"
        )

        val_rp = rp.get("avg_on_hand_value", 0.0)
        val_base = base.get("avg_on_hand_value", 0.0)
        val_diff = delta.get("avg_on_hand_value_diff", 0.0)
        typer.echo(
            f"  - Average Inventory Value:  RP £{val_rp:,.2f} vs Base £{val_base:,.2f} "
            f"(£{val_diff:+,.2f})"
        )

        h_rp = rp.get("holding_cost", 0.0)
        h_base = base.get("holding_cost", 0.0)
        h_diff = delta.get("holding_cost_diff", 0.0)
        typer.echo(
            f"  - Total Holding Cost:       RP £{h_rp:,.2f} vs Base £{h_base:,.2f} "
            f"(£{h_diff:+,.2f})"
        )

        ord_rp = rp.get("n_orders", 0)
        ord_base = base.get("n_orders", 0)
        typer.echo(f"  - Purchase Orders Placed:   RP {ord_rp:,} vs Base {ord_base:,}")

        if report_generated_path:
            typer.secho(f"\nReport written to: {report_generated_path}", fg=typer.colors.CYAN)
        typer.echo()


users_app = typer.Typer(
    name="users",
    help="User access and account management commands.",
    no_args_is_help=True,
)
app.add_typer(users_app, name="users")


@users_app.command("create")
def create_user_command(
    email: Annotated[
        str,
        typer.Option("--email", "-e", help="User email address."),
    ],
    role: Annotated[
        str,
        typer.Option("--role", "-r", help="Role: admin, analyst, or viewer."),
    ] = "analyst",
    full_name: Annotated[
        str | None,
        typer.Option("--full-name", help="User's full name."),
    ] = None,
    password: Annotated[
        str | None,
        typer.Option("--password", "-p", help="User password (prompted securely if omitted)."),
    ] = None,
) -> None:
    """Create a new user with Argon2id password hashing (J1 setup)."""
    clean_email = email.strip().lower()
    clean_role = role.strip().lower()

    if clean_role not in ("admin", "analyst", "viewer"):
        typer.secho(
            f"Error: Invalid role '{role}'. Must be admin, analyst, or viewer.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if password is None:
        password = typer.prompt("Password", hide_input=True, confirmation_prompt=True)

    try:
        validate_password(password)
    except Exception as exc:
        typer.secho(f"Password Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    pwd_hash = hash_password(password, validate=False)

    with session_scope() as session:
        existing = session.execute(
            sa.select(AppUser.user_id).where(AppUser.email == clean_email)
        ).scalar_one_or_none()
        if existing is not None:
            typer.secho(
                f"Error: User with email '{clean_email}' already exists.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        user = AppUser(
            user_id=uuid.uuid4(),
            email=clean_email,
            full_name=full_name.strip() if full_name else None,
            role=clean_role,
            password_hash=pwd_hash,
            is_active=True,
        )
        session.add(user)
        session.commit()

        typer.secho(
            f"Successfully created user {clean_email} ({clean_role}) [ID: {user.user_id}]",
            fg=typer.colors.GREEN,
        )


@users_app.command("list")
def list_users_command(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format."),
    ] = False,
) -> None:
    """List all registered users."""
    with session_scope() as session:
        users = (
            session.execute(sa.select(AppUser).order_by(AppUser.created_at.asc())).scalars().all()
        )

        if json_output:
            data = [
                {
                    "user_id": str(u.user_id),
                    "email": u.email,
                    "full_name": u.full_name,
                    "role": u.role,
                    "is_active": u.is_active,
                    "failed_login_count": u.failed_login_count,
                    "locked_until": u.locked_until.isoformat() if u.locked_until else None,
                    "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                }
                for u in users
            ]
            typer.echo(json.dumps(data, indent=2))
            return

        if not users:
            typer.echo("No registered users found.")
            return

        typer.echo(
            f"{'Email':<30} {'Role':<10} {'Active':<8} {'Locked':<8} {'Full Name':<25} {'User ID'}"
        )
        typer.echo("-" * 110)
        now_dt = utc_now()
        for u in users:
            is_locked = "Yes" if (u.locked_until and u.locked_until > now_dt) else "No"
            active_str = "Yes" if u.is_active else "No"
            name_str = (u.full_name or "-")[:24]
            typer.echo(
                f"{u.email:<30} {u.role:<10} {active_str:<8} "
                f"{is_locked:<8} {name_str:<25} {u.user_id}"
            )


@users_app.command("deactivate")
def deactivate_user_command(
    identifier: Annotated[
        str,
        typer.Argument(help="User email address or UUID to deactivate."),
    ],
) -> None:
    """Deactivate an active user account (with last-admin protection)."""
    clean_id = identifier.strip().lower()

    with session_scope() as session:
        user: AppUser | None = None
        try:
            target_uuid = uuid.UUID(clean_id)
            user = session.execute(
                sa.select(AppUser).where(AppUser.user_id == target_uuid)
            ).scalar_one_or_none()
        except ValueError:
            pass

        if user is None:
            user = session.execute(
                sa.select(AppUser).where(AppUser.email == clean_id)
            ).scalar_one_or_none()

        if user is None:
            typer.secho(f"Error: User '{identifier}' not found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        try:
            ensure_not_last_admin(session, user, new_is_active=False)
        except Exception as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        user.is_active = False
        session.commit()
        typer.secho(
            f"Successfully deactivated user {user.email} ({user.role}) [ID: {user.user_id}]",
            fg=typer.colors.GREEN,
        )
