"""Command-line interface for RetailPulse."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

import retailpulse
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
