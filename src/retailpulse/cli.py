"""Command-line interface for RetailPulse."""

from __future__ import annotations

import datetime
import json

import typer

import retailpulse
from retailpulse.db.session import check_database, session_scope
from retailpulse.errors import DatabaseError
from retailpulse.etl.calendar import ensure_dim_date, ensure_dim_store

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
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show application version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """RetailPulse CLI entrypoint."""


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
