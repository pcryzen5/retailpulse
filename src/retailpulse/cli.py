"""Command-line interface for RetailPulse."""

from __future__ import annotations

import json

import typer

import retailpulse
from retailpulse.db.session import check_database
from retailpulse.errors import DatabaseError

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
