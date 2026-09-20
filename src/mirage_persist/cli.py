"""``mirage`` command-line interface.

Subcommand groups are added as their drivers land:
    mirage version
    mirage check-substrate [--suite banking] [--version v1.2.2]
    mirage cv0 run|determinism|capability|potency|variance|report --config ...
    mirage smoke mock|hf

Every command that runs an experiment does so inside a RunContext (provenance +
terminal capture). CLI overrides are passed as ``-o key=value`` (dotted paths).
"""

from __future__ import annotations

import typer

from mirage_persist.run.terminal_capture import enable_utf8_stdio
from mirage_persist.version import __version__

enable_utf8_stdio()

app = typer.Typer(
    name="mirage",
    help="MIRAGE-Persist CV-0 calibration instrument.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the MIRAGE-Persist version and the pinned substrate revision."""
    from mirage_persist.version import EXPECTED_AGENTDOJO_SHA, EXPECTED_AGENTDOJO_VERSION

    typer.echo(f"mirage-persist {__version__}")
    typer.echo(f"substrate: agentdojo {EXPECTED_AGENTDOJO_VERSION} @ {EXPECTED_AGENTDOJO_SHA}")


@app.command("check-substrate")
def check_substrate(
    suite: str = typer.Option("banking", help="Suite to self-check."),
    benchmark_version: str = typer.Option("v1.2.2", "--version", help="AgentDojo benchmark version."),
    check_injectable: bool = typer.Option(
        False,
        help="Also run AgentDojo's canary injectability check (BROKEN in v0.1.35 for the "
        "content-block message format; MIRAGE verifies injection separately).",
    ),
) -> None:
    """Verify ground truth solves each task and injection tasks are achievable.

    Injectability defaults OFF: AgentDojo v0.1.35's ``is_task_injectable`` filters tool
    outputs by ``isinstance(content, str)``, but the new format stores content as a list
    of blocks, so it always reports "not injectable". MIRAGE's real injection is validated
    by the potency screen and ``tests/test_interventions.py`` instead.
    """
    from mirage_persist.substrate.dojo.adapter import check_suite

    ok = check_suite(benchmark_version, suite, check_injectable=check_injectable)
    typer.echo(f"substrate check ({benchmark_version}/{suite}): {'PASS' if ok else 'FAIL'}")
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def preflight(
    config: str = typer.Option(..., "--config", "-c", help="Experiment config YAML (its models are probed)."),
    json_out: str = typer.Option("", "--json-out", help="Write the preflight report to this JSON file."),
) -> None:
    """Probe each configured model server: /v1/models, one completion, one tool call.

    Cheap (a few dozen tokens) and read-only: it never mutates the configuration
    it is given. Intended to run right after a server starts and before an
    expensive campaign. Exit code 0 = every check passed.
    """
    import json as _json
    from pathlib import Path

    from mirage_persist.config.loader import load_config
    from mirage_persist.config.schema import CV0Config
    from mirage_persist.models.preflight import run_preflight

    cfg: CV0Config = load_config(config, CV0Config)
    reports = []
    ok = True
    for model in cfg.models:
        typer.echo(f"preflight: {model.name} ({model.model_id})")
        report = run_preflight(model)
        for check in report.checks:
            typer.echo(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
        reports.append(report.to_dict())
        ok = ok and report.ok

    if json_out:
        out = Path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps({"ok": ok, "models": reports}, indent=2), encoding="utf-8")
        typer.echo(f"preflight report: {out}")

    typer.echo(f"preflight: {'PASS' if ok else 'FAIL'}")
    raise typer.Exit(code=0 if ok else 1)


def _register_optional_groups() -> None:
    """Attach experiment subcommand groups if their modules import cleanly.

    Kept lazy so a partial checkout (or a missing heavy dep) still yields a
    working ``mirage version`` / ``mirage check-substrate``.
    """
    try:
        from mirage_persist.experiments.cv0.cli import cv0_app

        app.add_typer(cv0_app, name="cv0")
    except Exception:  # pragma: no cover - optional wiring
        pass
    try:
        from mirage_persist.experiments.cv0.cli import smoke_app

        app.add_typer(smoke_app, name="smoke")
    except Exception:  # pragma: no cover
        pass


_register_optional_groups()


if __name__ == "__main__":
    app()
