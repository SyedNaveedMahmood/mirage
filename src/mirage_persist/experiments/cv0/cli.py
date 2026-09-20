"""CLI for CV-0: ``mirage cv0 run`` and ``mirage smoke {mock,hf}``.

Every command runs inside a RunContext (provenance + terminal capture). Exit code:
0 = PASS, 1 = FAIL, 2 = KILL, so CI can gate on the CV-0 verdict.
"""

from __future__ import annotations

from pathlib import Path

import typer

from mirage_persist.config.loader import load_config
from mirage_persist.config.schema import CV0Config
from mirage_persist.experiments.cv0.report import run_cv0
from mirage_persist.run.run_context import RunContext

cv0_app = typer.Typer(name="cv0", help="CV-0 calibration instrument.", no_args_is_help=True, add_completion=False)
smoke_app = typer.Typer(name="smoke", help="Brief smoke tests.", no_args_is_help=True, add_completion=False)

_CONFIGS = Path(__file__).resolve().parents[4] / "configs"

_EXIT = {"PASS": 0, "FAIL": 1, "KILL": 2, "INFO": 0}


def _parse_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _run(config_path: Path, overrides: list[str], subcommand: str) -> int:
    cfg: CV0Config = load_config(config_path, CV0Config, overrides=_parse_overrides(overrides))
    with RunContext(cfg, subcommand=subcommand, config_path=config_path) as ctx:
        report = run_cv0(cfg, ctx)
    return _EXIT.get(report.overall, 0)


@cv0_app.command("run")
def cv0_run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="CV-0 config YAML."),
    override: list[str] = typer.Option([], "-o", "--override", help="Dotted override key=value."),
) -> None:
    """Run all four CV-0 sub-experiments, evaluate gates, write artifacts."""
    raise typer.Exit(code=_run(config, override, "cv0 run"))


@smoke_app.command("mock")
def smoke_mock(
    override: list[str] = typer.Option([], "-o", "--override", help="Dotted override key=value."),
) -> None:
    """GPU-free byte-exact engine + determinism smoke (mock backend)."""
    raise typer.Exit(code=_run(_CONFIGS / "smoke" / "cv0_smoke_mock.yaml", override, "smoke mock"))


@smoke_app.command("hf")
def smoke_hf(
    override: list[str] = typer.Option([], "-o", "--override", help="Dotted override key=value."),
) -> None:
    """Tiny real-model (Qwen2.5-1.5B) decoder-determinism smoke on the local GPU."""
    raise typer.Exit(code=_run(_CONFIGS / "smoke" / "cv0_smoke_hf.yaml", override, "smoke hf"))


__all__ = ["cv0_app", "smoke_app"]
