"""``codedupes info``: diagnostics layout and environment setup."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest
from click.testing import CliRunner

from codedupes import cli
from codedupes.devices import DeviceDiagnostics


@pytest.mark.parametrize("flag", ["--verbose", "-v"])
def test_cli_info_verbose_exit_zero(flag):
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["info", flag])
    assert result.exit_code == 0
    assert "codedupes" in result.output.lower()
    assert "NumPy" in result.output
    assert "PyTorch" in result.output
    assert "Tokenizers" in result.output
    assert "╭" in result.output and "│" in result.output
    assert result.stderr == ""
    assert "mps built/available" in result.output.lower()
    assert "mlx loaded in process" in result.output.lower()
    assert "built-in semantic model aliases" in result.output.lower()
    assert "Family" in result.output and "gte-modernbert" in result.output
    assert "Search threshold" in result.output and "0.68" in result.output
    assert (
        "python=0.87, c=0.84, rust=0.84, "
        "javascript=0.69, typescript=0.76 (fallback=0.87)" in result.output
    )
    default_revision = cli.resolve_model_profile(cli.DEFAULT_MODEL).default_revision
    assert "Default model revision" in result.output
    assert default_revision in result.output


def test_cli_info_configures_mps_environment_before_diagnostics(monkeypatch):
    order: list[str] = []

    def _record_configure(requested_device, *, fallback):
        order.append(f"configure:{requested_device}:{fallback}")

    def _record_diagnostics(requested_device):
        order.append(f"diagnostics:{requested_device}")
        return DeviceDiagnostics(
            requested=requested_device,
            resolved="cpu",
            torch_available=True,
            cuda_available=False,
            mps_built=False,
            mps_available=False,
            mps_fallback_env="1",
            mlx_loaded=False,
            cpu_name="Test CPU",
            cpu_architecture="arm64",
            cpu_bf16_isa=False,
            cpu_mkldnn_available=False,
            cpu_bf16_native=False,
        )

    monkeypatch.setattr(cli, "configure_mps_environment", _record_configure)
    monkeypatch.setattr(cli, "get_device_diagnostics", _record_diagnostics)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["info"])

    assert result.exit_code == 0
    assert order == [
        f"configure:{cli.DEFAULT_SEMANTIC_DEVICE}:None",
        f"diagnostics:{cli.DEFAULT_SEMANTIC_DEVICE}",
    ]


def test_cli_info_survives_cache_construction_failure(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("no home directory")

    monkeypatch.setattr(cli, "EmbeddingCache", _raise)

    result = CliRunner().invoke(cli.cli, ["info", "--verbose"])

    assert result.exit_code == 0
    assert "Unavailable" in result.output
    assert "no home directory" in result.output
    assert "Run with --help for CLI usage" in result.output


@pytest.mark.parametrize("width", [80, 160])
def test_cli_info_default_is_compact(monkeypatch, width):
    from importlib import import_module

    info_module = import_module("codedupes.cli.info")

    def unexpected_details(*_args, **_kwargs):
        pytest.fail("Compact info must not collect verbose-only details")

    monkeypatch.setattr(cli, "EmbeddingCache", unexpected_details)
    monkeypatch.setattr(info_module, "get_grammar_statuses", unexpected_details)
    monkeypatch.setattr(info_module, "list_supported_models", unexpected_details)
    result = CliRunner().invoke(cli.cli, ["info", "--output-width", str(width)])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert "Default model" in result.stdout and cli.DEFAULT_MODEL in result.stdout
    assert "Python" in result.stdout and "PyTorch" in result.stdout
    assert "Device" in result.stdout and "Supported languages" in result.stdout
    assert "--verbose" in result.stdout
    assert len(result.stdout.splitlines()) <= 12
    assert max(map(len, result.stdout.splitlines())) <= width
    for detail in (
        "Default exclusions",
        "Tree-sitter grammar packages",
        "Embedding cache",
        "Built-in semantic model aliases",
        "CPU bfloat16",
        "Default model revision",
    ):
        assert detail not in result.stdout


@pytest.mark.parametrize("terminal_width", [60, 120])
@pytest.mark.parametrize("width_args", [[], ["--output-width", "400"]])
def test_cli_info_fits_actual_terminal(terminal_width, width_args):
    """Catch fixed render widths that would wrap borders in a real terminal."""
    pty = pytest.importorskip("pty")
    termios = pytest.importorskip("termios")
    from rich.text import Text

    try:
        master, slave = pty.openpty()
    except OSError as exc:  # sandboxes may refuse to allocate a pseudo-terminal
        pytest.skip(f"no pty available: {exc}")
    termios.tcsetwinsize(slave, (40, terminal_width))
    env = dict(os.environ, TERM="xterm-256color")
    env.pop("COLUMNS", None)
    env.pop("LINES", None)

    # The PTY buffer is smaller than a narrow-width report, so drain the master
    # while the child runs or the child blocks on write and never exits.
    chunks: list[bytes] = []

    def drain() -> None:
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:  # EIO once the child's slave descriptor closes
                return
            if not chunk:
                return
            chunks.append(chunk)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from codedupes.cli import main; raise SystemExit(main())",
            "info",
            *width_args,
        ],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=subprocess.PIPE,
        env=env,
    )
    os.close(slave)  # only the child holds the slave now, so its exit ends the read loop
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        _, stderr = proc.communicate(timeout=60)
    finally:
        proc.kill()
        reader.join(timeout=5)
        os.close(master)
    raw = b"".join(chunks).decode()
    assert proc.returncode == 0, stderr.decode()
    assert stderr == b""
    lines = Text.from_ansi(raw.replace("\r\n", "\n")).plain.splitlines()
    assert max(map(len, lines)) <= terminal_width
    assert "Default model" in raw and cli.DEFAULT_MODEL in raw
    borders = [line for line in lines if line.startswith(("╭", "│", "╰"))]
    assert len(borders) >= 7
    assert len({len(line) for line in borders}) == 1
    assert borders[0].endswith("╮") and borders[-1].endswith("╯")
