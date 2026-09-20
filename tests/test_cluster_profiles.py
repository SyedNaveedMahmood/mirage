"""Model profiles, Slurm resource policy, and shell-syntax checks.

Everything here is static: no Slurm command is executed (this box has none) and
no model is contacted. `sbatch --test-only` is deliberately not used.
"""

from __future__ import annotations

import re
import subprocess

import pytest
import yaml

from shell_support import CLUSTER_DIR, REPO_ROOT, find_bash

_BASH = find_bash()
pytestmark = pytest.mark.skipif(_BASH is None, reason="a POSIX bash is required for shell-library tests")

PROFILES = ("qwen3p5_4b", "qwen3p5_9b", "qwen3p5_27b", "gemma4_26b_a4b", "gpt_oss_20b", "llama31_8b")

# Every shell/Slurm file that must at least parse.
SHELL_FILES = sorted(
    [p for p in CLUSTER_DIR.rglob("*.sh")] + [p for p in CLUSTER_DIR.rglob("*.sbatch")],
    key=lambda p: p.as_posix(),
)

PRODUCTION_JOBS = ("cv0_model.sbatch", "cv0_qwen3p5_9b.sbatch")


def _load_profile(name: str) -> dict[str, str]:
    """Source lib/profile.sh + one profile and dump the contract variables."""
    script = f"""
set -Eeuo pipefail
export MIRAGE_REPO_ROOT="{REPO_ROOT.as_posix()}"
source "{(CLUSTER_DIR / "lib" / "profile.sh").as_posix()}"
load_model_profile "$1"
for var in PROFILE_NAME MODEL_ID MODEL_REVISION SERVED_MODEL_NAME MODEL_FAMILY \\
           CV0_CONFIG_DEFAULT VLLM_TOOL_CALL_PARSER VLLM_STARTUP_TIMEOUT_S \\
           SBATCH_GPU_TIME_DEFAULT PROFILE_PARTITION_PREFERENCE PROFILE_MODEL_IS_GATED \\
           PROFILE_GPU_MEMORY_UTILIZATION PROFILE_DESCRIPTION; do
    printf '%s=%s\\n' "${{var}}" "${{!var-}}"
done
VLLM_ARGS=()
# llama31_8b resolves its chat template in profile_prepare(); stand in for it so
# the flag list can be inspected without touching the filesystem.
PROFILE_CHAT_TEMPLATE_PATH="${{PROFILE_CHAT_TEMPLATE_PATH:-/nonexistent/template.jinja}}"
profile_vllm_args
printf 'VLLM_ARGS=%s\\n' "${{VLLM_ARGS[*]}}"
"""
    out = subprocess.run(
        [_BASH, "-c", script, "bash", name],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line)


def _load_profile_rc(name: str) -> int:
    script = f"""
export MIRAGE_REPO_ROOT="{REPO_ROOT.as_posix()}"
source "{(CLUSTER_DIR / "lib" / "profile.sh").as_posix()}"
load_model_profile "$1"
"""
    return subprocess.run([_BASH, "-c", script, "bash", name], capture_output=True, text=True).returncode


# --------------------------------------------------------------------------- #
# shell syntax
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", SHELL_FILES, ids=lambda p: p.name)
def test_shell_files_parse(path):
    result = subprocess.run([_BASH, "-n", path.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, f"{path}: {result.stderr}"


# --------------------------------------------------------------------------- #
# profile contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads_and_declares_its_contract(name):
    profile = _load_profile(name)
    assert profile["PROFILE_NAME"] == name
    assert profile["MODEL_ID"]
    assert profile["MODEL_FAMILY"]
    assert profile["VLLM_TOOL_CALL_PARSER"]
    assert (REPO_ROOT / profile["CV0_CONFIG_DEFAULT"]).is_file()


@pytest.mark.parametrize("name", PROFILES)
def test_revisions_are_pinned_to_a_commit(name):
    revision = _load_profile(name)["MODEL_REVISION"]
    assert revision != "main"
    assert re.fullmatch(r"[0-9a-f]{40}", revision), revision


@pytest.mark.parametrize("name", PROFILES)
def test_walltime_defaults_are_below_48h(name):
    profile = _load_profile(name)
    hours, minutes, seconds = (int(part) for part in profile["SBATCH_GPU_TIME_DEFAULT"].split(":"))
    assert 0 < hours * 3600 + minutes * 60 + seconds < 48 * 3600


def test_expected_walltime_defaults():
    assert _load_profile("llama31_8b")["SBATCH_GPU_TIME_DEFAULT"] == "22:00:00"
    assert _load_profile("gpt_oss_20b")["SBATCH_GPU_TIME_DEFAULT"] == "30:00:00"
    assert _load_profile("gemma4_26b_a4b")["SBATCH_GPU_TIME_DEFAULT"] == "30:00:00"


@pytest.mark.parametrize("name", PROFILES)
def test_partition_preference_has_no_development_queue(name):
    preference = _load_profile(name)["PROFILE_PARTITION_PREFERENCE"].split()
    assert preference[0] == "gpu_h100"  # H100 preferred unless a profile says otherwise
    assert all(not p.startswith("dev_") for p in preference)
    assert set(preference) <= {"gpu_h100", "gpu_h100_il", "gpu_a100_il"}


def test_gated_models_are_flagged():
    assert _load_profile("gemma4_26b_a4b")["PROFILE_MODEL_IS_GATED"] == "1"
    assert _load_profile("llama31_8b")["PROFILE_MODEL_IS_GATED"] == "1"
    assert _load_profile("gpt_oss_20b")["PROFILE_MODEL_IS_GATED"] == "0"


def test_profile_allowlist_rejects_unknown_and_traversal():
    for bad in ("../../etc/passwd", "does_not_exist", "", "Qwen3P5", "a/b"):
        assert _load_profile_rc(bad) != 0, bad


# --------------------------------------------------------------------------- #
# per-model vLLM flags
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", ("4b", "9b", "27b"))
def test_qwen_flags(size):
    profile = _load_profile(f"qwen3p5_{size}")
    args = profile["VLLM_ARGS"]
    assert "--dtype bfloat16" in args
    assert "--tool-call-parser qwen3_xml" in args
    assert "--gdn-prefill-backend triton" in args
    assert "--limit-mm-per-prompt" in args
    assert '"image": 0' in args and '"video": 0' in args
    assert '"enable_thinking": false' in args
    assert profile["MODEL_FAMILY"] == "qwen3.5"
    if size == "27b":
        assert profile["PROFILE_PARTITION_PREFERENCE"].split() == ["gpu_h100", "gpu_h100_il"]
        assert profile["SBATCH_GPU_TIME_DEFAULT"] == "47:00:00"


def test_gemma_flags():
    args = _load_profile("gemma4_26b_a4b")["VLLM_ARGS"]
    assert "--dtype bfloat16" in args
    assert "--tool-call-parser gemma4" in args
    assert "enable_thinking" in args and "false" in args
    assert "--limit-mm-per-prompt" in args  # text-only run: no vision/audio paths
    assert "--gdn-prefill-backend" not in args  # Qwen-only
    assert _load_profile("gemma4_26b_a4b")["PROFILE_GPU_MEMORY_UTILIZATION"] == "0.90"


def test_gpt_oss_flags_keep_native_mxfp4():
    args = _load_profile("gpt_oss_20b")["VLLM_ARGS"]
    assert "--dtype" not in args, "gpt-oss must keep its native MXFP4 weights"
    assert "--quantization" not in args
    assert "--tool-call-parser openai" in args
    assert "reasoning_effort" in args and "low" in args
    assert "--reasoning-parser openai_gptoss" in args
    assert "--gdn-prefill-backend" not in args


def test_llama_flags_and_chat_template_check(tmp_path):
    args = _load_profile("llama31_8b")["VLLM_ARGS"]
    assert "--dtype bfloat16" in args
    assert "--tool-call-parser llama3_json" in args
    assert "--chat-template" in args
    assert "--gdn-prefill-backend" not in args

    # profile_prepare() must fail loudly (before vLLM starts) when the template
    # cannot be resolved, and succeed with an explicit override.
    script = f"""
export MIRAGE_REPO_ROOT="{REPO_ROOT.as_posix()}"
export VLLM_ROOT="{tmp_path.as_posix()}"
source "{(CLUSTER_DIR / "lib" / "profile.sh").as_posix()}"
load_model_profile llama31_8b
profile_prepare
"""
    missing = subprocess.run([_BASH, "-c", script], capture_output=True, text=True)
    template = tmp_path / "tool_chat_template_llama3.1_json.jinja"
    if not (CLUSTER_DIR / "chat_templates" / template.name).is_file():
        assert missing.returncode != 0
        assert "chat template was not found" in missing.stderr
        assert "fetch_chat_template.sh" in missing.stderr

    template.write_text("{% for m in messages %}{{ m }}{% endfor %}", encoding="utf-8")
    script_ok = f'export LLAMA31_CHAT_TEMPLATE="{template.as_posix()}"\n{script}'
    resolved = subprocess.run([_BASH, "-c", script_ok], capture_output=True, text=True)
    assert resolved.returncode == 0, resolved.stderr
    assert "sha256" in resolved.stdout


# --------------------------------------------------------------------------- #
# Slurm resource policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PRODUCTION_JOBS)
def test_production_job_resource_policy(name):
    text = (CLUSTER_DIR / name).read_text(encoding="utf-8")
    directives = [line.strip() for line in text.splitlines() if line.startswith("#SBATCH")]
    joined = "\n".join(directives)
    assert "#SBATCH --nodes=1" in joined
    assert "#SBATCH --ntasks=1" in joined
    assert "#SBATCH --gres=gpu:1" in joined
    assert "#SBATCH --cpus-per-task=4" in joined
    assert "--exclusive" not in joined
    assert "--mem" not in joined  # covers --mem and --mem-per-cpu
    assert "--output=" in joined and "--error=" in joined
    assert "logs/" in joined
    # No development partition for a production job.
    assert "dev_" not in joined
    # Walltime below 48h.
    walltime = re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", joined)
    assert walltime is not None
    assert int(walltime.group(1)) < 48


@pytest.mark.parametrize("path", SHELL_FILES, ids=lambda p: p.name)
def test_no_unresolved_placeholders(path):
    text = path.read_text(encoding="utf-8")
    for placeholder in ("MODEL_HERE", "TODO", "FIXME", "PLACEHOLDER", "revision: main", "REPLACE_ME"):
        assert placeholder not in text, f"{path} contains {placeholder!r}"


def test_config_files_have_no_unresolved_placeholders():
    for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for placeholder in ("MODEL_HERE", "TODO", "FIXME", "REPLACE_ME"):
            assert placeholder not in text, f"{path} contains {placeholder!r}"
        data = yaml.safe_load(text)
        assert isinstance(data, dict), path
        for model in data.get("models", []) or []:
            if isinstance(model, dict):
                assert model.get("revision") != "main", path


def test_shell_scripts_are_fail_fast():
    for path in SHELL_FILES:
        text = path.read_text(encoding="utf-8")
        if path.parent.name == "lib" or path.parent.name == "model_profiles":
            continue  # sourced fragments inherit the caller's shell options
        assert "set -Eeuo pipefail" in text, path
