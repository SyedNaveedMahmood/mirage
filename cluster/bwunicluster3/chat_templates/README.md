# Repository-controlled chat templates

Some tool-call parsers require a chat template that is *not* the one embedded in
the model checkpoint. `--tool-call-parser llama3_json` is the case that matters
here: it expects the official Llama 3.1 **JSON tool-use** template that ships in
the vLLM source tree under `examples/`, which Python wheels do not reliably
install.

This directory holds the pinned copy plus its provenance.

## Contents (after the first fetch)

| File | Purpose |
| --- | --- |
| `tool_chat_template_llama3.1_json.jinja` | the template passed to `vllm serve --chat-template` |
| `tool_chat_template_llama3.1_json.jinja.sha256` | pinned digest; later fetches must match it |
| `tool_chat_template_llama3.1_json.jinja.provenance.json` | source URL, vLLM version, digest, fetch timestamp |

The template is **not committed by default**: it is downloaded once from the
pinned vLLM release tag, on a machine with network access, by

```bash
bash cluster/bwunicluster3/fetch_chat_template.sh llama31_8b
```

which by default reads
`https://raw.githubusercontent.com/vllm-project/vllm/v${VLLM_VERSION}/examples/tool_chat_template_llama3.1_json.jinja`
(`VLLM_VERSION` defaults to the pinned `0.25.1`).

After the first fetch, **commit all three files**. From then on the digest is
pinned: a re-fetch that returns different bytes fails loudly instead of silently
changing the prompt format mid-campaign. `FORCE_CHAT_TEMPLATE_REFETCH=1` re-runs
the download, still subject to the digest check.

The CPU prefetch job for `llama31_8b` calls the fetch script automatically, and
the GPU job performs a static existence check (`profile_prepare`) **before**
launching vLLM, so a missing template fails in seconds rather than after model
load. `LLAMA31_CHAT_TEMPLATE=/path/to/template.jinja` overrides the resolution
for a one-off run.

The resolved path and its SHA-256 are written into every run's
`cluster_job_metadata.json`.
