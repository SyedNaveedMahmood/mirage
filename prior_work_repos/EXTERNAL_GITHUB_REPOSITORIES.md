# External GitHub Repositories for MIRAGE-Persist Experiments

**Based on:** `docs/MIRAGE-Persist_Experimental_Design.md` and `docs/FINAL_MIRAGE-Persist_Proposal.md`
**Repository check date:** 2026-07-19
**Scope:** Repositories that should be cloned or monitored to implement the experiments. Model weights are not included because they should be pulled from the selected model registry and pinned separately.

## 1. Clone summary

### Clone now — shared experimental stack

| Repository | Clone URL | Why it is needed | Experiments |
|---|---|---|---|
| **AgentDojo** (`ethz-spylab/agentdojo`) | `https://github.com/ethz-spylab/agentdojo.git` | Primary benchmark and environment underlying **MIRAGE-Dojo**. Provides the four stateful suites, task definitions, mutable environment objects, attacks/defenses, tool execution, and deterministic task/security checks. MIRAGE implements checkpointing, fork/continue execution, removal operators, reset operators, subgoal scoring, and branch manifests as a thin wrapper over the pinned dependency. | **CV-0, CV-1, CV-2, CV-3, CV-4, CV-5, OR-1, OR-2, OR-3, OR-6, OR-7** |
| **vLLM** (`vllm-project/vllm`) | `https://github.com/vllm-project/vllm.git` | High-throughput serving for the local open-weight agents and the H100 frontier run. Needed for batched stochastic continuations, OpenAI-compatible serving, tensor parallelism, and throughput accounting. | **CV-0 through CV-5, OR-1, OR-3, OR-6, OR-7**; optional for OR-4 |
| **Transformers** (`huggingface/transformers`) | `https://github.com/huggingface/transformers.git` | Model loading, chat/tool templates, token accounting, generation controls, local NLI scoring, and access to hidden states/hooks for mechanistic work. Even when vLLM serves the model, this repository is useful for tokenizer/template parity and non-serving utilities. | **CV-0 through CV-5, OR-1, OR-3, OR-5, OR-6, OR-7** |
| **Inspect Evals** (`UKGovernmentBEIS/inspect_evals`) | `https://github.com/UKGovernmentBEIS/inspect_evals.git` | Independent AgentDojo implementation/reference used to verify suite counts, task behavior, scoring, and compatibility. It is not the main MIRAGE runtime, but it is valuable for CV-0 validation and for detecting accidental deviations from the public benchmark. | **CV-0** primarily; recommended validation for **CV-5** |

### Clone when the corresponding experiment starts

| Repository | Clone URL | Why it is needed | Experiments |
|---|---|---|---|
| **MELON** (`kaijiezhu11/MELON`) | `https://github.com/kaijiezhu11/MELON.git` | Official compact implementation of MELON's masked re-execution detector for AgentDojo. Use it as a source-level reference and sanity baseline when implementing re-execution diagnostics and checking that comparisons are faithful. Its code is designed to be copied into an AgentDojo checkout. | **CV-3** baseline/reference fidelity; supporting comparison for **CV-1** and **OR-1** |
| **CHeaT** (`Daniel-Ayz/CHeaT`) | `https://github.com/Daniel-Ayz/CHeaT.git` | Proactive deception artifacts, datasets, and challenge environments relevant to the containerized **MIRAGE-Range** substrate. Useful for constructing the decoy, honeytoken, and false-hint intervention classes in a synthetic host-only range. | **OR-4** only |
| **DECEIVE** (`Cisco-Talos/DECEIVE`) | `https://github.com/Cisco-Talos/DECEIVE.git` | Reference implementation for LLM-assisted honeypot/decoy services. Use as an optional source of deterministic service behavior and interaction logging for MIRAGE-Range; do not make the experiment depend on its live LLM behavior. | **OR-4** only, optional |
| **NNsight** (`ndif-team/nnsight`) | `https://github.com/ndif-team/nnsight.git` | General intervention framework for capturing and modifying model activations. Best fit for activation patching and direction ablation across arbitrary Hugging Face model architectures. | **OR-5** only, if the residual trigger fires |
| **TransformerLens** (`TransformerLensOrg/TransformerLens`) | `https://github.com/TransformerLensOrg/TransformerLens.git` | Mechanistic-interpretability toolkit with hooks and activation patching utilities. Use for supported architectures or as a reference implementation for patching controls. It is an alternative/complement to NNsight, not necessarily a second mandatory backend. | **OR-5** only, if the residual trigger fires |

## 2. Repository-to-experiment map

| Experiment | Repositories to use | Notes |
|---|---|---|
| **CV-0 — Infrastructure, reproducibility, capability and potency calibration** | AgentDojo, vLLM, Transformers; Inspect Evals for cross-checking | Build MIRAGE-Dojo, checkpoint/fork logic, seed control, deterministic state digests, capability screens, and benchmark parity tests. |
| **CV-1 — Post-removal effect and persistence half-life** | AgentDojo, vLLM, Transformers; MELON as a methodological reference | The actual post-removal protocol is new code in the MIRAGE wrapper. MELON is not the primary runtime. |
| **CV-2 — Carrier decomposition and fresh-instance residual** | AgentDojo, vLLM, Transformers | Memory, environment, budget, scaffold/tool, context, and process resets must be implemented in MIRAGE-Dojo. No additional external carrier-decomposition repository is required. |
| **CV-3 — Recovery and security–fidelity frontier** | AgentDojo, vLLM, Transformers, MELON | MELON supports fidelity checks for masked re-execution. AgentSentry-style purification and Task-Shield-style verification must currently be reimplemented from their papers unless official code is released. |
| **CV-4 — Positive/negative-control battery** | AgentDojo, vLLM, Transformers | All planted carriers and sham controls are MIRAGE-specific additions to the AgentDojo wrapper. |
| **CV-5 — Held-out confirmatory replication** | AgentDojo, vLLM, Transformers; Inspect Evals optional | Run only frozen MIRAGE code and pinned revisions. Inspect Evals can provide a final benchmark-parity check. |
| **OR-1 — Self-laundering / trace re-derivation** | AgentDojo, vLLM, Transformers | Strict stripping, span redaction, length-matched filler, lexical overlap, and local NLI scoring are MIRAGE-specific. No verified public repository for *Plans Don't Persist* was found. |
| **OR-2 — Persistence triage** | Reuses MIRAGE outputs; no new mandatory repository | Use package dependencies such as scikit-learn/XGBoost rather than cloning a research repository. Boundary features come from the MIRAGE and AgentDojo logs. |
| **OR-3 — Cross-episode storage-vs-residual bridge** | AgentDojo, vLLM, Transformers | Implement the four memory regimes and identity/session manager in MIRAGE. No verified public repository for the 2026 cross-session benchmark or *Zombie Agents* was found. |
| **OR-4 — Deception-as-defense in MIRAGE-Range** | CHeaT; DECEIVE optional; vLLM/Transformers if agents are local | MIRAGE-Range remains a custom, host-only, containerized substrate. CHeaT supplies the closest reusable deception artifacts; DECEIVE can supply service ideas and logging patterns. |
| **OR-5 — Mechanistic audit** | Transformers plus **NNsight or TransformerLens** | Run only if all four preregistered trigger conditions hold. Use one instrumentation backend first; add the second only for replication or unsupported architectures. |
| **OR-6 — Frontier-scale capability check** | AgentDojo, vLLM, Transformers | vLLM tensor parallelism is the critical external dependency for the 70B/100B-class H100 run. |
| **OR-7 — Potency–persistence decoupling** | AgentDojo, vLLM, Transformers | Adaptive intervention search and held-out evaluation are MIRAGE-specific; no additional paper repository is required. |

## 3. Named methods with no verified public GitHub repository

Similarly named repositories are not substitutes for the methods below. As of the check date, no official public GitHub implementation was verified for the following papers or methods:

| Method/paper | Where it affects the design | Action |
|---|---|---|
| **AgentSentry** | CV-3 purification; OR-2 boundary diagnostics | Reimplement the four counterfactual regimes and causally gated `Purify` from the paper. Add the official repository later if the authors publish one. |
| **The Task Shield** | CV-3 verifier arm | Implement a small independent task-alignment verifier from the paper specification; label it “Task-Shield-style,” not the official method, unless fidelity is established. |
| **Plans Don't Persist** | OR-1 strict trace stripping | Implement strict stripping and the redaction controls directly from the paper description. |
| **Cross-Session Stored Prompt Injection** | OR-3 carrier taxonomy and stored-artifact baseline | Recreate the required storage regimes inside MIRAGE-Dojo. Monitor for the benchmark/sandbox release claimed by the paper. |
| **Zombie Agents** | CV-2 scrub-vs-snapshot ablation; OR-3 memory persistence | Use the paper to specify attack/memory conditions; do not depend on unavailable code. |
| **AttriGuard** | CV-1 related work; OR-2 attribution features | Implement only the diagnostics actually needed by MIRAGE, or add the official code if released. |
| **HoneyTrap** | OR-4 comparison metric | The design uses its resource-consumption framing, not its implementation. CHeaT is the more directly reusable codebase for the planned container deception substrate. |

## 4. Suggested checkout layout

```text
external/
├── agentdojo/              # required benchmark/runtime base
├── inspect_evals/          # independent benchmark validation
├── vllm/                   # model serving
├── transformers/           # model/tokenizer/tool templates
├── MELON/                  # defense reference implementation
├── CHeaT/                  # OR-4 only
├── DECEIVE/                # OR-4 optional
├── nnsight/                # OR-5 conditional
└── TransformerLens/        # OR-5 conditional/alternative
```

## 5. Clone commands

```bash
mkdir -p external
cd external

# Shared stack
git clone https://github.com/ethz-spylab/agentdojo.git
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/huggingface/transformers.git
git clone https://github.com/UKGovernmentBEIS/inspect_evals.git

# Baseline/reference code
git clone https://github.com/kaijiezhu11/MELON.git

# Clone only when OR-4 is approved
git clone https://github.com/Daniel-Ayz/CHeaT.git
git clone https://github.com/Cisco-Talos/DECEIVE.git

# Clone only if OR-5's preregistered trigger fires
git clone https://github.com/ndif-team/nnsight.git
git clone https://github.com/TransformerLensOrg/TransformerLens.git
```

## 6. Reproducibility rules after cloning

1. **Do not run from moving default branches.** Pin every repository to a tested commit SHA and record it in the experiment manifest.
2. **Maintain MIRAGE code separately.** Keep the MIRAGE integration in its thin wrapper rather than mixing custom logic into an untracked AgentDojo checkout.
3. **Record licenses before artifact release.** AgentDojo's license is explicitly called out in the design as a release-time check; repeat this for every vendored or modified repository.
4. **Hash the complete environment.** Record repository SHA, dirty-tree status, Python lockfile, CUDA/PyTorch versions, model revision, tokenizer revision, and container image digest.
5. **Treat unavailable methods honestly.** Use “style,” “paper-derived reimplementation,” or “not reproducible at this scale” where appropriate; do not present a local approximation as official code.
6. **Keep OR-4 isolated.** Run CHeaT/DECEIVE-derived components only in the authorized synthetic host-only range described by the experimental design.

## 7. Minimal clone set by paper scope

- **Minimum publishable core:** AgentDojo, vLLM, Transformers, Inspect Evals, MELON.
- **Core plus highest-value overreach OR-1:** same set; no additional repository.
- **Add Substrate B / OR-4:** add CHeaT; optionally add DECEIVE.
- **Add gated mechanistic OR-5:** add NNsight or TransformerLens, with the other reserved for replication.
