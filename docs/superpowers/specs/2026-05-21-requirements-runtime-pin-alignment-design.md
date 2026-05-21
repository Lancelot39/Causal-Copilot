# Requirements Runtime Pin Alignment Design

## Summary

The CPU and GPU requirement files share the same Gradio web-demo runtime, but
their shared pins have drifted. The next maintenance change will align only the
shared runtime packages that should not vary by hardware target and add a test
that prevents that drift from returning unnoticed.

## Goals

- Keep the CPU and GPU Gradio runtime pins coherent.
- Make shared dependency drift visible in verification.
- Preserve intentional CPU-only and GPU-only packages.
- Avoid restructuring the requirements layout in this small maintenance step.

## Non-Goals

- Split requirements into base, CPU, and GPU overlay files.
- Upgrade every dependency or regenerate the full requirement files.
- Change Docker, Gradio UI behavior, or model/provider configuration.

## Design

The change will update the GPU requirements to match the CPU file for the
shared Gradio runtime family:

- `gradio`
- `gradio-client`
- `huggingface-hub`
- `python-multipart`

The CPU file will receive an explicit `huggingface-hub` pin so the shared
runtime contract is deterministic instead of comparing one pinned dependency
against an unpinned one. The remaining shared `zipp` version drift will also be
aligned because it is currently different between CPU and GPU without a
hardware-specific reason.

The existing requirement checks will be extended to compare a small allowlist
of shared packages that are expected to use the same exact pinned version in
both files. This keeps the test focused and avoids incorrectly treating
hardware-specific dependencies such as `torch`, `triton`, `cupy-cuda12x`, or
GPU acceleration libraries as drift.

## Data Flow And Failure Handling

There is no runtime data-flow change. Installation behavior remains:

1. Docker or local setup selects either the CPU or GPU requirement file.
2. Pip installs the selected pins.
3. Verification fails early if the guarded shared runtime pins diverge again.

If a future hardware-specific exception becomes necessary for one guarded
package, maintainers must update the test contract intentionally instead of
silently changing one requirements file.

## Verification

- Add a focused test in `tests/test_requirements.py` for exact shared runtime
  pin alignment.
- Keep the existing resolvability assertions for Gradio-related pins.
- Run the focused requirements tests.
- Run `scripts/verify.sh` with the repo verification environment.
