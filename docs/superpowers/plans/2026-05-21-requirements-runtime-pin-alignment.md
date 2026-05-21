# Requirements Runtime Pin Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align shared CPU and GPU web-runtime requirement pins without changing hardware-specific dependency sets.

**Architecture:** Keep the existing `requirements_cpu.txt` and `requirements_gpu.txt` layout. Add a focused requirements contract test that compares only shared runtime pins, then make the minimum pin edits needed for that test to pass.

**Tech Stack:** Python 3.10-compatible pytest verification, pinned requirement text files, repository `scripts/verify.sh`.

---

## File Map

- Modify `tests/test_requirements.py` to encode the exact shared runtime-pin contract.
- Modify `requirements_cpu.txt` to pin the currently unpinned shared Hugging Face Hub dependency.
- Modify `requirements_gpu.txt` to align the selected shared runtime versions with CPU.

### Task 1: Add Shared Runtime Pin Contract

**Files:**
- Modify: `tests/test_requirements.py`

- [ ] **Step 1: Write the failing test**

Add a test that parses exact requirement pins and asserts the CPU and GPU files
share identical pins for:

```python
shared_runtime_packages = {
    "gradio",
    "gradio-client",
    "huggingface-hub",
    "python-multipart",
    "zipp",
}
```

- [ ] **Step 2: Run the requirements test to verify it fails**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/causal-copilot-verify-venv/bin/python -m pytest tests/test_requirements.py -q
```

Expected: FAIL because the requirement files currently disagree on the guarded
shared pins.

### Task 2: Align Minimum Shared Requirement Pins

**Files:**
- Modify: `requirements_cpu.txt`
- Modify: `requirements_gpu.txt`

- [ ] **Step 1: Pin the CPU Hugging Face Hub requirement**

Change CPU `huggingface-hub` from an unpinned line to the explicit pin selected
for the shared Gradio runtime.

- [ ] **Step 2: Align the guarded GPU pins**

Update only the GPU values for `gradio`, `gradio-client`,
`huggingface-hub`, `python-multipart`, and `zipp` so the shared test contract
matches CPU. Do not alter GPU-only acceleration packages or CPU-only packages.

- [ ] **Step 3: Run the focused requirements test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/causal-copilot-verify-venv/bin/python -m pytest tests/test_requirements.py -q
```

Expected: PASS.

### Task 3: Verify And Publish

**Files:**
- Verify: `tests/test_requirements.py`
- Verify: `scripts/verify.sh`

- [ ] **Step 1: Run repository verification**

Run:

```bash
PATH=/tmp/causal-copilot-verify-venv/bin:$PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 bash scripts/verify.sh
git diff --check
```

Expected: the verification script passes and `git diff --check` reports no
whitespace errors.

- [ ] **Step 2: Stage and commit implementation files**

```bash
git add tests/test_requirements.py requirements_cpu.txt requirements_gpu.txt
git commit -m "chore: align shared runtime requirement pins"
```

- [ ] **Step 3: Push the active branch**

```bash
git push origin frontend-improvement
```
