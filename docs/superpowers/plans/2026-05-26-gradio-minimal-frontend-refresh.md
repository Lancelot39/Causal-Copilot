# Gradio Minimal Frontend Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved minimal Gradio frontend refresh while keeping the existing Gradio app, callback wiring, dataset upload flow, chat flow, report flow, and download behavior intact.

**Architecture:** Keep `web_demo/demo.py` as the app entrypoint and callback owner. Extract visual CSS and deterministic HTML snippets into a new `web_demo/frontend_theme.py` helper module so the large Gradio file becomes easier to improve without mixing presentation code into causal workflow logic. Implement the visible refresh incrementally: theme helpers first, report gallery cleanup second, top-level layout third, then browser verification.

**Tech Stack:** Python, Gradio Blocks, pytest, ruff, Bash verification through `scripts/verify.sh`.

---

## Source Spec

- `docs/superpowers/specs/2026-05-24-gradio-minimal-frontend-refresh-design.md`

## File Structure

- Create: `web_demo/frontend_theme.py`
  - Owns `APP_CSS`, `APP_JS`, welcome text, lightweight HTML helpers, and report gallery rendering helpers.
  - Does not import Gradio.
  - Does not touch backend callback state.
- Create: `tests/test_frontend_theme.py`
  - Unit tests deterministic CSS/HTML helper behavior.
  - Avoids brittle screenshot testing.
- Modify: `web_demo/demo.py`
  - Imports theme helpers.
  - Replaces inline CSS/JS and inline report gallery construction with helper calls.
  - Reorganizes the Gradio Blocks layout into header, dataset rail, main chat workspace, and output/report rail.
  - Preserves existing callback functions and event chains unless a later test proves a narrow callback adaptation is needed.
- Modify: `tests/test_maintenance_smoke.py`
  - Add one smoke assertion that the app still launches from `web_demo/demo.py` and keeps the expected Gradio port only if this needs maintenance coverage.
- Do not modify:
  - Causal discovery algorithms.
  - Causal inference logic.
  - LLM provider behavior.
  - Dataset staging semantics in `web_demo/frontend_utils.py` unless a tiny import boundary cleanup becomes necessary.

## Implementation Tasks

### Task 1: Extract Presentational Theme Helpers

**Files:**
- Create: `web_demo/frontend_theme.py`
- Create: `tests/test_frontend_theme.py`
- Modify: `web_demo/demo.py`

- [ ] **Step 1: Write the failing import and CSS tests**

Create `tests/test_frontend_theme.py`:

```python
from web_demo.frontend_theme import APP_CSS, APP_JS, build_welcome_message


def test_theme_exports_gradio_safe_css_and_js():
    assert ".cc-app-shell" in APP_CSS
    assert ".report-gallery" in APP_CSS
    assert "footer{display:none" in APP_CSS
    assert "function createGradioAnimation" in APP_JS


def test_welcome_message_is_plain_and_dataset_first():
    message = build_welcome_message()

    assert "upload" in message.lower()
    assert "csv" in message.lower()
    assert "causal" in message.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_frontend_theme.py
```

Expected: `ModuleNotFoundError: No module named 'web_demo.frontend_theme'`.

- [ ] **Step 3: Add the minimal theme module**

Create `web_demo/frontend_theme.py` with:

```python
APP_JS = """
function createGradioAnimation() {
    return 'Animation disabled for minimal layout';
}
"""

APP_CSS = """
.cc-app-shell {
    max-width: 1440px;
    margin: 0 auto;
}
.cc-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.cc-panel {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #ffffff;
}
.report-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
}
footer{display:none !important}
"""


def build_welcome_message() -> str:
    return (
        "Welcome to Causal Copilot. Upload a CSV dataset or choose a demo "
        "dataset to begin causal discovery."
    )
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_frontend_theme.py
```

Expected: `2 passed`.

- [ ] **Step 5: Wire the theme constants into `web_demo/demo.py`**

In `web_demo/demo.py`:

```python
from web_demo.frontend_theme import APP_CSS, APP_JS, build_welcome_message
```

Find the `with gr.Blocks(...) as demo:` call that currently passes `js=js` and the large inline `css=` string. Change it to:

```python
with gr.Blocks(title="Causal Copilot", js=APP_JS, theme=gr.themes.Soft(), css=APP_CSS) as demo:
```

Replace the initial chatbot welcome text and `clear_chat()` welcome text with `build_welcome_message()`.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_frontend_theme.py tests/test_frontend_utils.py
```

Expected: all selected tests pass.

Commit:

```bash
git add web_demo/frontend_theme.py tests/test_frontend_theme.py web_demo/demo.py
git commit -m "refactor: extract Gradio frontend theme helpers"
```

### Task 2: Move Report Gallery HTML Into a Tested Helper

**Files:**
- Modify: `web_demo/frontend_theme.py`
- Modify: `tests/test_frontend_theme.py`
- Modify: `web_demo/demo.py`

- [ ] **Step 1: Write the failing report gallery helper test**

Add to `tests/test_frontend_theme.py`:

```python
from web_demo.frontend_theme import build_report_gallery_html


def test_report_gallery_html_renders_cards_without_inline_style_grid():
    html = build_report_gallery_html(
        [
            {
                "title": "Sachs Protein Signaling",
                "description": "Discovering causal structure",
                "author": "Causal Copilot",
                "file": "/gradio_api/file=asset/report_Sachs.pdf",
                "image": "/gradio_api/file=asset/logo.png",
            }
        ]
    )

    assert '<div class="report-gallery">' in html
    assert "Sachs Protein Signaling" in html
    assert "/gradio_api/file=asset/report_Sachs.pdf" in html
    assert "grid-template-columns" not in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_frontend_theme.py::test_report_gallery_html_renders_cards_without_inline_style_grid
```

Expected: import or missing function failure.

- [ ] **Step 3: Implement `build_report_gallery_html()`**

Add to `web_demo/frontend_theme.py`:

```python
from html import escape
from typing import Mapping, Sequence


def build_report_gallery_html(report_items: Sequence[Mapping[str, str]]) -> str:
    cards = []
    for item in report_items:
        title = escape(item["title"])
        description = escape(item["description"])
        author = escape(item["author"])
        file_url = escape(item["file"], quote=True)
        image_url = escape(item["image"], quote=True)
        cards.append(
            f"""
            <a class="report-card-link" href="{file_url}" target="_blank" rel="noopener noreferrer">
                <article class="report-card">
                    <div class="report-card-image-area" style="background-image: url('{image_url}');"></div>
                    <div class="report-card-content">
                        <h3 class="report-card-title">{title}</h3>
                        <p class="report-card-desc">{description}</p>
                        <p class="report-card-author">By {author}</p>
                    </div>
                </article>
            </a>
            """
        )
    return '<div class="report-gallery">' + "\n".join(cards) + "</div>"
```

- [ ] **Step 4: Replace inline gallery construction in `web_demo/demo.py`**

After `report_items = build_report_gallery_items(...)`, replace the large inline `gallery_html` style block and card-building loop with:

```python
gallery_html = build_report_gallery_html(report_items)
```

Import the helper:

```python
from web_demo.frontend_theme import APP_CSS, APP_JS, build_report_gallery_html, build_welcome_message
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_frontend_theme.py tests/test_frontend_utils.py
```

Expected: all selected tests pass.

Commit:

```bash
git add web_demo/frontend_theme.py tests/test_frontend_theme.py web_demo/demo.py
git commit -m "refactor: move report gallery markup into theme helper"
```

### Task 3: Build the Minimal Four-Part Gradio Layout

**Files:**
- Modify: `web_demo/demo.py`
- Modify: `web_demo/frontend_theme.py`
- Modify: `tests/test_frontend_theme.py`

- [ ] **Step 1: Write layout contract tests**

Add to `tests/test_frontend_theme.py`:

```python
from web_demo.frontend_theme import build_app_header_html, build_status_cards_html


def test_app_header_html_names_product_and_status():
    html = build_app_header_html()

    assert "Causal Copilot" in html
    assert "Gradio" in html
    assert "cc-topbar" in html


def test_status_cards_html_keeps_three_short_cards():
    html = build_status_cards_html()

    assert html.count('class="cc-status-card"') == 3
    assert "Dataset" in html
    assert "Analysis" in html
    assert "Report" in html
```

- [ ] **Step 2: Run the layout tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_frontend_theme.py::test_app_header_html_names_product_and_status tests/test_frontend_theme.py::test_status_cards_html_keeps_three_short_cards
```

Expected: missing helper failures.

- [ ] **Step 3: Implement deterministic header and status-card helpers**

Add to `web_demo/frontend_theme.py`:

```python
def build_app_header_html() -> str:
    return """
    <header class="cc-topbar">
        <div>
            <p class="cc-eyebrow">Causal analysis workspace</p>
            <h1>Causal Copilot</h1>
        </div>
        <div class="cc-runtime-pill">Gradio development app</div>
    </header>
    """


def build_status_cards_html() -> str:
    cards = [
        ("Dataset", "Upload CSV or use a demo"),
        ("Analysis", "Chat through causal discovery"),
        ("Report", "Export the generated package"),
    ]
    return (
        '<section class="cc-status-grid">'
        + "".join(
            f'<article class="cc-status-card"><span>{title}</span><p>{body}</p></article>'
            for title, body in cards
        )
        + "</section>"
    )
```

- [ ] **Step 4: Reorganize the Gradio component tree**

In `web_demo/demo.py`, keep existing component variables and event chains, but move where components are declared. Also move the `report_items = build_report_gallery_items(...)` and `gallery_html = build_report_gallery_html(report_items)` setup before the right rail uses `gallery_html`.

```python
from web_demo.frontend_theme import (
    APP_CSS,
    APP_JS,
    build_app_header_html,
    build_report_gallery_html,
    build_status_cards_html,
    build_welcome_message,
)

# Keep state declarations before visible components.
stage_state = gr.State("initial_process")
state = gr.State(None)
args = gr.State(type("Args", (), {})())
# Keep the existing REQUIRED_INFO = gr.State(...) dictionary unchanged.

report_items = build_report_gallery_items([
    {
        "title": "Abalone Causal Analysis",
        "description": "Discovering relationships between physical attributes and age of abalone",
        "author": "Causal Copilot",
        "file_path": "asset/report_Abalone.pdf",
    },
    {
        "title": "CCS Data Causal Analysis",
        "description": "Analyzing causal relationships in concrete compressive strength data",
        "author": "Causal Copilot",
        "file_path": "asset/report_CCS.pdf",
    },
    {
        "title": "Sachs Protein Signaling",
        "description": "Discovering causal structure between protein signaling molecules",
        "author": "Causal Copilot",
        "file_path": "asset/report_Sachs.pdf",
    },
])
gallery_html = build_report_gallery_html(report_items)

gr.HTML(build_app_header_html())

with gr.Row(elem_classes=["cc-app-shell"]):
    with gr.Column(scale=3, min_width=260, elem_classes=["cc-panel", "cc-dataset-rail"]):
        file_upload = gr.UploadButton(
            "Upload CSV",
            file_types=[".csv"],
            size="sm",
            elem_classes=["cc-primary-action"],
            file_count="single",
        )
        gr.Markdown("### Demo datasets")
        demo_btns = {}
        with gr.Column(elem_classes=["cc-demo-list"]):
            for dataset_name in DEMO_DATASETS:
                demo_btn = gr.Button(f"{DEMO_DATASETS[dataset_name]['name']} Demo")
                demo_btns[dataset_name] = demo_btn

    with gr.Column(scale=7, min_width=420, elem_classes=["cc-main-workspace"]):
        gr.HTML(build_status_cards_html())
        chatbot = gr.Chatbot(
            value=[(None, build_welcome_message())],
            height=620,
            show_label=False,
            show_share_button=False,
            bubble_full_width=False,
            elem_classes=["cc-chatbot"],
            render_markdown=True,
        )
        with gr.Row(elem_classes=["cc-composer"]):
            msg = gr.Textbox(
                placeholder="Ask what causal relationship to explore",
                elem_classes=["cc-input-box"],
                show_label=False,
                container=False,
                scale=10,
            )
            reset_btn = gr.Button("Reset", scale=1, elem_classes=["cc-secondary-action"], size="sm")

    with gr.Column(scale=3, min_width=260, elem_classes=["cc-panel", "cc-output-rail"]):
        download_btn = gr.DownloadButton(
            "Download result package",
            size="sm",
            elem_classes=["cc-secondary-action"],
            interactive=False,
        )
        gr.Markdown("### Case study reports")
        gr.HTML(gallery_html)
```

Keep the variable names `msg`, `file_upload`, `download_btn`, `reset_btn`, `chatbot`, and `demo_btns` so the existing event handlers require minimal edits.

- [ ] **Step 5: Extend CSS for responsive layout**

Update `APP_CSS`:

```css
.cc-app-shell { align-items: stretch; gap: 16px; }
.cc-dataset-rail, .cc-output-rail { padding: 16px; }
.cc-main-workspace { min-width: 0; }
.cc-status-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 12px;
}
.cc-status-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    background: #ffffff;
}
.cc-composer { align-items: center; }
@media (max-width: 920px) {
    .cc-app-shell { flex-direction: column; }
    .cc-status-grid { grid-template-columns: 1fr; }
}
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_frontend_theme.py tests/test_frontend_utils.py
```

Expected: all selected tests pass.

Commit:

```bash
git add web_demo/demo.py web_demo/frontend_theme.py tests/test_frontend_theme.py
git commit -m "feat: add minimal Gradio workspace layout"
```

### Task 4: Preserve Callback Behavior Through the Layout Move

**Files:**
- Modify: `tests/test_maintenance_smoke.py`
- Modify: `web_demo/demo.py` only if tests reveal missing callback variables.

- [ ] **Step 1: Add a smoke test for stable component/event names**

Add to `tests/test_maintenance_smoke.py`:

```python
def test_demo_keeps_core_gradio_component_bindings():
    source = Path("web_demo/demo.py").read_text(encoding="utf-8")

    for name in ("msg", "file_upload", "download_btn", "reset_btn", "chatbot", "demo_btns"):
        assert f"{name} =" in source

    assert "msg.submit(" in source
    assert "file_upload.upload(" in source
    assert "reset_btn.click(" in source
    assert "demo_btn.click(" in source
```

- [ ] **Step 2: Run the smoke test**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_maintenance_smoke.py::test_demo_keeps_core_gradio_component_bindings
```

Expected: pass if Task 3 preserved variable names; fail with a missing binding if the layout move dropped one.

- [ ] **Step 3: Fix any missing binding with the smallest code change**

If the test fails, restore the missing component variable in `web_demo/demo.py` without changing callback logic.

- [ ] **Step 4: Run maintenance smoke and commit**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_maintenance_smoke.py
```

Expected: all smoke tests pass.

Commit:

```bash
git add tests/test_maintenance_smoke.py web_demo/demo.py
git commit -m "test: lock Gradio callback bindings"
```

### Task 5: Full Repo Verification

**Files:**
- No source changes unless verification reveals a narrow issue.

- [ ] **Step 1: Run the full verification script**

Run:

```bash
PYTHON=.venv/bin/python scripts/verify.sh
```

Expected:

```text
All checks passed!
<pytest summary with zero failures>
```

The exact pytest count may increase after new tests. Confirm there are zero failures and zero unexpected errors.

- [ ] **Step 2: Fix only failures caused by this branch**

If verification fails, inspect the failing file and fix only the minimal issue introduced by the refresh work.

- [ ] **Step 3: Re-run the full verification script**

Run:

```bash
PYTHON=.venv/bin/python scripts/verify.sh
```

Expected: zero failures.

- [ ] **Step 4: Commit any verification-only fix**

If Step 2 required changes:

```bash
git add <changed-files>
git commit -m "fix: keep Gradio refresh verification clean"
```

### Task 6: Browser Verification on the Running Gradio App

**Files:**
- No source changes unless browser verification finds a layout issue.

- [ ] **Step 1: Install runtime dependencies if missing**

Use the project runtime dependency file needed to launch the Gradio demo. On this machine, a dev-only `.venv` may not include Gradio, so either install the app requirements into the existing local virtualenv or use a separate runtime environment.

Example:

```bash
.venv/bin/python -m pip install -r requirements_cpu.txt
```

If the install is too heavy or fails because of unavailable platform packages, record the exact blocked command and error.

- [ ] **Step 2: Launch the Gradio app**

Run:

```bash
.venv/bin/python web_demo/demo.py
```

Expected: Gradio starts on `http://localhost:7860`.

- [ ] **Step 3: Open desktop viewport in the browser**

Open:

```text
http://localhost:7860
```

Check:

- Header is visible.
- Dataset rail, chat workspace, and output rail do not overlap.
- Upload button, demo buttons, textbox, reset, download, and report cards are visible.

- [ ] **Step 4: Open a narrow viewport**

Use the in-app browser or Playwright viewport control to inspect mobile/narrow layout.

Check:

- Rails stack cleanly.
- Text does not overlap buttons or cards.
- Chat composer remains usable.

- [ ] **Step 5: Upload or select a sample dataset**

Use a small existing CSV from the repo, such as:

```text
data/dataset/example/data.csv
```

Check:

- Upload or demo dataset selection updates the chat.
- The app does not throw a visible Gradio error.
- Existing callback flow continues to enable inputs after processing.

- [ ] **Step 6: Record verification evidence**

In the final implementation summary, include:

- Focused tests run.
- Full `scripts/verify.sh` result.
- Browser URL tested.
- Dataset used.
- Any runtime dependency gaps.

### Task 7: Push the Completed Frontend Refresh

**Files:**
- No new source changes unless final review finds an issue.

- [ ] **Step 1: Confirm branch and diff**

Run:

```bash
git status --short --branch --untracked-files=all
git log --oneline --decorate --max-count=5
```

Expected: on `frontend-improvement`, with only intended files modified.

- [ ] **Step 2: Push after all verification passes**

Run:

```bash
git push origin frontend-improvement
```

Expected: push succeeds and remote `frontend-improvement` points to the final commit.

## Stop Rules

- Stop if the Gradio app cannot launch because runtime dependencies cannot be installed locally; report the exact command and error.
- Stop if callback behavior requires backend changes beyond component relocation.
- Stop if the layout starts drifting toward a full framework migration. This branch keeps Gradio.
- Stop if browser verification shows overlapping or unreadable UI after two quick layout fixes; revisit the design instead of piling on CSS.

## Final Verification Checklist

- `tests/test_frontend_theme.py` passes.
- `tests/test_frontend_utils.py` passes.
- `tests/test_maintenance_smoke.py` passes.
- `PYTHON=.venv/bin/python scripts/verify.sh` passes.
- Gradio app launches locally or the exact dependency blocker is documented.
- Browser desktop and narrow checks show no overlapping layout.
- Upload/demo dataset flow still reaches the chat.
- Branch is pushed to `origin/frontend-improvement`.
