# Gradio Minimal Frontend Refresh Design

## Summary

This change implements the approved Minimal Direction v2 as the first major frontend refresh slice. The current Gradio app remains the running development surface. We are not migrating to Next.js, replacing the backend, or changing the causal analysis workflow in this slice.

The goal is to make the app feel clean, simple, and credible while keeping the existing upload, chat, demo dataset, report, and download flows intact.

## Goals

- Keep `web_demo/demo.py` as the Gradio entrypoint during this development phase.
- Replace the visually cluttered surface with a minimal, non-overlapping layout.
- Preserve the existing backend callbacks and state flow.
- Make the interface easier to scan: dataset controls, conversation, progress, and outputs should each have a clear place.
- Move reusable frontend styling and small HTML helpers behind a focused helper boundary so future frontend work is less messy.
- Add focused tests around helper output where practical, then run the existing verification suite.

## Non-Goals

- No Next.js, React, or standalone frontend migration in this slice.
- No changes to model providers, causal algorithms, prompt behavior, or report generation logic.
- No broad backend refactor.
- No new authentication, persistence, deployment, or billing features.

## Proposed Layout

The refreshed Gradio screen will use a calm four-part structure:

- A compact top bar with the product name, simple status, and a small action area.
- A left dataset rail for upload, selected file status, and demo dataset choices.
- A main work area for the intro, key status cards, chat history, and the message composer.
- A right output rail for progress, report status, and download/clear actions.

This keeps the app familiar but removes the current crowded feel. The layout should be responsive enough to stack rails below the main work area on narrower screens instead of squeezing or overlapping content.

## Component Boundaries

The implementation should keep behavior changes narrow:

- `web_demo/demo.py` keeps the Gradio component tree and callback wiring.
- A new helper module, likely `web_demo/frontend_theme.py`, owns CSS and small presentational HTML snippets.
- Existing upload staging and utility behavior in `web_demo/frontend_utils.py` remains focused on data/file handling unless a tiny shared helper clearly belongs there.
- Tests should target deterministic helper output and avoid brittle screenshot-style assertions.

This gives us a cleaner frontend foundation without mixing visual polish into backend callbacks.

## Data Flow

Existing callback contracts stay the same:

- Uploading a CSV still stages the dataset and updates the required information panel.
- Selecting a demo dataset still prepares the chat state and visible dataset context.
- Chat submission still drives the existing response flow.
- Report generation and download outputs keep their current state shape.
- Clear/reset actions keep the same user-visible behavior.

The frontend refresh may rename labels and reorganize where components appear, but it should not change the meaning of values passed between callbacks.

## Error Handling

The refreshed UI should make current errors easier to see without inventing new error semantics:

- Dataset upload failures should remain visible near the dataset controls.
- Chat or report failures should remain visible in the main/output areas.
- Empty states should be plain and useful, not decorative.
- Buttons should not appear visually available when the underlying Gradio flow has disabled them.

## Verification

Before implementation is considered complete:

- Add or update focused tests for any new frontend helper module.
- Run the focused frontend/helper tests.
- Run `scripts/verify.sh`.
- Launch the Gradio app when the local runtime supports it.
- Use browser checks on the running Gradio app for at least desktop and narrow viewport layouts.
- Upload a sample dataset through the UI and confirm the existing chat/report flow still works.

If the local machine is missing a runtime dependency, record the exact blocked command and error instead of treating the flow as verified.

## Acceptance Criteria

- The app still runs as a Gradio app.
- No core backend behavior changes are required.
- The first screen is clean, minimal, and free of overlapping elements.
- Dataset upload, demo selection, chat, report, clear, and download controls remain available.
- The layout is readable on desktop and does not collapse into overlapping panels on narrow screens.
- Existing repo verification passes, or any environment-specific blocker is documented clearly.
