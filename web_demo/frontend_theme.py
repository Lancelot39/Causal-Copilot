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
