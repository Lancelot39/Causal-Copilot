from web_demo.frontend_theme import (
    APP_CSS,
    APP_JS,
    build_report_gallery_html,
    build_welcome_message,
)


def test_theme_exports_gradio_safe_css_and_js():
    assert ".cc-app-shell" in APP_CSS
    assert ".cc-topbar" in APP_CSS
    assert ".cc-panel" in APP_CSS
    assert ".report-gallery" in APP_CSS
    assert ".icon-button" in APP_CSS
    assert ".gallery-section" in APP_CSS
    assert ".gallery-heading" in APP_CSS
    assert ".message-wrap" in APP_CSS
    assert "footer{display:none" in APP_CSS
    assert "function createGradioAnimation" in APP_JS
    assert "Watch walk-through video on YouTube" in APP_JS
    assert "Animation created" in APP_JS


def test_welcome_message_is_plain_and_dataset_first():
    message = build_welcome_message()

    assert "upload" in message.lower()
    assert "csv" in message.lower()
    assert "causal" in message.lower()


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
