from html import escape

from web_demo.frontend_theme import (
    APP_CSS,
    APP_JS,
    build_app_header_html,
    build_report_gallery_html,
    build_status_cards_html,
    build_welcome_message,
)


def test_theme_exports_gradio_safe_css_and_js():
    assert ".cc-app-shell" in APP_CSS
    assert ".cc-topbar" in APP_CSS
    assert ".cc-panel" in APP_CSS
    assert ".cc-dataset-rail" in APP_CSS
    assert ".cc-output-rail" in APP_CSS
    assert ".cc-main-workspace" in APP_CSS
    assert ".cc-status-grid" in APP_CSS
    assert ".cc-status-card" in APP_CSS
    assert ".cc-composer" in APP_CSS
    assert ".cc-primary-action" in APP_CSS
    assert ".cc-secondary-action" in APP_CSS
    assert ".cc-demo-list" in APP_CSS
    assert ".report-gallery" in APP_CSS
    assert ".icon-button" in APP_CSS
    assert ".message-wrap" in APP_CSS
    assert ".report-card-link" in APP_CSS
    assert ".report-card-image" in APP_CSS
    assert "repeat(auto-fit, minmax(180px, 1fr))" in APP_CSS
    assert "repeat(6, 1fr)" not in APP_CSS
    assert "#gradio-animation" in APP_CSS
    assert ".gallery-section" not in APP_CSS
    assert ".gallery-heading" not in APP_CSS
    assert "footer{display:none" in APP_CSS
    assert "function createGradioAnimation" in APP_JS
    assert "Watch walk-through video on YouTube" in APP_JS
    assert "Animation created" in APP_JS


def test_welcome_message_is_plain_and_dataset_first():
    message = build_welcome_message()

    assert "upload" in message.lower()
    assert "csv" in message.lower()
    assert "causal" in message.lower()


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
    assert 'class="pdf-icon"' in html
    assert "grid-template-columns" not in html


def test_report_gallery_html_escapes_hostile_values_without_inline_image_css():
    title = '<script>alert("title")</script> ); title'
    description = 'Discovering <b>causes</b> with "quotes", \'apostrophes\', and );'
    author = 'Causal "Copilot" <team> \'alpha\' );'
    file_url = '/gradio_api/file=asset/report_"bad"_<script>.pdf?x=\');'
    image_url = '/gradio_api/file=asset/logo_"bad"_<script>.png?x=);'

    html = build_report_gallery_html(
        [
            {
                "title": title,
                "description": description,
                "author": author,
                "file": file_url,
                "image": image_url,
            }
        ]
    )

    assert "<script" not in html.lower()
    assert "background-image" not in html
    assert escape(title) in html
    assert escape(description) in html
    assert escape(author) in html
    assert escape(file_url, quote=True) in html
    assert escape(image_url, quote=True) in html
    assert '<img class="report-card-image"' in html
    assert "&#x27;" in html
