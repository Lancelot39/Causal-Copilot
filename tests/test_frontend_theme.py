from web_demo.frontend_theme import APP_CSS, APP_JS, build_welcome_message


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
