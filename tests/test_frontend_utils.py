from pathlib import Path

from web_demo.frontend_utils import build_upload_error_response, get_static_allowed_paths


def test_upload_error_response_preserves_callback_shape_and_state():
    required_info = {"data_uploaded": False}
    chatbot = [("hello", "world")]
    file_upload_btn = object()
    download_btn = object()

    response = build_upload_error_response(
        required_info=required_info,
        current_stage="initial_process",
        chatbot=chatbot,
        file_upload_btn=file_upload_btn,
        download_btn=download_btn,
        error="bad csv",
    )

    assert len(response) == 5
    assert response[0] is required_info
    assert response[1] == "initial_process"
    assert response[2] == [
        ("hello", "world"),
        (None, "\u274c Error loading file: bad csv"),
    ]
    assert response[3] is file_upload_btn
    assert response[4] is download_btn


def test_static_allowed_paths_are_absolute_existing_directories():
    allowed_paths = get_static_allowed_paths()

    assert str(Path("asset").resolve()) in allowed_paths
    for path in allowed_paths:
        static_path = Path(path)
        assert static_path.is_absolute()
        assert static_path.is_dir()
