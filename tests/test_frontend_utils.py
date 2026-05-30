from pathlib import Path

from web_demo.frontend_utils import (
    build_report_gallery_items,
    build_upload_error_response,
    get_static_allowed_paths,
    stage_dataset_file,
    try_report_generation,
)


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


def test_report_gallery_items_skip_missing_files_and_fallback_image(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_file = reports_dir / "available.pdf"
    fallback_image = reports_dir / "fallback.png"
    report_file.write_bytes(b"%PDF-1.4\n")
    fallback_image.write_bytes(b"png")

    gallery_items = build_report_gallery_items(
        [
            {
                "title": "Available report",
                "description": "Uses the fallback preview",
                "author": "Causal Copilot",
                "file_path": "reports/available.pdf",
                "image_path": "reports/missing.png",
            },
            {
                "title": "Missing report",
                "description": "Should not render",
                "author": "Causal Copilot",
                "file_path": "reports/missing.pdf",
                "image_path": "reports/fallback.png",
            },
        ],
        project_root=tmp_path,
        fallback_image_path="reports/fallback.png",
    )

    assert gallery_items == [
        {
            "title": "Available report",
            "description": "Uses the fallback preview",
            "author": "Causal Copilot",
            "file": "/gradio_api/file=reports/available.pdf",
            "image": "/gradio_api/file=reports/fallback.png",
        }
    ]


def test_stage_dataset_file_uses_unique_workspace_for_same_second(tmp_path):
    source_file = tmp_path / "uploaded.csv"
    source_file.write_text("x,y\n1,2\n", encoding="utf-8")
    upload_folder = tmp_path / "uploads"

    first = stage_dataset_file(
        source_file=source_file,
        upload_folder=upload_folder,
        required_info={"data_uploaded": False},
        timestamp="20260519_120000",
        upload_id="first",
    )
    second = stage_dataset_file(
        source_file=source_file,
        upload_folder=upload_folder,
        required_info={"data_uploaded": False},
        timestamp="20260519_120000",
        upload_id="second",
    )

    assert first["target_path"] != second["target_path"]
    assert first["output_dir"] != second["output_dir"]
    assert "20260519_120000_first" in first["output_dir"]
    assert "20260519_120000_second" in second["output_dir"]
    assert Path(first["target_path"]).read_text(encoding="utf-8") == "x,y\n1,2\n"
    assert Path(second["target_path"]).read_text(encoding="utf-8") == "x,y\n1,2\n"


def test_try_report_generation_returns_error_without_raising():
    error = RuntimeError("latex failed")

    def generate_report():
        raise error

    report_path, report_error = try_report_generation(generate_report)

    assert report_path is None
    assert report_error is error


def test_try_report_generation_requires_existing_report_file(tmp_path):
    missing_report = tmp_path / "missing.pdf"

    report_path, report_error = try_report_generation(lambda: missing_report)

    assert report_path is None
    assert report_error is None


def test_try_report_generation_returns_existing_report_file(tmp_path):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-1.4\n")

    report_path, report_error = try_report_generation(lambda: report)

    assert report_path == str(report)
    assert report_error is None
