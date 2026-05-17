from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote


ChatMessage = Tuple[Any, Any]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_static_allowed_paths() -> List[str]:
    root = get_project_root()
    candidates = [
        root / "web_demo" / "public",
        root / "asset",
    ]
    return [str(path) for path in candidates if path.is_dir()]


def build_gradio_file_url(path: Union[str, Path], project_root: Optional[Path] = None) -> str:
    root = project_root or get_project_root()
    file_path = Path(path)
    if file_path.is_absolute():
        try:
            file_path = file_path.relative_to(root)
        except ValueError:
            pass
    normalized_path = file_path.as_posix()
    return f"/gradio_api/file={quote(normalized_path, safe='/')}"


def build_report_gallery_items(
    report_items: Sequence[Dict[str, str]],
    project_root: Optional[Path] = None,
    fallback_image_path: str = "asset/logo.png",
) -> List[Dict[str, str]]:
    root = project_root or get_project_root()
    fallback_image = root / fallback_image_path
    gallery_items = []

    for item in report_items:
        report_path = root / item["file_path"]
        if not report_path.is_file():
            continue

        image_path = root / item.get("image_path", "")
        if not image_path.is_file() and fallback_image.is_file():
            image_path = fallback_image

        gallery_items.append(
            {
                "title": item["title"],
                "description": item["description"],
                "author": item["author"],
                "file": build_gradio_file_url(report_path, project_root=root),
                "image": build_gradio_file_url(image_path, project_root=root),
            }
        )

    return gallery_items


def build_upload_error_response(
    required_info: Dict[str, Any],
    current_stage: str,
    chatbot: List[ChatMessage],
    file_upload_btn: Any,
    download_btn: Any,
    error: Union[Exception, str],
):
    updated_chatbot = list(chatbot or [])
    updated_chatbot.append((None, f"\u274c Error loading file: {error}"))
    return required_info, current_stage, updated_chatbot, file_upload_btn, download_btn
