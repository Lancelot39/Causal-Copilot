import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
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


def stage_dataset_file(
    source_file: Union[str, Path],
    upload_folder: Union[str, Path],
    required_info: Dict[str, Any],
    timestamp: Optional[str] = None,
    upload_id: Optional[str] = None,
) -> Dict[str, Any]:
    source_path = Path(source_file)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    upload_id = upload_id or uuid.uuid4().hex[:8]
    dataset_name = source_path.stem or "dataset"
    output_dir = Path(upload_folder) / f"{timestamp}_{upload_id}" / dataset_name
    target_path = output_dir / source_path.name

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(source_path, target_path)

    updated_info = required_info.copy()
    updated_info["target_path"] = str(target_path)
    updated_info["output_dir"] = str(output_dir)
    return updated_info


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


def try_report_generation(
    generate_report: Callable[[], Union[str, Path, None]],
) -> Tuple[Optional[str], Optional[Exception]]:
    try:
        report_path = generate_report()
    except Exception as error:
        return None, error

    if not report_path:
        return None, None

    report_file = Path(report_path)
    if not report_file.is_file():
        return None, None

    return str(report_file), None
