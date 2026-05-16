from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


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
