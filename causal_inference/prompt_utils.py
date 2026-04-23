from pathlib import Path


def dataset_prompt_replacements(global_state, algo_text="", target_node=None, extra_replacements=None):
    """Build shared replacements for dataset-aware causal inference prompts."""
    columns = '\t'.join(str(column) for column in global_state.user_data.processed_data.columns)
    statistics_desc = global_state.statistics.description

    replacements = {
        "[COLUMNS]": columns,
        "[STATISTICS_DESC]": statistics_desc,
        # Some older prompt code used this spelling; keep it as a compatibility alias.
        "[STATISTICS INFO]": statistics_desc,
        "[ALGO_CONTEXT]": algo_text,
    }
    if target_node is not None:
        replacements["[TARGET_NODE]"] = target_node
    if extra_replacements:
        replacements.update(extra_replacements)
    return replacements


def render_prompt(template_path, replacements):
    prompt = Path(template_path).read_text()
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, "" if value is None else str(value))
    return prompt


def render_dataset_prompt(template_path, algo_text_path, global_state, target_node=None, extra_replacements=None):
    algo_text = Path(algo_text_path).read_text()
    return render_prompt(
        template_path,
        dataset_prompt_replacements(
            global_state,
            algo_text=algo_text,
            target_node=target_node,
            extra_replacements=extra_replacements,
        ),
    )
