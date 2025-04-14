from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import numpy as np

# Update logic and priority of the state initialization
# 1. All values are intialized to None, later would be set its valid values by information exaction from the user query
# 2. The priority of the state initialization is from the user query > from the data > default values, if previously value is set, the later corresponding operation would be skipped

@dataclass
class UserData:
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None
    ground_truth: Optional[np.ndarray] = None
    initial_query: Optional[str] = None
    knowledge_docs: Optional[str] = None
    knowledge_docs_for_user: Optional[str] = None
    output_report_dir: Optional[str] = None
    output_graph_dir: Optional[str] = None
    selected_features: Optional[object] = None
    important_features: Optional[object] = None
    high_corr_feature_groups: Optional[object] = None
    visual_selected_features: Optional[object] = None
    user_drop_features: Optional[object] = None
    llm_drop_features: Optional[object] = None
    high_corr_drop_features: Optional[object] = None
    nan_indicator: Optional[str] = None
    drop_important_var: Optional[bool] = None
    meaningful_feature: Optional[bool] = None
    heterogeneity: Optional[str] = None
    accept_CPDAG: Optional[bool] = True
    initial_query_type: Optional[str] = None

@dataclass
class Statistics:
    miss_ratio: List[Dict] = field(default_factory=list)
    sparsity_dict: Optional[Dict] = None
    linearity: Optional[bool] = None
    gaussian_error: Optional[bool] = None
    missingness: Optional[bool] = None
    sample_size: Optional[int] = None
    feature_number: Optional[int] = None
    boot_num: int = 20
    alpha: float = 0.1
    num_test: int = 100
    ratio: float = 0.5
    data_type: Optional[str] = None
    data_type_column: Optional[str] = None
    heterogeneous: Optional[bool] = None
    domain_index: Optional[str] = None
    description: Optional[str] = None
    time_series: Optional[bool] = False # indicator of time-series data
    time_lag: int = 50
    time_index: Optional[str] = None

@dataclass
class Logging:
    query_conversation: List[Dict] = field(default_factory=list)
    knowledge_conversation: List[Dict] = field(default_factory=list)
    filter_conversation: List[Dict] = field(default_factory=list)
    select_conversation: List[Dict] = field(default_factory=list)
    argument_conversation: List[Dict] = field(default_factory=list)
    errors_conversion: List[Dict] = field(default_factory=list)
    graph_conversion: Optional[Dict] = field(default_factory=dict)
    downstream_discuss: List[Dict] = field(default_factory=list)
    final_discuss: List[Dict] = field(default_factory=list)
    global_state_logging: List[Dict] = field(default_factory=list)

@dataclass
class Algorithm:
    handle_correlated_features: Optional[bool] = True
    correlation_threshold: Optional[float] = 0.95
    selected_algorithm: Optional[str] = None
    selected_reason: Optional[str] = None
    algorithm_candidates: Optional[Dict] = None
    algorithm_optimum: Optional[str] = None
    algorithm_arguments: Optional[Dict] = None
    waiting_minutes: float = 1440.0
    algorithm_arguments_json: Optional[object] = None
    gpu_available: Optional[bool] = False

@dataclass
class Results:
    raw_result: Optional[object] = None
    raw_pos: Optional[object] = None
    raw_edges: Optional[Dict] = None
    raw_info: Optional[Dict] = None
    converted_graph: Optional[str] = None
    lagged_graph: Optional[object] = None
    metrics: Optional[Dict] = None
    revised_graph: Optional[np.ndarray] = None
    revised_edges: Optional[Dict] = None
    revised_metrics: Optional[Dict] = None
    bootstrap_probability: Optional[np.ndarray] = None
    bootstrap_check_dict: Optional[Dict] = None
    llm_errors: Optional[Dict] = None
    bootstrap_errors: List[Dict] = field(default_factory=list)
    eda_result: Optional[Dict] = None
    prior_knowledge: Optional[object] = None
    refutation_analysis: Optional[object] = None
    report_selected_index: Optional[object] = None

@dataclass
class Inference:
    hte_algo_json: Optional[Dict] = None
    hte_model_y_json: Optional[Dict] = None
    hte_model_T_json: Optional[Dict] = None
    hte_model_param: Optional[Dict] = None
    cycle_detection_result: Optional[Dict] = field(default_factory=dict)  # 🔹 Stores detected cycles
    editing_history: List[Dict] = field(default_factory=list)  # 🔹 Tracks cycle resolution steps
    inference_result: Optional[Dict] = field(default_factory=dict)  # 🔹 Stores final inference output
    task_index: Optional[int] = -1
    task_info: Optional[Dict] = None

@dataclass
class GlobalState:
    user_data: UserData = field(default_factory=UserData)
    statistics: Statistics = field(default_factory=Statistics)
    logging: Logging = field(default_factory=Logging)
    algorithm: Algorithm = field(default_factory=Algorithm)
    inference: Inference = field(default_factory=Inference)
    results: Results = field(default_factory=Results)

