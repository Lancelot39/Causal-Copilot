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
    output_report_dir: Optional[str] = None
    output_graph_dir: Optional[str] = None
    selected_features: Optional[object] = None
    important_features: Optional[object] = None
    visual_selected_features: Optional[object] = None
    system_drop_features: Optional[object] = None
    user_drop_features: Optional[object] = None
    llm_drop_features: Optional[object] = None
    high_corr_drop_features: Optional[object] = None

@dataclass
class Statistics:
    linearity: Optional[bool] = None
    gaussian_error: Optional[bool] = None
    missingness: Optional[bool] = None
    sample_size: Optional[int] = None
    feature_number: Optional[int] = None
    time_series: Optional[bool] = False
    time_lag: List[Dict] = field(default_factory=list)
    boot_num: int = 100
    alpha: float = 0.1
    num_test: int = 100
    ratio: float = 0.5
    data_type: Optional[str] = None
    heterogeneous: Optional[bool] = None
    domain_index: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Logging:
    query_conversation: List[Dict] = field(default_factory=list)
    knowledge_conversation: List[Dict] = field(default_factory=list)
    filter_conversation: List[Dict] = field(default_factory=list)
    select_conversation: List[Dict] = field(default_factory=list)
    argument_conversation: List[Dict] = field(default_factory=list)
    errors_conversion: List[Dict] = field(default_factory=list)
    graph_conversion: Optional[Dict] = field(default_factory=dict)

@dataclass
class Algorithm:
    selected_algorithm: Optional[str] = None
    selected_reason: Optional[str] = None
    algorithm_candidates: Optional[Dict] = None
    algorithm_arguments: Optional[Dict] = None
    waiting_minutes: float = 1440.0
    algorithm_arguments_json: Optional[object] = None

@dataclass
class Results:
    raw_result: Optional[object] = None
    raw_edges: Optional[Dict] = None
    raw_info: Optional[Dict] = None
    converted_graph: Optional[str] = None
    metrics: Optional[Dict] = None
    revised_graph: Optional[np.ndarray] = None
    revised_edges: Optional[Dict] = None
    revised_metrics: Optional[Dict] = None
    bootstrap_probability: Optional[np.ndarray] = None
    bootstrap_check_dict: Optional[Dict] = None
    llm_errors: List[Dict] = field(default_factory=list)
    bootstrap_errors: List[Dict] = field(default_factory=list)
    eda_result: Optional[Dict] = None
    prior_knowledge: Optional[object] = None
    refutation_analysis: Optional[object] = None

@dataclass
class GlobalState:
    user_data: UserData = field(default_factory=UserData)
    statistics: Statistics = field(default_factory=Statistics)
    logging: Logging = field(default_factory=Logging)
    algorithm: Algorithm = field(default_factory=Algorithm)
    results: Results = field(default_factory=Results)

