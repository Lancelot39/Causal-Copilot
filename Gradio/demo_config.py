import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DemoConfig:
    demo_mode: bool = True
    # Input/Output paths
    data_file: Optional[str] = None  # Will be set when file is uploaded
    output_report_dir: str = 'output_report'
    output_graph_dir: str = 'output_graph'

    # LLM Settings - Read from environment variables
    llm_provider: str = os.environ.get('LLM_PROVIDER', 'ollama')  # 'openai' or 'ollama'
    model_name: str = os.environ.get('LLM_MODEL', 'llama3.2')     # 'llama2', 'llama3.2', 'mistral', etc.
    
    # OpenAI Settings (only used if llm_provider == "openai")
    organization: str = "org-gw7mBMydjDsOnDlTvNQWXqPL"
    project: str = "proj_SIDtemBJMHUWG7CPdU7yRjsn" 
    apikey: str = "sk-***************************"
    
    # Analysis Settings
    simulation_mode: str = "offline"
    data_mode: str = "real"
    debug: bool = False
    initial_query: Optional[str] = None  # Will be set when user inputs query
    parallel: bool = True

    # Statistical Analysis Settings
    alpha: float = 0.1
    ratio: float = 0.5
    num_test: int = 100

    def __post_init__(self):
        # Create default output directories if they don't exist
        os.makedirs(self.output_report_dir, exist_ok=True)
        os.makedirs(self.output_graph_dir, exist_ok=True)

def get_demo_config() -> DemoConfig:
    """
    Creates and returns a DemoConfig instance with default values.
    The instance can be modified as needed after creation.
    """
    return DemoConfig() 