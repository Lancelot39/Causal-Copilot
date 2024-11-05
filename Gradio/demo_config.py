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

    # OpenAI Settings
    organization: str = "org-5NION61XDUXh0ib0JZpcppqS"
    project: str = "proj_Ry1rvoznXAMj8R2bujIIkhQN"
    apikey: str = "sk-l4ETwy_5kOgNvt5OzHf_YtBevR1pxQyNrlW8NRNPw2T3BlbkFJdKpqpbcDG0IhInYcsS3CXdz_EMHkJO7s1Bo3e4BBcA"

    # Analysis Settings
    simulation_mode: str = "offline"
    data_mode: str = "real"
    debug: bool = False
    initial_query: Optional[str] = None  # Will be set when user inputs query
    parallel: bool = False

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