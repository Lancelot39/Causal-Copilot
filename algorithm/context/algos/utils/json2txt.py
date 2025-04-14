import json
import os
import re

def add_benchmarking_results_to_all(benchmarking_json, directory="."):
    """Add benchmarking results to all algorithm description text files in directory.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        directory (str): Directory containing algorithm description text files
    """
    
    # Find all .txt files in directory
    for filename in os.listdir(directory):
        if re.match(r'.*\.txt$', filename):
            # Extract algorithm name from filename
            algo_name = filename.replace('.txt', '')
            print(algo_name, list(benchmarking_json.values())[0].keys())
            if not algo_name in list(benchmarking_json.values())[0].keys():
                continue
            # Build full file path
            file_path = os.path.join(directory, filename)
            
            # Add benchmarking results to this file
            add_benchmarking_results(file_path, benchmarking_json)
    
    # Create a consolidated benchmarking results file
    # create_consolidated_benchmarking_results(benchmarking_json, directory)

def add_benchmarking_results(algorithm_desc_path, benchmarking_json):
    """Add benchmarking results to algorithm description text file.
    
    Args:
        algorithm_desc_path (str): Path to algorithm description text file
        benchmarking_json (dict): Dictionary containing benchmarking results
    """
    with open(algorithm_desc_path, 'r') as f:
        content = f.read()
        
    # Remove any existing benchmarking sections to prevent duplication
    if "Benchmarking Results" in content:
        # Find all occurrences of benchmarking sections
        pattern = r"────+\nBenchmarking Results\n────+[\s\S]*?(?=────+|$)"
        # Replace with empty string
        content = re.sub(pattern, "", content)
        print("Found and removed existing benchmarking sections")
        
    # Clean up any potential double newlines or trailing whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.rstrip()
        
    with open(algorithm_desc_path, 'w') as f:
        f.write(content + "\n\n")
        f.write("────────────────────────────────────────────────────────\n")
        f.write("Benchmarking Results\n") 
        f.write("────────────────────────────────────────────────────────\n\n")

        # Simplified benchmarking context
        f.write("• Benchmarking Context\n")
        f.write("  – Simulations with various node counts (5-100), sample sizes (500-5000), edge probabilities (0.1-0.3)\n")
        f.write("  – Tests included linear/non-linear functions, different noise types, discrete variables\n")
        f.write("  – Multiple scenarios: measurement error, missing values, multi-domain data\n\n")
        
        f.write("• Performance Metrics\n")
        f.write("  – Performance level (1-10): Based on F1 score, higher is better\n")
        f.write("  – Efficiency level (0-5): Based on runtime, higher is better\n")
        f.write("  – Overall Score (1-10): Combined measure of performance and efficiency\n\n")
        
        # Rankings table with simplified format
        f.write("• Algorithm Performance by Scenario\n\n")
        
        # First determine which metrics are available for this algorithm
        algo_name = algorithm_desc_path.split('/')[-1].replace('.txt', '')
        has_efficiency = False
        has_composite = False
        
        for scenario in benchmarking_json:
            if algo_name in benchmarking_json[scenario]:
                result = benchmarking_json[scenario][algo_name]
                if 'efficiency' in result['levels'] and result['levels']['efficiency'] != 'N/A':
                    has_efficiency = True
                if 'composite' in result['levels'] and result['levels']['composite'] != 'N/A':
                    has_composite = True
        
        # Create the appropriate table header based on available metrics
        header = "| Scenario | Performance |"
        if has_efficiency:
            header += " Efficiency |"
        if has_composite:
            header += " Overall Score |"
        f.write(header + "\n")
        
        # Create the separator line with the right number of columns
        separator = "|----------|------------|"
        if has_efficiency:
            separator += "------------|"
        if has_composite:
            separator += "------------|"
        f.write(separator + "\n")
        
        # Add data rows - group scenarios to keep related ones together
        scenario_groups = {
            "general": [],
            "scaling": ["Variable Scaling", "Sample Scaling"],
            "function": ["Linear Function", "Non-Linear Function"],
            "noise": ["Gaussian Noise", "Non-Gaussian Noise"],
            "other": ["Heterogeneity", "Measurement Error", "Missing Data", "Edge Probability",
                    "Discrete Ratio", "Dense Graph", "Sparse Graph", 
                    "High Missing Data", "High Measurement Error", "Highly Heterogeneous"]
        }
        
        # First, determine which scenarios are actually available
        available_scenarios = {}
        for group, scenarios in scenario_groups.items():
            if group == "general":
                continue
            available = [s for s in scenarios if s in benchmarking_json and algo_name in benchmarking_json[s]]
            if available:
                available_scenarios[group] = available
            
            # Add scenarios that aren't in any specific group to general
            for scenario in benchmarking_json:
                is_in_group = False
                for group_scenarios in scenario_groups.values():
                    if scenario in group_scenarios:
                        is_in_group = True
                        break
                if not is_in_group and algo_name in benchmarking_json[scenario]:
                    scenario_groups["general"].append(scenario)
            
        # Now print tables in groups
        groups_to_print = ["scaling", "function", "noise", "other", "general"]
        
        for group in groups_to_print:
            scenarios = scenario_groups[group]
            if not scenarios:
                continue
                
            # Only add section headers for non-empty groups
            if group == "scaling":
                f.write("\n• Scaling Scenarios\n")
            elif group == "function":
                f.write("\n• Function Type Scenarios\n")
            elif group == "noise":
                f.write("\n• Noise Type Scenarios\n")
            elif group == "other" and scenario_groups["other"]:
                f.write("\n• Other Scenarios\n")
            elif group == "general" and scenario_groups["general"]:
                f.write("\n• General Scenarios\n")
                
            for scenario in scenarios:
                if scenario in benchmarking_json and algo_name in benchmarking_json[scenario]:
                    result = benchmarking_json[scenario][algo_name]
                    
                    # Get performance level (treat 'N/A' appropriately)
                    performance = result['levels'].get('performance', 'N/A')
                    performance_str = f"{float(performance):.1f}" if performance != 'N/A' else 'N/A'
                    
                    # Create the row with performance
                    row = f"| {scenario} | {performance_str} |"
                    
                    # Add efficiency if available
                    if has_efficiency:
                        efficiency = result['levels'].get('efficiency', 'N/A')
                        eff_str = f"{float(efficiency):.1f}" if efficiency != 'N/A' else 'N/A'
                        row += f" {eff_str} |"
                    
                    # Add overall score (formerly composite) if available
                    if has_composite:
                        composite = result['levels'].get('composite', 'N/A')
                        comp_str = f"{float(composite):.1f}" if composite != 'N/A' else 'N/A'
                        row += f" {comp_str} |"
                    
                    f.write(row + "\n")
        
        # Add a special linear vs non-linear comparison if both exist
        if ("Linear Function" in benchmarking_json and algo_name in benchmarking_json["Linear Function"] and
            "Non-Linear Function" in benchmarking_json and algo_name in benchmarking_json["Non-Linear Function"]):
            
            linear_perf = benchmarking_json["Linear Function"][algo_name]['levels'].get('performance', 'N/A')
            nonlinear_perf = benchmarking_json["Non-Linear Function"][algo_name]['levels'].get('performance', 'N/A')
            
            if linear_perf != 'N/A' and nonlinear_perf != 'N/A':
                linear_perf = float(linear_perf)
                nonlinear_perf = float(nonlinear_perf)
                f.write("\n• Linear vs Non-Linear Performance\n")
                f.write(f"  – Linear function performance: {linear_perf:.1f}\n")
                f.write(f"  – Non-linear function performance: {nonlinear_perf:.1f}\n")
                
                if linear_perf > nonlinear_perf:
                    diff = linear_perf - nonlinear_perf
                    f.write(f"  – This algorithm performs {diff:.1f} points better on linear functions\n")
                elif nonlinear_perf > linear_perf:
                    diff = nonlinear_perf - linear_perf
                    f.write(f"  – This algorithm performs {diff:.1f} points better on non-linear functions\n")
                else:
                    f.write("  – This algorithm performs equally well on linear and non-linear functions\n")
        
        # Simplified Analysis section
        f.write("\n• Analysis\n")
        
        # Calculate overall stats
        total_scenarios = len(benchmarking_json)
        avg_ranks = []
        for scenario in benchmarking_json:
            if algo_name in benchmarking_json[scenario]:
                avg_ranks.append(benchmarking_json[scenario][algo_name]['ranking']['mean'])
        
        if avg_ranks:
            overall_rank = sum(avg_ranks) / len(avg_ranks)
            f.write(f"\n  – Overall ranking across scenarios: {overall_rank:.2f}\n")
        
        # Add scenario-specific insights (keeping the most relevant ones)
        # Check if algorithm is in top performers
        top_performers = ["GRaSP", "XGES", "FCI"]
        if algo_name in top_performers:
            f.write("  – Consistently ranks among top performers across multiple scenarios\n")
            
        # Add simplified specific scenario strengths
        for scenario_type, threshold in [
            ("Variable Scaling", 5),
            ("Sample Scaling", 5),
            ("Non-Gaussian Noise", 5),
            ("Heterogeneity", 5)
        ]:
            if (scenario_type in benchmarking_json and 
                algo_name in benchmarking_json[scenario_type] and 
                benchmarking_json[scenario_type][algo_name]["ranking"]["mean"] < threshold):
                f.write(f"  – Strong performance in {scenario_type.lower()} scenarios\n")

def create_consolidated_benchmarking_results(benchmarking_json, directory):
    """Create a consolidated file with benchmarking results for all algorithms.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        directory (str): Directory to save the consolidated file
    """
    output_path = os.path.join(directory, "consolidated_benchmarking_results.txt")
    
    with open(output_path, 'w') as f:
        f.write("# Consolidated Benchmarking Results for All Algorithms\n\n")
        f.write("────────────────────────────────────────────────────────\n")
        f.write("Overview\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        # Simplified benchmarking context
        f.write("• Benchmarking Context\n")
        f.write("  – Simulations with various node counts (5-100), sample sizes (500-5000), edge probabilities (0.1-0.3)\n")
        f.write("  – Tests included linear/non-linear functions, different noise types, discrete variables\n")
        f.write("  – Multiple scenarios: measurement error, missing values, multi-domain data\n\n")
        
        f.write("• Performance Metrics\n")
        f.write("  – Performance level (1-10): Based on F1 score, higher is better\n")
        f.write("  – Efficiency level (0-5): Based on runtime, higher is better\n")
        f.write("  – Overall Score (1-10): Combined measure of performance and efficiency\n\n")
        
        # Get all algorithms across all scenarios
        all_algorithms = set()
        for scenario in benchmarking_json:
            all_algorithms.update(benchmarking_json[scenario].keys())
        
        # Calculate overall stats for each algorithm
        algorithm_overall_stats = {}
        for algo_name in all_algorithms:
            scenarios_present = 0
            avg_rank = 0
            avg_performance = 0
            has_efficiency = False
            avg_efficiency = 0
            has_composite = False
            avg_composite = 0
            
            # Track linear and non-linear performance separately
            linear_perf = None
            nonlinear_perf = None
            
            for scenario in benchmarking_json:
                if algo_name in benchmarking_json[scenario]:
                    result = benchmarking_json[scenario][algo_name]
                    scenarios_present += 1
                    avg_rank += result['ranking']['mean']
                    
                    # Handle performance
                    performance = result['levels']['performance']
                    if performance != 'N/A':
                        avg_performance += float(performance)
                    
                    # Check and handle efficiency
                    if 'efficiency' in result['levels'] and result['levels']['efficiency'] != 'N/A':
                        has_efficiency = True
                        avg_efficiency += float(result['levels']['efficiency'])
                    
                    # Check and handle composite
                    if 'composite' in result['levels'] and result['levels']['composite'] != 'N/A':
                        has_composite = True
                        avg_composite += float(result['levels']['composite'])
                    
                    # Store linear/non-linear performance specifically
                    if scenario == "Linear Function" and performance != 'N/A':
                        linear_perf = float(performance)
                    elif scenario == "Non-Linear Function" and performance != 'N/A':
                        nonlinear_perf = float(performance)
            
            if scenarios_present > 0:
                stats = {
                    'avg_rank': avg_rank / scenarios_present,
                    'avg_performance': avg_performance / scenarios_present,
                    'scenarios_present': scenarios_present,
                }
                
                if has_efficiency:
                    stats['avg_efficiency'] = avg_efficiency / scenarios_present
                
                if has_composite:
                    stats['avg_composite'] = avg_composite / scenarios_present
                
                if linear_perf is not None:
                    stats['linear_perf'] = linear_perf
                
                if nonlinear_perf is not None:
                    stats['nonlinear_perf'] = nonlinear_perf
                
                algorithm_overall_stats[algo_name] = stats
        
        # Sort algorithms by overall rank for the summary table
        sorted_algos_by_rank = sorted(
            algorithm_overall_stats.items(),
            key=lambda x: x[1]['avg_rank']
        )
        
        # For each algorithm, display overall stats
        f.write("────────────────────────────────────────────────────────\n")
        f.write("Overall Algorithm Performance\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        # Create header based on available metrics
        header = "| Algorithm | Performance |"
        if any('avg_efficiency' in stats for _, stats in sorted_algos_by_rank):
            header += " Efficiency |"
        if any('avg_composite' in stats for _, stats in sorted_algos_by_rank):
            header += " Overall Score |"
        f.write(header + "\n")
        
        # Create separator
        separator = "|-----------|-------------|"
        if any('avg_efficiency' in stats for _, stats in sorted_algos_by_rank):
            separator += "-------------|"
        if any('avg_composite' in stats for _, stats in sorted_algos_by_rank):
            separator += "-------------|"
        f.write(separator + "\n")
        
        # Add data rows
        for algo_name, stats in sorted_algos_by_rank:
            row = f"| {algo_name} | {stats['avg_performance']:.2f} |"
            
            if any('avg_efficiency' in s for _, s in sorted_algos_by_rank):
                eff_val = stats.get('avg_efficiency', 'N/A')
                eff_str = f"{eff_val:.2f}" if eff_val != 'N/A' else 'N/A'
                row += f" {eff_str} |"
            
            if any('avg_composite' in s for _, s in sorted_algos_by_rank):
                comp_val = stats.get('avg_composite', 'N/A')
                comp_str = f"{comp_val:.2f}" if comp_val != 'N/A' else 'N/A'
                row += f" {comp_str} |"
            
            f.write(row + "\n")
        
        # Add a section that compares linear vs non-linear performance
        f.write("\n\n────────────────────────────────────────────────────────\n")
        f.write("Linear vs Non-Linear Function Performance\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        f.write("| Algorithm | Linear | Non-Linear | Difference |\n")
        f.write("|-----------|--------|------------|------------|\n")
        
        # Only include algorithms with both linear and non-linear data
        for algo_name, stats in sorted_algos_by_rank:
            if 'linear_perf' in stats and 'nonlinear_perf' in stats:
                linear = stats['linear_perf']
                nonlinear = stats['nonlinear_perf']
                diff = linear - nonlinear
                diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
                f.write(f"| {algo_name} | {linear:.1f} | {nonlinear:.1f} | {diff_str} |\n")
        
        # Explain the difference values
        f.write("\nNote: Positive difference values indicate better performance on linear functions,\n")
        f.write("negative values indicate better performance on non-linear functions.\n")
        
        # Simplified scenario-specific results
        f.write("\n\n────────────────────────────────────────────────────────\n")
        f.write("Top and Bottom Performers by Scenario\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        for scenario in benchmarking_json:
            f.write(f"\n• {scenario}\n")
            
            # Sort algorithms by their ranking in this scenario
            sorted_algos = sorted(
                [(algo, benchmarking_json[scenario][algo]['ranking']['mean']) 
                 for algo in benchmarking_json[scenario]],
                key=lambda x: x[1]
            )
            
            # Show top 5 algorithms for each scenario
            f.write("  Top performers:\n")
            top_algos = sorted_algos[:5]
            for i, (algo_name, rank) in enumerate(top_algos):
                result = benchmarking_json[scenario][algo_name]
                perf = result['levels']['performance']
                perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
                
                # For Variable Scaling and Sample Scaling, also show efficiency
                if scenario in ["Variable Scaling", "Sample Scaling"]:
                    eff = result['levels'].get('efficiency', 'N/A')
                    eff_str = f"{float(eff):.1f}" if eff != 'N/A' else 'N/A'
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}, Efficiency {eff_str}\n")
                else:
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}\n")
            
            # Show bottom 5 algorithms for each scenario
            f.write("\n  Bottom performers:\n")
            bottom_algos = sorted_algos[-5:] if len(sorted_algos) >= 5 else sorted_algos
            bottom_algos.reverse()  # Display from worst to better
            for i, (algo_name, rank) in enumerate(bottom_algos):
                result = benchmarking_json[scenario][algo_name]
                perf = result['levels']['performance']
                perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
                
                # For Variable Scaling and Sample Scaling, also show efficiency
                if scenario in ["Variable Scaling", "Sample Scaling"]:
                    eff = result['levels'].get('efficiency', 'N/A')
                    eff_str = f"{float(eff):.1f}" if eff != 'N/A' else 'N/A'
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}, Efficiency {eff_str}\n")
                else:
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}\n")
    
    print(f"Consolidated benchmarking results saved to {output_path}")

def create_filtered_benchmarking_results(benchmarking_json, algorithm_list=None, output_filename=None):
    """Create a consolidated benchmarking results text with only the specified algorithms.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        algorithm_list (list): List of algorithm names to include in the results
        output_filename (str, optional): Name of the output file if saving is desired
    
    Returns:
        str: Formatted text containing the filtered benchmarking results
    """
    if algorithm_list is None or len(algorithm_list) == 0:
        print("No algorithms specified for filtering. Using all algorithms.")
        return None
    
    # Mapping table for algorithms that don't have benchmarking results
    # Maps missing algorithms to similar ones with available benchmarking data
    algorithm_mapping = {
        # "GOLEM": "NOTEARSLinear",
        # "PC_stable": "PC",
        # "PC_max": "PC",
        # "GES_BIC": "GES",
        # "GES_AIC": "GES",
        # "MMHC": "MMPC",
        # "LINGAM_ICA": "LINGAM",
        # "NOTEARS_L1": "NOTEARS",
        # "DAGMA_BIC": "DAGMA",
        # "DAGMA_AIC": "DAGMA",
        # "GOLEM_EV": "GOLEM",
        # "GOLEM_NV": "GOLEM",
        # "GIES": "GES",
        # "MCMC_BIC": "MCMC",
        # "MCMC_AIC": "MCMC",
        # "CAM_pruned": "CAM",
        # "DirectLiNGAM": "LINGAM",
        # "FCI_stable": "FCI",
        # "RFCI": "FCI",
        # "GFCI": "FCI",
        # "DECI_nonlinear": "DECI",
        # "DECI_linear": "DECI",
        # "DYNOTEARS": "NOTEARS",
        # "RL": "RL_Sober",
        # "CORL": "RL_Sober"
    }
    
    # Expert bias adjustments for mapped algorithms
    # Adjusts performance and efficiency scores based on expert knowledge
    algorithm_bias = {
        # "GOLEM": {"performance": 0.2, "efficiency": -2},  # GOLEM performs slightly better but is less efficient than NOTEARSLinear
        # "PC_stable": {"performance": 0.3, "efficiency": -0.1},
        # "PC_max": {"performance": 0.2, "efficiency": -0.2},
        # "GES_BIC": {"performance": 0.4, "efficiency": 0.0},
        # "GES_AIC": {"performance": -0.2, "efficiency": 0.0},
        # "MMHC": {"performance": 0.1, "efficiency": 0.2},
        # "LINGAM_ICA": {"performance": -0.3, "efficiency": -0.1},
        # "NOTEARS_L1": {"performance": 0.2, "efficiency": -0.1},
        # "DAGMA_BIC": {"performance": 0.3, "efficiency": 0.0},
        # "DAGMA_AIC": {"performance": -0.2, "efficiency": 0.0},
        # "GOLEM_EV": {"performance": 0.2, "efficiency": -0.1},
        # "GOLEM_NV": {"performance": -0.1, "efficiency": 0.1},
        # "GIES": {"performance": 0.2, "efficiency": -0.2},
        # "MCMC_BIC": {"performance": 0.3, "efficiency": 0.0},
        # "MCMC_AIC": {"performance": -0.2, "efficiency": 0.0},
        # "CAM_pruned": {"performance": 0.1, "efficiency": 0.2},
        # "DirectLiNGAM": {"performance": 0.2, "efficiency": -0.1},
        # "FCI_stable": {"performance": 0.1, "efficiency": -0.2},
        # "RFCI": {"performance": -0.1, "efficiency": 0.3},
        # "GFCI": {"performance": 0.2, "efficiency": -0.3},
        # "DECI_nonlinear": {"performance": 0.4, "efficiency": -0.3},
        # "DECI_linear": {"performance": -0.2, "efficiency": 0.2},
        # "DYNOTEARS": {"performance": 0.1, "efficiency": -0.1},
        # "RL": {"performance": -0.3, "efficiency": 0.0},
        # "CORL": {"performance": 0.3, "efficiency": -0.2}
    }
    
    # Filter the benchmarking_json to only include the specified algorithms
    from copy import deepcopy
    filtered_json = {}
    for scenario, algorithms in benchmarking_json.items():
        filtered_algorithms = {}
        for algo in algorithm_list:
            # Check if algorithm exists directly in the benchmarking results
            matching_algos = [a for a in algorithms.keys() if a.startswith(algo)]
            
            if matching_algos:
                for a in matching_algos:        
                    filtered_algorithms[a] = deepcopy(algorithms[a])
            elif algo in algorithm_mapping and any(a.startswith(algorithm_mapping[algo]) for a in algorithms.keys()):
                # If algorithm doesn't exist but has a mapping, use the mapped algorithm's data
                mapped_algo = algorithm_mapping[algo]
                matching_mapped_algos = [a for a in algorithms.keys() if a.startswith(mapped_algo)]
                
                if matching_mapped_algos:
                    for a in matching_mapped_algos:
                        # Create a deep copy of the mapped algorithm's data
                        mapped_data = deepcopy(algorithms[a])
                        
                        # Apply expert bias adjustments if available
                        if algo in algorithm_bias:
                            # Adjust performance score
                            if 'levels' in mapped_data and 'performance' in mapped_data['levels']:
                                perf = mapped_data['levels']['performance']
                                if perf != 'N/A':
                                    adjusted_perf = float(perf) + algorithm_bias[algo]['performance']
                                    # Cap performance between 0 and 10
                                    mapped_data['levels']['performance'] = max(0, min(10, adjusted_perf))
                            
                            # Adjust efficiency score
                            if 'levels' in mapped_data and 'efficiency' in mapped_data['levels']:
                                eff = mapped_data['levels']['efficiency']
                                if eff != 'N/A':
                                    adjusted_eff = float(eff) + algorithm_bias[algo]['efficiency']
                                    # Cap efficiency between 0 and 5
                                    mapped_data['levels']['efficiency'] = max(0, min(5, adjusted_eff))
                            
                            # Adjust composite score if present
                            if 'levels' in mapped_data and 'composite' in mapped_data['levels']:
                                comp = mapped_data['levels']['composite']
                                if comp != 'N/A':
                                    # Recalculate composite based on adjusted performance and efficiency
                                    perf = mapped_data['levels']['performance']
                                    eff = mapped_data['levels'].get('efficiency', 0)
                                    if perf != 'N/A' and eff != 'N/A':
                                        mapped_data['levels']['composite'] = 0.9 * float(perf) + 0.1 * float(eff)
                        
                        # Store the adjusted data with the original algorithm name
                        filtered_algorithms[algo] = mapped_data
        
        if filtered_algorithms:  # Only include scenarios that have at least one of the specified algorithms
            filtered_json[scenario] = filtered_algorithms
    
    if not filtered_json:
        return "No results found for the specified algorithms."
    
    # Create the output text
    output_text = ""
    output_text += "# ALGORITHM BENCHMARKING RESULTS\n\n"
    
    output_text += "• CAUTIONARY NOTE\n"
    output_text += "  – These benchmarking results should be used as guidelines, not definitive judgments\n"
    output_text += "  – Performance may vary significantly with real-world data compared to simulations\n"
    output_text += "  – Consider your specific domain knowledge and data characteristics when selecting algorithms\n"
    
    output_text += "• Simulation Settings\n"
    output_text += "  – Network sizes: 5 to 1000 nodes\n"
    output_text += "  – Sample sizes: 500 to 10000 data points\n"
    output_text += "  – Edge density: 0.11 to 0.78 probability (avg. degree 1 to 7)\n"
    output_text += "  – Data types: Continuous and mixed (0-20% discrete variables)\n"
    output_text += "  – Function types: Linear and non-linear (MLP) relationships\n"
    output_text += "  – Noise types: Gaussian and uniform distributions\n\n"
    
    output_text += "• Challenge Scenarios\n"
    output_text += "  – Measurement error: 10%, 30%, 50% noise in observations\n"
    output_text += "  – Missing data: 10%, 20%, 30% missing values\n"
    output_text += "  – Multi-domain data: 1, 2, 5, or 10 heterogeneous domains\n"
    output_text += "  – Each configuration tested with 3 different random seeds\n\n"
    
    output_text += "• Key Terms\n"
    output_text += "  – (linear): Scenarios where relationships between variables follow linear functions\n"
    output_text += "  – (mlp): Scenarios where relationships are non-linear (using multilayer perceptron models)\n\n"
    
    output_text += "• Scenario Types\n"
    output_text += "  – Robustness scenarios (e.g., Variable Scaling, Edge Probability): Test algorithm performance across varying levels of a property\n"
    output_text += "  – Specific scenarios (e.g., Gaussian Noise, Dense Graph): Test performance at a fixed specific setting\n\n"
    
    output_text += "• Performance Metrics\n"
    output_text += "  – Performance level (1-10): Based on F1 score, higher is better\n"
    output_text += "  – Efficiency level (0-5): Based on runtime, higher is better (only relevant for scaling scenarios)\n"
    output_text += "  – Stability: Standard deviation of performance, lower values indicate more consistent results\n\n"
    
    output_text += "• Important Note on Efficiency Scoring\n"
    output_text += "  – Benchmarks include large-scale systems with up to 1000 nodes and may timeout for some algorithms\n"
    output_text += "  – For large-scale systems (node size > 200), prioritize algorithms that can utilize available GPUs\n"
    output_text += "  – GPU-accelerated methods provide significant efficiency advantages in large-scale scenarios\n\n"
    output_text += "────────────────────────────────────────────────────────\n"
    output_text += "Filtered Benchmarking Results\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    
    output_text += f"Algorithms included: {', '.join(algorithm_list)}\n\n"
    
    # Overall algorithm performance across all scenarios
    output_text += "────────────────────────────────────────────────────────\n"
    output_text += "Overall Algorithm Performance\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    output_text += "⚠️ IMPORTANT: Overall rankings can be misleading! ⚠️\n"
    output_text += "- An algorithm with high average performance may perform poorly on specific scenarios\n"
    output_text += "- Always check scenario-specific performance for your use case\n\n"
    
    # Get all unique algorithm names across all scenarios
    all_algos = set()
    for scenario in filtered_json:
        all_algos.update(filtered_json[scenario].keys())
    
    # Calculate average performance for each algorithm
    algo_avg_performance = {}
    for algo in all_algos:
        performances = []
        for scenario in filtered_json:
            if algo in filtered_json[scenario]:
                perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                if perf != 'N/A':
                    performances.append(float(perf))
        
        if performances:
            algo_avg_performance[algo] = sum(performances) / len(performances)
    
    # Sort algorithms by average performance
    sorted_algos = sorted(algo_avg_performance.items(), key=lambda x: x[1], reverse=True)
    
    # Display overall ranking
    output_text += "Overall ranking based on average performance across all scenarios:\n\n"
    for i, (algo_name, avg_perf) in enumerate(sorted_algos):
        output_text += f"{i+1}. {algo_name}: {avg_perf:.1f}\n"
    # Efficiency comparison section (only for scaling scenarios)
    output_text += "\n\n────────────────────────────────────────────────────────\n"
    output_text += "Efficiency Comparison\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    output_text += "Note: Efficiency scores are primarily measured in Variable Scaling and Sample Scaling scenarios.\n\n"
    
    # Calculate variable scaling efficiency for ranking
    algo_var_scaling_efficiency = {}
    for algo_name in all_algos:
        var_scaling_efficiencies = []
        for scenario in filtered_json.keys():
            if "variable scaling" in scenario.lower():
                if scenario in filtered_json and algo_name in filtered_json[scenario]:
                    eff = filtered_json[scenario][algo_name]['levels'].get('efficiency', 'N/A')
                    if eff != 'N/A':
                        var_scaling_efficiencies.append(float(eff))
        
        if var_scaling_efficiencies:
            algo_var_scaling_efficiency[algo_name] = sum(var_scaling_efficiencies) / len(var_scaling_efficiencies)
    
    # Sort algorithms by variable scaling efficiency
    sorted_algos_by_var_scaling = sorted(algo_var_scaling_efficiency.items(), key=lambda x: x[1], reverse=True)
    
    # Create a table for efficiency comparison, ranked by variable scaling efficiency
    output_text += "| Algorithm | Variable Scaling (linear) | Sample Scaling (linear) | Variable Scaling (mlp) | Sample Scaling (mlp) | Average |\n"
    output_text += "|-----------|---------------------------|--------------------------|------------------------|----------------------|--------|\n"
    
    for algo_name, _ in sorted_algos_by_var_scaling:
        efficiency_row = f"| {algo_name} |"
        efficiencies = []
        
        for scenario in filtered_json.keys():
            if "scaling" in scenario.lower():
                if scenario in filtered_json and algo_name in filtered_json[scenario]:
                    eff = filtered_json[scenario][algo_name]['levels'].get('efficiency', 'N/A')
                    if scenario == "Variable Scaling (linear)" and algo_name == "NOTEARSLinear":
                        print(scenario, algo_name, eff)
                    if eff != 'N/A':
                        eff_value = float(eff)
                        efficiencies.append(eff_value)
                        efficiency_row += f" {eff_value:.1f} |"
                    else:
                        efficiency_row += " N/A |"
                else:
                    efficiency_row += " N/A |"
        
        # Calculate and add average efficiency
        if efficiencies:
            if algo_name == "PC_indep_test=fisherz_gpu":
                print(scenario, algo_name, efficiencies)
            avg_eff = sum(efficiencies) / len(efficiencies)
            efficiency_row += f" {avg_eff:.1f} |"
        else:
            efficiency_row += " N/A |"
            
        output_text += efficiency_row + "\n"
    
    # Top algorithm recommendations
    output_text += "\n\n────────────────────────────────────────────────────────\n"
    output_text += "Algorithm Recommendations by Scenario Type\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    
    # Define scenario categories
    scenario_categories = {
        "Linear Relationships": [s for s in filtered_json.keys() if "(linear)" in s or "Linear Function" in s],
        "Non-Linear Relationships": [s for s in filtered_json.keys() if "(mlp)" in s or "Non-Linear Function" in s],
        "Data with Missing Values": ["Missing Data (linear)", "Missing Data (mlp)", "High Missing Data"],
        "Data with Measurement Error": ["Measurement Error (linear)", "Measurement Error (mlp)", "High Measurement Error"],
        "Dense vs Sparse Graphs": ["Dense Graph", "Sparse Graph", "Edge Probability (linear)", "Edge Probability (mlp)"],
        "Heterogeneous Data": ["Heterogeneity (linear)", "Heterogeneity (mlp)", "Highly Heterogeneous", "Highly Mixed Data"]
    }
    
    for category, scenarios in scenario_categories.items():
        valid_scenarios = [s for s in scenarios if s in filtered_json]
        if not valid_scenarios:
            continue
            
        output_text += f"• {category}\n"
        
        # Calculate average performance per algorithm for this category
        category_performance = {}
        for algo in all_algos:
            performances = []
            for scenario in valid_scenarios:
                if algo in filtered_json[scenario]:
                    perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                    if perf != 'N/A':
                        performances.append(float(perf))
            
            if performances:
                category_performance[algo] = sum(performances) / len(performances)
        
        # Sort and display top algorithms for this category
        sorted_category = sorted(category_performance.items(), key=lambda x: x[1], reverse=True)
        for i, (algo, perf) in enumerate(sorted_category[:3]):  # Show top 3
            output_text += f"  {i+1}. {algo}: Performance {perf:.1f}\n"
        output_text += "\n"
    
    # Scenario-specific results
    output_text += "\n────────────────────────────────────────────────────────\n"
    output_text += "Performance by Scenario\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    
    output_text += "### ROBUSTNESS SCENARIOS\n"
    output_text += "These scenarios test algorithm performance across varying levels of a property.\n\n"
    
    # Define which scenarios are robustness scenarios (with variable levels)
    robustness_scenarios = [
        "Variable Scaling (linear)", "Sample Scaling (linear)",
        "Heterogeneity (linear)", "Measurement Error (linear)",
        "Noise Type (linear)", "Missing Data (linear)",
        "Edge Probability (linear)", "Discrete Ratio (linear)",
        "Variable Scaling (mlp)", "Sample Scaling (mlp)",
        "Heterogeneity (mlp)", "Measurement Error (mlp)",
        "Noise Type (mlp)", "Missing Data (mlp)",
        "Edge Probability (mlp)", "Discrete Ratio (mlp)"
    ]
    
    # Define which scenarios are specific settings
    specific_scenarios = [
        "Linear Function", "Non-Linear Function", 
        "Gaussian Noise", "Non-Gaussian Noise", 
        "Dense Graph", "Sparse Graph",
        "High Missing Data", "High Measurement Error",
        "Highly Mixed Data", "Highly Heterogeneous"
    ]
    
    # First process robustness scenarios
    for scenario in filtered_json:
        if scenario not in robustness_scenarios:
            continue
            
        output_text += f"\n• {scenario}\n"
        
        # Create a comparison table for this scenario
        # For scaling scenarios, include efficiency
        is_scaling_scenario = "Scaling" in scenario
        
        if is_scaling_scenario:
            output_text += "| Algorithm | Performance | Stability | Efficiency | Overall Score |\n"
            output_text += "|-----------|------------|-----------|------------|-------------|\n"
        else:
            output_text += "| Algorithm | Performance | Stability |\n"
            output_text += "|-----------|------------|----------|\n"
        
        # Sort algorithms by performance for this scenario
        sorted_algos = sorted(
            [(algo, filtered_json[scenario][algo]['levels'].get('performance', 0)) 
             for algo in filtered_json[scenario]],
            key=lambda x: x[1],
            reverse=True  # Higher performance is better
        )
        
        for algo_name, perf in sorted_algos:
            result = filtered_json[scenario][algo_name]
            perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
            
            # Get standard deviation for stability
            stability = result['ranking'].get('std', 'N/A')
            stability_str = f"{float(stability):.1f}" if stability != 'N/A' else 'N/A'
            
            if is_scaling_scenario:
                # Add row with efficiency and overall score for scaling scenarios
                eff = result['levels'].get('efficiency', 'N/A')
                eff_str = f"{float(eff):.1f}" if eff != 'N/A' else 'N/A'
                
                comp = result['levels'].get('composite', 'N/A')
                comp_str = f"{float(comp):.1f}" if comp != 'N/A' else 'N/A'
                
                output_text += f"| {algo_name} | {perf_str} | {stability_str} | {eff_str} | {comp_str} |\n"
            else:
                # Only performance and stability for non-scaling scenarios
                output_text += f"| {algo_name} | {perf_str} | {stability_str} |\n"
    
    # Then process specific scenarios
    output_text += "\n### SPECIFIC SCENARIOS\n"
    output_text += "These scenarios test algorithm performance at specific settings rather than variable levels.\n\n"
    
    for scenario in filtered_json:
        if scenario not in specific_scenarios:
            continue
            
        output_text += f"\n• {scenario}\n"
        output_text += "| Algorithm | Performance | Stability |\n"
        output_text += "|-----------|------------|----------|\n"
        
        # Sort algorithms by performance for this scenario
        sorted_algos = sorted(
            [(algo, filtered_json[scenario][algo]['levels'].get('performance', 0)) 
             for algo in filtered_json[scenario]],
            key=lambda x: x[1],
            reverse=True  # Higher performance is better
        )
        
        for algo_name, perf in sorted_algos:
            result = filtered_json[scenario][algo_name]
            perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
            
            # Get standard deviation for stability
            stability = result['ranking'].get('std', 'N/A')
            stability_str = f"{float(stability):.1f}" if stability != 'N/A' else 'N/A'
            
            output_text += f"| {algo_name} | {perf_str} | {stability_str} |\n"
    
    # Save to file if output_filename is provided
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(output_text)
        print(f"Filtered benchmarking results saved to {output_filename}")
    
    return output_text

def create_ranking_benchmarking_results(benchmarking_json, algorithm_list=None, output_filename=None):
    """Create a consolidated benchmarking results text optimized for LLM readability,
    focusing on rankings rather than raw values.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        algorithm_list (list): List of algorithm names to include in the results
        output_filename (str, optional): Name of the output file if saving is desired
    
    Returns:
        str: Formatted text containing the ranking-based benchmarking results
    """
    if algorithm_list is None or len(algorithm_list) == 0:
        print("No algorithms specified for filtering. Using all algorithms.")
        return None
    
    # Filter the benchmarking_json to only include the specified algorithms
    filtered_json = {}
    for scenario, algorithms in benchmarking_json.items():
        filtered_algorithms = {}
        for algo, data in algorithms.items():
            # Check if any algorithm in algorithm_list is a substring of algo
            if any(algo.startswith(target_algo) for target_algo in algorithm_list):
                filtered_algorithms[algo] = data
        
        if filtered_algorithms:  # Only include scenarios that have at least one of the specified algorithms
            filtered_json[scenario] = filtered_algorithms
    
    if not filtered_json:
        return "No results found for the specified algorithms."
    
    # Create the output text
    output_text = ""
    output_text += "# ALGORITHM BENCHMARKING RESULTS\n\n"
    output_text += f"Algorithms included: {', '.join(algorithm_list)}\n\n"
    
    # Get all unique algorithm names across all scenarios
    all_algos = set()
    for scenario in filtered_json:
        all_algos.update(filtered_json[scenario].keys())
    all_algos = list(all_algos)
    
    # Format algorithm names to be more readable
    def format_algorithm_name(algo_name):
        # Handle names like 'FCI_indep_test=fisherz'
        if '_' in algo_name:
            parts = algo_name.split('_')
            base_algo = parts[0]
            params = []
            for part in parts[1:]:
                if '=' in part:
                    param, value = part.split('=')
                    # Format as "param: value" instead of "param=value"
                    params.append(f"{param}: {value}")
                else:
                    params.append(part)
            
            if params:
                return f"{base_algo} ({'_'.join(params)})"
            else:
                return base_algo
        return algo_name
    
    # 1. Overall algorithm ranking across all scenarios
    output_text += "## 1. OVERALL ALGORITHM RANKING\n\n"
    
    # Calculate average performance for each algorithm
    algo_avg_performance = {}
    for algo in all_algos:
        performances = []
        for scenario in filtered_json:
            if algo in filtered_json[scenario]:
                perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                if perf != 'N/A':
                    performances.append(float(perf))
        
        if performances:
            algo_avg_performance[algo] = sum(performances) / len(performances)
    
    # Sort algorithms by average performance
    sorted_algos = sorted(algo_avg_performance.items(), key=lambda x: x[1], reverse=True)
    
    # Create a ranking table
    output_text += "| Rank | Algorithm | Overall Score |\n"
    output_text += "|------|-----------|---------------|\n"
    for i, (algo, avg_perf) in enumerate(sorted_algos):
        formatted_algo = format_algorithm_name(algo)
        output_text += f"| {i+1} | {formatted_algo} | {avg_perf:.1f} |\n"
    
    output_text += "\n"
    
    # 2. Best algorithm by scenario category
    output_text += "## 2. BEST ALGORITHM BY SCENARIO CATEGORY\n\n"
    
    # Define scenario categories
    scenario_categories = {
        "Linear Function": "Function Type",
        "Non-Linear Function": "Function Type",
        "Gaussian Noise": "Noise Type",
        "Non-Gaussian Noise": "Noise Type",
        "Variable Scaling": "Scaling",
        "Sample Scaling": "Scaling",
        "Dense Graph": "Graph Structure",
        "Sparse Graph": "Graph Structure",
        "High Missing Data": "Data Quality",
        "High Measurement Error": "Data Quality",
        "Highly Heterogeneous": "Data Quality",
        "Heterogeneity": "Data Quality",
        "Measurement Error": "Data Quality",
        "Missing Data": "Data Quality",
        "Edge Probability": "Graph Structure",
        "Discrete Ratio": "Data Type"
    }
    
    # Get unique categories
    categories = sorted(set(scenario_categories.values()))
    
    # Find best algorithm for each category
    for category in categories:
        output_text += f"### {category}\n\n"
        output_text += "| Scenario | Best Algorithm | Score | Runner-up | Score |\n"
        output_text += "|----------|---------------|-------|-----------|-------|\n"
        
        # Get all scenarios in this category
        category_scenarios = [s for s in filtered_json if any(s.startswith(sc) for sc in scenario_categories if scenario_categories[sc] == category)]
        
        # Process each scenario
        for scenario in sorted(category_scenarios):
            # Sort algorithms by performance for this scenario
            scenario_algos = []
            for algo, data in filtered_json[scenario].items():
                perf = data['levels'].get('performance', 'N/A')
                if perf != 'N/A':
                    scenario_algos.append((algo, float(perf)))
            
            if not scenario_algos:
                continue
                
            # Sort algorithms by performance (highest first)
            scenario_algos.sort(key=lambda x: x[1], reverse=True)
            
            # Get the best and runner-up algorithms
            best_algo, best_score = scenario_algos[0]
            runner_up = "N/A"
            runner_up_score = "N/A"
            if len(scenario_algos) > 1:
                runner_up, runner_up_score = scenario_algos[1]
                
            formatted_best_algo = format_algorithm_name(best_algo)
            formatted_runner_up = format_algorithm_name(runner_up) if runner_up != "N/A" else "N/A"
            
            output_text += f"| {scenario} | {formatted_best_algo} | {best_score:.1f} | {formatted_runner_up} | {runner_up_score if runner_up_score == 'N/A' else f'{runner_up_score:.1f}'} |\n"
        
        output_text += "\n"
    
    # 3. Algorithm Performance Matrix
    output_text += "## 3. ALGORITHM PERFORMANCE MATRIX\n\n"
    output_text += "Performance rating: Excellent (9-10), Good (7-8.9), Moderate (5-6.9), Poor (3-4.9), Very Poor (0-2.9)\n\n"
    
    # Group scenarios by type
    scenario_groups = {
        "Function Type": ["Linear Function", "Non-Linear Function"],
        "Noise Type": ["Gaussian Noise", "Non-Gaussian Noise"],
        "Scaling": ["Variable Scaling", "Sample Scaling"],
        "Graph Structure": ["Dense Graph", "Sparse Graph", "Edge Probability"],
        "Data Quality": ["High Missing Data", "High Measurement Error", "Highly Heterogeneous", 
                         "Heterogeneity", "Measurement Error", "Missing Data"],
        "Data Type": ["Discrete Ratio"]
    }
    
    # Create a matrix table with algorithms as rows and scenario types as columns
    output_text += "| Algorithm |"
    for group_name in scenario_groups:
        output_text += f" {group_name} |"
    output_text += "\n|-----------|"
    for _ in scenario_groups:
        output_text += "------------|"
    output_text += "\n"
    
    # For each algorithm, calculate average performance for each scenario group
    for algo in sorted(all_algos):
        formatted_algo = format_algorithm_name(algo)
        output_text += f"| {formatted_algo} |"
        
        for group_name, group_scenarios in scenario_groups.items():
            # Calculate average performance for this algorithm in this scenario group
            performances = []
            for scenario in filtered_json:
                if algo in filtered_json[scenario] and any(scenario.startswith(s) for s in group_scenarios):
                    perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                    if perf != 'N/A':
                        performances.append(float(perf))
            
            if performances:
                avg_perf = sum(performances) / len(performances)
                # Convert numerical score to rating
                if avg_perf >= 9:
                    rating = "Excellent"
                elif avg_perf >= 7:
                    rating = "Good"
                elif avg_perf >= 5:
                    rating = "Moderate"
                elif avg_perf >= 3:
                    rating = "Poor"
                else:
                    rating = "Very Poor"
                output_text += f" {rating} ({avg_perf:.1f}) |"
            else:
                output_text += " N/A |"
        
        output_text += "\n"
    
    output_text += "\n"
    
    # 4. Special Considerations
    output_text += "## 4. SPECIAL CONSIDERATIONS\n\n"
    
    # Efficiency analysis for scaling scenarios
    output_text += "### Efficiency Analysis (for scaling scenarios)\n\n"
    output_text += "| Algorithm | Efficiency Score | Notes |\n"
    output_text += "|-----------|-----------------|-------|\n"
    
    scaling_scenarios = ["Variable Scaling", "Sample Scaling"]
    for algo in sorted(all_algos):
        efficiency_scores = []
        for scenario in filtered_json:
            if (any(scenario.startswith(s) for s in scaling_scenarios) and 
                algo in filtered_json[scenario] and
                'efficiency' in filtered_json[scenario][algo]['levels'] and 
                filtered_json[scenario][algo]['levels']['efficiency'] != 'N/A'):
                efficiency_scores.append(float(filtered_json[scenario][algo]['levels']['efficiency']))
        
        if efficiency_scores:
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
            if avg_efficiency >= 4.5:
                notes = "Excellent efficiency, suitable for large-scale problems"
            elif avg_efficiency >= 3.5:
                notes = "Good efficiency, handles most scaling challenges well"
            elif avg_efficiency >= 2.5:
                notes = "Moderate efficiency, may struggle with very large problems"
            elif avg_efficiency >= 1.5:
                notes = "Poor efficiency, not recommended for large-scale problems"
            else:
                notes = "Very poor efficiency, will likely fail on large-scale problems"
            
            formatted_algo = format_algorithm_name(algo)
            output_text += f"| {formatted_algo} | {avg_efficiency:.1f} | {notes} |\n"
    
    output_text += "\n"
    
    # 5. Recommendations
    output_text += "## 5. ALGORITHM RECOMMENDATIONS\n\n"
    
    # Best overall algorithm
    if sorted_algos:
        best_algo, _ = sorted_algos[0]
        formatted_best_algo = format_algorithm_name(best_algo)
        output_text += f"- **Best Overall Algorithm**: {formatted_best_algo}\n"
    
    # Best algorithms for specific scenario types
    output_text += "- **Best for Specific Use Cases**:\n"
    
    for category in categories:
        category_scores = {}
        
        # Get all scenarios in this category
        category_scenarios = [s for s in filtered_json if any(s.startswith(sc) for sc in scenario_categories if scenario_categories[sc] == category)]
        
        # Calculate average performance for each algorithm in this category
        for algo in all_algos:
            scores = []
            for scenario in category_scenarios:
                if algo in filtered_json[scenario]:
                    perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                    if perf != 'N/A':
                        scores.append(float(perf))
            
            if scores:
                category_scores[algo] = sum(scores) / len(scores)
        
        # Find the best algorithm for this category
        if category_scores:
            best_algo_for_category = max(category_scores.items(), key=lambda x: x[1])
            formatted_best_algo = format_algorithm_name(best_algo_for_category[0])
            output_text += f"  - **{category}**: {formatted_best_algo} (Score: {best_algo_for_category[1]:.1f})\n"
    
    # Save to file if output_filename is provided
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(output_text)
        print(f"Ranking benchmarking results saved to {output_filename}")
    
    return output_text

def test_create_filtered_benchmarking_results():
    # Load the benchmarking JSON data
    with open('algorithm/context/benchmarking/algorithm_performance_analysis.json', 'r') as f:
        benchmarking_json = json.load(f)
    
    # Select two algorithms to test
    algorithm_list = ["GOLEM", "XGES"] #["PC", "GES", "FCI", "CDNOD", "XGES", "FGES", "GRaSP", "NOTEARSLinear", "GOLEM",
                    #   "DirectLiNGAM", "InterIAMB", "BAMB", "HITONMB", "IAMBnPC", "MBOR"]
    print(benchmarking_json['Sample Scaling (linear)']['PC_indep_test=fisherz_gpu'])


    # Call the function to generate filtered results
    output_text = create_filtered_benchmarking_results(benchmarking_json, algorithm_list)
    
    # Save output for inspection
    with open('test_filtered_results.txt', 'w') as f:
        f.write(output_text)
    
    # Verify the output contains all expected information
    
    # 1. Check if both algorithms are included
    for algo in algorithm_list:
        assert algo in output_text, f"Algorithm {algo} not found in output"
    
    # 2. Check if all scenarios are included for these algorithms
    expected_scenarios = [
        "Variable Scaling (linear)", "Sample Scaling (linear)", 
        "Heterogeneity (linear)", "Measurement Error (linear)",
        "Noise Type (linear)", "Missing Data (linear)",
        "Edge Probability (linear)", "Discrete Ratio (linear)",
        "Variable Scaling (mlp)", "Sample Scaling (mlp)",
        "Heterogeneity (mlp)", "Measurement Error (mlp)",
        "Noise Type (mlp)", "Missing Data (mlp)",
        "Edge Probability (mlp)", "Discrete Ratio (mlp)",
        "Linear Function", "Non-Linear Function", "Gaussian Noise",
        "Non-Gaussian Noise", "Dense Graph", "Sparse Graph",
        "High Missing Data", "High Measurement Error",
        "Highly Mixed Data", "Highly Heterogeneous"
    ]
    
    # Check for scenarios where both algorithms exist
    for scenario in expected_scenarios:
        if scenario in benchmarking_json and all(algo in benchmarking_json[scenario] for algo in algorithm_list):
            assert scenario in output_text, f"Scenario {scenario} not found in output"
    
    # 3. Check if performance, efficiency, and composite scores are included
    metrics = ["Performance", "Efficiency", "Overall Score"]
    for metric in metrics:
        assert metric in output_text, f"Metric {metric} not found in output"
    
    # 4. Check if ranking information is included
    assert "Overall ranking" in output_text, "Overall ranking section not found"

    print(output_text)
    
    # 5. Check if scenario groups are correctly formatted
    scenario_groups = ["Scaling Scenarios", "Function Type Scenarios", 
                       "Noise Type Scenarios", "Other Scenarios"]
    for group in scenario_groups:
        assert group in output_text, f"Scenario group {group} not found"
    
    # 6. Check if top/bottom performers sections exist
    assert "Top and Bottom Performers by Scenario" in output_text
    assert "Top performers:" in output_text
    assert "Bottom performers:" in output_text
    
    print("Test passed! All expected information is present in the output.")

# Run the test
if __name__ == "__main__":
    test_create_filtered_benchmarking_results()
