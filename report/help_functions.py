import re

def fix_latex_itemize(text):
    """
    Fix missing itemize environment in LaTeX text
    """
    # Check if there are \item entries without proper environment
    items_pattern = r'\\item\s+'
    
    # If there are \item entries
    if re.search(items_pattern, text):
        # Check if itemize environment exists
        has_begin = '\\begin{itemize}' in text
        has_end = '\\end{itemize}' in text
        
        if not (has_begin and has_end):
            # Split text into lines for better processing
            lines = text.split('\n')
            new_lines = []
            items_started = False
            items_block = []
            
            for line in lines:
                # If line contains \item and itemize hasn't started
                if line.strip().startswith('\\item') and not items_started:
                    items_started = True
                    items_block.append('\\begin{itemize}')
                    items_block.append(line)
                # If line contains \item and itemize has started
                elif line.strip().startswith('\\item'):
                    items_block.append(line)
                # If no \item but items have started
                elif items_started and not line.strip().startswith('\\item'):
                    if line.strip():  # If line is not empty
                        items_block.append('\\end{itemize}')
                        items_started = False
                        new_lines.extend(items_block)
                        items_block = []
                        new_lines.append(line)
                    else:
                        items_block.append(line)
                else:
                    new_lines.append(line)
            
            # If items block is still open, close it
            if items_started:
                items_block.append('\\end{itemize}')
                new_lines.extend(items_block)
            
            return '\n'.join(new_lines)
    
    return text

def fix_latex_itemize_LLM(client, text):
    prompt = rf"""
    The following text is a LaTeX code. Please check if there are any \item entries that are not properly enclosed in an itemize environment. 
    If there are, please add the necessary \begin{{itemize}} and \end{{itemize}} commands to fix the document. If the document is already correct, please return it unchanged.
    Only return the corrected LaTeX code without any additional text or explanation.
    Do not include any titles, subsections, \begin{{document}}, or \end{{document}} commands.
    """
    response = LLM_parse_query(client, None, prompt, text)
    return response

def list_conversion(text):
    # Split the text into lines
    lines = text.strip().split('\n')
    lines = [l for l in lines if l not in ['', '\n']]
    latex_lines = []
    item_num = 0
    # Process each line
    for line_ind in range(len(lines)):
        line = lines[line_ind].strip()
        if line.startswith('-') or line.startswith('*') or line.startswith('+'):
            # Convert bullet points to LaTeX itemize
            item_num += 1
            if item_num > 1:  # Not the first list item
                latex_lines.append(r"  \item " + line[2:].strip())
                if line_ind != len(lines) - 1:  # Not Last line
                    if not lines[line_ind + 1].strip().startswith('-') and not lines[line_ind + 1].strip().startswith('*') and not lines[line_ind + 1].strip().startswith('+'):
                        latex_lines.append(r"\end{itemize}" + "\n")
                        item_num = 0
                else:
                    latex_lines.append(r"\end{itemize}" + "\n")
                    item_num = 0
            else:  # Starting a new itemize list
                latex_lines.append(r"\begin{itemize}" + "\n")
                latex_lines.append(r"  \item " + line[2:].strip())
        else:
            latex_lines.append(line + "\n")
    
    return "\n".join(latex_lines)

def bold_conversion(text):
    while '**' in text:
        text = text.replace('**', r'\textbf{', 1).replace('**', '}', 1)
    return text

def load_context(filepath):
    with open(filepath, "r") as f:
        return f.read()
        
# Function to replace Greek letters in text
def replace_unicode_with_latex( text):
    greek_to_latex = {
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "γ": r"$\gamma$",
        "δ": r"$\delta$",
        "ε": r"$\epsilon$",
        "ζ": r"$\zeta$",
        "η": r"$\eta$",
        "θ": r"$\theta$",
        "ι": r"$\iota$",
        "κ": r"$\kappa$",
        "λ": r"$\lambda$",
        "μ": r"$\mu$",
        "ν": r"$\nu$",
        "ξ": r"$\xi$",
        "ο": r"$o$",
        "π": r"$\pi$",
        "ρ": r"$\rho$",
        "σ": r"$\sigma$",
        "τ": r"$\tau$",
        "υ": r"$\upsilon$",
        "φ": r"$\phi$",
        "χ": r"$\chi$",
        "ψ": r"$\psi$",
        "ω": r"$\omega$",
        '↔': r'$\leftrightarrow$',
        '→': r'$\rightarrow$',
        '←': r'$\leftarrow$',
        '⇔': r'$\Leftrightarrow$',
        '⇒': r'$\Rightarrow$',
        '⇐': r'$\Leftarrow$',
        '↑': r'$\uparrow$',
        '↓': r'$\downarrow$',
        '⟶': r'$\longrightarrow$',
        '⟵': r'$\longleftarrow$'
    }
    # Use regular expressions to find and replace Greek letters
    pattern = "|".join(map(re.escape, greek_to_latex.keys()))
    return re.sub(pattern, lambda match: greek_to_latex[match.group()], text)

def remove_redundant_point(text):
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('\item'):  # Regular bullet point
            if not r'\textbf' in line:  # Not a header
                line = line.replace('\item','\n') # Remove the hyphen but keep indentation
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def remove_redundant_title(text):
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('#'):  # Regular bullet point
            line = ''
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def LLM_parse_query(client, format, prompt, message):
    if format:
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        response_format=format,
        )
        parsed_response = completion.choices[0].message.parsed
    else: 
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        )
        parsed_response = completion.choices[0].message.content
    return parsed_response

def granger_causality_to_latex(potential_granger_list):
    """
    Transforms a list of potential Granger causality relationships into LaTeX format.
    
    :param potential_granger_list: List of dictionaries containing potential Granger causality relationships
    :return: LaTeX formatted string with paragraph and itemize environment
    """
    if not potential_granger_list:
        return "No potential Granger causality relationships were identified."
    
    # Start with an introductory paragraph
    latex = "We calculated both time-lagged and instantaneous correlation coefficients between variables. Here  we list some variable pairs which have large absolute lagged correlation coefficients. \n\n"
    # Add itemize environment for bullet points
    latex += "\\begin{itemize}\n"
    
    # Add bullet points for each relationship
    for item in potential_granger_list[:5]:
        corr_str = f"{item['correlation']:.3f}"
        # Add + sign for positive correlations for clarity
        if item['correlation'] > 0:
            corr_str = "+" + corr_str
            
        latex += f"  \\item {item['cause'].replace('_', ' ')} $\\rightarrow$ {item['effect'].replace('_', ' ')} at Lag {item['lag']}: {corr_str}\n"
    
    # Close itemize environment
    latex += "\\end{itemize}\n"
    
    return latex

def stationarity_summary_to_latex(summary):
    """
    Convert stationarity summary dictionary to LaTeX formatted text.
    
    :param summary: Dictionary containing stationarity analysis results
    :return: LaTeX formatted text describing the stationarity results
    """
    if not summary or "total_variables_analyzed" not in summary:
        return "No stationarity analysis results available."
    
    latex = ""
    
    # Overall statistics
    latex += "\\begin{itemize}\n"
    latex += f"  \\item \\textbf{{Variables analyzed:}} {summary['total_variables_analyzed']}\n"
    latex += f"  \\item \\textbf{{Stationary variables:}} {summary['total_stationary']} "
    latex += f"({summary['total_stationary']/summary['total_variables_analyzed']*100:.1f}\\%)\n"
    latex += f"  \\item \\textbf{{Non-stationary variables:}} {summary['total_non_stationary']} "
    latex += f"({summary['total_non_stationary']/summary['total_variables_analyzed']*100:.1f}\\%)\n"
    latex += "\\end{itemize}\n\n"
    
    # List of stationary variables
    if summary["stationary_variables"]:
        latex += "\\paragraph{Stationary Variables}\n\n"
        latex += "\\begin{itemize}\n"
        for var in summary["stationary_variables"]:
            # Escape underscores for LaTeX
            var_latex = var.replace("_", "\\_")
            latex += f"  \\item {var_latex}\n"
        latex += "\\end{itemize}\n\n"
    else:
        latex += "\\paragraph{Stationary Variables}\n\n"
        latex += "No stationary variables were identified.\n\n"
    
    # List of non-stationary variables
    if summary["non_stationary_variables"]:
        latex += "\\paragraph{Non-stationary Variables}\n\n"
        latex += "\\begin{itemize}\n"
        for var in summary["non_stationary_variables"]:
            # Escape underscores for LaTeX
            var_latex = var.replace("_", "\\_")
            latex += f"  \\item {var_latex}\n"
        latex += "\\end{itemize}\n\n"
    else:
        latex += "\\paragraph{Non-stationary Variables}\n\n"
        latex += "No non-stationary variables were identified.\n\n"
    
    # Add conclusion or recommendation based on the results

    if summary["total_non_stationary"] > summary["total_stationary"]:
        latex += "Most variables in this dataset are \\textbf{non-stationary}. "
        latex += "Consider differencing or other transformations before modeling.\n"
    else:
        latex += "Most variables in this dataset are \\textbf{stationary}, "
        latex += "which is favorable for time series analysis without further transformations.\n"
    
    return latex

def eda_summary_to_latex(eda_result):
        dist_input_num = eda_result['dist_analysis_num']
        dist_input_cat = eda_result['dist_analysis_cat']
        corr_input = eda_result['corr_analysis']
        
        # Description of distribution
        response_dist_doc = ""
        if dist_input_num != {}:
            response_dist_doc += "Numerical Variables \n \\begin{itemize} \n"
            left_skew_list = []
            right_skew_list = []
            symmetric_list = []
            for feature in dist_input_num.keys():
                if dist_input_num[feature]['mean']<dist_input_num[feature]['median']:
                    left_skew_list.append(feature.replace('_', ' '))
                elif dist_input_num[feature]['mean']>dist_input_num[feature]['median']:
                    right_skew_list.append(feature.replace('_', ' '))
                else:
                    symmetric_list.append(feature.replace('_', ' '))
            response_dist_doc += f"\item Slight left skew distributed variables: {', '.join(left_skew_list) if left_skew_list != [] else 'None'} \n"
            response_dist_doc += f"\item Slight right skew distributed variables: {', '.join(right_skew_list) if right_skew_list != [] else 'None'} \n"
            response_dist_doc += f"\item Symmetric distributed variables: {', '.join(symmetric_list) if symmetric_list != [] else 'None'} \n"
            response_dist_doc += "\end{itemize} \n"
        if dist_input_cat != {}:
            response_dist_doc += "Categorical Variables \n"
            response_dist_doc += "\\begin{itemize} \n"
            for feature in dist_input_cat.keys():
                response_dist_doc += f"\item {feature}: {dist_input_cat[feature]} \n"
            response_dist_doc += "\end{itemize} \n"
        #print('response_dist_doc: ', response_dist_doc)
        # Description of Correlation
        response_corr_doc = "\\begin{itemize} \n"
        high_corr_list = [f"{key[0]}".replace('_', ' ') + " - " + f"{key[1]}".replace('_', ' ') for key, value in corr_input.items() if abs(value) > 0.8]
        if len(high_corr_list)>10:
            response_corr_doc += f"\item \\textbf{{Strongly Correlated Variables}} ($\geq 0.9$): {', '.join(high_corr_list)}"
            response_corr_doc += ", etc. \n"
        else:
            response_corr_doc += f"\item \\textbf{{Strongly Correlated Variables}} ($\geq 0.9$): {', '.join(high_corr_list) if high_corr_list != [] else 'None'} \n"
        med_corr_list = [f"{key[0]}".replace('_', ' ') + " - " + f"{key[1]}".replace('_', ' ') for key, value in corr_input.items() if (abs(value) <= 0.8 and abs(value) > 0.5)]
        if len(med_corr_list)>10:
            response_corr_doc += f"\item \\textbf{{Moderately Correlated Variables}} ($0.1-0.9$): {', '.join(med_corr_list)}"
            response_corr_doc += ", etc. \n"
        else:
            response_corr_doc += f"\item \\textbf{{Moderately Correlated Variables}} ($0.1-0.9$): {', '.join(med_corr_list) if med_corr_list != [] else 'None'} \n"
        low_corr_list = [f"{key[0]}".replace('_', ' ') + " - " + f"{key[1]}".replace('_', ' ') for key, value in corr_input.items() if abs(value) <= 0.5]
        if len(low_corr_list)>10:
            response_corr_doc += f"\item \\textbf{{Weakly Correlated Variables}} ($\leq 0.1$): {', '.join(low_corr_list)}"
            response_corr_doc += ", etc. \n"
        else:
            response_corr_doc += f"\item \\textbf{{Weakly Correlated Variables}} ($\leq 0.1$): {', '.join(low_corr_list) if low_corr_list != [] else 'None'} \n"
        response_corr_doc += "\end{itemize} \n"
        #print('response_corr_doc: ',response_corr_doc)
        return response_dist_doc, response_corr_doc