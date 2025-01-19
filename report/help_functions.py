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

def list_conversion(text):
    # Split the text into lines
    lines = text.strip().split('\n')
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

