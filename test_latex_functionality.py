#!/usr/bin/env python3
"""
LaTeX Functionality Test for Causality-Copilot

This script tests all LaTeX-related functionality needed for the project.
"""

import sys
import os
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.latex]

if os.getenv("RUN_LATEX_TESTS") != "1":
    pytest.skip("set RUN_LATEX_TESTS=1 to run LaTeX integration checks", allow_module_level=True)

import subprocess
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def test_latex_compilation():
    """Test basic LaTeX compilation"""
    print("🔧 Testing LaTeX compilation...")
    
    # Create a simple LaTeX document
    latex_content = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\title{Test Document}
\author{Causality-Copilot}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This is a test document to verify LaTeX functionality.

\section{Mathematical Formulas}
Here's a mathematical formula:
\begin{equation}
E = mc^2
\end{equation}

\section{Table Example}
\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
Variable & Value \\
\midrule
$\alpha$ & 0.05 \\
$\beta$ & 0.02 \\
\bottomrule
\end{tabular}
\caption{Example table with booktabs}
\end{table}

\section{Code Listing}
\begin{lstlisting}[language=Python]
import numpy as np
import pandas as pd

def causal_analysis(data):
    return "Causal relationship found"
\end{lstlisting}

\end{document}
"""
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = os.path.join(temp_dir, "test.tex")
            with open(tex_file, 'w') as f:
                f.write(latex_content)
            
            # Compile LaTeX to PDF
            result = subprocess.run([
                'pdflatex', '-output-directory', temp_dir, tex_file
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pdf_file = os.path.join(temp_dir, "test.pdf")
                if os.path.exists(pdf_file):
                    print("✅ LaTeX compilation successful - PDF generated")
                    return True
                else:
                    print("❌ LaTeX compilation finished but no PDF found")
                    return False
            else:
                print(f"❌ LaTeX compilation failed: {result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("❌ LaTeX compilation timed out")
        return False
    except Exception as e:
        print(f"❌ LaTeX compilation error: {e}")
        return False

def test_pylatex():
    """Test PyLaTeX functionality"""
    print("🐍 Testing PyLaTeX...")
    
    try:
        from pylatex import Document, Section, Subsection, Command
        from pylatex.base_classes import Environment
        from pylatex.package import Package
        
        # Create a document
        doc = Document()
        doc.packages.append(Package('booktabs'))
        doc.packages.append(Package('amsmath'))
        
        with doc.create(Section('Test Section')):
            doc.append('This is a test section created with PyLaTeX.')
            
            with doc.create(Subsection('Math')):
                doc.append('Mathematical formula: ')
                doc.append(Command('begin', arguments='equation'))
                doc.append(r'y = mx + b')
                doc.append(Command('end', arguments='equation'))
        
        print("✅ PyLaTeX document creation successful")
        return True
        
    except Exception as e:
        print(f"❌ PyLaTeX test failed: {e}")
        return False

def test_pandas_latex():
    """Test pandas to LaTeX conversion"""
    print("📊 Testing pandas to LaTeX conversion...")
    
    try:
        # Create sample data similar to what the project uses
        df = pd.DataFrame({
            'Variable': ['X1', 'X2', 'X3', 'X4'],
            'Mean': [1.25, 2.34, 0.87, 3.12],
            'Std': [0.45, 0.67, 0.23, 0.89],
            'Type': ['Continuous', 'Discrete', 'Binary', 'Continuous']
        })
        
        # Convert to LaTeX (this is what the project does)
        latex_table = df.to_latex(index=False)
        
        # Check if the conversion worked
        if 'tabular' in latex_table and 'Variable' in latex_table:
            print("✅ Pandas to LaTeX conversion successful")
            print("   Sample output:")
            print(f"   {latex_table[:100]}...")
            return True
        else:
            print("❌ Pandas to LaTeX conversion failed - invalid output")
            return False
            
    except Exception as e:
        print(f"❌ Pandas to LaTeX test failed: {e}")
        return False

def test_matplotlib_latex():
    """Test matplotlib with LaTeX integration"""
    print("📈 Testing matplotlib LaTeX integration...")
    
    try:
        # Create a simple plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        
        ax.plot(x, y, label=r'$y = \sin(x)$')
        ax.set_xlabel(r'$x$ (radians)')
        ax.set_ylabel(r'$y = \sin(x)$')
        ax.set_title(r'Sine Function: $f(x) = \sin(x)$')
        ax.legend()
        ax.grid(True)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            fig.savefig(tmp.name, format='pdf', bbox_inches='tight')
            if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                print("✅ Matplotlib LaTeX integration successful")
                os.unlink(tmp.name)  # Clean up
                plt.close(fig)
                return True
            else:
                print("❌ Matplotlib LaTeX integration failed - no output file")
                plt.close(fig)
                return False
                
    except Exception as e:
        print(f"❌ Matplotlib LaTeX test failed: {e}")
        plt.close('all')
        return False

def test_project_latex_integration():
    """Test the project's specific LaTeX integration"""
    print("🎯 Testing project-specific LaTeX integration...")
    
    try:
        # Test plumbum latexmk (used in the project)
        from plumbum.cmd import latexmk
        print("✅ Plumbum latexmk import successful")
        
        # Test if we can create a simple LaTeX workflow like the project does
        latex_template = r"""
\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}

\begin{document}
\title{Causal Discovery Report}
\author{Causality-Copilot}
\maketitle

\section{Data Overview}
[DATA_PREVIEW]

\section{Results}
[RESULTS]

\end{document}
"""
        
        # Sample data preview (like in the project)
        sample_df = pd.DataFrame({
            'Feature1': [1.2, 2.3, 3.4],
            'Feature2': [0.8, 1.9, 2.1],
            'Target': [0, 1, 0]
        })
        data_preview = sample_df.to_latex(index=False)
        
        # Replace placeholders (like in the project)
        final_latex = latex_template.replace('[DATA_PREVIEW]', data_preview)
        final_latex = final_latex.replace('[RESULTS]', "Causal relationships discovered successfully.")
        
        print("✅ Project LaTeX template processing successful")
        return True
        
    except Exception as e:
        print(f"❌ Project LaTeX integration test failed: {e}")
        return False

def test_additional_packages():
    """Test additional LaTeX-related packages"""
    print("📚 Testing additional LaTeX packages...")
    
    success_count = 0
    total_tests = 0
    
    # Test pypandoc
    try:
        import pypandoc
        print("✅ pypandoc import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ pypandoc import failed: {e}")
    total_tests += 1
    
    # Test nbconvert
    try:
        import nbconvert
        print("✅ nbconvert import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ nbconvert import failed: {e}")
    total_tests += 1
    
    # Test sphinx
    try:
        import sphinx
        print("✅ sphinx import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ sphinx import failed: {e}")
    total_tests += 1
    
    # Test bibtexparser
    try:
        import bibtexparser
        print("✅ bibtexparser import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ bibtexparser import failed: {e}")
    total_tests += 1
    
    print(f"📊 Additional packages: {success_count}/{total_tests} successful")
    return success_count == total_tests

def main():
    print("📄 LaTeX Functionality Test for Causality-Copilot")
    print("=" * 60)
    
    tests = [
        ("LaTeX Compilation", test_latex_compilation),
        ("PyLaTeX Integration", test_pylatex),
        ("Pandas LaTeX Conversion", test_pandas_latex),
        ("Matplotlib LaTeX", test_matplotlib_latex),
        ("Project LaTeX Integration", test_project_latex_integration),
        ("Additional Packages", test_additional_packages),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 LaTeX Functionality Test Results")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All LaTeX functionality tests passed!")
        print("   Your LaTeX setup is ready for Causality-Copilot.")
        print("\n💡 Next steps:")
        print("   1. Run the full demo: python web_demo/demo.py")
        print("   2. Generate a test report to verify PDF generation")
    else:
        print("\n⚠️  Some LaTeX functionality tests failed.")
        print("   Please check the error messages above and install missing components.")
        print("\n💡 Common solutions:")
        print("   - Ensure LaTeX is properly installed: pdflatex --version")
        print("   - Install missing LaTeX packages: tlmgr install <package-name>")
        print("   - Check Python package installations: pip list | grep -E '(latex|pandoc|sphinx)'")

if __name__ == "__main__":
    main()
