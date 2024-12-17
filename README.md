<h1 align="center">
<img src="asset/logo.png" width="200" alt="Causality" />
<br>
Causal-Copilot: An Autonomous Causal Analysis Agent
</h1>
<p align="center">
  <a href="https://huggingface.co/spaces/Causal-Copilot/Causal-Copilot"><b>[Demo]</b></a> •
  <a href="https://github.com/Lancelot39/Causal-Copilot"><b>[Code]</b></a>
</p>


---

## Introduction
Understanding causal relationships is fundamental to scientific discovery, enabling researchers to move beyond mere correlation and establish the underlying mechanisms that drive natural and social phenomena. Recent years have witnessed significant theoretical advancements in causal discovery, yielding a diverse array of sophisticated methodologies. However, the complexity of these methods—each with its distinct assumptions, applicability conditions, and technical nuances—has created substantial barriers for scientists outside the field of causal analysis, often deterring them from adopting these powerful analytical tools in their research.

Causal-Copilot is a LLM-oriented toolkit for **automatic causal analysis** that uniquely integrates domain knowledge from large language models with established expertise from causal discovery researchers. Designed for scientific researchers and data scientists, it facilitates the identification, analysis, and interpretation of causal relationships within real-world datasets through natural dialogue. The system autonomously orchestrates the entire analytical pipeline-analyzing statistics, selecting optimal causal analysis algorithms, configuring appropriate hyperparameters, synthesizing executable code, conducting uncertainty quantification, and generating comprehensive PDF reports—while requiring minimal expertise in causal methods. This seamless integration of conversational interaction and rigorous methodology culminates enables researchers across disciplines to focus on domain-specific insights rather than technical implementation details.

<p align="center">
  🔍 Try out our interactive demo: <a href="https://huggingface.co/spaces/Causal-Copilot/Causal-Copilot"><b>Causal-Copilot Live Demo</b></a>
</p>

---

## Demo

### Video Demo

[![Demo Video](asset/chatbot.png)](https://www.youtube.com/watch?v=A6j80I97Slg)

### Report Examples

We provide some examples of our system automatically generated reports for open-source datasets generated as follows:

- [1. Bioinformatics-Abalone](asset/report_Abalone.pdf)
- [2. Architecture-CCS](asset/report_CCS.pdf)
- [3. Bioinformatics-Sachs](asset/report_Sachs.pdf)

---

## Table of Contents

- [Demo](#Demo)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)
- [Contact](#Contact)

---

## Features

- **Automated Causal Analysis**: Harnesses the power of large language models combined with domain expertise to select optimal causal analysis algorithms and hyperparameters. Incorporates proven methodological insights from causal discovery researchers to ensure the analytical reliability, without the requirements in expertise about causality and extensive parameter tuning. 
- **Statistical-LLM Hybrid Post Processing**: Present the edge uncertainty examination (bootstrap), as well as graph pruning and direction revision driven by LLM's prior knowledge.
- **Chat-based User-friendly Interface**: Navigate complex causal analysis through natural dialogue, and visualize data statistics and causal graphs through clear, intuitive figures, without wrestling with technical details. 
- **Comprehensive Analysis Report**: Provide well-formulated scientific report for the whole causal analysis process, containing detailed explanation documenting the complete analytical process, intuitive visualization and in-depth interpretation of the findings.
- **Extensibility**: Maintain open interfaces for integrating new causal analysis algorithms and support seamless incorporation of emerging causality-related libraries and methodologies

### Architecture Details

- Our Causal-Copilot consists of four components, namely preprocessing, decision making, post processing and intepretation parts, which are all supported by SOTA LLMs (e.g., GPT-4o, GPT-4o-mini).

<h1 align="center">
<div style="text-align: center;">
    <img src="asset/architecture.png" width="700" alt="Causality" />
</div>
</h1>

### Evaluation on Simulated Data

- We evaluate the automatic causal discovery ability of our Causal-Copilot on in total 180 simulated datasets including different types of functional forms, graph sparsity, noise types and heterogeneity, compared with a robust baseline, PC algorithm with the default setting.
- The results show that our Causal-Copilot can achieve much better performance, indicating the effectiveness of its automatic algorithm selection and hyper-parameter setting strategy, in a autonomous manner.

| Metric    | Baseline | Causal-Copilot |
|-----------|----------|----------------|
| Precision | 78.6%    | **81.6%**      |
| Recall    | 78.2%    | **81.0%**      |
| F1-score  | 76.1%    | **79.3%**      |

---

## Getting Started

### Online Demo

<p align="center">
  🔍 Try out our interactive demo: <a href="https://huggingface.co/spaces/Causal-Copilot/Causal-Copilot"><b>Causal-Copilot Live Demo</b></a>
</p>


### Local Deployment

- **Python 3.8+**
- Required Python libraries (specified in `requirements.txt`)

Ensure you have the necessary dependencies installed by running:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py --data_file your_data --apikey your_openai_apikey --initial_query your_user_query
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Resource 

- Our codes for causal discovery are from the [causal-learn](https://causal-learn.readthedocs.io/en/latest/index.html) and [CausalNex](https://causalnex.readthedocs.io/en/latest/) projects, currently including [PC](https://link.springer.com/book/10.1007/978-1-4612-2748-9), [FCI](https://arxiv.org/abs/1302.4983), [CDNOD](https://arxiv.org/abs/1903.01672), [GES](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf), [NOTEARS](https://arxiv.org/abs/1803.01422), [DirectLiNGAM](https://jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf), [ICALiNGAM](https://jmlr.org/papers/volume7/shimizu06a/shimizu06a.pdf)
- Our PDF template is based on this [overleaf project](https://www.overleaf.com/latex/templates/style-and-template-for-preprints-arxiv-bio-arxiv/fxsnsrzpnvwc)
- Our example datasets are from [Bioinformatics-Abalone](https://archive.ics.uci.edu/dataset/1/abalone), [Architecture-CCS](https://netl.doe.gov/carbon-management/carbon-storage/worldwide-ccs-database), [Bioinformatics-Sachs](https://www.science.org/doi/10.1126/science.1105809)
- Our codes for deployment are from [Gradio](https://www.gradio.app/)

---

## Contributor

- [Xinyue Wang*](https://github.com/CharonWangg), [Kun Zhou*](https://github.com/Lancelot39)(Equal Contribution), [Wenyi Wu](https://github.com/WenyiWU0111), [Fang Nan](https://github.com/Fangn06), Shivam Singh, [Biwei Huang](https://biweihuang.com/)


## Contact

For additional information, questions, or feedback, please contact ours at **xiw159@ucsd.edu**, **franciskunzhou@gmail.com**, **wew058@ucsd.edu**, fnan@ucsd.edu and bih007@ucsd.edu. We welcome contributions! Come and join us now!

If you use Causal-Copilot in your research, please cite it as follows:

```
@inproceedings{causalcopilot,
  title={Causal-Copilot: An Autonomous Causal Analysis Agent},
  author={Wang, Xinyue and Zhou, Kun and Wu, Wenyi and Nan, Fang and Huang, Biwei},
  year={2024}
}
```
