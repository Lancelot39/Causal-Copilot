# Causal Discovery Report for Variables X0 through X7

## Background
The purpose of this report is to explore the relationships among a set of variables denoted as X0, X1, X2, X3, X4, X5, X6, and X7, offering insights into their interdependencies and potential causal interactions. These variables may represent a range of concepts depending on the context of the study, such as demographic factors (e.g., age, income), behavioral indicators (e.g., spending habits, time spent on activities), or environmental influences (e.g., temperature, pollution levels). 

Understanding the relationships among these variables can provide valuable insights into underlying mechanisms and support decision-making in various fields, such as economics, healthcare, and social sciences. For example, X0 could represent a treatment or intervention variable, while X1 to X7 might include various outcome measures or confounding factors that interact with X0. By systematically analyzing the connections between these variables, this report aims to identify significant patterns, potential causal pathways, and the broader implications of these findings in practical applications.

## Dataset Descriptions
The dataset used for this analysis consists of categorical data without missing values. The basic structure is as follows:

| X0 | X1 | X2 | X3 | X4 | X5 | X6 | X7 |
|----|----|----|----|----|----|----|----|
| 1  | 1  | 0  | 1  | 1  | 0  | 0  | 1  |
| 1  | 0  | 0  | 0  | 1  | 1  | 1  | 0  |
| 1  | 0  | 1  | 1  | 1  | 0  | 0  | 1  |
| 0  | 1  | 0  | 0  | 1  | 1  | 1  | 0  |
| 0  | 0  | 0  | 0  | 1  | 1  | 1  | 1  |

This dataset does not have missing values, is categorical in nature, violates the linearity assumption, and satisfies the Gaussian error assumption. 

## Discovery Procedure
To identify the causal relationships among the variables, we employed a structured discovery procedure that includes:
1. **Initial Graph Construction** - Using causal discovery algorithms, an initial graph representation of the relationships among variables was generated.
2. **Bootstrapping and LLM Pruning** - The initial graph was then refined through bootstrapping and suggestions from a large language model (LLM) to improve accuracy and interpretability.
3. **Validation of Results** - The produced causal graph was evaluated for coherence with the data and the directionality of relationships.

## Results Summary

The following are result graphs produced by our algorithm. The initial graph is the one observed in the first attempt, while the revised graph is pruned based on bootstrapping and LLM suggestions.

| <center> True Graph | <center> Initial Graph | <center> Revised Graph |
| -- | -- | -- |
| ![Initial Graph](/postprocess/test_data/20241007_184921_base_nodes8_samples1500/output_graph/True_Graph.jpg) | ![Initial Graph](/postprocess/test_data/20241007_184921_base_nodes8_samples1500/output_graph/Initial_Graph.jpg) | ![Revised Graph](/postprocess/test_data/20241007_184921_base_nodes8_samples1500/output_graph/Revised_Graph.jpg) |

### Interpretation of Causal Relationships
- **X1 → X0**: There is a direct causal influence of X1 on X0, meaning that changes in X1 will directly impact X0.
- **X2 → X0**: X2 also exerts a causal effect on X0, indicating that X2 impacts X0 independently of X1.
- **X2 → X7**: Variations in X2 will affect X7, suggesting a predictive relationship.
- **X4 → (X3, X5)**: X4 serves as a significant causal factor for both X3 and X5, indicating its central role in influencing these variables.
- **X6 → (X1, X4, X5)**: X6 is a causal agent for multiple outcomes, showing extensive influence across the network.

Overall, this network of causal relationships highlights the interconnectedness of the variables, suggesting potential pathways for interventions or analysis regarding how altering one variable may impact the others. The insights derived from this causal structure can inform future research and practice, demonstrating the importance of understanding these complex interactions.

### Metrics Evaluation
The following are some metrics measuring the difference between the true graph and the estimated one.
- **SHD (Structural Hamming Distance)**: 0
- **precision**: 1
- **Recall**: 1
- **f1**: 1
