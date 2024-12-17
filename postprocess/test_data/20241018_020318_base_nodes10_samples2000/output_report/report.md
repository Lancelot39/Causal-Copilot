# Causal Discovery Report on Variables X0 through X9

## Background
The purpose of this report is to explore and analyze the potential relationships among the variables X0, X1, X2, X3, X4, X5, X6, X7, X8, and X9. In the absence of prior knowledge about these variables, we will treat them generically as distinct entities that may exhibit interdependencies or causal links. The variable names — X0 to X9 — suggest they could represent a range of phenomena, such as different measurements, experimental conditions, or characteristics within a particular dataset. The exploration will aim to uncover any underlying patterns, correlations, or causal structures among these variables, which could shed light on their interactions and help inform future research directions. Without specific insights into what each variable represents, we will focus on identifying statistical relationships and potential causal directions that may arise from the data. This systematic investigation will serve as a foundation for understanding how these variables interact, potentially leading to valuable conclusions and hypotheses for further study.

## Dataset Descriptions
The following is a preview of our dataset:

|    |   X0     |   X1     |   X2     |   X3     |   X4     |   X5     |   X6     |   X7     |    X8    |   X9     |
|----|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 0  |  0.190591| -0.021391| -0.170866|  0.070673| -0.323332|  0.456985| -0.514653| -0.337884| -1.126015|  1.715769 |
| 1  | -0.009655| -0.213175| -0.033270|  0.121998| -0.131856|  0.092601| -0.177431| -0.217623| -0.111022| -0.174257 |
| 2  |  0.018706|  0.020344|  0.012444|  0.046595|  0.124365|  0.058195| -0.183826| -0.351986|  0.125315| -0.412959 |
| 3  |  0.132656| -0.100911| -0.001887|  0.208837| -0.409481|  0.329989| -0.495770| -0.314155| -0.589905|  0.842987 |
| 4  | -0.137661| -0.011168| -0.102827| -0.150752|  0.286250| -0.072213|  0.000216| -0.245480|  0.275955| -0.246922 |

### Dataset Characteristics:
- Sample Size: 2000
- Features: 10
- Data Type: Continuous
- Data Quality: No missing values
- Statistical Properties: 
  - Linearity: Predominantly linear relationships
  - Gaussian Errors: Errors follow a Gaussian distribution
  - Homogeneity: The dataset is not heterogeneous

### Implications for Analysis:
1. The data is well-suited for linear modeling techniques.

## Discovery Procedure
### Step-by-Step Causal Discovery Procedure
1. **Data Preprocessing**:  
   - Ensured integrity by checking for missing values, which were absent.
   - Assessed basic statistical characteristics (sample size, number of features, types, linearity, and error distribution).

2. **Algorithm Selection**:  
   - Chose the **PC Algorithm** due to its efficiency in identifying causal relationships in large datasets and its ability to leverage predominantly linear relationships.
   - Other algorithms such as GES and NOTEARS were considered but not selected due to their specific requirements or resource intensiveness.

3. **Hyperparameter Values Proposal**:  
   - Below are the selected hyperparameters for the PC algorithm:
   ```json
   {
     "algorithm": "PC",
     "hyperparameters": {
       "alpha": {
         "value": 0.01,
         "explanation": "Lower significance level is chosen to reduce false positives."
       },
       "indep_test": {
         "value": "fisherz",
         "explanation": "Fisher's Z test is suitable for continuous data with Gaussian errors."
       },
       "depth": {
         "value": -1,
         "explanation": "Setting depth to -1 allows exploration of all conditional independencies."
       }
     }
   }
   ```

4. **Graph Tuning**:  
   - Further refined the causal structure using bootstrapping methods to assess stability and robustness of identified causal links.

## Results Summary
The following graphs were produced by our analysis:

| <center> True Graph | <center> Initial Graph | <center> Revised Graph |
|--|--|--|
| ![True Graph](/postprocess/test_data/20241018_020318_base_nodes10_samples2000/output_graph/True_Graph.jpg) | ![Initial Graph](/postprocess/test_data/20241018_020318_base_nodes10_samples2000/output_graph/Initial_Graph.jpg) | ![Revised Graph](/postprocess/test_data/20241018_020318_base_nodes10_samples2000/output_graph/Revised_Graph.jpg) |

### Analysis of Causal Relationships:
- **Key Findings**: 
  - X1 is a significant influencer, causing both X0 and X9.
  - X4 affects both X0 and X8, indicating its role as a critical link.
  - X5 influences X8 and X9, showing a broad impact.
  - X6 serves as a mediator affecting X7 and contributing to causal dynamics involving X8 and X9.

### Metrics Evaluation
Below is a comparison of metrics for the original and revised graphs:

<table>
    <tr>
        <td>
        <table>
    <tr>
        <th>Metric</th>
        <th>Original Graph</th>
        <th>Revised Graph</th>
    </tr>
    <tr>
        <td>SHD</td>
        <td>38.0</td>
        <td>36.0</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>0.4286</td>
        <td>0.5000</td>
    </tr>
    <tr>
        <td>Recall</td>
        <td>0.1667</td>
        <td>0.1667</td>
    </tr>
    <tr>
        <td>F1 Score</td>
        <td>0.2400</td>
        <td>0.2500</td>
    </tr> 
          </table>
    </td>
    <td>
            <img src="/postprocess/test_data/20241018_020318_base_nodes10_samples2000/output_graph/metrics.jpg" alt="Metric Graph" width="400"/>
    </td>
    </tr>
</table>


### Analysis of Metrics:
- **Structural Hamming Distance (SHD)**: A reduction from 38.0 to 36.0 indicates an improvement in the accuracy of causal discovery.
- **Precision**: Increased from 0.4286 to 0.5000, suggesting that the revised model has a higher proportion of true positive relationships.
- **Recall**: Remained constant at 0.1667, indicating that while the model's precision improved, its ability to identify all true causal relationships did not.
- **F1 Score**: A slight improvement from 0.2400 to 0.2500 reflects the balancing act between precision and recall.

Overall, the results indicate that the revised graph presents a more refined understanding of the causal relationships among variables, highlighting key influencers while maintaining a focus on the completeness of the model. Further research could build upon these insights to elucidate the underlying dynamics of the system represented by these variables.
