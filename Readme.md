# PredictMIC
This tool is developed based on the framework of H2O AutoML, and is used to automatically train the MIC (minimum inhibitory concentration) prediction model of compounds. By inputting a CSV file containing the SMILES structure and the corresponding MIC value of the compound, the tool can automatically complete the processes of feature extraction, data set division, multi-model training, performance evaluation and model preservation, and is suitable for quantitative prediction scenarios in drug screening or microbiology research.

![Process](images/Process.svg)

## Getting Started

### Install

```bash
git clone https://github.com/xyMagicField/PredictMIC.git
cd PredictMIC
```
### Environment

```bash
conda env create -f environment.yml
conda activate PredictMIC
```

### Usage Train

- Data Preparation: The input CSV file should contain two columns: SMILES and MIC. The SMILES column contains the SMILES structure of the compound, and the MIC column contains the MIC value of the compound.


- Command:
    ```bash
    python train.py -i input_file.csv [other parameters]
    ```

- Parameter Explanation:

    | Parameter | Short | Description | Default |
    |----------------|------------|------------------|--------------|
    | --input | -i | Path to input CSV file | Required |
    | --output | -o | Path to save trained models | "output_info" |
    | --smiles_column | -s | Name of the SMILES column | "Smiles" |
    | --mic_column | -m | Name of the MIC column | "MIC" |
    | --max_models | -x | Maximum number of models to train | 10 |
    | --save_models | -v | Number of top models to save | 5 |
    | --seed | -r | Random seed value | 1 |

-  Example:
    ```bash
    python train.py -i data.csv -o model_output -x 20 -v 10 -r 42
    ```

### Usage Predict

- Data Preparation: The input CSV file should contain SMILES column. The SMILES column contains the SMILES structure of the compound.


- Command:
    ```bash
    python predict.py -i input_file.csv [other parameters]
    ```
  
- Parameter Explanation:

    | Parameter | Short | Description | Default |
    |--------------|------------|----------------|--------------|
    | --input | -i | Path to input CSV file | Required |
    | --output | -o | Path to save the predict CSV file | "output_info" |
    | --smiles_column | -s | Name of the SMILES column | "Smiles" |
    | --model-path | -m | Path to the H2O model | Required |

-  Example:
    ```bash
    python predict.py -i data.csv -m './models/model_1/GBM_5_AutoML_1'
    ```

## Issues

If you come across any bugs, have feature requests, or require assistance, please file an issue on our GitHub Issues page. When submitting an issue, kindly:

1. Check whether a similar issue already exists
2. Provide a clear description of the problem
3. Add steps to reproduce the issue if applicable
4. Specify details of your environment (such as OS, Python version, etc.)
5. Include any relevant error messages or screenshots

We welcome contributions and feedback from the community.
