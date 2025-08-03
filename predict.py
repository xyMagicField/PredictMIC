import os
import h2o
import numpy as np
import pandas as pd
import argparse
from feature import file_prediction_features


def main():
    parser = argparse.ArgumentParser(description='Use H2O model for prediction and process results')
    parser.add_argument('-m', '--model-path', required=True, help='Path to the H2O model')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-o', '--output', default="output_info", help='Path to save the predict CSV file')
    parser.add_argument('-s', '--smiles_column', default="Smiles", help='Name of the smiles column')

    args = parser.parse_args()
    feature_path = os.path.join(args.output, "feature")
    os.makedirs(feature_path, exist_ok=True)
    processed_file = file_prediction_features(args.input,
                                              feature_path,
                                              smiles_column=args.smiles_column)
    print(f"Processed features saved to: {processed_file}")

    h2o.init()

    try:
        print(f"Loading model: {args.model_path}")
        model = h2o.load_model(args.model_path)

        print(f"Reading input data: {processed_file}")
        new_data_pd = pd.read_csv(processed_file)
        new_data_h2o = h2o.H2OFrame(new_data_pd)

        print("Performing prediction...")
        predictions = model.predict(new_data_h2o)

        predictions_df = predictions.as_data_frame()

        predictions_df['predict_MIC'] = np.exp(predictions_df['predict'])

        if 'predict' in predictions_df.columns:
            predictions_df = predictions_df.drop('predict', axis=1)
            print("predict column removed")

        original_df = pd.read_csv(args.input)
        smiles_series = original_df[args.smiles_column]

        predictions_df.insert(0, args.smiles_column, smiles_series)
        print(f"{args.smiles_column} column added to the first column of results")

        print("First few rows of prediction results:")
        print(predictions_df.head())

        predict_file = os.path.join(args.output, "predict")
        os.makedirs(predict_file, exist_ok=True)
        output_path = os.path.join(predict_file, "predict.csv")
        print(f"Saving results to: {output_path}")
        predictions_df.to_csv(output_path, index=False)

    finally:
        print("Shutting down H2O cluster")
        h2o.cluster().shutdown()


if __name__ == "__main__":
    main()
