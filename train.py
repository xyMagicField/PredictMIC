import h2o
from h2o.automl import H2OAutoML
import argparse
import os

from feature import file_train_features


def main():
    parser = argparse.ArgumentParser(description='Model training using H2O AutoML')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-o', '--output', default="output_info", help='Path to save the trained models')
    parser.add_argument('-s', '--smiles_column', default="Smiles", help='Name of the smiles column')
    parser.add_argument('-m', '--mic_column', default="MIC", help='Name of the mic column')
    parser.add_argument('-x', '--max_models', type=int, default=10, help='Maximum number of models to train')
    parser.add_argument('-v', '--save_models', type=int, default=5, help='Number of top models to save')
    parser.add_argument('-r', '--seed', type=int, default=1, help='Random seed value')
    args = parser.parse_args()

    feature_path = os.path.join(args.output, "feature")
    os.makedirs(feature_path, exist_ok=True)
    processed_file = file_train_features(args.input,
                                         feature_path,
                                         smiles_column=args.smiles_column,
                                         mic_column=args.mic_column)
    print(f"Processed features saved to: {processed_file}")

    h2o.init()

    data = h2o.import_file(processed_file)

    data[args.mic_column] = data[args.mic_column].asnumeric()

    train, test = data.split_frame(ratios=[0.8], seed=args.seed)
    print(f"Number of training samples: {train.nrows}, Number of test samples: {test.nrows}")

    response = args.mic_column
    features = data.columns
    features.remove(response)

    aml = H2OAutoML(
        max_models=args.max_models,
        seed=args.seed,
        project_name="predict_mic_project"
    )

    aml.train(x=features, y=response, training_frame=train)

    lb = aml.leaderboard
    print("\nLeaderboard of all trained models:")
    lb.show()

    print(f"\nActual number of models trained: {lb.nrows}")

    all_model_ids = lb['model_id'].as_data_frame()['model_id'].tolist()
    print("\nList of trained models:")
    for i, model_id in enumerate(all_model_ids, 1):
        print(f"{i}. {model_id}")

    save_count = min(args.save_models, lb.nrows)
    top_model_ids = lb[0:save_count, 'model_id'].as_data_frame()['model_id'].tolist()

    for i, model_id in enumerate(top_model_ids):
        model_path = os.path.join(args.output, f"model_{i+1}")
        h2o.save_model(h2o.get_model(model_id), model_path)
        print(f"Saved model {i+1} to {model_path}")

    best_model = aml.leader
    predictions = best_model.predict(test).as_data_frame()
    print("\nPrediction results of the best model on the test set (first 5 rows):")
    print(predictions.head())

    h2o.cluster().shutdown()

if __name__ == "__main__":
    main()
