import argparse
import os
from sklearn.neighbors import KNeighborsRegressor
import joblib

import argparse
import os
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    parser.add_argument('--nneighbors', type=int, default=5)
    parser.add_argument('--weights', type=str, default='uniform')
    parser.add_argument('--algorithm', type=str, default='auto')
    parser.add_argument('--leaf_size', type=int, default=30)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--metric', type=str, default='minkowski')
    parser.add_argument('--metricparams', type=dict, default=None)
    parser.add_argument('--njobs', type=int, default=None)


    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    
    parser.add_argument('--train', type=str, 
                        default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    """ # Local taining manually
    parser.add_argument('--output-data-dir', type=str,
                        default=os.getcwd())
    parser.add_argument('--model-dir', type=str,
                        default=os.getcwd())
    parser.add_argument('--train', type=str,
                        default=os.getcwd()) """

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file)
                   for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

                    
    raw_data = [pd.read_csv(file, engine="python")
                for file in input_files]
    train_data = pd.concat(raw_data)
    
    organized_columns = ['daily_container_quanity', 'fbx_diff', 'd_quote_search_amount', 'conversion_rate', 'peak_season']
    train_data = train_data[organized_columns]
    train_data = train_data[(train_data["d_quote_search_amount"] != 0) & (train_data["daily_container_quanity"] != 0)]

    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]

    # Now use scikit-learn's KNeighborsRegressor to train the model.
    clf = KNeighborsRegressor()
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
