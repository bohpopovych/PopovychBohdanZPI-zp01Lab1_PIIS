import argparse
import time
import logging
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

from src.TextPreprocessor import TextPreprocessor
from src.constants import ALL_MINI_LM_L6_V2_PATH, BATCH_SIZE, CLASSIFIER_PATH

logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s %(message)s', filemode='w')


def train_vectoriser(train_data, test_data):
    transformer = SentenceTransformer(ALL_MINI_LM_L6_V2_PATH)
    tv_train_reviews = transformer. \
        encode(train_data, batch_size=BATCH_SIZE, convert_to_numpy=True)
    tv_test_reviews = transformer. \
        encode(test_data, batch_size=BATCH_SIZE, convert_to_numpy=True)

    return transformer, tv_train_reviews, tv_test_reviews


def train_classifier(train_X, train_y, test_X, test_y):
    xgb_model = XGBClassifier(
        learning_rate=0.1,
        max_depth=8,
        n_estimators=300,
        seed=42,
    )

    xgb_model.fit(train_X, train_y)

    train_predictions = xgb_model.predict(train_X)
    test_predictions = xgb_model.predict(test_X)

    logging.info(f'Train accuracy: {accuracy_score(train_y, train_predictions)}')

    return xgb_model, accuracy_score(test_y, test_predictions)


def prepare_data(df):
    df['class_index'] = df['class_index'] - 1
    df['review'] = df['review_title'] + df['review_text']
    df = df.dropna()
    X = TextPreprocessor.transform(df['review'].tolist())
    y = df['class_index'].tolist()

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.csv')
    parser.add_argument('--test', default='./data/test.csv')
    parser.add_argument('--train_count', default=100000, type=int)
    parser.add_argument('--test_count', default=20000, type=int)
    args = parser.parse_args()

    try:
        logging.info('Start data preparation')
        start_time = time.time()
        train_df = pd.read_csv(args.train)
        train_df = train_df.groupby('class_index').sample(args.train_count)
        train_X, train_y = prepare_data(train_df)
        logging.info('Prepared train data')

        test_df = pd.read_csv(args.test)
        test_df = test_df.groupby('class_index').sample(args.test_count)
        test_X, test_y = prepare_data(test_df)
        logging.info('Prepared test data')
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f'Data preparation of {len(test_df) + len(train_df)} documents took: {elapsed_time} seconds')
    except Exception as e:
        logging.error(f'Exception during data preparation: {e}')

    try:
        logging.info('Start training vectoriser model')
        start_time = time.time()
        vectoriser_model, train_X_vec, test_X_vec = train_vectoriser(train_X, test_X)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f'Vectoriser training took: {elapsed_time} seconds')
    except Exception as e:
        logging.error(f'Exception during vectoriser model training: {e}')

    try:
        logging.info('Start training classifier model')
        start_time = time.time()
        classifier_model, classifier_accuracy = train_classifier(
            train_X_vec,
            train_y,
            test_X_vec,
            test_y,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f'Classifier training took: {elapsed_time} seconds')
        logging.info(f'Test accuracy: {classifier_accuracy}')
        classifier_model.save_model(CLASSIFIER_PATH)
        logging.info('Classifier model saved successfully')

    except Exception as e:
        logging.error(f'Exception during clasifier model training: {e}')
