import yake
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

from src.TextPreprocessor import TextPreprocessor
from src.constants import ALL_MINI_LM_L6_V2_PATH, BATCH_SIZE, CLASSIFIER_PATH


class SenseExtractor:
    kw_model = yake.KeywordExtractor(
        lan='en',
        top=10,
        n=3,
    )
    vectoriser_moder = SentenceTransformer(ALL_MINI_LM_L6_V2_PATH)
    classifier_model = XGBClassifier()
    classifier_model.load_model(CLASSIFIER_PATH)
    SENTIMENTS = ['negative', 'positive']
    TOP_K_KEYWORDS = 10

    @classmethod
    def extract_keywords(cls, text):
        yake_keywords = cls.kw_model.extract_keywords(text)

        return [keyword for keyword, score in yake_keywords]

    @classmethod
    def extract_sentiment(cls, vector):
        prediction = cls.classifier_model.predict(vector)

        return cls.SENTIMENTS[prediction[0]]

    @classmethod
    def analyse_text(cls, text):
        preprocessed_text = TextPreprocessor.transform_text(text)
        text_vector = cls.vectoriser_moder. \
            encode([preprocessed_text], batch_size=BATCH_SIZE, convert_to_numpy=True)

        sentiment = cls.extract_sentiment(text_vector)
        keywords = cls.extract_keywords(preprocessed_text)

        return sentiment, keywords
