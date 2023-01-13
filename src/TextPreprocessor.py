import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    @classmethod
    def strip_html(cls, text):
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    @classmethod
    def remove_between_square_brackets(cls, text):
        return re.sub('\[[^]]*\]', '', text)

    @classmethod
    def denoise_text(cls, text):
        text = cls.strip_html(text)
        text = cls.remove_between_square_brackets(text)
        return text

    @classmethod
    def remove_special_characters(cls, text):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern,'',text)
        return text

    @classmethod
    def remove_stopwords_and_lower(cls, text):
        stop_words = stopwords.words('english')
        text_tokens = word_tokenize(text)
        text = [word.lower() for word in text_tokens if word.lower() not in stop_words]
        return str(' '.join(text))

    @classmethod
    def transform_text(cls, text):
        text = cls.denoise_text(text)
        text = cls.remove_special_characters(text)
        text = cls.remove_stopwords_and_lower(text)

        return text

    @classmethod
    def transform(cls, documents):
        return [cls.transform_text(document) for document in documents]
