import os
from typing import Iterable, Set

import numpy as np
from fasttext import load_model
from fasttext.util import download_model
from sklearn.feature_extraction.text import TfidfVectorizer

from d3l.utils.constants import STOPWORDS
from d3l.utils.functions import shingles


class EmbeddingTransformer:
    def __init__(
        self,
        token_pattern: str = r"(?u)\b\w\w+\b",
        max_df: float = 0.5,
        stop_words: Iterable[str] = STOPWORDS,
        embedding_model_lang="en",
    ):
        """
        Instantiate a new embedding-based transformer
        Parameters
        ----------
        token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        max_df : float
            Percentage of values the token can appear in before it is ignored.
        stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        embedding_model_lang : str
            The embedding model language.
        """

        self._token_pattern = token_pattern
        self._max_df = max_df
        self._stop_words = stop_words
        self._embedding_model_lang = embedding_model_lang

        self._embedding_model = EmbeddingTransformer.get_embedding_model(
            self._embedding_model_lang, overwrite=False
        )

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != '_embedding_model'}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._embedding_model = EmbeddingTransformer.get_embedding_model(
            self._embedding_model_lang, overwrite=False
        )

    @staticmethod
    def get_embedding_model(model_lang: str = "en", overwrite: bool = False):
        """
        Download, if not exists, and load the pretrained FastText embedding model in the working directory.
        Note that the default gzipped English Common Crawl FastText model has 4.2 GB
        and its unzipped version has 6.7 GB.
        Parameters
        ----------
        model_lang : str
            The model language.
        overwrite : bool
            If True overwrites the model if exists.
        Returns
        -------

        """
        if_exists = "strict" if not overwrite else "overwrite"
        file_name = "cc.%s.300.bin" % model_lang

        download_model(lang_id=model_lang, if_exists=if_exists)

        if os.path.isfile("./{}.gz".format(file_name)):
            os.remove("./{}.gz".format(file_name))
        embedding_model = load_model("./{}".format(file_name))
        return embedding_model

    def get_embedding_dimension(self) -> int:
        """
        Retrieve the embedding dimensions of the underlying model.
        Returns
        -------
        int
            The dimensions of each embedding
        """
        return self._embedding_model.get_dimension()

    def get_vector(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding of the given word.
        If the word is out of vocabulary a zero vector is returned.
        Parameters
        ----------
        word : str
            The word to retrieve the vector for.

        Returns
        -------
        np.ndarray
            A vector of float numbers.
        """
        try:
            vector = self._embedding_model.get_word_vector(str(word).strip().lower())
        except KeyError:
            vector = np.zeros(self.get_embedding_dimension())
        return vector

    def get_tokens(self, input_values: Iterable[str]) -> Set[str]:
        """
        Extract the most representative tokens of each value and return the token set.
        Here, the most representative tokens are the ones with the lowest TF/IDF scores -
        tokens that describe what the values are about.
        Parameters
        ----------
        input_values : Iterable[str]
            The collection of values to extract tokens from.

        Returns
        -------
        Set[str]
            A set of representative tokens
        """

        if len(input_values) < 1:
            return set()

        try:
            vectorizer = TfidfVectorizer(
                decode_error="ignore",
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                stop_words=self._stop_words,
                token_pattern=self._token_pattern,
                max_df=self._max_df,
                use_idf=True,
            )
            vectorizer.fit_transform(input_values)
        except ValueError:
            return set()

        weight_map = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tokenset = set()
        tokenizer = vectorizer.build_tokenizer()
        for value in input_values:
            value = value.lower().replace("\n", " ").strip()
            for shingle in shingles(value):
                tokens = [t for t in tokenizer(shingle)]

                if len(tokens) < 1:
                    continue

                token_weights = [weight_map.get(t, 0.0) for t in tokens]
                min_tok_id = np.argmin(token_weights)
                tokenset.add(tokens[min_tok_id])

        return tokenset

    def transform(self, input_values: Iterable[str]) -> np.ndarray:
        """
         Extract the embeddings of the most representative tokens of each value and return their **mean** embedding.
         Here, the most representative tokens are the ones with the lowest TF/IDF scores -
         tokens that describe what the values are about.
         Given that the underlying embedding model is a n-gram based one,
         the number of out-of-vocabulary tokens should be relatively small or zero.
         Parameters
         ----------
        input_values : Iterable[str]
             The collection of values to extract tokens from.

         Returns
         -------
         np.ndarray
             A Numpy vector representing the mean of all token embeddings.
        """

        embeddings = [self.get_vector(token) for token in self.get_tokens(input_values)]
        if len(embeddings) == 0:
            return np.empty(0)
        return np.mean(np.array(embeddings), axis=0)
