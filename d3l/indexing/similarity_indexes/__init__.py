from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Tuple

from d3l.indexing.feature_extraction.schema.qgram_transformer import QGramTransformer
from d3l.indexing.feature_extraction.values.distribution_transformer import (
    DistributionTransformer,
)

from d3l.indexing.feature_extraction.values.fd_transformer import FDTransformer
from d3l.indexing.feature_extraction.values.glove_embedding_transformer import GloveTransformer
from d3l.indexing.feature_extraction.values.token_transformer import TokenTransformer
from d3l.indexing.lsh.lsh_index import LSHIndex
from d3l.input_output.dataloaders import DataLoader
from d3l.utils.constants import STOPWORDS
from d3l.utils.functions import is_numeric


class SimilarityIndex(ABC):
    def __init__(self, dataloader: DataLoader, data_root: Optional[str] = None):
        """
        The constructor of the generic similarity index.
        A similarity index is just a wrapper over an LSH index
        that provides extra functionality for index creation and querying.
        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : Optional[str]
            A schema name if the data is being loaded from a database.
        """
        super().__init__()
        self._dataloader = dataloader
        self._data_root = data_root

    @property
    def dataloader(self) -> DataLoader:
        return self._dataloader

    @property
    def data_root(self) -> str:
        return self._data_root

    @abstractmethod
    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """
        pass

    @abstractmethod
    def query(self, query: Any, k: Optional[int] = None) -> Iterable[Tuple[str, float]]:
        """
        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : Any
            The query that can be a simple string for name similarity or a set of values for value-based similarities.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """
        pass


class NameIndex(SimilarityIndex):
    def __init__(
        self,
        dataloader: DataLoader,
        data_root: Optional[str] = None,
        transformer_qgram_size: int = 3,
        index_hash_size: int = 256,
        index_similarity_threshold: float = 0.5,
        index_fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        index_seed: int = 12345,
    ):
        """

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : str
            A schema name if the data is being loaded from a database.
        transformer_qgram_size : int
            The size of name qgrams to extract.
            Defaults to 3.
        index_hash_size : int
            The expected size of the input hashcodes.
        index_similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        index_fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        index_seed : int
            The random seed for the underlying hash generator.
        """
        super(NameIndex, self).__init__(dataloader=dataloader, data_root=data_root)

        self.transformer_qgram_size = transformer_qgram_size
        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed

        self.transformer = QGramTransformer(qgram_size=self.transformer_qgram_size)
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """

        lsh_index = LSHIndex(
            hash_size=self.index_hash_size,
            similarity_threshold=self.index_similarity_threshold,
            fp_fn_weights=self.index_fp_fn_weights,
            seed=self.index_seed,
            dimension=None,
        )

        for table in tqdm(self.dataloader.get_tables(self.data_root)):
            table_signature = self.transformer.transform(table)
            lsh_index.add(input_id=str(table), input_set=table_signature)
            column_data = self.dataloader.get_columns(table_name=table)

            column_signatures = [(c, self.transformer.transform(c)) for c in column_data]
            for c, signature in column_signatures:
                if len(signature) > 0:
                    lsh_index.add(input_id=str(table) + "." + str(c), input_set=signature)

        return lsh_index

    def query(self, query: str, k: Optional[int] = None) -> Iterable[Tuple[str, float]]:
        """

        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : str
            A string to search the underlying LSH index with.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """

        query_signature = self.transformer.transform(query)
        if len(query_signature) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query_signature, k=k, with_scores=True
        )


class FormatIndex(SimilarityIndex):
    def __init__(
        self,
        dataloader: DataLoader,
        data_root: Optional[str] = None,
        index_hash_size: int = 256,
        index_similarity_threshold: float = 0.5,
        index_fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        index_seed: int = 12345,
    ):
        """

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : Optional[str]
            A schema name if the data is being loaded from a database.
        index_hash_size : int
            The expected size of the input hashcodes.
        index_similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        index_fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        index_seed : int
            The random seed for the underlying hash generator.
        """
        super(FormatIndex, self).__init__(dataloader=dataloader, data_root=data_root)

        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed

        self.transformer = FDTransformer()
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """

        lsh_index = LSHIndex(
            hash_size=self.index_hash_size,
            similarity_threshold=self.index_similarity_threshold,
            fp_fn_weights=self.index_fp_fn_weights,
            seed=self.index_seed,
            dimension=None,
        )
        for table in tqdm(self.dataloader.get_tables(self.data_root)):
            table_data = self.dataloader.read_table(table_name=table)

            column_signatures = [
                (c, self.transformer.transform(table_data[c].tolist()))
                for c in table_data.columns
                if not is_numeric(table_data[c]) and table_data[c].count() > 0
            ]
            for c, signature in column_signatures:
                if len(signature) > 0:
                    lsh_index.add(input_id=str(table) + "." + str(c), input_set=signature)
        return lsh_index

    def query(
        self, query: Iterable[Any], k: Optional[int] = None
    ) -> Iterable[Tuple[str, float]]:
        """

        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : Iterable[Any]
            A collection of values representing the query set.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """
        if is_numeric(query):
            return []

        query_signature = self.transformer.transform(query)
        if len(query_signature) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query_signature, k=k, with_scores=True
        )


class ValueIndex(SimilarityIndex):
    def __init__(
        self,
        dataloader: DataLoader,
        data_root: Optional[str] = None,
        transformer_token_pattern: str = r"(?u)\b\w\w+\b",
        transformer_max_df: float = 0.5,
        transformer_stop_words: Iterable[str] = STOPWORDS,
        index_hash_size: int = 256,
        index_similarity_threshold: float = 0.5,
        index_fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        index_seed: int = 12345,
    ):
        """

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : Optional[str]
            A schema name if the data is being loaded from a database.
        transformer_token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        transformer_max_df : float
            Percentage of values the token can appear in before it is ignored.
        transformer_stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        index_hash_size : int
            The expected size of the input hashcodes.
        index_similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        index_fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        index_seed : int
            The random seed for the underlying hash generator.
        """
        super(ValueIndex, self).__init__(dataloader=dataloader, data_root=data_root)

        self.transformer_token_pattern = transformer_token_pattern
        self.transformer_max_df = transformer_max_df
        self.transformer_stop_words = transformer_stop_words
        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed

        self.transformer = TokenTransformer(
            token_pattern=self.transformer_token_pattern,
            max_df=self.transformer_max_df,
            stop_words=self.transformer_stop_words,
        )
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """

        lsh_index = LSHIndex(
            hash_size=self.index_hash_size,
            similarity_threshold=self.index_similarity_threshold,
            fp_fn_weights=self.index_fp_fn_weights,
            seed=self.index_seed,
            dimension=None,
        )

        for table in tqdm(self.dataloader.get_tables(self.data_root)):
            table_data = self.dataloader.read_table(table_name=table)

            column_signatures = [
                (c, self.transformer.transform(table_data[c].tolist()))
                for c in table_data.columns
                if not is_numeric(table_data[c]) and table_data[c].count() > 0
            ]
            for c, signature in column_signatures:
                if len(signature) > 0:
                    lsh_index.add(input_id=str(table) + "." + str(c), input_set=signature)

        return lsh_index

    def query(
        self, query: Iterable[Any], k: Optional[int] = None
    ) -> Iterable[Tuple[str, float]]:
        """

        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : Iterable[Any]
            A collection of values representing the query set.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """
        if is_numeric(query):
            return []

        query_signature = self.transformer.transform(query)
        if len(query_signature) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query_signature, k=k, with_scores=True
        )


class EmbeddingIndex(SimilarityIndex):
    def __init__(
        self,
        dataloader: DataLoader,
        data_root: Optional[str] = None,
        transformer_token_pattern: str = r"(?u)\b\w\w+\b",
        transformer_max_df: float = 0.5,
        transformer_stop_words: Iterable[str] = STOPWORDS,
        transformer_embedding_model_lang: str = "en",
        index_hash_size: int = 1024,
        index_similarity_threshold: float = 0.5,
        index_fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        index_seed: int = 12345,
        index_cache_dir: Optional[str] = None
    ):
        """

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : Optional[str]
            A schema name if the data is being loaded from a database.
        transformer_token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        transformer_max_df : float
            Percentage of values the token can appear in before it is ignored.
        transformer_stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        transformer_embedding_model_lang : str
            The embedding model language.
        index_hash_size : int
            The expected size of the input hashcodes.
        index_similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        index_fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        index_seed : int
            The random seed for the underlying hash generator.
        index_cache_dir : str
            A file system path for storing the embedding model.

        """
        super(EmbeddingIndex, self).__init__(dataloader=dataloader, data_root=data_root)

        self.transformer_token_pattern = transformer_token_pattern
        self.transformer_max_df = transformer_max_df
        self.transformer_stop_words = transformer_stop_words
        self.transformer_embedding_model_lang = transformer_embedding_model_lang
        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed

        self.index_cache_dir = index_cache_dir

        # self.transformer = FasttextTransformer(
        #     token_pattern=self.transformer_token_pattern,
        #     max_df=self.transformer_max_df,
        #     stop_words=self.transformer_stop_words,
        #     embedding_model_lang=self.transformer_embedding_model_lang,
        #     cache_dir=self.index_cache_dir
        # )

        self.transformer = GloveTransformer(
            token_pattern=self.transformer_token_pattern,
            max_df=self.transformer_max_df,
            stop_words=self.transformer_stop_words,
            cache_dir=self.index_cache_dir
        )
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """

        lsh_index = LSHIndex(
            hash_size=self.index_hash_size,
            dimension=self.transformer.get_embedding_dimension(),
            similarity_threshold=self.index_similarity_threshold,
            fp_fn_weights=self.index_fp_fn_weights,
            seed=self.index_seed,
        )

        for table in tqdm(self.dataloader.get_tables(self.data_root)):
            table_data = self.dataloader.read_table(table_name=table)

            column_signatures = [
                (c, self.transformer.transform(table_data[c].tolist()))
                for c in table_data.columns
                if not is_numeric(table_data[c]) and table_data[c].count() > 0
            ]
            for c, signature in column_signatures:
                if len(signature) > 0:
                    lsh_index.add(input_id=str(table) + "." + str(c), input_set=signature)

        return lsh_index

    def query(
        self, query: Iterable[Any], k: Optional[int] = None
    ) -> Iterable[Tuple[str, float]]:
        """

        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : Iterable[Any]
            A collection of values representing the query set.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """
        if is_numeric(query):
            return []

        query_signature = self.transformer.transform(query)
        if len(query_signature) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query_signature, k=k, with_scores=True
        )


class DistributionIndex(SimilarityIndex):
    def __init__(
        self,
        dataloader: DataLoader,
        data_root: Optional[str] = None,
        transformer_num_bins: int = 300,
        transformer_use_density: bool = True,
        index_hash_size: int = 1024,
        index_similarity_threshold: float = 0.5,
        index_fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        index_seed: int = 12345,
    ):
        """

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : Optional[str]
            A schema name if the data is being loaded from a database.
        transformer_num_bins : int
            Defines the dimension of the resulting distribution representation and the number of equal-width bins.
        transformer_use_density : bool
            If True the distribution representation defines a probability density function
            rather than just a count histogram.
        index_hash_size : int
            The expected size of the input hashcodes.
        index_similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        index_fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        index_seed : int
            The random seed for the underlying hash generator.
        """
        super(DistributionIndex, self).__init__(
            dataloader=dataloader, data_root=data_root
        )

        self.transformer_num_bins = transformer_num_bins
        self.transformer_use_density = transformer_use_density
        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed

        self.transformer = DistributionTransformer(
            num_bins=self.transformer_num_bins, use_density=self.transformer_use_density
        )
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """

        lsh_index = LSHIndex(
            hash_size=self.index_hash_size,
            dimension=self.transformer_num_bins,
            similarity_threshold=self.index_similarity_threshold,
            fp_fn_weights=self.index_fp_fn_weights,
            seed=self.index_seed,
        )

        for table in tqdm(self.dataloader.get_tables(self.data_root)):
            table_data = self.dataloader.read_table(table_name=table)

            column_signatures = [
                (c, self.transformer.transform(table_data[c].tolist()))
                for c in table_data.columns
                if is_numeric(table_data[c]) and table_data[c].count() > 0
            ]
            for c, signature in column_signatures:
                if len(signature) > 0:
                    lsh_index.add(input_id=str(table) + "." + str(c), input_set=signature)

        return lsh_index

    def query(
        self, query: Iterable[Any], k: Optional[int] = None
    ) -> Iterable[Tuple[str, float]]:
        """

        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : Iterable[Any]
            A collection of values representing the query set.
            These should be numeric.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """
        if not is_numeric(query):
            return []

        query_signature = self.transformer.transform(query)
        if len(query_signature) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query_signature, k=k, with_scores=True
        )
