import numpy as np
from d3l.indexing.similarity_indexes import (
    NameIndex,
    FormatIndex,
    ValueIndex,
    EmbeddingIndex,
    DistributionIndex,
)
from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object

if __name__ == "__main__":
    csvdl = CSVDataLoader(
        root_path="/Users/alexteodor/Downloads/evaluation/groundTruth/",
        header=0,
        sep=",",
    )
    name_index = unpickle_python_object("./name.lsh")
    for ht in name_index.lsh_index._hashtables:
        for _, bucket in ht.items():
            if len(bucket) > 1:
                print(bucket)

    name_index = NameIndex(dataloader=csvdl)
    pickle_python_object(name_index, "./name.lsh")
    print("Name: SAVED!")

    format_index = FormatIndex(dataloader=csvdl)
    pickle_python_object(format_index, "./format.lsh")
    print("Format: SAVED!")

    value_index = ValueIndex(dataloader=csvdl)
    pickle_python_object(value_index, "./value.lsh")
    print("Value: SAVED!")

    embedding_index = EmbeddingIndex(dataloader=csvdl)
    pickle_python_object(embedding_index, "./embedding.lsh")
    print("Embedding: SAVED!")

    distribution_index = DistributionIndex(dataloader=csvdl)
    pickle_python_object(distribution_index, "./distribution.lsh")
    print("Distribution: SAVED!")

    name_index = unpickle_python_object("./name.lsh")
    format_index = unpickle_python_object("./format.lsh")
    value_index = unpickle_python_object("./value.lsh")
    embedding_index = unpickle_python_object("./embedding.lsh")
    distribution_index = unpickle_python_object("./distribution.lsh")

    qe = QueryEngine(
        name_index, format_index, value_index, embedding_index, distribution_index
    )

    results = qe.table_query(
        table=csvdl.read_table(table_name="TEgbBqq_Food_Hygiene_Data"),
        aggregator=lambda scores: np.mean(scores),
        k=10,
        verbose=False,
    )
    print(results)
