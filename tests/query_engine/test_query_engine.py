"""
Copyright (C) 2021 Alex.
This file is part of the D3L Data Discovery Framework.

Notes
-----
This module defines querying tests for D3L.
"""

import pytest

import os
import numpy as np
from d3l.input_output.dataloaders import CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.indexing.similarity_indexes import (
    NameIndex,
    FormatIndex,
    ValueIndex,
    DistributionIndex,
)


@pytest.fixture
def csv_data_loader(root_path="./resources/data/", **csv_kwargs):
    """
    Return a new CSVDataLoader instance.
    """
    if not os.path.isdir(root_path):
        raise FileNotFoundError("No such local data root: <{}> !".format(root_path))

    return CSVDataLoader(root_path=root_path, **csv_kwargs)


@pytest.fixture
def name_index(csv_data_loader):
    return NameIndex(dataloader=csv_data_loader)


@pytest.fixture
def format_index(csv_data_loader):
    return FormatIndex(dataloader=csv_data_loader)


@pytest.fixture
def value_index(csv_data_loader):
    return ValueIndex(dataloader=csv_data_loader)


@pytest.fixture
def distribution_index(csv_data_loader):
    return DistributionIndex(dataloader=csv_data_loader)


@pytest.fixture
def query_engine(name_index, format_index, value_index, distribution_index):
    return QueryEngine(name_index, format_index, value_index, distribution_index)


def test_column_query(query_engine, csv_data_loader):
    column_results = query_engine.column_query(
        column=csv_data_loader.read_table(table_name="tableA", table_columns=["title"])[
            "title"
        ],
        aggregator=np.mean,
        k=2,
    )
    assert "tableB.title" in [t[0] for t in column_results]


def test_table_query(query_engine, csv_data_loader):
    table_results = query_engine.table_query(
        table=csv_data_loader.read_table(table_name="tableA"),
        aggregator=np.mean,
        k=2,
        verbose=False,
    )
    assert table_results[0][0] == "tableA"
