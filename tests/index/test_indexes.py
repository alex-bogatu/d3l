"""
Copyright (C) 2021 Alex.
This file is part of the D3L Data Discovery Framework.

Notes
-----
This module defines indexing tests for D3L.
Note that embedding indexing is not included in the tests as it requires external word2vec-like models.
"""

import pytest

import os
from d3l.input_output.dataloaders import CSVDataLoader
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


def test_name_index_creation(name_index):
    assert len(name_index.lsh_index._keys) > 0


def test_format_index_creation(format_index):
    assert len(format_index.lsh_index._keys) > 0


def test_value_index_creation(value_index):
    assert len(value_index.lsh_index._keys) > 0


def test_distribution_index_creation(distribution_index):
    assert len(distribution_index.lsh_index._keys) > 0
