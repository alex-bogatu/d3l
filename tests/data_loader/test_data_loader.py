"""
Copyright (C) 2021 Alex.
This file is part of the D3L Data Discovery Framework.

Notes
-----
This module defines data loading tests for D3L.
"""

import os

import pytest

from d3l.input_output.dataloaders import CSVDataLoader


@pytest.fixture
def csv_data_loader(root_path="./resources/data/", **csv_kwargs):
    """
    Return a new CSVDataLoader instance.
    """
    if not os.path.isdir(root_path):
        raise FileNotFoundError("No such local data root: <{}> !".format(root_path))

    return CSVDataLoader(root_path=root_path, **csv_kwargs)


def test_data_loading(csv_data_loader):
    tables = csv_data_loader.get_tables()

    assert len(tables) > 0
    assert len(csv_data_loader.get_columns(table_name=tables[0])) > 0


def test_table_loading(csv_data_loader):
    tables = csv_data_loader.get_tables()
    table_data = csv_data_loader.read_table(table_name=tables[0])

    assert table_data.shape[0] > 0
