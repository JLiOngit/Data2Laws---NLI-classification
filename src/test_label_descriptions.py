import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from label_descriptions import LABEL_DESCRIPTIONS, IMPROVED_GROUP_DESCRIPTIONS
from data_utils import load_data

ROOT = os.path.join(os.path.dirname(__file__), '..')
TRAIN_PATH = os.path.join(ROOT, "code_review_comments", "crc_train.jsonl")
TEST_PATH = os.path.join(ROOT, "code_review_comments", "crc_test.jsonl")
DESCRIPTION_MAP = {"label": LABEL_DESCRIPTIONS,
                   "group": IMPROVED_GROUP_DESCRIPTIONS}

@pytest.mark.parametrize("label_name", DESCRIPTION_MAP.keys())
def test_label_consistency(label_name):
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    dataset_labels = set(full_df[label_name].unique())
    description_df = DESCRIPTION_MAP[label_name]
    description_labels = set(description_df[label_name].unique())
    missing_descriptions = dataset_labels - description_labels
    assert missing_descriptions == set(), f"Labels in dataset but missing from descriptions: {missing_descriptions}"
    unused_descriptions = description_labels - dataset_labels
    assert unused_descriptions == set(), f"Labels in descriptions but never seen in dataset: {unused_descriptions}"
