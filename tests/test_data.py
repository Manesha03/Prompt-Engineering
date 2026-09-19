import pandas as pd

from customer_review_prompt_engineering.data import dataset_insights, rating_to_label


def test_rating_to_label():
    assert rating_to_label(1) == "Negative"
    assert rating_to_label(2) == "Negative"
    assert rating_to_label(3) == "Neutral"
    assert rating_to_label(4) == "Positive"
    assert rating_to_label(5) == "Positive"


def test_dataset_insights():
    df = pd.DataFrame({"Score": [1, 3, 5], "reference_label": ["Negative", "Neutral", "Positive"]})
    result = dataset_insights(df)
    assert result["review_count"] == 3
    assert result["average_score"] == 3.0
    assert result["score_distribution"] == {1: 1, 3: 1, 5: 1}

