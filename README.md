# Customer Review Prompt Engineering

A Streamlit dashboard for designing, testing, comparing, and evaluating prompt versions for Amazon Fine Food customer reviews.

The project supports two run modes:

- **Local free mode**: works without API credits and gives rule-based demo responses.
- **OpenAI API mode**: sends prompts to an OpenAI model for real LLM responses.

## What This Project Does

- Loads and cleans customer review data.
- Converts review ratings into reference sentiment labels.
- Provides prompt versions V1, V2, and V3 for each task.
- Lets you compare prompt versions side by side.
- Shows dataset metrics and rating distribution.
- Provides rubric templates for qualitative prompt evaluation.
- Supports automated classification evaluation with accuracy, confusion matrix, and classification report.

## Prompt Engineering Workflow

```text
Design -> Test -> Evaluate -> Identify Weaknesses -> Improve -> Compare
```

## Tasks Covered

| Task | Purpose |
| --- | --- |
| Data Analysis | Find rating patterns, review themes, risks, and business insights. |
| Summarization | Summarize positive points, negative points, concerns, and overall opinion. |
| Classification | Classify reviews as Positive, Neutral, or Negative. |
| Content Generation | Generate product improvement and customer experience recommendations. |

## Prompt Versions

Each task has three prompt versions in:

```text
src/customer_review_prompt_engineering/prompts/prompt_versions.yaml
```

| Version | Description |
| --- | --- |
| V1 | Basic baseline prompt with minimal instruction. |
| V2 | Improved prompt with clearer context, sections, and constraints. |
| V3 | Optimized prompt with role prompting, explicit rules, structured output, and evaluation-friendly formatting. |

## Project Structure

```text
.
|-- app.py
|-- data/
|   |-- raw/
|   |   |-- .gitkeep
|   |   `-- Reviews.csv
|   `-- processed/
|       `-- .gitkeep
|-- experiments/
|   |-- process_dataset.py
|   |-- run_classification_eval.py
|   `-- run_rubric_template.py
|-- reports/
|   `-- .gitkeep
|-- src/
|   `-- customer_review_prompt_engineering/
|       |-- app.py
|       |-- config.py
|       |-- data.py
|       |-- evaluation.py
|       |-- llm.py
|       |-- prompts.py
|       |-- runner.py
|       `-- prompts/
|           `-- prompt_versions.yaml
|-- tests/
|   |-- conftest.py
|   `-- test_data.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the Amazon Fine Food Reviews CSV here:

```text
data/raw/Reviews.csv
```

The loader expects these columns:

- `Score`
- `Summary`
- `Text`

It creates:

- a cleaned `review` field from summary and review text
- a `reference_label` from rating
- an optional `review_date` when the dataset has a Unix `Time` column

Rating labels:

| Score | Label |
| --- | --- |
| 1-2 | Negative |
| 3 | Neutral |
| 4-5 | Positive |

Create a processed dataset:

```bash
python -m experiments.process_dataset
```

Output:

```text
data/processed/reviews_cleaned.csv
```

## Run The Dashboard

```bash
streamlit run app.py
```

In the sidebar, choose:

- task
- prompt version
- dataset rows or custom text
- Local free mode or OpenAI API mode

## Local Free Mode

Local free mode does not call OpenAI and does not use API credits.

Use this mode when:

- you want to test the dashboard
- your API credits are finished
- you want to compare V1, V2, and V3 without paid API calls

Local mode gives simple rule-based responses. It is useful for demonstration, but it is not a real LLM response.

## OpenAI API Mode

OpenAI API mode sends the selected prompt and review text to the configured model.

Create a `.env` file in the project root only if you want to use OpenAI API mode:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
DATA_PATH=data/raw/Reviews.csv
```

If API credits are exhausted, the dashboard will show a quota message. Switch back to **Local free mode** to continue using the app without API calls.

## Custom Text Input

When you choose **Custom text**, the dashboard uses the text you type as the prompt input for the current session.

Custom text is not saved to the dataset.

In Local free mode, classification works best with dataset rows because ratings are available. Custom text without ratings may return an unknown or low-confidence classification.

## Evaluation

Run tests:

```bash
python -m pytest
```

Run classification evaluation in local free mode:

```bash
python -m experiments.run_classification_eval --version v3 --sample-size 25 --mode local
```

Output:

```text
reports/classification_metrics.json
```

To evaluate with live OpenAI predictions instead, use:

```bash
python -m experiments.run_classification_eval --version v3 --sample-size 25 --mode openai
```

Create a blank rubric sheet:

```bash
python -m experiments.run_rubric_template --task summarization --version v3
```

Rubric criteria:

- relevance
- completeness
- consistency
- instruction following
- output quality

## Useful Commands

```bash
streamlit run app.py
python -m pytest
python -m experiments.process_dataset
python -m experiments.run_classification_eval --version v3 --sample-size 25 --mode local
python -m experiments.run_rubric_template --task summarization --version v3
```

## Notes

- Local free mode is for testing and demonstration.
- OpenAI API mode requires valid API credits.
- Generated reports and processed CSV files are ignored by Git.
- Raw dataset CSV files are not committed to keep the repository lightweight.
