# Customer Review Prompt Engineering

Professional Prompt Engineering project for Customer Review Analysis using Python, Streamlit, an LLM API, and the Amazon Fine Food Reviews dataset.

The project demonstrates the lifecycle:

Design -> Test -> Evaluate -> Identify weaknesses -> Improve -> Compare results

## Tasks Covered

- Data Analysis: Finds rating patterns, themes, risks, and business insights.
- Summarization: Produces positive points, negative points, concerns, and overall opinion.
- Classification: Classifies reviews as Positive, Neutral, or Negative.
- Content Generation: Produces product improvement and customer experience recommendations.

Each task has three prompt versions in `src/customer_review_prompt_engineering/prompts/prompt_versions.yaml`:

- V1: Basic prompt.
- V2: Improved prompt with clearer context, instructions, and constraints.
- V3: Optimized prompt with role prompting, structured output, rules, and few-shot examples where useful.

## Project Structure

```text
.
├── app.py
├── data/
│   ├── raw/Reviews.csv
│   └── processed/
├── experiments/
│   ├── run_classification_eval.py
│   └── run_rubric_template.py
├── reports/
├── src/customer_review_prompt_engineering/
│   ├── app.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── llm.py
│   ├── prompts.py
│   ├── runner.py
│   └── prompts/prompt_versions.yaml
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`:

```bash
copy .env.example .env
```

Then set:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
DATA_PATH=data/raw/Reviews.csv
```

## Dataset

The app expects the Amazon Fine Food Reviews CSV at:

```text
data/raw/Reviews.csv
```

The loader uses `Score`, `Summary`, and `Text`. It drops rows without scores or review text, fills missing summaries, combines summary and text into a `review` field, converts Unix timestamps into `review_date`, and creates rating-derived reference labels:

- 1-2 = Negative
- 3 = Neutral
- 4-5 = Positive

Create a processed CSV with:

```bash
python -m experiments.process_dataset
```

Output is written to:

```text
data/processed/reviews_cleaned.csv
```

## Run the Dashboard

```bash
streamlit run app.py
```

In the dashboard you can:

- Select a task.
- Select V1, V2, or V3.
- Use dataset reviews or paste custom review text.
- Run a prompt.
- Compare all prompt versions side by side.
- View rubric templates for qualitative tasks.

## Run Classification Evaluation

This calculates actual metrics using rating-derived reference labels and LLM predictions:

```bash
python -m experiments.run_classification_eval --version v3 --sample-size 25
```

Output is written to:

```text
reports/classification_metrics.json
```

Metrics include accuracy, confusion matrix, and a classification report. Results are not included by default because they depend on live LLM outputs and should not be fabricated.

## Qualitative Evaluation Rubric

For data analysis, summarization, and content generation, use the rubric criteria in `evaluation.py`:

- Relevance
- Completeness
- Consistency
- Instruction-following
- Output quality

Create a blank rubric sheet:

```bash
python -m experiments.run_rubric_template --task summarization --version v3
```

Score each criterion from 1 to 5 after running prompts and reviewing the actual responses.

## Prompt Version Comparison

V1 establishes a baseline with minimal instruction. It is useful for showing common weaknesses such as vague output, inconsistent formatting, and missing evaluation hooks.

V2 improves task framing by adding audience, expected sections, constraints, and rating-label rules. This usually improves relevance, completeness, and instruction-following.

V3 is optimized for repeatability and evaluation. It uses role prompting, structured output, explicit rules, evidence grounding, confidence language, and few-shot examples for classification. This makes outputs easier to compare and easier to score.

## Notes for Submission

Include these files and folders:

- Python source files under `src/`
- Prompt versions YAML
- Streamlit dashboard
- Dataset file or dataset download instructions
- Evaluation scripts
- `requirements.txt`
- `.env.example`
- `README.md`

Do not submit your real `.env` file or API key.
