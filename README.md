# Active-Critic

Active-Critic is a reference-free evaluator for natural language generation. It first infers the task and evaluation criteria from human-rated examples, then scores responses under those criteria.

Two modes are available. `fine` scores each criterion before producing an overall score. `coarse` produces the overall score directly.

## Setup

Use Python 3.10–3.14.

```bash
python -m pip install -r requirements.txt
```

The default target model is `gpt-3.5-turbo-1106`. Set `OPENAI_API_KEY` before running with this target.

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

The full pipeline also requires a locally served Orca2-7B proxy with an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`. Use `--model`, `--api-base`, and `--proxy-api-base` to change the target model and endpoints.

## Run

```bash
python run_active_critic.py --dataset summeval --mode fine
```

Datasets: `summeval`, `topicalchat`, `sfres`, `openmeva`.

Modes: `coarse`, `fine`.

Datasets are included in `dataset/`. Each run uses a source-grouped 25%/75% train/test split, builds the evaluator on the training partition, and reports correlations on the test partition. Outputs are written to `output/`.
