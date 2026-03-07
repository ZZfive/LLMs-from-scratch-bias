# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-07
**Commit:** 6e6ccc1
**Branch:** (current)

## OVERVIEW
Educational Python codebase implementing LLMs from scratch. Translation/extension of Sebastian Raschka's "Build a Large Language Model From Scratch" book. Uses PyTorch + Jupyter Notebooks.

## STRUCTURE
```
./
├── ch01/      # Introduction/Setup
├── ch02/      # Text data & tokenization
├── ch03/      # Attention mechanisms
├── ch04/      # GPT implementation
├── ch05/      # Pretraining (largest, most bonus content)
├── ch06/      # Finetuning
└── ch07/      # Instruction finetuning & RLHF
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Train GPT | `ch05/01_main-chapter-code/gpt_train.py` | Main training script |
| Generate text | `ch05/01_main-chapter-code/gpt_generate.py` | Inference |
| Fine-tune | `ch07/01_main-chapter-code/gpt_instruction_finetuning.py` | RLHF/DPO |
| Gradio UI | `ch07/06_user_interface/app.py` | Interactive web interface |
| Tests | `ch05/*/tests/test_*.py` | Distributed test files |

## CODE MAP
No LSP available. Key modules:
- `previous_chapters.py` — Import cascade (each chX imports from chX-1)
- `gpt.py` (ch04) — Core GPT model
- `gpt_train.py` (ch05) — Pretraining loop

## CONVENTIONS (THIS PROJECT)
- Decentralized deps: Each module has `requirements-extra.txt`
- No root requirements.txt
- No centralized test config (pytest.ini, etc.)
- Test files: `tests.py` (standalone) or `tests/test_*.py` (bonus modules)
- Notebook-driven: Tests import from `.ipynb` via `import_definitions_from_notebook()`

## ANTI-PATTERNS (THIS PROJECT)
- **NONE** — Clean codebase (no TODO/FIXME/DEPRECATED markers)
- No CI/CD (no GitHub Actions, Makefile, Dockerfile)
- No root package config (not a pip-installable library)

## UNIQUE STYLES
- Chapter-based organization mirrors book structure
- Bonus content in `ch0X/0X_bonus_*/` subdirectories
- Memory estimator scripts for each optimization technique

## COMMANDS
```bash
# Train
python ch05/01_main-chapter-code/gpt_train.py

# Generate
python ch05/01_main-chapter-code/gpt_generate.py

# Fine-tune
python ch07/01_main-chapter-code/gpt_instruction_finetuning.py

# UI
python ch07/06_user_interface/app.py

# Tests (per module)
pytest ch05/07_gpt_to_llama/tests/
```

## NOTES
- Install extras as needed: `pip install -r ch07/02_dataset-utilities/requirements-extra.txt`
- OpenAI API keys go in module-specific `config.json` (gitignored)
- No Docker/reproducibility setup — run locally with PyTorch
