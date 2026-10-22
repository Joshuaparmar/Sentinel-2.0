# SENTINEL 2.0 synthetic generator starter package

Contents:
- `src/sentinel/generate_synthetic.py`
- `tests/test_generate_synthetic.py`

## Requirements
Python 3.10+, pandas, numpy, pyarrow, pytest.

Install:
```bash
python -m pip install pandas numpy pyarrow pytest
```

Run tests:
```bash
python -m pytest -v tests/test_generate_synthetic.py
```

Run generator:
```bash
python src/sentinel/generate_synthetic.py --source PATH_TO_SOURCE_PARQUET --output-dir PATH_TO_SEPARATE_OUTPUT_DIR
```

Important:
- This is a first implementation to test and review, not yet certified for real-data use.
- The training-derived envelope is a synthetic plausibility guardrail, not a hardware specification.
- Review the feature list and source schema before trusting model inputs.
- Output must be in a separate directory; never point it at the original source directory.
