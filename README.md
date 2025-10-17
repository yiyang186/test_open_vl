```bash

pip install --upgrade "openai>=1.0"
pip install google-genai

# modify your promt in scripts.
python doubao_seed_1_6.py --key <your key> --input-dir /path/to/input_dir --output-dir /path/to/output_dir
python qwen_3_vl.py --key <your key> --input-dir /path/to/input_dir --output-dir /path/to/output_dir
```