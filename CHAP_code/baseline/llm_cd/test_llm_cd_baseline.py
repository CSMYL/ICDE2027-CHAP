"""Command-line entry point for the LLM-CD baseline.

Usage (from CHAP_code/):
  cd baseline/llm_cd
  python test_llm_cd_baseline.py --dataset creditcard --sample_size 200 --predictor rf
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # CHAP_code/
sys.path.insert(0, project_root)

from baseline.llm_cd.llm_cd_baseline import main


if __name__ == "__main__":
    main()
