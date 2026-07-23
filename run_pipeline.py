#!/usr/bin/env python
"""
run_pipeline.py
================
Master orchestration script — runs all 7 pipeline stages in order.
Each stage is a subprocess call so failures are isolated.

Usage:
  python run_pipeline.py                          # full pipeline
  python run_pipeline.py --from-step 3            # resume from Stage 3a
  python run_pipeline.py --skip-persona           # skip Stage 2 (reuse existing)
  python run_pipeline.py --skip-training          # skip Stage 4 (reuse adapter)
  python run_pipeline.py --only-evaluate          # run Stage 7 only

All scripts are located in src/ and run with PYTHONPATH=src so they
can import config and utils without package installation.
"""

import argparse
import os
import subprocess
import sys
import time

# ── Ensure src/ is on the path so config.py and utils.py are importable ───────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import validate_config  # noqa: E402 (must come after sys.path insert)
from utils import get_logger        # noqa: E402

logger = get_logger("pipeline", log_file="pipeline.log")

# ── Pipeline stage definitions ─────────────────────────────────────────────────
# Each entry: (display name, script path relative to project root)
STAGES = [
    (1,  "Stage 1:  Data Preparation",                  "src/prepare_data.py"),
    (2,  "Stage 2:  Persona Generation (Ollama LLaMA)", "src/infer_persona_llm.py"),
    (3,  "Stage 3a: Generate Instruction Templates",    "src/generate_instructions.py"),
    (4,  "Stage 3b: Build Instruction Dataset",         "src/build_instructions.py"),
    (5,  "Stage 4:  LoRA Fine-Tuning",                  "src/fine_tune_lora.py"),
    (6,  "Stage 5+6: Test Predict",                     "src/test_predict.py"),
    (7,  "Stage 7:  Evaluate",                          "src/evaluate.py"),
]


def run_stage(name: str, script: str) -> bool:
    """
    Run one pipeline stage as a subprocess.
    PYTHONPATH is set so src/ imports work correctly.

    Returns True on success, False on failure.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RUNNING: {name}")
    logger.info("=" * 60)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        env=env,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        logger.info(f"SUCCESS: {name} ({elapsed:.1f}s)")
        return True
    else:
        logger.error(f"FAILED : {name} (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apollo Mode Choice Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                  # Run all stages
  python run_pipeline.py --from-step 3   # Resume from Stage 3a
  python run_pipeline.py --skip-persona  # Skip Stage 2 (reuse existing personas)
  python run_pipeline.py --skip-training # Skip Stage 4 (reuse LoRA adapter)
  python run_pipeline.py --only-evaluate # Run only Stage 7
        """
    )
    parser.add_argument("--from-step",      type=int, choices=range(1, 8),
                        help="Start from stage N (1-7). Skips all prior stages.")
    parser.add_argument("--skip-persona",   action="store_true",
                        help="Skip Stage 2: persona inference (uses existing CSV)")
    parser.add_argument("--skip-templates", action="store_true",
                        help="Skip Stage 3a: template generation (uses existing JSON)")
    parser.add_argument("--skip-build",     action="store_true",
                        help="Skip Stage 3b: instruction build (uses existing JSONL)")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip Stage 4: LoRA fine-tuning (uses existing adapter)")
    parser.add_argument("--skip-predict",   action="store_true",
                        help="Skip Stages 5+6: prediction (uses existing predictions CSV)")
    parser.add_argument("--only-evaluate",  action="store_true",
                        help="Run Stage 7 only")
    args = parser.parse_args()

    # ── Determine which stages to skip ────────────────────────────────────────
    from_step = args.from_step or 1

    skip = {
        1: from_step > 1,
        2: from_step > 2 or args.skip_persona,
        3: from_step > 3 or args.skip_templates,
        4: from_step > 4 or args.skip_build,
        5: from_step > 5 or args.skip_training,
        6: from_step > 6 or args.skip_predict,
        7: False,  # Evaluate always runs unless only_evaluate overrides above
    }
    if args.only_evaluate:
        skip = {k: True for k in skip}
        skip[7] = False

    # ── Print plan ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("APOLLO MODE CHOICE PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Validate config before starting
    logger.info("Validating configuration...")
    validate_config()

    logger.info("\nPipeline plan:")
    for idx, (num, name, _) in enumerate(STAGES, 1):
        status = "SKIP" if skip[idx] else " RUN"
        logger.info(f"  [{status}] {name}")

    # ── Execute stages ─────────────────────────────────────────────────────────
    logger.info("\nStarting pipeline...\n")
    pipeline_start = time.time()

    for idx, (num, name, script) in enumerate(STAGES, 1):
        if skip[idx]:
            logger.info(f"Skipping: {name}")
            continue

        success = run_stage(name, script)
        if not success:
            logger.error(f"\nPipeline stopped at {name}.")
            logger.error("Fix the error above and re-run with --from-step to resume.")
            sys.exit(1)

    total = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE ({total/60:.1f} min)")
    logger.info("=" * 60)
    logger.info("\nKey output files:")
    logger.info("  outputs/test_predictions_full.csv  — predictions")
    logger.info("  outputs/evaluation_report.txt      — metrics summary")
    logger.info("  outputs/confusion_matrix.png       — confusion matrix")
    logger.info("  outputs/persona_accuracy.png       — accuracy by persona")
    logger.info("  outputs/lora_adapter_final/        — fine-tuned model")


if __name__ == "__main__":
    main()
