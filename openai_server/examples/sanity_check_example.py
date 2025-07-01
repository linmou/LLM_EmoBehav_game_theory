#!/usr/bin/env python3
"""
Sanity check script for testing neural emotion activation with PromptExperiment.
This script runs a quick validation test with limited data to verify the setup works.
"""

import logging
import sys

from prompt_experiment import PromptExperiment

# Configure logging for sanity check
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_neutral_sanity_check():
    """Test neutral server configuration."""
    logger.info("=" * 50)
    logger.info("SANITY CHECK: Testing Neutral Server")
    logger.info("=" * 50)

    try:
        engine = PromptExperiment(
            "config/priDeli_neural_test_config.yaml", experiment_id="sanity_check_neutral"
        )

        # Run sanity check with very limited data
        engine.run_sanity_check(max_scenarios=3, max_repeat=1)

        logger.info("✅ Neutral server sanity check PASSED")
        return True

    except ConnectionError as e:
        logger.error(f"❌ Neutral server sanity check FAILED:\n{e}")
        return False
    except Exception as e:
        logger.error(f"❌ Neutral server sanity check FAILED: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def run_anger_sanity_check():
    """Test anger-activated server configuration."""
    logger.info("=" * 50)
    logger.info("SANITY CHECK: Testing Anger-Activated Server")
    logger.info("=" * 50)

    try:
        engine = PromptExperiment(
            "config/priDeli_neural_anger_test_config.yaml", experiment_id="sanity_check_anger"
        )

        # Run sanity check with very limited data
        engine.run_sanity_check(max_scenarios=3, max_repeat=1)

        logger.info("✅ Anger server sanity check PASSED")
        return True

    except ConnectionError as e:
        logger.error(f"❌ Anger server sanity check FAILED:\n{e}")
        return False
    except Exception as e:
        logger.error(f"❌ Anger server sanity check FAILED: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def main():
    """Run complete sanity check for both configurations."""
    logger.info("Starting Neural Emotion Activation Sanity Check")
    logger.info("Make sure both servers are running:")
    logger.info("  Neutral: python -m openai_server --model <model_path> --port 8000")
    logger.info(
        "  Anger:   python -m openai_server --model <model_path> --emotion anger --port 8001"
    )
    logger.info("")

    # Test both configurations
    neutral_ok = run_neutral_sanity_check()
    anger_ok = run_anger_sanity_check()

    # Summary
    logger.info("=" * 50)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Neutral Server: {'✅ PASS' if neutral_ok else '❌ FAIL'}")
    logger.info(f"Anger Server:   {'✅ PASS' if anger_ok else '❌ FAIL'}")

    if neutral_ok and anger_ok:
        logger.info("🎉 All sanity checks passed! Ready for full experiments.")
        logger.info("")
        logger.info("Next steps:")
        logger.info(
            "  1. Run full neutral experiment: "
            "python prompt_experiment.py config/priDeli_neural_test_config.yaml"
        )
        logger.info(
            "  2. Run full anger experiment: "
            "python prompt_experiment.py config/priDeli_neural_anger_test_config.yaml"
        )
        logger.info("  3. Compare results in results/neural_test/")
    else:
        logger.error(
            "❌ Some sanity checks failed. Please fix issues before running full experiments."
        )
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
