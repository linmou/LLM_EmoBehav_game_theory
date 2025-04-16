import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

import api_configs
from api_infer_engine import run_tests
from constants import Emotions
from data_creation.create_scenario import ScenarioGenerator
from games.game import Game
from games.game_configs import get_game_config
from statistical_engine import analyze_emotion_and_intensity_effects
from stimulis.emotions import emotion2stimulus

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PromptExperiment:
    def __init__(self, config_path: str, experiment_id: str = None):
        """Initialize the experiment engine with configuration from YAML."""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.experiment_id = (
            experiment_id
            or f"{self.config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.output_dir = (
            Path(self.config["experiment"]["output"]["base_dir"]) / self.experiment_id
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save the experiment config file
        self._save_experiment_config()

    def _save_experiment_config(self):
        """Save a copy of the experiment configuration file with the results."""
        dest_path = self.output_dir / "exp_cfg.yaml"
        shutil.copy2(self.config_path, dest_path)
        logger.info(f"Saved experiment configuration to {dest_path}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load and merge LLM config from api_configs.py
        llm_config = config["experiment"]["llm"]
        config_name = llm_config.pop("llm_config")

        try:
            # Get the base config from api_configs
            base_config = getattr(api_configs, config_name).copy()
            # Update with any additional parameters from YAML
            # Put the merged config back
            config["experiment"]["llm"] = {
                "llm_config": base_config,
                "generation_config": llm_config["generation_config"],
            }

            logger.info(
                f"Loaded LLM config from {config_name} with additional parameters"
            )

        except AttributeError:
            raise ValueError(f"Config '{config_name}' not found in api_configs.py")

        return config

    def _get_game_instance(self) -> Game:
        """Create game instance based on configuration."""
        exp_game_config = self.config["experiment"]["game"]
        game_name = exp_game_config["name"]
        game_config = get_game_config(game_name)

        game = Game(
            name=game_name,
            scenario_class=game_config["scenario_class"],
            decision_class=game_config["decision_class"],
            payoff_matrix=game_config["payoff_matrix"],
            data_folder=game_config.get("data_folder", None),
        )
        for key, value in exp_game_config.items():
            if key != "name":
                game.add_extra_attr(key, value)
        return game

    def should_generate_data(self) -> bool:
        """Check if data generation is needed based on config."""
        data_config = self.config["experiment"].get("data", {})
        return data_config.get("generate", False)

    def run_data_creation(self):
        """Step 1: Create game scenarios if needed."""
        if not self.should_generate_data():
            logger.info("Skipping data creation as per configuration")
            return

        logger.info("Step 1: Creating game scenarios")

        game = self._get_game_instance()
        data_config = self.config["experiment"]["data"]

        generator = ScenarioGenerator(
            game=game,
            participants=["Alice", "Bob"],
            config_path="config/OAI_CONFIG_LIST",
        )

        generator.generate_simultaneous_scenarios(
            ttl_number=data_config["num_scenarios"],
            batch_size=data_config["batch_size"],
        )

        logger.info("Scenario creation completed")

    def run_api_tests(self):
        """Step 2: Run API tests for each emotion."""
        logger.info("Step 2: Running API tests")

        game = self._get_game_instance()
        llm_config = self.config["experiment"]["llm"]

        # run api tests without emotion as baseline
        output_files = []
        output_file = run_tests(
            game=game,
            llm_config=llm_config["llm_config"],
            generation_config=llm_config["generation_config"],
            output_dir=self.output_dir,
            emotion="Neutral",
            intensity="Neutral",
            repeat=self.config["experiment"]["repeat"],
        )
        output_files.append(output_file)

        # run api tests with emotion
        for emotion_str in self.config["experiment"]["emotions"]:
            emotion = Emotions.from_string(emotion_str)
            stimulus = emotion2stimulus[emotion]
            for intensity in self.config["experiment"]["intensity"]:
                logger.info(
                    f"Testing scenarios with emotion: {emotion.value} and intensity: {intensity}"
                )

                # Use system message template from config
                system_message = self.config["experiment"][
                    "system_message_template"
                ].format(emotion=emotion.value, stimulus=stimulus, intensity=intensity)

                output_file = run_tests(
                    game=game,
                    llm_config=llm_config["llm_config"],
                    generation_config=llm_config["generation_config"],
                    output_dir=self.output_dir,
                    emotion=emotion.value,
                    intensity=intensity,
                    system_message=system_message,
                    repeat=self.config["experiment"]["repeat"],
                )
                output_files.append(output_file)

        self.output_files = output_files

        return output_files

    def run_statistical_analysis(self):
        """Step 3: Perform statistical analysis."""
        logger.info("Step 3: Running statistical analysis")

        output_samples = []
        for output_file in self.output_files:
            with open(output_file, "r") as f:
                output_samples.extend(json.load(f))

        df = pd.DataFrame(output_samples).replace("None", "Neutral")
        csv_output_file = self.output_dir / "all_output_samples.csv"
        df.to_csv(csv_output_file, index=False)

        results = analyze_emotion_and_intensity_effects(
            str(csv_output_file), self.output_dir
        )

        # Save analysis results
        analysis_output = self.output_dir / "analysis_results.json"
        with open(analysis_output, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Analysis results saved to {analysis_output}")

    def run_experiment(self):
        """Run all three steps of the experiment."""
        logger.info(f"Starting experiment: {self.experiment_id}")

        try:
            self.run_data_creation()
            self.run_api_tests()
            self.run_statistical_analysis()
            logger.info("Experiment completed successfully")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise


if __name__ == "__main__":
    # engine = PromptExperiment("config/trusteeGame_experiment_config.yaml", )
    engine = PromptExperiment("./config/priDeli_experiment_config.yaml")
    engine.run_experiment()
