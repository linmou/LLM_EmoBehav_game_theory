from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from .payoff_matrices import PayoffMatrix


class BehaviorChoices(BaseModel, ABC):
    """Abstract base class for behavior choices in a game"""

    @abstractmethod
    def is_valid_choice(self, choice: str) -> bool:
        """Check if a choice is valid"""
        pass

    @abstractmethod
    def get_choices(self) -> list[str]:
        """Get the choices"""
        pass

    @staticmethod
    @abstractmethod
    def example() -> dict:
        """Provide an example of behavior choices"""
        pass

    def __str__(self):
        return ", ".join(self.get_choices())


class GameScenario(BaseModel, ABC):
    """Abstract base class for game scenarios"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payoff_matrix: Union[Dict[str, Dict[str, Dict[str, float]]], PayoffMatrix] = Field(
        default=None
    )

    @abstractmethod
    def get_scenario_info(self) -> dict:
        """Get the scenario information"""
        pass

    def get_participants(self) -> list[dict]:
        return self.participants

    @abstractmethod
    def get_behavior_choices(self) -> BehaviorChoices:
        """Get the behavior choices"""
        pass

    @abstractmethod
    def find_behavior_from_decision(self, decision: str) -> str:
        """Convert a decision string to a behavior identifier"""
        pass

    @staticmethod
    @abstractmethod
    def example() -> dict:
        """Provide an example scenario"""
        pass

    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participants()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        """


class SequentialGameScenario(GameScenario, ABC):
    """Base class for sequential game scenarios"""

    previous_actions_length: int

    @property
    @abstractmethod
    def previous_actions(self) -> list:
        """Get the previous actions"""
        pass


class GameDecision(BaseModel, ABC):
    """Abstract base class for game decisions"""

    @abstractmethod
    def validate_decision(self, decision: str) -> bool:
        """Validate if a decision is valid for the current scenario"""
        pass

    @staticmethod
    def example() -> dict:
        return {"rational": "<rational for the decision>", "decision": "<decision>"}


class Game:
    """Main class to handle different types of games"""

    def __init__(
        self,
        name: str,
        scenario_class: Union[GameScenario, SequentialGameScenario],
        decision_class: Type[GameDecision],
        payoff_matrix: Union[Dict[str, Any], PayoffMatrix],
        extra_attrs: Dict[str, Any] = {},
        data_path: str = None,
        data_folder: str = None,
    ):
        self.name = name
        self.scenario_class = scenario_class
        self.decision_class = decision_class
        self.payoff_matrix = payoff_matrix
        self.data_path = data_path
        self.data_folder = data_folder if data_folder else self.folder_path
        self.extra_attrs = extra_attrs

    def add_extra_attr(self, key: str, value: Any):
        self.extra_attrs.update({key: value})

    @property
    def folder_path(self) -> str:
        """Get the default folder path for scenario files"""
        return f"groupchat/scenarios/{self.name}"

    def create_scenario(self, data: dict) -> GameScenario:
        """Create a new scenario instance"""
        data["payoff_matrix"] = self.payoff_matrix
        data.update(self.extra_attrs)
        scenario = self.scenario_class(**data)
        self.decision_class.set_scenario(scenario)
        return scenario

    def create_decision(self, **data) -> GameDecision:
        """Create a new decision instance"""
        return self.decision_class(**data)

    @property
    def example_scenario(self) -> dict:
        """Get an example scenario for this game type"""
        return self.scenario_class.example()
