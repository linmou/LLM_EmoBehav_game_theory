# Tests for data_creation/scenario_creation/langgraph_creation/wordnet_affect_behavior_choice_portion.py
# Purpose: validate WordNet-Affect hierarchy mapping, lexicon building, and emotion portion computation.

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_creation.scenario_creation.langgraph_creation.wordnet_affect_behavior_choice_portion as wna


def test_map_category_to_basic_emotion():
    parent_map = {
        "wrath": "anger",
        "anger": "emotion",
        "joy": "emotion",
    }
    assert wna.map_category_to_basic_emotion("wrath", parent_map) == "anger"
    assert wna.map_category_to_basic_emotion("joy", parent_map) == "happiness"
    assert wna.map_category_to_basic_emotion("emotion", parent_map) is None


def test_build_emotion_lexicon_from_xml(tmp_path: Path):
    hierarchy_xml = """<categ-list>
  <categ name="anger" isa="emotion"/>
  <categ name="wrath" isa="anger"/>
  <categ name="joy" isa="emotion"/>
</categ-list>
"""
    synsets_xml = """<syn-list>
  <noun-syn-list>
    <noun-syn id="n#00000001" categ="wrath"/>
    <noun-syn id="n#00000002" categ="joy"/>
  </noun-syn-list>
</syn-list>
"""
    hierarchy_path = tmp_path / "a-hierarchy.xml"
    synsets_path = tmp_path / "a-synsets.xml"
    hierarchy_path.write_text(hierarchy_xml)
    synsets_path.write_text(synsets_xml)

    def resolver(synset_id: str) -> list[str]:
        return {"n#00000001": ["rage"], "n#00000002": ["delight"]}.get(
            synset_id, []
        )

    parent_map = wna.parse_affect_hierarchy(hierarchy_path)
    lexicon = wna.build_emotion_lexicon(
        synsets_path, parent_map, synset_resolver=resolver
    )

    assert "rage" in lexicon["anger"]
    assert "delight" in lexicon["happiness"]


def test_choice_portions():
    choices = [
        "Release a stable update",
        "Act in rage",
        "Delight the user",
    ]
    lexicon = {
        "anger": {"rage"},
        "happiness": {"delight"},
        "sadness": set(),
        "disgust": set(),
        "fear": set(),
        "surprise": set(),
    }
    portions = wna.compute_choice_emotion_portions(choices, lexicon)

    assert portions["anger"] == 1 / 3
    assert portions["happiness"] == 1 / 3
    assert portions["any_emotion"] == 2 / 3


def test_analyze_choice_marks_tokens():
    lexicon = {
        "anger": {"rage"},
        "happiness": {"delight"},
        "sadness": set(),
        "disgust": set(),
        "fear": set(),
        "surprise": set(),
    }
    result = wna.analyze_choice("Rage and delight", lexicon)

    assert "anger" in result["emotions"]
    assert "happiness" in result["emotions"]
    assert "rage" in result["tokens"]["anger"]
    assert "delight" in result["tokens"]["happiness"]


def test_parallel_choice_details_match_sequential():
    lexicon = {
        "anger": {"rage"},
        "happiness": {"delight"},
        "sadness": set(),
        "disgust": set(),
        "fear": set(),
        "surprise": set(),
    }
    choices = ["Rage now", "Delight later", "Nothing here"]
    sequential = wna.compute_choice_emotion_details(choices, lexicon, max_workers=1)
    parallel = wna.compute_choice_emotion_details(choices, lexicon, max_workers=2)

    assert sequential == parallel


def test_find_game_names(tmp_path: Path):
    (tmp_path / "Game_A_all_data_samples.json").write_text("[]")
    (tmp_path / "Game_B_all_data_samples.json").write_text("[]")
    names = wna.find_game_names(tmp_path)

    assert set(names) == {"Game_A", "Game_B"}


def test_progress_wrapper_iterates():
    items = [1, 2, 3]
    wrapped = list(wna.progress_iter(items, enabled=False, desc="test"))
    assert wrapped == items


def test_progress_wrapper_enabled_iterates():
    items = [1, 2, 3]
    wrapped = list(wna.progress_iter(items, enabled=True, desc="test"))
    assert wrapped == items


def test_synset_lemmas_invalid_returns_empty():
    assert wna.synset_lemmas_from_wordnet("n#00000000") == []


def test_wordnet_affect_real_lexicon_marks_choice(tmp_path: Path):
    wna.ensure_nltk_resource("corpora/wordnet", "wordnet")
    wna.ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    wna.download_wordnet_affect(tmp_path)
    parent_map = wna.parse_affect_hierarchy(tmp_path / "a-hierarchy.xml")
    lexicon = wna.build_emotion_lexicon(tmp_path / "a-synsets.xml", parent_map)

    picked_emotion = None
    picked_word = None
    for emotion in wna.Emotions.get_emotions():
        words = sorted(lexicon.get(emotion, []))
        if words:
            picked_emotion = emotion
            picked_word = words[0]
            break

    assert picked_word is not None
    result = wna.analyze_choice(f"This is {picked_word}", lexicon)
    assert picked_emotion in result["emotions"]


def test_affect_hierarchy_contains_labels(tmp_path: Path):
    wna.download_wordnet_affect(tmp_path)
    categories = wna.parse_affect_categories(tmp_path / "a-hierarchy.xml")

    required = {
        "emotion",
        "mood",
        "trait",
        "cognitive-state",
        "physical-state",
        "emotion-eliciting-situation",
        "emotional-response",
        "behaviour",
        "attitude",
        "sensation",
    }
    assert ("edonic-signal" in categories) or ("hedonic-signal" in categories)
    assert required - {"attitude", "emotional-response"} <= categories
    assert {"attitude", "emotional-response"} - categories == {
        "attitude",
        "emotional-response",
    }


def test_natural_language_examples_for_emotions(tmp_path: Path):
    wna.ensure_nltk_resource("corpora/wordnet", "wordnet")
    wna.ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    wna.download_wordnet_affect(tmp_path)
    parent_map = wna.parse_affect_hierarchy(tmp_path / "a-hierarchy.xml")
    lexicon = wna.build_emotion_lexicon(tmp_path / "a-synsets.xml", parent_map)

    neutral_pool = [
        "The package arrived on Tuesday with the correct label.",
        "Please review the quarterly budget before Friday.",
        "She organized the files in alphabetical order.",
        "The meeting starts at nine and ends at ten.",
        "He calibrated the instrument and recorded the values.",
        "They walked to the station and caught the early train.",
    ]
    neutral_sentences = []
    for sentence in neutral_pool:
        if not wna.analyze_choice(sentence, lexicon)["emotions"]:
            neutral_sentences.append(sentence)
        if len(neutral_sentences) == 3:
            break

    assert len(neutral_sentences) == 3

    for emotion in wna.Emotions.get_emotions():
        words = sorted(
            [word for word in lexicon.get(emotion, []) if word.isalpha()]
        )
        assert words
        picked = words[0]
        positives = [
            f"I felt {picked} today.",
            f"Her {picked} was obvious to everyone.",
            f"The news brought {picked} to the team.",
        ]
        for sentence in positives:
            result = wna.analyze_choice(sentence, lexicon)
            assert emotion in result["emotions"]
            assert picked in result["tokens"][emotion]
        for sentence in neutral_sentences:
            result = wna.analyze_choice(sentence, lexicon)
            assert emotion not in result["emotions"]
