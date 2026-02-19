"""
Simple emotion check dataset for validating emotion manipulation.
Tests basic emotional state awareness with direct questions and classifies responses by emotion category.
"""

import re
from typing import Any, Dict, List

from ..data_models import BenchmarkItem
from .base import BaseBenchmarkDataset


class EmotionCheckDataset(BaseBenchmarkDataset):
    """
    Simple dataset to test emotion manipulation by asking direct questions
    about the model's current emotional state and classifying responses.

    Returns detected emotion category instead of binary scoring.
    """

    # Emotion expression dictionary for classification
    EMOTION_EXPRESSIONS = {
        "anger": [
            "angry",
            "mad",
            "furious",
            "irritated",
            "annoyed",
            "rage",
            "upset",
            "frustrated",
            "pissed",
            "livid",
        ],
        "happiness": [
            "happy",
            "joyful",
            "pleased",
            "content",
            "cheerful",
            "delighted",
            "excited",
            "glad",
            "blissful",
            "ecstatic",
        ],
        "sadness": [
            "sad",
            "depressed",
            "melancholy",
            "down",
            "upset",
            "disappointed",
            "blue",
            "gloomy",
            "miserable",
            "dejected",
        ],
        "fear": [
            "afraid",
            "scared",
            "anxious",
            "worried",
            "nervous",
            "terrified",
            "panicked",
            "frightened",
            "fearful",
            "alarmed",
        ],
        "disgust": [
            "disgusted",
            "revolted",
            "repulsed",
            "sick",
            "nauseous",
            "appalled",
            "disturbed",
            "repelled",
            "grossed",
            "nauseated",
        ],
        "surprise": [
            "surprised",
            "amazed",
            "astonished",
            "shocked",
            "startled",
            "stunned",
            "bewildered",
            "astounded",
            "flabbergasted",
            "dumbfounded",
        ],
        "neutral": [
            "neutral",
            "calm",
            "peaceful",
            "fine",
            "okay",
            "normal",
            "relaxed",
            "stable",
            "balanced",
            "composed",
        ],
    }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index with prompt formatting.

        For academic_scale items, returns a composite ground_truth dict:
        {"active": <activated emotion>, "target": <item target emotion>, "keying": "direct"|"reverse"}
        """
        item = self.items[idx]

        # Extract options from metadata for multiple choice questions
        options = None
        if item.metadata and "options" in item.metadata:
            options = item.metadata["options"]

        # Create prompt using wrapper or default format
        if self.prompt_wrapper:
            prompt = self.prompt_wrapper(
                context=item.context if item.context else "",
                question=item.input_text,
                answer=item.ground_truth,
                options=options,
            )

        else:
            # Default prompt format
            if item.context:
                prompt = (
                    f"Context: {item.context}\nQuestion: {item.input_text}\nAnswer:"
                )
            else:
                prompt = f"{item.input_text}\nAnswer:"

        # Transform ground truth if answer wrapper provided
        active_emotion = (
            self.answer_wrapper(item.ground_truth)
            if self.answer_wrapper
            else item.ground_truth
        )

        # For academic_scale, provide composite ground truth with active/target/keying
        if item.metadata and item.metadata.get("category") == "academic_scale":
            target_emotion = item.ground_truth
            keying = (item.metadata.get("keying") or "direct").lower()
            ground_truth: Any = {
                "active": active_emotion,
                "target": target_emotion,
                "keying": "reverse" if keying == "reverse" else "direct",
                "options": options,
            }
        else:
            # Default pass-through for non-academic tasks
            ground_truth = active_emotion

        return {"item": item, "prompt": prompt, "ground_truth": ground_truth}

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """
        Load emotion check questions from JSON/JSONL file.

        Supports three schemas:
        1) Basic validation schema (id, input, ground_truth, category)
        2) Academic scales schema (emotion, Instructions, question, source)
        3) Emotion scale schema (id, sentence) for subjective sentence prompts
        """

        raw_data = self._load_raw_data()

        items: List[BenchmarkItem] = []
        if not raw_data:
            return items

        # Detect schema by keys present in the first record
        first = raw_data[0]
        basic_schema = all(k in first for k in ("id", "input", "ground_truth"))
        academic_schema = all(k in first for k in ("emotion", "Instructions", "question"))

        if basic_schema:
            for item_data in raw_data:
                items.append(
                    BenchmarkItem(
                        id=item_data["id"],
                        input_text=item_data["input"],
                        context=None,
                        ground_truth=item_data["ground_truth"],
                        metadata={
                            "category": item_data.get("category", "emotion_check"),
                            "expects_emotion": True,
                            "response_type": "single_word",
                        },
                    )
                )
            return items

        if academic_schema:
            # Build items by combining Instructions + question; ground truth is the target emotion
            for idx, item_data in enumerate(raw_data):
                instruction = item_data.get("Instructions", "").strip()
                question = item_data.get("question", "").strip()
                joined = (instruction + " \n" + question).strip() if instruction else question
                target_emotion = item_data.get("emotion", "neutral").strip().lower()
                # Optional choice anchors; keep as provided
                options = item_data.get("choices")
                if isinstance(options, list):
                    options = [str(x).strip() for x in options]

                items.append(
                    BenchmarkItem(
                        id=idx,
                        input_text=joined,
                        context=None,
                        ground_truth=target_emotion,
                        metadata={
                            "category": "academic_scale",
                            "expects_emotion": False,
                            "response_type": "scale_choice",
                            "emotion_dimension": target_emotion,
                            "scale_source": item_data.get("source"),
                            "options": options,
                        },
                    )
                )
            return items

        # Subjective sentence schema for emotion-scale task
        # Each record provides a sentence prompt that should be answered in free text.
        subjective_schema = "sentence" in first
        if subjective_schema:
            for idx, item_data in enumerate(raw_data):
                sentence = str(item_data.get("sentence", "")).strip()
                if not sentence:
                    continue
                item_id = item_data.get("id", idx)
                items.append(
                    BenchmarkItem(
                        id=item_id,
                        input_text=sentence,
                        context=None,
                        ground_truth=item_data.get("ground_truth", "neutral"),
                        metadata={
                            "category": "emotion_scale",
                            "expects_emotion": True,
                            "response_type": "free_text",
                            "source": item_data.get("source", "style_vectors_subjective_sentences"),
                        },
                    )
                )
            return items

        # Fallback: unknown schema - try to coerce minimal fields
        for idx, item_data in enumerate(raw_data):
            text = item_data.get("input") or item_data.get("question") or "Describe your current emotion."
            items.append(
                BenchmarkItem(
                    id=idx,
                    input_text=text,
                    context=None,
                    ground_truth=item_data.get("ground_truth") or item_data.get("emotion") or "neutral",
                    metadata={"category": item_data.get("category", "emotion_check")},
                )
            )
        return items

    def _classify_emotion_response(self, response: str) -> str:
        """
        Classify response into one of the 6 emotion categories + neutral.

        Args:
            response: Model's response text

        Returns:
            Detected emotion category name or "unknown"
        """
        if not response:
            return "unknown"

        # Clean and extract first meaningful word
        response_clean = response.strip().lower()
        # Remove punctuation and get first word
        first_word = (
            re.sub(r"[^\w\s]", "", response_clean).split()[0]
            if response_clean.split()
            else ""
        )

        if not first_word:
            return "unknown"

        # Check which emotion category this word belongs to
        for emotion_category, expressions in self.EMOTION_EXPRESSIONS.items():
            if first_word in expressions:
                return emotion_category

        return "unknown"

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        """Normalize judge confidence into [0.0, 1.0]."""
        try:
            conf = float(value)
        except Exception:
            conf = 0.5
        if conf < 0.0:
            return 0.0
        if conf > 1.0:
            return 1.0
        return conf

    def _score_emotion_response(
        self, response: str, ground_truth: Any, prompt: str = ""
    ) -> tuple[float, Dict[str, Any]]:
        """Return score and detailed evaluation metadata for emotion tasks."""
        ground_truth_text = str(ground_truth).strip().lower()

        if hasattr(self, "llm_eval_config") and self.llm_eval_config is not None:
            from ..evaluation_utils import llm_evaluate_response

            eval_prompt = self.llm_eval_config.get("evaluation_prompt", "")
            query = eval_prompt.format(question=prompt, response=response)
            eval_config = dict(self.llm_eval_config)
            eval_config.pop("evaluation_prompt", None)
            eval_config.setdefault("model", "gpt-4o-mini")
            eval_config.setdefault("temperature", 0.1)

            try:
                result = llm_evaluate_response(
                    system_prompt="You are an expert emotion classifier. Always respond with valid JSON format.",
                    query=query,
                    llm_eval_config=eval_config,
                )
                detected_emotion = str(result.get("emotion", "neutral")).strip().lower()
                confidence = self._normalize_confidence(result.get("confidence", 0.5))
                matched = detected_emotion == ground_truth_text
                details = {
                    "predicted_emotion": detected_emotion,
                    "confidence": confidence,
                    "matched_ground_truth": matched,
                    "judge_client": str(eval_config.get("client", "openai")).lower(),
                    "judge_model": str(eval_config.get("model", "gpt-4o-mini")),
                    "judge_method": "llm",
                }
                return float(matched) * confidence, details
            except Exception as exc:
                detected_emotion = self._classify_emotion_response(response)
                matched = detected_emotion == ground_truth_text
                details = {
                    "predicted_emotion": detected_emotion,
                    "confidence": 1.0 if matched else 0.0,
                    "matched_ground_truth": matched,
                    "judge_client": str(eval_config.get("client", "openai")).lower(),
                    "judge_model": str(eval_config.get("model", "gpt-4o-mini")),
                    "judge_method": "rule_based_fallback",
                    "evaluation_error": str(exc),
                }
                return float(matched), details

        detected_emotion = self._classify_emotion_response(response)
        matched = detected_emotion == ground_truth_text
        details = {
            "predicted_emotion": detected_emotion,
            "confidence": 1.0 if matched else 0.0,
            "matched_ground_truth": matched,
            "judge_client": "rule_based",
            "judge_model": "rule_based",
            "judge_method": "rule_based",
        }
        return float(matched), details

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str = ""
    ) -> float:
        """
        Evaluate single response.

        - For academic_scale: interpret Likert rating (1-5) and compute alignment with
          the activated emotion vs the item's target emotion, handling reverse keying.
        - Else: Use emotion classification (LLM or rule-based) vs ground truth string.
        """
        # Academic scale per-item scoring
        if isinstance(ground_truth, dict) and {"active", "target"}.issubset(ground_truth.keys()):
            resp_text = self._extract_response_text(response)
            options = ground_truth.get("options") or []
            if options:
                idx = self._parse_choice_index(resp_text, options)
                if idx is None:
                    idx = self._parse_numeric_index(resp_text, options)
                if idx is None:
                    raise ValueError(
                        f"Unable to map response '{resp_text}' to any provided option"
                    )
                N = max(1, len(options))
                z = (2.0 * float(idx) / float(max(1, N - 1))) - 1.0
            else:
                # No options provided; attempt Likert inference (legacy behavior)
                rating = self._parse_likert_rating(resp_text)
                idx = float(rating - 1)
                N = 5
                z = (2.0 * idx / float(N - 1)) - 1.0

            keying_sign = -1.0 if ground_truth.get("keying") == "reverse" else 1.0
            target_sign = 1.0 if ground_truth.get("active") == ground_truth.get("target") else -1.0
            score = 0.5 * (1.0 + keying_sign * target_sign * z)
            return max(0.0, min(1.0, float(score)))

        score, _ = self._score_emotion_response(response, ground_truth, prompt)
        return score

    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> List[float]:
        """
        Compatibility wrapper returning scores only.

        Detailed outputs are available via evaluate_batch_with_details().
        """
        scores, eval_errors, eval_details = self.evaluate_batch_with_details(
            responses=responses,
            ground_truths=ground_truths,
            task_names=task_names,
            prompts=prompts,
        )
        self._last_eval_errors = eval_errors  # type: ignore[attr-defined]
        self._last_eval_details = eval_details  # type: ignore[attr-defined]
        return scores

    def evaluate_batch_with_details(
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> tuple[List[float], List[Any], List[Any]]:
        """
        Batch evaluation with per-item detail persistence.

        Returns:
        - scores
        - eval_errors: list[str | None]
        - eval_details: list[dict | None]
        """
        from concurrent.futures import ThreadPoolExecutor

        last_eval_errors: List[Any] = [None] * len(responses)
        last_eval_details: List[Any] = [None] * len(responses)

        scores: List[float] = [float("nan")] * len(responses)
        max_workers = getattr(self, "eval_workers", 64)
        max_workers = max(1, min(int(max_workers), len(responses)))

        def _evaluate_one(idx: int, resp: str, gt: Any, task: str, prompt: str):
            if isinstance(gt, dict) and {"active", "target"}.issubset(gt.keys()):
                score = self.evaluate_response(resp, gt, task, prompt)
                return idx, score, None
            score, details = self._score_emotion_response(resp, gt, prompt)
            return idx, score, details

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_evaluate_one, i, resp, gt, task, prompt): i
                for i, (resp, gt, task, prompt) in enumerate(
                    zip(responses, ground_truths, task_names, prompts)
                )
            }

            for future, item_idx in future_to_idx.items():
                try:
                    idx, score, details = future.result()
                except Exception as exc:
                    last_eval_errors[item_idx] = str(exc)
                    continue
                scores[idx] = score
                last_eval_details[idx] = details

        return scores, last_eval_errors, last_eval_details

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> dict[str, Any]:
        """
        Return detailed emotion classification metrics.
        """
        detected_emotion = self._classify_emotion_response(response)

        return {
            "overall_score": float(detected_emotion == ground_truth),
            "detected_emotion": detected_emotion,
            "response_word": (
                response.strip().lower().split()[0] if response.strip() else ""
            ),
            "classification_success": detected_emotion != "unknown",
        }

    def get_task_metrics(self, task_name: str) -> List[str]:
        """Available metrics for emotion check"""
        if task_name == "academic_scale":
            return ["likert_alignment"]
        return ["emotion_classification", "detection_rate", "response_relevance"]

    # Internal helpers
    def _parse_likert_rating(self, response: str) -> int:
        """Extract a Likert rating 1..5 from a response.

        Strategy:
        - First digit 1..5 wins
        - Else map common words to numbers
        - Else return 3 (neutral)
        """
        if not response:
            return 3
        text = response.strip().lower()
        # Extract first digit 1..5
        m = re.search(r"[1-5]", text)
        if m:
            try:
                n = int(m.group(0))
                if 1 <= n <= 5:
                    return n
            except Exception:
                pass
        # Word mapping
        mapping = {
            "strongly agree": 5,
            "agree": 4,
            "neutral": 3,
            "neither": 3,
            "undecided": 3,
            "disagree": 2,
            "strongly disagree": 1,
        }
        for k, v in mapping.items():
            if k in text:
                return v
        return 3

    def _extract_response_text(self, response: str) -> str:
        """Extract answer content from potential JSON/Python-dict wrapper."""
        if not response:
            return ""
        t = response.strip()

        # Strip markdown code fences (common in model outputs)
        if "```" in t:
            m = re.search(
                r"```(?:json)?\s*(.*?)\s*```", t, flags=re.IGNORECASE | re.DOTALL
            )
            if m:
                t = m.group(1).strip()

        if t.startswith("{") and t.endswith("}"):
            # Try strict JSON
            try:
                import json
                obj = json.loads(t)
                val = obj.get("response")
                if isinstance(val, str):
                    return val.strip()
            except Exception:
                # Try Python literal dict parsing
                try:
                    import ast
                    obj = ast.literal_eval(t)
                    if isinstance(obj, dict):
                        val = obj.get("response") or obj.get("Response")
                        if isinstance(val, str):
                            return val.strip()
                except Exception:
                    # Regex fallback to capture 'response': '...'
                    m = re.search(r"['\"]response['\"]\s*:\s*['\"]([^'\"]+)['\"]", t, flags=re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
        return t

    def _sanitize_option(self, s: str) -> str:
        t = str(s).strip().lower()
        t = re.sub(r"^\s*\d+\s*[.=)\-:]\s*", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    def _parse_choice_index(self, response_text: str, options: List[str]):
        if not response_text:
            return None
        # Exact match
        for i, opt in enumerate(options):
            if response_text.strip() == str(opt):
                return i
        # Case-insensitive
        rt = response_text.strip().lower()
        for i, opt in enumerate(options):
            if rt == str(opt).strip().lower():
                return i
        # Sanitize numeric prefixes and compare
        rt_s = self._sanitize_option(rt)
        for i, opt in enumerate(options):
            if rt_s == self._sanitize_option(opt):
                return i
        # Flexible prefix matching (either can be a prefix of the other)
        for i, opt in enumerate(options):
            os = self._sanitize_option(opt)
            if os.startswith(rt_s) or rt_s.startswith(os):
                return i
        return None

    def _parse_numeric_index(self, response_text: str, options: List[str]):
        if not response_text:
            return None
        m = re.search(r"\d+", response_text)
        if not m:
            return None
        num = int(m.group(0))
        N = len(options)
        if N <= 0:
            return None
        zero_based = any(str(opt).lstrip().startswith("0") for opt in options)
        idx = num if zero_based else (num - 1)
        if 0 <= idx < N:
            return idx
        return None
