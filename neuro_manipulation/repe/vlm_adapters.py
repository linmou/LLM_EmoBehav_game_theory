from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import torch


@dataclass
class AdapterContext:
    """Context passed to adapters with ready resources.

    - processor: usually an AutoProcessor or image processor for VLMs
    - tokenizer: tokenizer instance (some processors provide this)
    - model: optional model reference (not required for preprocessing)
    """
    processor: Any
    tokenizer: Any
    model: Any = None


class BaseVLMAdapter:
    """Base adapter interface for processing multimodal inputs.

    Implementations should avoid heavy logic and rely on provided processor
    whenever possible to prevent token/feature mismatches.
    """

    # Identification keywords in model/tokenizer path
    MATCH_KEYWORDS: Sequence[str] = ()

    def matches(self, name_or_path: str) -> bool:
        low = (name_or_path or "").lower()
        return all(k in low for k in self.MATCH_KEYWORDS)

    # --- Policies ---
    def rep_token_policy(self) -> Union[int, str]:
        """Default rep token index or symbolic position for this model.
        Most models work with the last token (-1)."""
        return -1

    def optimal_layers(self) -> List[int]:
        """Heuristic default layers for steering extraction.
        Keep small and conservative; callers may override."""
        return [-1, -2, -3]

    def supports_images(self) -> bool:
        return True

    # --- Processing ---
    def process_multimodal(
        self,
        text: str,
        images: Optional[List[Any]],
        ctx: AdapterContext,
        **tokenizer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Default unified path: try processor chat template if available,
        otherwise pass text/images directly to processor. Implementations can
        override for model-specific message formatting.
        """
        processor = ctx.processor

        # Try universal messages format if processor supports apply_chat_template
        if hasattr(processor, "apply_chat_template"):
            messages = self._default_messages(text, images)
            formatted = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return processor(
                text=[formatted],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )

        # Fallback: rely on processor direct call
        return processor(
            text=[text] if isinstance(text, str) else text,
            images=images if images else None,
            padding=True,
            return_tensors="pt",
            **tokenizer_kwargs,
        )

    @staticmethod
    def _default_messages(text: str, images: Optional[List[Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text or ""})
        return [{"role": "user", "content": content}]


class QwenVLAdapter(BaseVLMAdapter):
    MATCH_KEYWORDS = ("qwen", "vl")

    def optimal_layers(self) -> List[int]:
        return [-5, -6, -7, -8, -9]

    def process_multimodal(
        self,
        text: str,
        images: Optional[List[Any]],
        ctx: AdapterContext,
        **tokenizer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        processor = ctx.processor

        # Build messages in Qwen expected format
        messages = self._default_messages(text, images)

        # Prefer qwen_vl_utils if available
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            formatted = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            return processor(
                text=[formatted],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
        except Exception:
            # Fallback to unified processor-only path
            if hasattr(processor, "apply_chat_template"):
                formatted = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return processor(
                    text=[formatted],
                    images=images if images else None,
                    padding=True,
                    return_tensors="pt",
                    **tokenizer_kwargs,
                )
            return super().process_multimodal(text, images, ctx, **tokenizer_kwargs)


class MiniCPMV4Adapter(BaseVLMAdapter):
    MATCH_KEYWORDS = ("minicpm", "v-4")

    def optimal_layers(self) -> List[int]:
        return [-3, -4, -5, -6]

    def process_multimodal(
        self,
        text: str,
        images: Optional[List[Any]],
        ctx: AdapterContext,
        **tokenizer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        processor = ctx.processor
        # MiniCPM-V-4 processors generally support unified calls; prefer template if present
        if hasattr(processor, "apply_chat_template"):
            messages = self._default_messages(text, images)
            formatted = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return processor(
                text=[formatted],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
        return super().process_multimodal(text, images, ctx, **tokenizer_kwargs)


class GLMEdgeV2bAdapter(BaseVLMAdapter):
    MATCH_KEYWORDS = ("glm", "edge", "v-2b")

    def optimal_layers(self) -> List[int]:
        return [-3, -4, -5]

    def process_multimodal(
        self,
        text: str,
        images: Optional[List[Any]],
        ctx: AdapterContext,
        **tokenizer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        processor = ctx.processor
        # Try to use chat template if available, otherwise direct call
        if hasattr(processor, "apply_chat_template"):
            messages = self._default_messages(text, images)
            formatted = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return processor(
                text=[formatted],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
        return super().process_multimodal(text, images, ctx, **tokenizer_kwargs)


class GemmaTextAdapter(BaseVLMAdapter):
    MATCH_KEYWORDS = ("gemma-3",)

    def supports_images(self) -> bool:
        return False

    def optimal_layers(self) -> List[int]:
        return [-3, -4, -5, -6]

    def process_multimodal(
        self,
        text: str,
        images: Optional[List[Any]],
        ctx: AdapterContext,
        **tokenizer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Ignore images gracefully; text-only
        tokenizer = getattr(ctx, "tokenizer", None) or getattr(ctx.processor, "tokenizer", None)
        if tokenizer is None:
            # Fall back to processor
            return ctx.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
        return tokenizer(text, return_tensors="pt", padding=True, **tokenizer_kwargs)


class PhiTextAdapter(BaseVLMAdapter):
    MATCH_KEYWORDS = ("phi-3.5", "mini", "instruct")

    def supports_images(self) -> bool:
        return False

    def optimal_layers(self) -> List[int]:
        return [-2, -3, -4]

    def process_multimodal(
        self,
        text: str,
        images: Optional[List[Any]],
        ctx: AdapterContext,
        **tokenizer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        tokenizer = getattr(ctx, "tokenizer", None) or getattr(ctx.processor, "tokenizer", None)
        if tokenizer is None:
            return ctx.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
        return tokenizer(text, return_tensors="pt", padding=True, **tokenizer_kwargs)


class GenericTextAdapter(BaseVLMAdapter):
    MATCH_KEYWORDS = ()

    def supports_images(self) -> bool:
        return False


class AdapterRegistry:
    """Simple registry to route to a best-effort adapter based on name patterns.

    Keep this local and stateless; detection is done per call.
    """

    def __init__(self) -> None:
        self.adapters: List[BaseVLMAdapter] = [
            QwenVLAdapter(),
            MiniCPMV4Adapter(),
            GLMEdgeV2bAdapter(),
            GemmaTextAdapter(),
            PhiTextAdapter(),
        ]

    def get(self, name_or_path: str) -> Optional[BaseVLMAdapter]:
        for adapter in self.adapters:
            try:
                if adapter.matches(name_or_path):
                    return adapter
            except Exception:
                continue
        return None

    def get_or_default(self, name_or_path: str, allow_images: bool = True) -> BaseVLMAdapter:
        adapter = self.get(name_or_path)
        if adapter is not None:
            if allow_images or not adapter.supports_images():
                return adapter
        # Fallback to generic
        return GenericTextAdapter()

