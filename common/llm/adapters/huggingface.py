import asyncio
import logging
import os
from functools import partial
from typing import Any, TypeVar

from pydantic import BaseModel

from common.settings import get_settings

settings = get_settings()
T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# Module-level cache for loaded models
_model_cache: dict[str, tuple[Any, Any]] = {}


def setup_huggingface_auth() -> bool:
    """Setup HuggingFace authentication from settings or environment.

    Returns:
        True if authenticated successfully, False otherwise.
    """
    from huggingface_hub import login, whoami

    # Check for token in settings first, then environment
    hf_token = settings.HF_TOKEN or os.environ.get("HF_TOKEN")

    if not hf_token:
        logger.warning("HF_TOKEN not found in settings or environment. Gated models will not be accessible.")
        print("[HuggingFace Auth] WARNING: HF_TOKEN not found - gated models will fail to download")  # noqa: T201
        return False

    try:
        # Login with token
        login(token=hf_token, add_to_git_credential=False)
        user_info = whoami()
        logger.info("HuggingFace authenticated as: %s", user_info.get("name", "unknown"))
        print(f"[HuggingFace Auth] SUCCESS: Authenticated as {user_info.get('name', 'unknown')}")  # noqa: T201
        return True
    except Exception as e:
        logger.exception("HuggingFace authentication failed: %s", e)
        print(f"[HuggingFace Auth] FAILED: {e}")  # noqa: T201
        return False


class HuggingFaceModelAdapter:
    """Adapter for HuggingFace transformers models like MedGemma.

    This adapter supports multimodal models that use AutoModelForImageTextToText
    and AutoProcessor, running them in text-only mode for chat completion.

    The model is lazy-loaded on first use and cached at the module level to
    avoid reloading on subsequent requests.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
    ) -> None:
        self._model_id = model
        self._device = device
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _is_quantized_model(self) -> bool:
        """Check if this is a quantized model (e.g., bitsandbytes 4-bit)."""
        quantized_indicators = ["bnb-4bit", "4bit", "bnb-8bit", "8bit", "gptq", "awq"]
        model_lower = self._model_id.lower()
        return any(indicator in model_lower for indicator in quantized_indicators)

    def _is_text_only_model(self) -> bool:
        """Check if this is a text-only model (not multimodal).

        Text-only models like Qwen3, Llama, etc. use AutoModelForCausalLM.
        Multimodal models like MedGemma use AutoModelForImageTextToText.
        """
        # Known text-only model families
        text_only_indicators = [
            "qwen",
            "llama",
            "mistral",
            "phi",
            "gemma-2",  # Note: gemma-2 is text-only, medgemma is multimodal
            "smollm",
            "olmo",
        ]
        model_lower = self._model_id.lower()

        # MedGemma is explicitly multimodal
        if "medgemma" in model_lower:
            return False

        return any(indicator in model_lower for indicator in text_only_indicators)

    def _load_model(self) -> None:
        """Lazy-load model on first use."""
        if self._model_id in _model_cache:
            logger.info("Using cached model: %s", self._model_id)
            print(f"[HuggingFace LLM] Using cached model: {self._model_id}")  # noqa: T201
            self._model, self._processor = _model_cache[self._model_id]
            return

        # Import here to avoid loading torch/transformers if not using this adapter
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        logger.info("Loading HuggingFace model: %s", self._model_id)
        print(f"[HuggingFace LLM] Loading model: {self._model_id}")  # noqa: T201

        # Ensure HuggingFace auth is set up before downloading gated models
        setup_huggingface_auth()

        # Determine if this is a text-only or multimodal model
        is_text_only = self._is_text_only_model()
        print(f"[HuggingFace LLM] Model type: {'text-only' if is_text_only else 'multimodal'}")  # noqa: T201

        try:
            print(f"[HuggingFace LLM] Downloading processor/tokenizer for {self._model_id}...")  # noqa: T201
            if is_text_only:
                # Text-only models use AutoTokenizer
                self._processor = AutoTokenizer.from_pretrained(self._model_id)
            else:
                # Multimodal models use AutoProcessor
                self._processor = AutoProcessor.from_pretrained(self._model_id)
            print("[HuggingFace LLM] Processor/tokenizer loaded successfully")  # noqa: T201
        except Exception as e:
            logger.exception("Failed to load processor/tokenizer: %s", e)
            print(f"[HuggingFace LLM] FAILED to load processor/tokenizer: {e}")  # noqa: T201
            raise

        # Check if this is a quantized model
        is_quantized = self._is_quantized_model()

        # Determine device mapping
        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[HuggingFace LLM] Target device: {device}, CUDA available: {torch.cuda.is_available()}")  # noqa: T201
        if is_quantized:
            print("[HuggingFace LLM] Detected quantized model, using device_map='auto'")  # noqa: T201

        # Select the appropriate model class
        model_class = AutoModelForCausalLM if is_text_only else AutoModelForImageTextToText

        try:
            print(f"[HuggingFace LLM] Downloading model {self._model_id} using {model_class.__name__}...")  # noqa: T201

            if is_quantized:
                # For quantized models (e.g., bitsandbytes 4-bit), use device_map="auto"
                # This lets the quantization library handle device placement
                self._model = model_class.from_pretrained(
                    self._model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            else:
                # For non-quantized models, load and move to device explicitly
                self._model = model_class.from_pretrained(
                    self._model_id,
                    torch_dtype=torch.bfloat16,
                ).to(device)

            print(f"[HuggingFace LLM] Model loaded successfully on device: {self._model.device}")  # noqa: T201
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            print(f"[HuggingFace LLM] FAILED to load model: {e}")  # noqa: T201
            raise

        _model_cache[self._model_id] = (self._model, self._processor)
        logger.info("Model loaded successfully on device: %s", self._model.device)

    def _normalize_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Normalize messages for models that require strict role alternation.

        MedGemma's chat template requires alternating user/assistant roles.
        This method:
        1. Merges consecutive system messages into one
        2. Prepends system content to the first user message
        3. Ensures strict user/assistant alternation
        """
        if not messages:
            return messages

        # Separate system messages and conversation messages
        system_parts = []
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append(msg)

        # If we have system content, prepend it to the first user message
        if system_parts and conversation:
            system_text = "\n\n".join(system_parts)
            if conversation[0]["role"] == "user":
                conversation[0] = {
                    "role": "user",
                    "content": f"{system_text}\n\n{conversation[0]['content']}",
                }
            else:
                # Insert a user message with system content at the start
                conversation.insert(0, {"role": "user", "content": system_text})

        # Ensure conversation starts with user and alternates
        normalized = []
        expected_role = "user"
        for msg in conversation:
            if msg["role"] == expected_role:
                normalized.append(msg)
                expected_role = "assistant" if expected_role == "user" else "user"
            elif msg["role"] == "user" and expected_role == "user":
                # Merge consecutive user messages
                if normalized:
                    normalized[-1]["content"] += "\n\n" + msg["content"]
                else:
                    normalized.append(msg)
            elif msg["role"] == "assistant" and expected_role == "assistant":
                # Merge consecutive assistant messages
                if normalized:
                    normalized[-1]["content"] += "\n\n" + msg["content"]

        return normalized

    def _get_thinking_token_ids(self) -> tuple[int | None, int | None]:
        """Get thinking token IDs from the tokenizer if available.

        Some models (e.g., Qwen3, SmolLM3) have dedicated thinking tokens.
        See: https://huggingface.co/docs/transformers/en/chat_templating

        Returns:
            Tuple of (begin_thinking_token_id, end_thinking_token_id), or (None, None)
            if the model doesn't have thinking tokens.
        """
        tokenizer = self._processor.tokenizer if hasattr(self._processor, "tokenizer") else self._processor

        # Check for thinking tokens - try common formats
        think_tokens = [
            ("<think>", "</think>"),  # Qwen3, SmolLM3 format
            ("<thinking>", "</thinking>"),  # Alternative format
        ]

        for begin_token, end_token in think_tokens:
            begin_id = tokenizer.convert_tokens_to_ids(begin_token)
            end_id = tokenizer.convert_tokens_to_ids(end_token)

            # If both tokens exist and aren't the unknown token, we have thinking support
            unk_id = tokenizer.unk_token_id
            if begin_id != unk_id and end_id != unk_id:
                logger.info("Found thinking tokens: %s=%d, %s=%d", begin_token, begin_id, end_token, end_id)
                return begin_id, end_id

        return None, None

    def _strip_thinking_tokens(self, token_ids: list[int], begin_id: int, end_id: int) -> list[int]:
        """Remove thinking blocks from generated token IDs.

        Args:
            token_ids: List of generated token IDs
            begin_id: Token ID for start of thinking block
            end_id: Token ID for end of thinking block

        Returns:
            Token IDs with thinking blocks removed
        """
        result = []
        in_thinking = False

        for token_id in token_ids:
            if token_id == begin_id:
                in_thinking = True
            elif token_id == end_id:
                in_thinking = False
            elif not in_thinking:
                result.append(token_id)

        return result

    def _generate_sync(self, messages: list[dict[str, str]]) -> str:
        """Synchronous generation (runs in thread pool)."""
        import torch

        self._load_model()

        # Normalize messages for strict role alternation (required by MedGemma)
        normalized_messages = self._normalize_messages(messages)

        # Check if this is a text-only model
        is_text_only = self._is_text_only_model()

        if is_text_only:
            # Text-only models (Qwen3, Llama, etc.) use simple message format
            formatted_messages = normalized_messages
        else:
            # Multimodal models (MedGemma) expect content as a list of content parts
            formatted_messages = [
                {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]} for msg in normalized_messages
            ]

        # Check if tokenizer supports enable_thinking parameter
        # Some models (Qwen3, SmolLM3) support this to disable chain-of-thought
        import inspect

        template_sig = inspect.signature(self._processor.apply_chat_template)
        supports_enable_thinking = "enable_thinking" in template_sig.parameters

        # Get thinking setting from config
        enable_thinking = settings.LOCAL_LLM_ENABLE_THINKING

        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }

        if supports_enable_thinking:
            # Pass through the thinking setting for models that support it (e.g., Qwen3)
            template_kwargs["enable_thinking"] = enable_thinking
            logger.info("Model supports enable_thinking, setting to %s", enable_thinking)

        inputs = self._processor.apply_chat_template(formatted_messages, **template_kwargs)

        # Move inputs to device - BatchEncoding (from tokenizer) doesn't support dtype arg
        if is_text_only:
            # Text-only models use tokenizer which returns BatchEncoding
            inputs = inputs.to(self._model.device)
        else:
            # Multimodal models use processor which may return tensors that support dtype
            inputs = inputs.to(self._model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Check for thinking tokens to strip from output (only if thinking is disabled)
        begin_thinking_id, end_thinking_id = None, None
        if not enable_thinking:
            begin_thinking_id, end_thinking_id = self._get_thinking_token_ids()

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._temperature > 0,
                temperature=self._temperature if self._temperature > 0 else None,
            )
            generation = generation[0][input_len:]

        # Strip thinking tokens if thinking is disabled and the model has dedicated tokens
        if begin_thinking_id is not None and end_thinking_id is not None:
            generation_list = generation.tolist()
            generation_list = self._strip_thinking_tokens(generation_list, begin_thinking_id, end_thinking_id)
            generation = torch.tensor(generation_list, device=generation.device)

        response = self._processor.decode(generation, skip_special_tokens=True)

        # Strip any XML-style thinking tags from the text response (only if thinking is disabled)
        if not enable_thinking:
            response = self._strip_thinking_xml_tags(response)

        return response

    def _strip_thinking_xml_tags(self, text: str) -> str:
        """Strip XML-style thinking tags from model output.

        Some models output chain-of-thought reasoning in XML tags like:
        - <thought>...</thought>
        - <thinking>...</thinking>
        - <think>...</think>

        These tags may not have dedicated token IDs but appear in the text output.
        The browser will render unknown HTML tags, showing their content.

        Args:
            text: Decoded model output that may contain thinking XML tags.

        Returns:
            The response with thinking XML tags and their content removed.
        """
        import re

        original_len = len(text)

        # Remove <thought>...</thought> tags and content
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove <thinking>...</thinking> tags and content
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove <think>...</think> tags and content
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        text = text.strip()

        if len(text) < original_len:
            logger.info("Stripped thinking XML tags from output (%d -> %d chars)", original_len, len(text))

        return text

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """Run inference asynchronously via thread pool.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The generated text response.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._generate_sync, messages))

    def _extract_json(self, text: str) -> str:
        """Extract JSON from model output that may contain other text.

        The model may output thinking/reasoning before the JSON, or wrap it in markdown.
        This method attempts to extract the JSON portion.
        """
        import json
        import re

        # Try to parse as-is first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        code_block_patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        ]
        for pattern in code_block_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        # Try to find JSON object or array in the text
        # Look for { ... } or [ ... ]
        json_patterns = [
            r"(\{[\s\S]*\})",  # Object
            r"(\[[\s\S]*\])",  # Array
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for candidate in reversed(matches):  # Prefer later matches (often the actual output)
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        # Return original if no JSON found - will fail at validation
        return text

    async def structured_chat(self, messages: list[dict[str, str]], response_format: type[T]) -> T:
        """Generate structured output by prompting for JSON.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            response_format: A Pydantic model class to parse the response into.

        Returns:
            An instance of response_format parsed from the model's JSON output.
        """
        # Add JSON schema instruction to system prompt
        schema = response_format.model_json_schema()
        json_instruction = {
            "role": "system",
            "content": f"Respond with valid JSON matching this schema:\n{schema}\nOutput only the JSON, no other text.",
        }
        augmented_messages = [json_instruction, *messages]

        response = await self.chat(augmented_messages)

        # Extract JSON from response (model may include thinking/markdown)
        json_str = self._extract_json(response)

        # Parse and validate response
        return response_format.model_validate_json(json_str)
