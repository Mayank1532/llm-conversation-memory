from pathlib import Path

from transformers import AutoTokenizer


class HuggingFaceTokenCounter:
    """Count tokens using a Hugging Face tokenizer."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model path does not exist: {model_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

    def count(self, messages: list[dict[str, str]]) -> int:
        if not messages:
            return 0

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
        )["input_ids"]

        return int(tokens.shape[1])
