from pathlib import Path
import time

from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    """Local Hugging Face LLM responsible only for model inference."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {self.model_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 64,
    ) -> str:
        """Generate an assistant response from the supplied messages."""

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        start = time.perf_counter()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generation_time = time.perf_counter() - start

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        print(f"Generation time: {generation_time:.2f}s")

        return response
