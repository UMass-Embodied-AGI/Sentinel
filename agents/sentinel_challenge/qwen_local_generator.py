import logging
from typing import Union, List

from PIL import Image

from tools.model_manager import global_model_manager


class QwenLocalGenerator:
    """
    Local Qwen generator that mimics tools.generator.Generator.generate,
    but routes requests through tools.model_manager's qwen_mm client.
    """

    def __init__(self, lm_id, server_port=8000, max_tokens=4096, temperature=0.0, top_p=1.0, logger=None):
        self.lm_id = lm_id
        self.server_port = server_port
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logger = logger or logging.getLogger(__name__)

        self._mm_client = global_model_manager.get_model("qwen_mm")
        self._mm_client.server_port = server_port

    def _normalize_images(self, img) -> List[Image.Image]:
        # Accept str, PIL.Image, or list of them → list[Image.Image]
        if img is None:
            return []
        if not isinstance(img, list):
            imgs = [img]
        else:
            imgs = img
        out: List[Image.Image] = []
        for each in imgs:
            if isinstance(each, Image.Image):
                out.append(each.convert("RGB"))
            else:
                out.append(Image.open(each).convert("RGB"))
        return out

    def generate(
        self,
        prompt,
        max_tokens=None,
        temperature=None,
        top_p=None,
        img=None,
        json_mode=False,
        chat_history=None,
        caller="none",
    ):
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p

        prompt = prompt + """You must NOT call any tools.
Do NOT output <tool_call>.
Respond ONLY with a valid JSON object, enclosed in ```json```.
No explanations.
No markdown.
No extra text."""

        # Flatten chat history into a single prompt string, similar to how prompts are written now
        if chat_history is not None:
            history_text = ""
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"[{role.upper()}]: {content}\n"
            full_prompt = history_text + f"[USER]: {prompt}"
        else:
            full_prompt = prompt

        images = self._normalize_images(img)

        # Route everything (text-only or multimodal) through qwen_mm; empty image list is allowed.
        # self.logger.debug(f"Generating with Qwen-VL: what happened here? images: {images}, img: {img}")
        texts = self._mm_client.multimodal_chat(
            full_prompt,
            images,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.logger.debug(f"Generating with Qwen-VL: and here? texts: {texts[0] if isinstance(texts, list) else texts}")
        return texts[0] if isinstance(texts, list) else texts

