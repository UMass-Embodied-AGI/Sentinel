import os
from typing import List, Union
from PIL import Image
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from transformers import AutoProcessor


class QwenMultimodalWrapper:
    def __init__(
        self,
        model_id: str,
        local_path: str = "/scratch4/workspace/xiangyelin_umass_edu-simple/weights",
        tensor_parallel_size: int = 1,
    ):
        self.model_id = model_id
        self.local_path = os.path.join(local_path, model_id)

        # Download if not exists
        if not self._weights_exist(self.local_path):
            print(f"Downloading {model_id}...")
            os.makedirs(self.local_path, exist_ok=True)
            snapshot_download(
                repo_id=model_id,
                local_dir=self.local_path,
                local_dir_use_symlinks=False,
            )
            print("Download complete.")
        else:
            print(f"Loading weights from {self.local_path}")

        # Processor (still needed for chat template)
        self.processor = AutoProcessor.from_pretrained(
            self.local_path,
            trust_remote_code=True,
        )

        # vLLM engine
        self.llm = LLM(
            model=self.local_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.85,
        )

    def _weights_exist(self, path: str) -> bool:
        required_files = ["config.json", "tokenizer_config.json"]
        return all(os.path.exists(os.path.join(path, f)) for f in required_files)

    def _to_pil(self, img: Union[str, Image.Image]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, str):
            return Image.open(img).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def _flatten_images(self, images):
        if images is None:
            return []
        if isinstance(images, list):
            result = []
            for i in images:
                if isinstance(i, list):
                    result.extend(i)
                else:
                    result.append(i)
            return result
        return [images]

    def chat_mm(self, texts, images, sampling_params):

        def get_param(sp, name, default):
            if isinstance(sp, dict):
                return sp.get(name, default)
            return getattr(sp, name, default)

        max_tokens = get_param(sampling_params, "max_tokens", 512)
        temperature = get_param(sampling_params, "temperature", 0.0)
        top_p = get_param(sampling_params, "top_p", 1.0)

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        requests = []

        for prompt, img_list in zip(texts, images):
            pil_images = [self._to_pil(i) for i in self._flatten_images(img_list)]

            if pil_images:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }]
            else:
                messages = [{"role": "user", "content": prompt}]

            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            request = {
                "prompt": prompt_text,
            }

            if pil_images:
                request["multi_modal_data"] = {
                    "image": pil_images
                }

            requests.append(request)

        # vLLM batched generation
        outputs = self.llm.generate(requests, sampling)

        results = []
        for out in outputs:
            results.append(out.outputs[0].text)

        return results