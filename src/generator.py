# a unified generator adapter for HF, vLLM, and OpenAI ---
from typing import Optional
import os

class UnifiedGenerator:
    """
    A thin adapter exposing a HF-like .__call__ interface for:
      - HF transformers.pipeline("text-generation")
      - vLLM (local python API)
      - OpenAI Responses API
    Returns: [{"generated_text": str}]
    """
    def __init__(
        self,
        backend: str,
        model: str,
        hf_device_map: str = "auto",
        hf_dtype: str = "auto",
        vllm_gpu_mem_util: float = 0.9,
        vllm_tensor_parallel_size: Optional[int] = None,
        default_max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = False,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.default_max_new_tokens = default_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.verbose = verbose
        self.model = model

        if self.backend == "hf":
            from transformers import pipeline
            self.pipe = pipeline(
                "text-generation",
                model=model,
                device_map=hf_device_map,
                dtype=hf_dtype,
            )
            if self.verbose:
                print(f"[UnifiedGenerator] Using HF transformers pipeline @ {model}")

        elif self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
            except Exception as e:
                raise RuntimeError(
                    "vLLM not installed or import failed. `pip install vllm`"
                ) from e
            init_kwargs = {
                "model": model,
                "gpu_memory_utilization": vllm_gpu_mem_util,
            }
            if vllm_tensor_parallel_size is not None:
                init_kwargs["tensor_parallel_size"] = vllm_tensor_parallel_size

            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.llm = LLM(**init_kwargs)

            if self.verbose:
                print(f"[UnifiedGenerator] Using vLLM @ {model} "
                      f"(gpu_mem_util={vllm_gpu_mem_util}, tp={vllm_tensor_parallel_size})")

        elif self.backend == "openai":
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e

            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")  # 可选
            if api_key is None:
                raise ValueError("OPENAI_API_KEY is required for backend='openai'.")

            self._oa_client = OpenAI(api_key=api_key, base_url=base_url)
            if self.verbose:
                print(f"[UnifiedGenerator] Using OpenAI Responses API @ {model} (base_url={base_url or 'default'})")

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'hf', 'vllm', or 'openai'.")

    def __call__(self, prompt: str, max_new_tokens: Optional[int] = None, return_full_text: bool = False):
        """
        Returns: list[dict] -> [{"generated_text": "..."}]
        """
        max_new_tokens = max_new_tokens or self.default_max_new_tokens

        if self.backend == "hf":
            return self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                return_full_text=return_full_text,
                do_sample=(self.temperature > 0.0),
                temperature=self.temperature,
                top_p=self.top_p,
            )

        elif self.backend == "vllm":
            sp = self.SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_new_tokens,
                seed=self.seed,
            )
            outs = self.llm.generate([prompt], sp)
            o = outs[0]
            text = o.outputs[0].text if o.outputs else ""
            return [{"generated_text": text}]

        elif self.backend == "openai":
            resp = self._oa_client.responses.create(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=max_new_tokens,
                # seed=self.seed,
            )
            text = getattr(resp, "output_text", None)
            if text is None:
                try:
                    parts = []
                    for item in getattr(resp, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", "") == "output_text" and hasattr(c, "text"):
                                parts.append(getattr(c.text, "value", "") or "")
                    text = "".join(parts) if parts else ""
                except Exception:
                    text = ""
            return [{"generated_text": text}]
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'hf', 'vllm', or 'openai'.")
