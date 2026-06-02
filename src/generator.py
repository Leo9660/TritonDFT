# unified_generator.py
from typing import Optional, Tuple
import os


class UnifiedGenerator:
    """
    A thin adapter exposing a HF-like .__call__ interface for:
      - HF transformers.pipeline("text-generation")
      - vLLM (local python API)
      - OpenAI Responses API
      - Google Gemini API
      - Anthropic Claude API

    Returns: [{"generated_text": str}]

    When *backend* is ``"auto"`` (the default) the backend is inferred from the
    model name and any explicitly supplied API credentials.
    """

    _OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt")

    @staticmethod
    def _infer_backend(
        model: str,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ) -> str:
        """Resolve a backend name from *model* and optional credentials."""
        model_lower = model.lower()

        if openai_api_key or openai_base_url:
            return "openai"

        if any(model_lower.startswith(p) for p in UnifiedGenerator._OPENAI_PREFIXES):
            return "openai"

        if model_lower.startswith("claude"):
            return "claude"

        if "gemini" in model_lower:
            return "gemini"

        if "/" in model:
            return "vllm"

        raise ValueError(
            f"Cannot infer backend for model '{model}'. "
            f"Please specify backend explicitly: 'openai', 'claude', 'gemini', 'hf', or 'vllm'."
        )

    def __init__(
        self,
        model: str,
        backend: str = "auto",
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
        if backend.lower() == "auto":
            self.backend = self._infer_backend(model, openai_api_key, openai_base_url)
        else:
            self.backend = backend.lower()
        self.model = model
        self.default_max_new_tokens = default_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.verbose = verbose

        # token accounting
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.last_prompt_tokens = 0
        self.last_output_tokens = 0
        # USD billed by the Claude Code CLI backend (it reports exact cost per
        # call); used to bill credits precisely for claude_cli models.
        self.total_cost_usd = 0.0

        self._tokenizer = None

        # ---------------- HF ----------------
        if self.backend == "hf":
            from transformers import pipeline

            self.pipe = pipeline(
                "text-generation",
                model=model,
                device_map=hf_device_map,
                dtype=hf_dtype,
            )
            self._tokenizer = getattr(self.pipe, "tokenizer", None)
            if self.verbose:
                print(f"[UnifiedGenerator] HF backend @ {model}")

        # ---------------- vLLM ----------------
        elif self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
            except Exception as e:
                raise RuntimeError("vLLM not installed: pip install vllm") from e

            init_kwargs = {
                "model": model,
                "gpu_memory_utilization": vllm_gpu_mem_util,
            }
            if vllm_tensor_parallel_size is not None:
                init_kwargs["tensor_parallel_size"] = vllm_tensor_parallel_size

            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.llm = LLM(**init_kwargs)

            try:
                self._tokenizer = self.llm.get_tokenizer()
            except Exception:
                self._tokenizer = None

            if self.verbose:
                print(f"[UnifiedGenerator] vLLM backend @ {model}")

        # ---------------- OpenAI ----------------
        elif self.backend == "openai":
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError("OpenAI SDK not installed: pip install openai") from e

            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY is required.")

            self._oa_client = OpenAI(api_key=api_key, base_url=base_url)
            if self.verbose:
                print(f"[UnifiedGenerator] OpenAI backend @ {model}")

        # ---------------- Gemini ----------------
        elif self.backend == "gemini":
            try:
                import google.generativeai as genai
            except Exception as e:
                raise RuntimeError("Gemini SDK not installed: pip install google-generativeai") from e

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("GEMINI_API_KEY is required.")

            genai.configure(api_key=api_key)
            self._genai = genai
            self._gemini_model = genai.GenerativeModel(model)
            if self.verbose:
                print(f"[UnifiedGenerator] Gemini backend @ {model}")

        # ---------------- Claude ----------------
        elif self.backend == "claude":
            try:
                from anthropic import Anthropic
            except Exception as e:
                raise RuntimeError("Anthropic SDK not installed: pip install anthropic") from e

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is required.")

            self._claude_client = Anthropic(api_key=api_key)
            if self.verbose:
                print(f"[UnifiedGenerator] Claude backend @ {model}")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ============================================================

    def __call__(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        return_full_text: bool = False,
    ):
        max_new_tokens = max_new_tokens or self.default_max_new_tokens

        # ---------- Claude via the Claude Code CLI (any "claude*" model) ----------
        # Routed regardless of the configured backend so a single agent can mix
        # OpenAI (gpt-*) and Claude (claude-*) models per job.
        if self.model and self.model.lower().startswith("claude"):
            return self._call_claude_cli(prompt, max_new_tokens)

        # ---------- HF ----------
        if self.backend == "hf":
            out = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                return_full_text=return_full_text,
                do_sample=(self.temperature > 0.0),
                temperature=self.temperature,
                top_p=self.top_p,
            )
            text = out[0].get("generated_text", "") if out else ""
            self._update_token_counts(prompt, text)
            return out

        # ---------- vLLM ----------
        elif self.backend == "vllm":
            sp = self.SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_new_tokens,
                seed=self.seed,
            )
            outs = self.llm.generate([prompt], sp)
            text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
            self._update_token_counts(prompt, text)
            return [{"generated_text": text}]

        # ---------- OpenAI ----------
        elif self.backend == "openai":
            resp = self._oa_client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_new_tokens,
            )

            text = getattr(resp, "output_text", "") or ""
            usage = getattr(resp, "usage", None)
            self._update_token_counts(prompt, text, usage)
            return [{"generated_text": text}]

        # ---------- Gemini ----------
        elif self.backend == "gemini":
            resp = self._gemini_model.generate_content(
                prompt,
                generation_config={
                    # "temperature": self.temperature,
                    # "top_p": self.top_p,
                    "max_output_tokens": max_new_tokens,
                },
            )

            # Never use resp.text (may raise ValueError when no text parts are returned)
            text = self._extract_text_from_gemini(resp)

            # Optional: print finish_reason / safety info when empty or verbose
            if self.verbose or not text:
                self._debug_gemini_response(resp)

            usage = getattr(resp, "usage_metadata", None)
            self._update_token_counts(prompt, text, usage)
            return [{"generated_text": text}]

        # ---------- Claude ----------
        # Default decoding
        elif self.backend == "claude":
            resp = self._claude_client.messages.create(
                model=self.model,
                max_tokens=max_new_tokens,
                # temperature=self.temperature,
                # top_p=self.top_p,
                messages=[{"role": "user", "content": prompt}],
            )

            parts = []
            for blk in resp.content:
                if blk.type == "text":
                    parts.append(blk.text)
            text = "".join(parts)
            usage = getattr(resp, "usage", None)
            self._update_token_counts(prompt, text, usage)
            return [{"generated_text": text}]

        else:
            raise RuntimeError("Invalid backend")

    # ============================================================

    # ---------------- Gemini helpers ----------------
    def _extract_text_from_gemini(self, resp) -> str:
        """Safely extract text from google-generativeai response without using resp.text."""
        if resp is None:
            return ""

        texts = []

        # Preferred: iterate candidates -> content -> parts
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    texts.append(t)

        if texts:
            return "\n".join(texts)

        # Fallback: some SDK versions may expose "content" at top-level
        content = getattr(resp, "content", None)
        if content is not None:
            parts = getattr(content, "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    texts.append(t)

        return "\n".join(texts) if texts else ""

    def _debug_gemini_response(self, resp) -> None:
        """Verbose debug info to understand empty outputs / finish reasons."""
        if not self.verbose or resp is None:
            return

        try:
            candidates = getattr(resp, "candidates", None) or []
            for i, cand in enumerate(candidates):
                fr = getattr(cand, "finish_reason", None)
                print(f"[UnifiedGenerator][Gemini] cand[{i}].finish_reason={fr}")
                sr = getattr(cand, "safety_ratings", None)
                if sr is not None:
                    print(f"[UnifiedGenerator][Gemini] cand[{i}].safety_ratings={sr}")
        except Exception as e:
            print(f"[UnifiedGenerator][Gemini] debug failed: {e}")

        try:
            pf = getattr(resp, "prompt_feedback", None)
            if pf is not None:
                print(f"[UnifiedGenerator][Gemini] prompt_feedback={pf}")
        except Exception as e:
            print(f"[UnifiedGenerator][Gemini] prompt_feedback read failed: {e}")

        try:
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                print(f"[UnifiedGenerator][Gemini] usage_metadata={um}")
        except Exception as e:
            print(f"[UnifiedGenerator][Gemini] usage_metadata read failed: {e}")

    # ---------------- Claude Code CLI ----------------
    def _call_claude_cli(self, prompt: str, max_new_tokens: int):
        """Generate via the Claude Code CLI in headless mode.

        Uses the official client (authenticated by CLAUDE_CODE_OAUTH_TOKEN in the
        environment) with ALL tools disabled, so it behaves as a plain text
        generator and a user query can never drive it to touch the container.
        The CLI reports exact USD cost per call, which we accumulate for precise
        credit billing.
        """
        import subprocess
        import json as _json

        sys_prompt = (
            "You are a precise text/JSON generator inside an automated DFT "
            "pipeline. Follow the user's instructions exactly and output only "
            "what is requested — no tools, no preamble, no markdown code fences."
        )
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--allowed-tools", "",
            "--system-prompt", sys_prompt,
        ]
        if self.model:
            cmd += ["--model", self.model]

        text = ""
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, env=os.environ.copy()
            )
        except Exception as e:
            if self.verbose:
                print(f"[claude_cli] subprocess failed: {e}")
            return [{"generated_text": ""}]

        if not res.stdout:
            if self.verbose:
                print(f"[claude_cli] empty stdout; stderr={(res.stderr or '')[:300]}")
            return [{"generated_text": ""}]

        try:
            data = _json.loads(res.stdout)
            text = data.get("result", "") or ""
            usage = data.get("usage") or {}
            pt = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0)
            )
            ot = int(usage.get("output_tokens", 0) or 0)
            self.last_prompt_tokens = pt
            self.last_output_tokens = ot
            self.total_prompt_tokens += pt
            self.total_output_tokens += ot
            cost = data.get("total_cost_usd")
            if isinstance(cost, (int, float)):
                self.total_cost_usd += float(cost)
            if self.verbose and data.get("is_error"):
                print(f"[claude_cli] api_error_status={data.get('api_error_status')}")
        except Exception as e:
            if self.verbose:
                print(f"[claude_cli] JSON parse failed: {e}; using raw stdout")
            text = res.stdout

        return [{"generated_text": text}]

    def reset_token_counters(self) -> None:
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.last_prompt_tokens = 0
        self.last_output_tokens = 0
        self.total_cost_usd = 0.0

    def _update_token_counts(
        self,
        prompt: str,
        output: str,
        usage: Optional[object] = None,
    ) -> None:
        pt, ot = self._count_tokens(prompt, output, usage)
        self.last_prompt_tokens = pt
        self.last_output_tokens = ot
        self.total_prompt_tokens += pt
        self.total_output_tokens += ot

    def _count_tokens(
        self,
        prompt: str,
        output: str,
        usage: Optional[object] = None,
    ) -> Tuple[int, int]:

        if usage is not None:
            # OpenAI / Claude
            pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            ot = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)

            # Gemini
            if pt is None:
                pt = getattr(usage, "prompt_token_count", None)
            if ot is None:
                ot = getattr(usage, "candidates_token_count", None)

            if pt is not None and ot is not None:
                return int(pt), int(ot)

        return self._count_tokens_text(prompt), self._count_tokens_text(output)

    def _count_tokens_text(self, text: str) -> int:
        if not text:
            return 0

        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass

        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return len(text.split())
