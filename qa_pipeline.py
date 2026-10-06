"""Hugging Face pipeline integration with an automatic Llama-compatible fallback."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from transformers import __version__ as transformers_version
from transformers import pipeline
from transformers.pipelines import get_supported_tasks

import config
from utils import LOGGER, get_device_info, safe_filename


class QuestionAnsweringService:
    """Load a QA pipeline, run single/batch questions, and save supported models."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        token: str | None = None,
        fallback_model_name: str = config.FALLBACK_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self.active_model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.token = token or config.HF_TOKEN
        self.fallback_reason = ""
        self.pipe: Any | None = None
        self.pipeline_task = "not loaded"
        self.mode = "not loaded"
        self.device_info = get_device_info()

    @property
    def pipeline_device(self) -> int:
        """Return pipeline's PyTorch device index (-1 means CPU)."""
        return 0 if self.device_info.get("torch_cuda") else -1

    def _pipeline_kwargs(self, model_name: str) -> dict[str, Any]:
        """Build PyTorch inference arguments while keeping authentication out of logs.

        TensorFlow remains available for environment/GPU detection, but explicitly
        selecting PyTorch prevents Transformers from attempting the unsupported
        Keras 3 backend when both frameworks are installed.
        """
        kwargs: dict[str, Any] = {
            "model": model_name,
            "device": self.pipeline_device,
            "framework": "pt",
        }
        if self.token:
            kwargs["token"] = self.token
        return kwargs

    @staticmethod
    def _is_access_error(error: Exception) -> bool:
        """Recognise Hub access failures that cannot be fixed by task fallback."""
        message = str(error).lower()
        indicators = (
            "gated repo", "access is restricted", "401 client error", "403 client error",
            "you must have access", "not authorized",
        )
        return any(indicator in message for indicator in indicators)

    def _load_public_fallback(self, reason: str) -> None:
        """Load the configured public extractive QA fallback after source failure."""
        if self.fallback_model_name == self.model_name:
            raise RuntimeError(reason)
        try:
            LOGGER.warning("Loading public QA fallback %s: %s", self.fallback_model_name, reason)
            self.pipe = pipeline(
                "question-answering", **self._pipeline_kwargs(self.fallback_model_name)
            )
            self.active_model_name = self.fallback_model_name
            self.pipeline_task = "question-answering"
            self.mode = "extractive question-answering fallback"
            self.fallback_reason = reason
        except Exception as fallback_error:
            message = (
                f"Could not load requested model '{self.model_name}' or public fallback "
                f"'{self.fallback_model_name}'. Requested model error: {reason}; "
                f"fallback error: {fallback_error}"
            )
            LOGGER.exception(message)
            raise RuntimeError(message) from fallback_error

    @staticmethod
    def _ensure_qa_pipeline_supported() -> None:
        """Fail clearly if a Transformers major version removed the QA task."""
        if "question-answering" not in get_supported_tasks():
            message = (
                f"Transformers {transformers_version} does not provide the required "
                "'question-answering' Pipeline API. Install the project-pinned "
                "Transformers 4.57.6 release, then restart Streamlit."
            )
            LOGGER.error(message)
            raise RuntimeError(message)

    def load(self) -> None:
        """Load extractive QA first; use text generation for causal Llama checkpoints."""
        if self.pipe is not None:
            return
        self._ensure_qa_pipeline_supported()
        kwargs = self._pipeline_kwargs(self.model_name)
        try:
            LOGGER.info("Loading question-answering model: %s", self.model_name)
            self.pipe = pipeline("question-answering", **kwargs)
            self.pipeline_task = "question-answering"
            self.mode = "extractive question-answering"
        except Exception as qa_error:
            if self._is_access_error(qa_error):
                self._load_public_fallback(
                    "The requested Hugging Face model is gated or inaccessible. "
                    "Using the configured public QA fallback."
                )
                LOGGER.info("Fallback model loaded on %s", self.device_info["device"])
                return
            # Decoder-only Llama weights cannot instantiate AutoModelForQuestionAnswering.
            # Generation is the closest Pipeline API approach for such checkpoints.
            LOGGER.warning("QA pipeline unavailable (%s); attempting text-generation fallback.", qa_error)
            try:
                self.pipe = pipeline("text-generation", **kwargs)
                self.pipeline_task = "text-generation"
                self.mode = "generative extraction fallback"
            except Exception as generation_error:
                self._load_public_fallback(
                    "The requested model supports neither the configured QA pipeline nor "
                    f"the generation fallback. QA error: {qa_error}; generation error: {generation_error}"
                )
        LOGGER.info("Model loaded in %s mode on %s", self.mode, self.device_info["device"])

    def _trim_context(self, context: str) -> str:
        """Token-trim context to configured length when a tokenizer is available."""
        if not self.pipe:
            return context
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            return context
        token_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(token_ids) <= config.MAX_CONTEXT_LENGTH:
            return context
        LOGGER.warning("Context trimmed from %d to %d tokens", len(token_ids), config.MAX_CONTEXT_LENGTH)
        return tokenizer.decode(token_ids[: config.MAX_CONTEXT_LENGTH], skip_special_tokens=True)

    def answer(self, question: str, context: str) -> dict[str, Any]:
        """Answer one question and always return a result-shaped dictionary."""
        question, context = question.strip(), context.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if not context:
            raise ValueError("Context cannot be empty.")
        self.load()
        assert self.pipe is not None
        context = self._trim_context(context)
        started = time.perf_counter()
        try:
            if self.pipeline_task == "question-answering":
                # Transformers 4.x QA pipelines accept question/context keyword
                # inputs and internally convert them to a SQuAD-style example.
                response = self.pipe(question=question, context=context)
                result = {
                    "question": question, "answer": response.get("answer", ""),
                    "confidence": float(response.get("score", 0.0)),
                    "start": response.get("start"), "end": response.get("end"),
                    "mode": self.mode, "error": "",
                }
            else:
                prompt = (
                    "Answer the question using only the context. Return a short direct answer.\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                )
                response = self.pipe(prompt, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
                generated = response[0]["generated_text"]
                answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
                result = {
                    "question": question, "answer": answer, "confidence": None,
                    "start": None, "end": None, "mode": self.mode,
                    "error": "Confidence is not available for generative extraction.",
                }
            result["processing_time_seconds"] = round(time.perf_counter() - started, 4)
            LOGGER.info("Answered question=%r confidence=%s time=%ss", question, result["confidence"], result["processing_time_seconds"])
            return result
        except Exception as exc:
            LOGGER.exception("Inference failed for question=%r", question)
            raise RuntimeError(f"Inference failed: {exc}") from exc

    def answer_batch(self, questions: list[str], context: str) -> list[dict[str, Any]]:
        """Answer each question independently so one failure does not abort the batch."""
        if not questions:
            raise ValueError("Provide at least one batch question.")
        results = []
        for question in questions:
            try:
                results.append(self.answer(question, context))
            except (ValueError, RuntimeError) as exc:
                results.append({"question": question, "answer": "", "confidence": None,
                                "processing_time_seconds": 0.0, "start": None, "end": None,
                                "mode": self.mode, "error": str(exc)})
        return results

    def save_model(self) -> Path:
        """Save loaded pipeline artifacts locally, when supported by its components."""
        self.load()
        assert self.pipe is not None
        destination = config.MODEL_DIR / safe_filename(self.model_name)
        try:
            self.pipe.save_pretrained(destination)
        except Exception as exc:
            LOGGER.exception("Unable to save model")
            raise RuntimeError(f"Model could not be saved: {exc}") from exc
        LOGGER.info("Model saved to %s", destination)
        return destination

    def info(self) -> dict[str, Any]:
        """Return UI-safe metadata without forcing an expensive model load."""
        return {"requested_model": self.model_name, "active_model": self.active_model_name,
                "mode": self.mode, "fallback_reason": self.fallback_reason,
                "pipeline_task": self.pipeline_task,
                "device": self.device_info["device"],
                "torch_cuda": self.device_info.get("torch_cuda"),
                "tensorflow_gpus": self.device_info.get("tensorflow_gpus", [])}
