from __future__ import annotations
from dataclasses import dataclass
import torch

TEACHER_MODEL = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = """Eres un profesor que genera ejemplos de entrenamiento para ButterflyAI, un asistente local en desarrollo.
Responde principalmente en espanol salvo que el usuario use otro idioma. Usa lenguaje natural, claro y variado.
No digas que eres Qwen ni suplantes a Butterfly. No muestres cadena de pensamiento privada.
Si falta informacion, no la inventes. Distingue hechos, inferencias y dudas cuando corresponda.
Para tareas de computadora, prioriza metodos estructurados y eficientes sobre clicks visuales cuando sea posible.
Para saludos o charla casual, responde como una conversacion normal y no conviertas todo en un plan de trabajo.
"""

@dataclass
class Teacher:
    model_name: str = TEACHER_MODEL

    def __post_init__(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Falta transformers. Ejecuta SETUP_WINDOWS.bat.") from e
        print(f"Loading local teacher: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        kwargs = {"torch_dtype": "auto"}
        if torch.cuda.is_available(): kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if not torch.cuda.is_available(): self.model = self.model.to("cpu")
        self.model.eval(); self.device = next(self.model.parameters()).device
        print(f"Teacher device: {self.device}")

    def _format(self, prompt: str):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.inference_mode()
    def answer_batch(self, prompts: list[str], max_new_tokens: int = 110) -> list[str]:
        texts = [self._format(p) for p in prompts]
        old_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.72, top_p=0.85, top_k=30,
            repetition_penalty=1.10, pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        answers = [self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in outputs]
        self.tokenizer.padding_side = old_padding
        return answers

    def answer(self, prompt: str, max_new_tokens: int = 110) -> str:
        return self.answer_batch([prompt], max_new_tokens=max_new_tokens)[0]

    def repair_legacy(self, prompt: str, old_answer: str) -> str:
        task = f"""ButterflyAI anterior intento responder esta conversacion:
Usuario: {prompt}
Respuesta anterior: {old_answer}

Escribe una respuesta corregida, natural y breve que conserve cualquier idea util, pero elimina texto incoherente, repeticiones y afirmaciones no justificadas. Devuelve solo la respuesta corregida."""
        return self.answer(task, max_new_tokens=100)
