import json
import re
from typing import Dict, Any, Callable, Tuple

def extract_and_validate_json(raw_text: str) -> Tuple[bool, Any]:
    """
    Validates whether the LLM output is strict, valid JSON.
    Returns (is_valid: bool, parsed_data_or_error: Any).
    """
    text = raw_text.strip()
    
    # Check 1: Did the small model hallucinate markdown fences when raw was demanded?
    if text.startswith("```"):
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text).strip()
    
    # Check 2: Strict JSON parsing
    try:
        data = json.loads(text)
        return True, data
    except (json.JSONDecodeError, ValueError) as err:
        return False, str(err)

def is_output_valid(prompt: str, response_text: str) -> bool:
    """
    Deterministic validator that checks response integrity.
    If the prompt required JSON, enforces strict JSON validation.
    """
    if not response_text or len(response_text.strip()) == 0:
        return False  # Empty output hallucination
        
    p_lower = prompt.lower()
    demands_json = any(keyword in p_lower for keyword in ["json", "schema", "raw json"])
    
    if demands_json:
        is_valid, _ = extract_and_validate_json(response_text)
        return is_valid

    # For general text, ensure the model didn't get stuck in a repetitive loop
    words = response_text.split()
    if len(words) > 20 and len(set(words)) < 5:
        return False  # Degenerate repetition loop
        
    return True

class SpeculativeCascade:
    """
    Executes speculative cascading:
    Calls Small model first -> Validates -> Falls back to Frontier only on failure.
    """

    def __init__(self, small_caller: Callable, frontier_caller: Callable):
        self.small_caller = small_caller
        self.frontier_caller = frontier_caller

    async def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Executes the speculative cascade workflow.
        Returns: {
            "content": str,
            "model_used": str,
            "fallback_triggered": bool,
            "cost_saved": bool
        }
        """
        # 1. Speculatively invoke the cheap 8B model first
        small_response = await self.small_caller(prompt, **kwargs)
        
        # 2. Fast deterministic validation (< 2ms)
        if is_output_valid(prompt, small_response):
            return {
                "content": small_response,
                "model_used": "small_speculative",
                "fallback_triggered": False,
                "cost_saved": True
            }

        # 3. Fallback: Small model failed validation -> Escalate to Frontier
        frontier_response = await self.frontier_caller(prompt, **kwargs)
        return {
            "content": frontier_response,
            "model_used": "frontier_fallback",
            "fallback_triggered": True,
            "cost_saved": False
        }