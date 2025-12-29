"""
gemini_service.py
Gemini AI integration for natural language enhancement of gesture sequences.

Features:
- Converts raw gesture words into natural English sentences
- Improves grammar, tone, and clarity
- Supports multiple tones: natural, polite, formal, casual
- Robust fallback handling if API fails
- Modular and optional integration
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        # Fallback to deprecated package
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        print("[WARN] Using deprecated google-generativeai. Consider upgrading to google-genai")
    except ImportError:
        GEMINI_AVAILABLE = False
        print("[WARN] Neither google-genai nor google-generativeai installed. Gemini features disabled.")

from config import GEMINI


@dataclass
class GeminiResponse:
    """Container for Gemini API response"""
    success: bool
    enhanced_text: str
    original_text: str
    error_message: Optional[str] = None
    processing_time: float = 0.0


class GeminiService:
    """
    Service class for Gemini AI natural language enhancement.
    
    Usage:
        service = GeminiService()
        if service.is_available():
            result = service.enhance_sentence("hello water please")
            if result.success:
                print(result.enhanced_text)
    """
    
    def __init__(self, api_key: Optional[str] = None, tone: str = "polite"):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Gemini API key (defaults to config value)
            tone: Response tone - "natural", "polite", "formal", or "casual"
        """
        self.api_key = api_key or GEMINI.api_key
        self.tone = tone
        self.model = None
        self.enabled = False
        self.last_error = None
        
        if not GEMINI_AVAILABLE:
            self.last_error = "google-genai package not installed"
            return
        
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            self.last_error = "Invalid API key"
            return
        
        try:
            # Try new google-genai package first
            try:
                client = genai.Client(api_key=self.api_key)
                self.model = client
                self.model_name = GEMINI.model_name
            except AttributeError:
                # Fallback to old google-generativeai package
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    GEMINI.model_name,
                    generation_config={
                        "temperature": GEMINI.temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 150,
                    }
                )
            
            self.enabled = True
            print(f"[INFO] Gemini AI initialized successfully (tone: {self.tone})")
        except Exception as e:
            self.last_error = str(e)
            print(f"[ERROR] Gemini initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini service is available and ready"""
        return self.enabled and self.model is not None
    
    def set_tone(self, tone: str) -> None:
        """
        Change the response tone.
        
        Args:
            tone: "natural", "polite", "formal", or "casual"
        """
        valid_tones = ["natural", "polite", "formal", "casual"]
        if tone.lower() in valid_tones:
            self.tone = tone.lower()
            print(f"[INFO] Gemini tone changed to: {self.tone}")
        else:
            print(f"[WARN] Invalid tone '{tone}'. Valid options: {valid_tones}")
    
    def _build_prompt(self, gesture_sequence: str) -> str:
        """
        Build the prompt for Gemini based on gesture sequence and tone.
        
        Args:
            gesture_sequence: Raw gesture words (e.g., "hello water please")
            
        Returns:
            Formatted prompt string
        """
        tone_instructions = {
            "natural": "Keep it conversational and natural.",
            "polite": "Make it polite and respectful.",
            "formal": "Use formal, professional language.",
            "casual": "Keep it casual and friendly."
        }
        
        tone_instruction = tone_instructions.get(self.tone, tone_instructions["natural"])
        
        prompt = f"""{GEMINI.system_prompt}

Tone: {tone_instruction}

Gesture sequence: "{gesture_sequence}"

Enhanced sentence:"""
        
        return prompt
    
    def enhance_sentence(
        self, 
        gesture_sequence: str, 
        timeout: Optional[int] = None
    ) -> GeminiResponse:
        """
        Enhance a gesture sequence into natural English.
        
        Args:
            gesture_sequence: Raw gesture words separated by spaces
            timeout: Optional timeout in seconds (defaults to config value)
            
        Returns:
            GeminiResponse object with results and metadata
        """
        start_time = time.time()
        
        # Validation
        if not gesture_sequence or not gesture_sequence.strip():
            return GeminiResponse(
                success=False,
                enhanced_text="",
                original_text=gesture_sequence,
                error_message="Empty input",
                processing_time=0.0
            )
        
        # Check if service is available
        if not self.is_available():
            return GeminiResponse(
                success=False,
                enhanced_text=gesture_sequence,  # Fallback to original
                original_text=gesture_sequence,
                error_message=self.last_error or "Service not available",
                processing_time=time.time() - start_time
            )
        
        # If already looks like a complete sentence, maybe skip enhancement
        # (optional - can be disabled)
        words = gesture_sequence.strip().split()
        if len(words) == 1:
            # Single word - capitalize and return
            return GeminiResponse(
                success=True,
                enhanced_text=gesture_sequence.strip().capitalize() + ".",
                original_text=gesture_sequence,
                processing_time=time.time() - start_time
            )
        
        # Build prompt
        prompt = self._build_prompt(gesture_sequence)
        timeout_val = timeout or GEMINI.timeout_seconds
        
        # Try API call with retries
        for attempt in range(GEMINI.max_retries + 1):
            try:
                # Try new API format first
                try:
                    # New google-genai package
                    response = self.model.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=GEMINI.temperature,
                            top_p=0.95,
                            top_k=40,
                            max_output_tokens=150,
                        )
                    )
                    # Access the text from the response - get full text
                    if response and response.text:
                        enhanced = response.text.strip()
                        # Take only the first line if multi-line response
                        enhanced = enhanced.split('\n')[0].strip()
                    else:
                        enhanced = ""
                except (AttributeError, TypeError):
                    # Fallback - try as old API
                    enhanced = ""
                
                if enhanced:
                    # Clean up response (remove quotes, extra whitespace)
                    enhanced = enhanced.strip('"\'')
                    enhanced = ' '.join(enhanced.split())
                    
                    # Ensure it ends with punctuation
                    if enhanced and enhanced[-1] not in '.!?':
                        enhanced += '.'
                    
                    processing_time = time.time() - start_time
                    
                    return GeminiResponse(
                        success=True,
                        enhanced_text=enhanced,
                        original_text=gesture_sequence,
                        processing_time=processing_time
                    )
                else:
                    error_msg = "Empty response from API"
                    
            except Exception as e:
                error_msg = str(e)
                # Check for specific quota errors
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                    error_msg = "API quota exceeded - using original text"
                    print(f"[WARN] Gemini quota exceeded, system will continue without enhancement")
                    break  # Don't retry quota errors
                elif attempt < GEMINI.max_retries:
                    time.sleep(0.5)  # Brief delay before retry
                    continue
        
        # All retries failed - return fallback
        processing_time = time.time() - start_time
        return GeminiResponse(
            success=False,
            enhanced_text=gesture_sequence,  # Fallback to original
            original_text=gesture_sequence,
            error_message=error_msg,
            processing_time=processing_time
        )
    
    def enhance_sentence_simple(self, gesture_sequence: str) -> str:
        """
        Simplified interface - returns enhanced text directly.
        Falls back to original text on error.
        
        Args:
            gesture_sequence: Raw gesture words
            
        Returns:
            Enhanced sentence string
        """
        result = self.enhance_sentence(gesture_sequence)
        return result.enhanced_text


# Singleton instance for easy access
_gemini_instance: Optional[GeminiService] = None


def get_gemini_service(tone: str = "polite") -> GeminiService:
    """
    Get or create the global Gemini service instance.
    
    Args:
        tone: Default tone for responses
        
    Returns:
        GeminiService instance
    """
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiService(tone=tone)
    return _gemini_instance


def quick_enhance(text: str, tone: str = "polite") -> str:
    """
    Quick function to enhance text using Gemini.
    Creates service instance if needed.
    
    Args:
        text: Gesture sequence to enhance
        tone: Response tone
        
    Returns:
        Enhanced text (or original on failure)
    """
    service = get_gemini_service(tone=tone)
    return service.enhance_sentence_simple(text)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Gemini Service Test")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        "hello water please",
        "thanks help",
        "I love food",
        "stop no bad",
        "hello",
        "yes okay good",
        "sorry help please",
    ]
    
    service = GeminiService(tone="polite")
    
    if not service.is_available():
        print(f"[ERROR] Service unavailable: {service.last_error}")
        print("\nMake sure to:")
        print("1. Install: pip install google-generativeai")
        print("2. Set valid API key in config.py")
    else:
        print(f"Service ready! Model: {GEMINI.model_name}\n")
        
        for test in test_cases:
            print(f"Input:  '{test}'")
            result = service.enhance_sentence(test)
            
            if result.success:
                print(f"Output: '{result.enhanced_text}'")
                print(f"Time:   {result.processing_time:.2f}s")
            else:
                print(f"Error:  {result.error_message}")
                print(f"Fallback: '{result.enhanced_text}'")
            print()
        
        # Test different tones
        print("\n" + "=" * 60)
        print("Testing different tones:")
        print("=" * 60)
        test_input = "water please"
        
        for tone in ["natural", "polite", "formal", "casual"]:
            service.set_tone(tone)
            result = service.enhance_sentence(test_input)
            print(f"{tone.upper():10} | '{result.enhanced_text}'")
