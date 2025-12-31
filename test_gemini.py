"""Quick test for Gemini AI enhancement"""

from gemini_ai import GeminiService

def test_gemini():
    print("Testing Gemini AI Enhancement...")
    print("="*60)
    
    gemini = GeminiService()
    
    if not gemini.is_available():
        print(f"❌ Gemini not available: {gemini.last_error}")
        return
    
    print("✓ Gemini AI is available\n")
    
    # Test cases
    test_phrases = [
        "water",
        "hello",
        "hello water",
        "hello water please",
        "2 water",
        "help",
        "thanks",
        "hello help please",
    ]
    
    for phrase in test_phrases:
        print(f"\nInput:  '{phrase}'")
        result = gemini.enhance_sentence(phrase)
        if result.success:
            print(f"Output: '{result.enhanced_text}'")
            print(f"Time:   {result.processing_time:.2f}s")
        else:
            print(f"❌ Failed: {result.error_message}")
    
    print("\n" + "="*60)
    print("Test complete!")

if __name__ == "__main__":
    test_gemini()
