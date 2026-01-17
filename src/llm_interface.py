"""
LLM Interface Module

This module handles:
1. Gemini API interaction
2. Prompt construction
3. Response parsing
4. Hallucination detection

CRITICAL: The system prompt is the last line of defense against hallucination.
Every word matters.
"""

import os
import google.generativeai as genai
from typing import Dict, Optional
from config import (
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    SYSTEM_PROMPT,
    ESCALATION_MESSAGE
)


class GeminiInterface:
    """
    Handles all interactions with Google's Gemini API.
    
    Key Design Decisions:
    - Temperature = 0.0 (deterministic, less hallucination)
    - Explicit system prompt (defines behavior)
    - Response validation (detect refusals)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = GEMINI_MODEL,
        temperature: float = GEMINI_TEMPERATURE,
        max_tokens: int = GEMINI_MAX_TOKENS
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (or reads from GEMINI_API_KEY env var)
            model_name: Model to use (e.g., "gemini-pro")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        # Get API key
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"Gemini initialized: {model_name}")
    
    def generate_response(
        self, 
        context: str, 
        question: str,
        system_prompt: str = SYSTEM_PROMPT
    ) -> str:
        """
        Generate answer using Gemini.
        
        Args:
            context: Retrieved documents (formatted)
            question: User's question
            system_prompt: Instructions for the LLM
        
        Returns:
            LLM's response (string)
        
        Prompt Structure:
        We embed the system prompt, context, and question into a single message.
        This is different from OpenAI's chat format but works well for Gemini.
        """
        # Construct the full prompt
        full_prompt = system_prompt.format(
            context=context,
            question=question
        )
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            candidate_count=1,  # Only need one response
        )
        
        # Safety settings: We want to allow all content since this is QA
        # (We're not generating harmful content, just answering questions)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        try:
            # Call Gemini
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Extract text
            # Note: response.text raises an exception if blocked by safety filters
            answer = response.text.strip()
            
            return answer
        
        except Exception as e:
            # Handle API errors gracefully
            error_msg = str(e)
            print(f"Gemini API error: {error_msg}")
            
            # Check if it's a safety filter block
            if "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                return (
                    "I apologize, but I cannot generate a response due to content "
                    "safety filters. Please rephrase your question."
                )
            
            # Generic error
            return (
                "I encountered an error while generating a response. "
                "Please try again or contact support."
            )


class RAGPipeline:
    """
    Complete RAG pipeline: Retrieval + Generation.
    
    This is the main class your Streamlit app will use.
    """
    
    def __init__(self, retriever, gemini_interface: Optional[GeminiInterface] = None):
        """
        Args:
            retriever: Retriever instance
            gemini_interface: GeminiInterface instance (or None to create)
        """
        self.retriever = retriever
        
        if gemini_interface is None:
            self.gemini = GeminiInterface()
        else:
            self.gemini = gemini_interface
    
    def answer_question(self, question: str) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            question: User's question
        
        Returns:
            Dictionary with:
            {
                'answer': str,              # LLM's answer or escalation message
                'sources': List[Dict],      # Retrieved documents
                'confidence': float,        # Top similarity score (or 0.0)
                'should_escalate': bool     # True if no confident match
            }
        
        Flow:
        1. Retrieve relevant documents
        2. Check if we have confident matches
        3a. If yes: Send to LLM
        3b. If no: Return escalation message
        4. Return structured response
        """
        print(f"\n{'='*80}")
        print(f"Processing question: {question}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve documents
        context, results, has_confident_match = self.retriever.retrieve_and_format(question)
        
        # Step 2: Check confidence
        if not has_confident_match:
            # No good matches - escalate immediately
            # We do NOT send to LLM to avoid hallucination
            print("⚠️  No confident matches - escalating to human")
            
            return {
                'answer': ESCALATION_MESSAGE,
                'sources': [],
                'confidence': 0.0,
                'should_escalate': True
            }
        
        # Step 3: We have confident matches - generate answer
        print(f"✓ Found {len(results)} confident matches - generating answer")
        
        answer = self.gemini.generate_response(context, question)
        
        # Step 4: Validate the response
        # Sometimes LLM might still say "I don't know" even with context
        # Treat this as an escalation
        if self._is_refusal_response(answer):
            print("⚠️  LLM refused to answer - escalating")
            return {
                'answer': ESCALATION_MESSAGE,
                'sources': [r.to_dict() for r in results],
                'confidence': results[0].score,
                'should_escalate': True
            }
        
        # Step 5: Success - return answer with sources
        print("✓ Answer generated successfully")
        
        return {
            'answer': answer,
            'sources': [r.to_dict() for r in results],
            'confidence': results[0].score,  # Top match score
            'should_escalate': False
        }
    
    def _is_refusal_response(self, answer: str) -> bool:
        """
        Detect if LLM is refusing to answer or expressing uncertainty.
        
        Common refusal patterns:
        - "I don't know"
        - "I cannot find"
        - "The context doesn't provide"
        - "I'm unable to answer"
        
        Why this matters:
        Even with a good system prompt, LLMs sometimes refuse to answer.
        We want to catch this and escalate instead of showing an uncertain answer.
        """
        refusal_phrases = [
            "i don't know",
            "i cannot find",
            "i'm not sure",
            "i don't have",
            "context does not",
            "context doesn't",
            "unable to answer",
            "cannot answer",
            "no information",
            "not mentioned"
        ]
        
        answer_lower = answer.lower()
        
        # Check if answer contains refusal phrases
        for phrase in refusal_phrases:
            if phrase in answer_lower:
                return True
        
        # Check if answer is suspiciously short (might be a refusal)
        if len(answer.split()) < 10:
            # Very short answers are often "I don't know" variations
            # But allow short answers if they look like valid responses
            if any(word in answer_lower for word in ["yes", "no", "$", "step", "click"]):
                return False  # Probably a valid short answer
            return True
        
        return False


# ============================================================================
# PROMPT ENGINEERING DEEP DIVE
# ============================================================================

"""
Let's dissect our system prompt:

SYSTEM_PROMPT = '''You are a helpful customer support assistant for a company.

STRICT RULES:
1. Answer ONLY using information from the provided context documents below.
2. If the context does not contain enough information to answer the question, 
   respond EXACTLY with: "I cannot find an answer in the knowledge base. 
   Your query will be escalated to a human agent."
3. Do NOT use external knowledge or make assumptions.
4. Always cite which document(s) you used (e.g., "According to Document 2...").
5. Keep answers concise and professional.
6. If multiple documents provide conflicting information, acknowledge this.

Context Documents:
{context}

Question: {question}

Answer:'''

Why each part exists:

1. "You are a helpful customer support assistant"
   → Sets the tone and role. LLM will adopt this persona.

2. "STRICT RULES" + numbering
   → Grabs attention. LLMs respond well to explicit rule lists.
   → Numbering makes rules clear and memorable.

3. "Answer ONLY using information from the provided context"
   → The core anti-hallucination instruction.
   → "ONLY" is key - it's a hard constraint.

4. "respond EXACTLY with: [specific phrase]"
   → Gives LLM a scripted response for edge cases.
   → "EXACTLY" means no variation - important for parsing.

5. "Do NOT use external knowledge"
   → Reinforces rule #1. Repetition helps with LLMs.

6. "Always cite which document"
   → Enables verification and builds trust.
   → "According to Document 2" format is easy to parse.

7. "Keep answers concise"
   → Prevents rambling. Users want quick answers.

8. "If conflicting information, acknowledge"
   → Handles edge case where corpus has inconsistencies.

9. {context} and {question} placeholders
   → We'll fill these at runtime with retrieved docs and user query.

Alternative prompts we REJECTED:

❌ "Answer the question using the context"
   → Too weak. LLM will use external knowledge too.

❌ "If you can't answer, say 'I don't know'"
   → Too vague. LLM might say "I don't know" in many ways.
   → Hard to detect programmatically.

❌ Using separate system/user messages (OpenAI style)
   → Gemini works better with a single combined prompt.

Tuning tips:
- Test with adversarial questions (unrelated to your corpus)
- If LLM still hallucinates, add more "DO NOT" instructions
- If LLM is too cautious, soften "ONLY" to "primarily"
- Monitor real queries to find edge cases
"""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from retriever import Retriever
    
    # Initialize pipeline
    retriever = Retriever()
    pipeline = RAGPipeline(retriever)
    
    # Test questions
    test_questions = [
        "How do I reset my password?",
        "What does the premium plan include?",
        "What's the weather like today?",  # Should escalate
        "Tell me about quantum physics"     # Should escalate
    ]
    
    for question in test_questions:
        result = pipeline.answer_question(question)
        
        print(f"\nQuestion: {question}")
        print(f"Should escalate: {result['should_escalate']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])} documents")
        print("-" * 80)