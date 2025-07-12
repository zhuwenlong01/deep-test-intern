"""API client module for calling various LLM APIs like Claude, Gemini, etc."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import random
from loguru import logger

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available. Install with: pip install anthropic")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")

class BaseAPIClient(ABC):
    """Base class for API clients."""
    
    def __init__(self, api_key: str, config):
        self.api_key = api_key
        self.config = config
        self.max_retries = config.api.max_retries
        self.retry_delay = config.api.retry_delay
        self.request_count = 0
        self.last_request_time = 0
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the API."""
        ...
    
    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Ensure minimum delay between requests
        min_delay = 60.0 / self.config.api.max_requests_per_minute
        if time_since_last < min_delay:
            time.sleep(min_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def generate_with_retry(self, prompt: str, **kwargs) -> str:
        """Generate response with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = await self.generate_response(prompt, **kwargs)
                return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise e

class ClaudeAPIClient(BaseAPIClient):
    """Claude API client."""
    
    def __init__(self, api_key: str, config):
        super().__init__(api_key, config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-sonnet-20240229"
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from Claude API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.config.generation.max_tokens),
                temperature=kwargs.get('temperature', self.config.generation.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

class GeminiAPIClient(BaseAPIClient):
    """Gemini API client."""
    
    def __init__(self, api_key: str, config):
        super().__init__(api_key, config)
        if not GENAI_AVAILABLE:
            raise ImportError("Google Generative AI library not available")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from Gemini API."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', self.config.generation.temperature),
                    max_output_tokens=kwargs.get('max_tokens', self.config.generation.max_tokens),
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

class OpenAIAPIClient(BaseAPIClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, config):
        super().__init__(api_key, config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4"
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', self.config.generation.max_tokens),
                temperature=kwargs.get('temperature', self.config.generation.temperature),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class MultiAPIClient:
    """Client that manages multiple API providers."""
    
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.active_clients = []
        
        # Initialize available clients
        if config.api.claude_api_key and ANTHROPIC_AVAILABLE:
            self.clients['claude'] = ClaudeAPIClient(config.api.claude_api_key, config)
            self.active_clients.append('claude')
        
        if config.api.gemini_api_key and GENAI_AVAILABLE:
            self.clients['gemini'] = GeminiAPIClient(config.api.gemini_api_key, config)
            self.active_clients.append('gemini')
        
        if config.api.openai_api_key and OPENAI_AVAILABLE:
            self.clients['openai'] = OpenAIAPIClient(config.api.openai_api_key, config)
            self.active_clients.append('openai')
        
        if not self.active_clients:
            raise ValueError("No API clients available. Please check your API keys and dependencies.")
        
        logger.info(f"Initialized API clients: {self.active_clients}")
    
    def get_preferred_client(self) -> BaseAPIClient:
        """Get the preferred API client based on configuration."""
        # Try to get client based on model preferences
        for model in self.config.generation.model_preferences:
            if "claude" in model.lower() and "claude" in self.active_clients:
                return self.clients['claude']
            elif "gemini" in model.lower() and "gemini" in self.active_clients:
                return self.clients['gemini']
            elif "gpt" in model.lower() and "openai" in self.active_clients:
                return self.clients['openai']
        
        # Fallback to first available client
        return self.clients[self.active_clients[0]]
    
    def get_random_client(self) -> BaseAPIClient:
        """Get a random API client for load balancing."""
        client_name = random.choice(self.active_clients)
        return self.clients[client_name]
    
    async def generate_response(self, prompt: str, use_random: bool = True, **kwargs) -> Tuple[str, str]:
        """Generate response using available API clients."""
        client = self.get_random_client() if use_random else self.get_preferred_client()
        client_name = [k for k, v in self.clients.items() if v == client][0]
        
        try:
            response = await client.generate_with_retry(prompt, **kwargs)
            return response, client_name
        except Exception as e:
            logger.error(f"Failed to generate response with {client_name}: {e}")
            
            # Try with other clients
            for name in self.active_clients:
                if name != client_name:
                    try:
                        backup_client = self.clients[name]
                        response = await backup_client.generate_with_retry(prompt, **kwargs)
                        logger.info(f"Successfully generated response with backup client {name}")
                        return response, name
                    except Exception as backup_e:
                        logger.warning(f"Backup client {name} also failed: {backup_e}")
            
            raise Exception("All API clients failed to generate response")
    
    async def generate_multiple_responses(self, prompts: List[str], **kwargs) -> List[Tuple[str, str]]:
        """Generate multiple responses concurrently."""
        semaphore = asyncio.Semaphore(self.config.api.max_concurrent_requests)
        
        async def generate_single(prompt: str) -> Tuple[str, str]:
            async with semaphore:
                return await self.generate_response(prompt, **kwargs)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate response for prompt {i}: {result}")
                processed_results.append(("", "failed"))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get statistics about API client usage."""
        stats = {}
        for name, client in self.clients.items():
            stats[name] = {
                "request_count": client.request_count,
                "last_request_time": client.last_request_time,
                "available": name in self.active_clients
            }
        return stats

class ConversationManager:
    """Manager for multi-round conversations."""
    
    def __init__(self, api_client: MultiAPIClient, config):
        self.api_client = api_client
        self.config = config
        self.max_rounds = config.generation.max_rounds
    
    async def run_conversation(self, initial_prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a multi-round conversation."""
        conversation_history = []
        current_prompt = initial_prompt
        
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Starting conversation round {round_num}")
            
            try:
                response, client_name = await self.api_client.generate_response(current_prompt)
                
                # Validate response
                if not self._validate_response(response):
                    logger.warning(f"Invalid response in round {round_num}, skipping")
                    continue
                
                conversation_turn = {
                    "round": round_num,
                    "prompt": current_prompt,
                    "response": response,
                    "client": client_name,
                    "timestamp": time.time()
                }
                
                conversation_history.append(conversation_turn)
                
                # Check if conversation should continue
                if self._should_end_conversation(response, conversation_history):
                    logger.info(f"Conversation ended naturally at round {round_num}")
                    break
                
                # Prepare next round prompt
                current_prompt = self._prepare_next_prompt(conversation_history, context)
                
            except Exception as e:
                logger.error(f"Error in conversation round {round_num}: {e}")
                break
        
        return conversation_history
    
    def _validate_response(self, response: str) -> bool:
        """Validate if response meets quality criteria."""
        if not response or len(response.strip()) < self.config.generation.min_answer_length:
            return False
        
        if len(response) > self.config.generation.max_answer_length:
            return False
        
        return True
    
    def _should_end_conversation(self, response: str, history: List[Dict[str, Any]]) -> bool:
        """Determine if conversation should end."""
        # End conversation if response indicates completion
        end_indicators = ["that's all", "no more", "finished", "完成", "结束"]
        response_lower = response.lower()
        
        for indicator in end_indicators:
            if indicator in response_lower:
                return True
        
        # End if response is too short and we've had several rounds
        if len(history) >= 3 and len(response.strip()) < 20:
            return True
        
        return False
    
    def _prepare_next_prompt(self, history: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare prompt for next round based on conversation history."""
        if not history:
            return ""
        
        last_response = history[-1]['response']
        
        # Simple continuation prompt - can be made more sophisticated
        next_prompts = [
            "请继续详细说明。",
            "能否进一步解释？",
            "还有其他要补充的吗？",
            "Please provide more details.",
            "Could you elaborate further?",
            "Are there any additional points to consider?"
        ]
        
        return random.choice(next_prompts)
    
    async def process_batch_conversations(self, prompts: List[str], contexts: Optional[List[Dict[str, Any]]] = None) -> List[List[Dict[str, Any]]]:
        """Process multiple conversations in batch."""
        if contexts is None:
            contexts = [{}] * len(prompts)
        
        tasks = []
        for prompt, context in zip(prompts, contexts):
            task = self.run_conversation(prompt, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process conversation {i}: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)
        
        return processed_results