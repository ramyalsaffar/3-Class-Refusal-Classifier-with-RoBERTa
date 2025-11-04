# ResponseCollector Module
#-------------------------
# Collects responses from multiple LLM APIs (Claude, GPT-5, Gemini).
# All imports are in 01-Imports.py
###############################################################################


class ResponseCollector:
    """Collect responses from multiple LLM models."""

    def __init__(self, claude_key: str, openai_key: str, google_key: str):
        """
        Initialize response collector.

        Args:
            claude_key: Anthropic API key
            openai_key: OpenAI API key
            google_key: Google API key
        """
        self.anthropic_client = Anthropic(api_key=claude_key)
        self.openai_client = OpenAI(api_key=openai_key)
        genai.configure(api_key=google_key)
        self.gemini_model = genai.GenerativeModel(API_CONFIG['response_models']['gemini'])

        # WHY: Derive models from API_CONFIG - single source of truth
        self.models = list(API_CONFIG['response_models'].values())
        self.max_tokens = API_CONFIG['max_tokens_response']
        self.rate_delay = API_CONFIG['rate_limit_delay']
        self.max_retries = API_CONFIG['max_retries']

        # Token counter for API usage tracking
        # WHY: cl100k_base is used by GPT-4/GPT-5 and provides good approximation for Claude/Gemini
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def collect_all_responses(self, prompts: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Collect responses from all models for all prompts.

        Args:
            prompts: Dictionary with keys: 'hard_refusal', 'soft_refusal', 'no_refusal'

        Returns:
            DataFrame with columns: [prompt, response, model, expected_label, timestamp]
        """
        all_data = []

        # Flatten prompts with expected labels
        prompt_data = []
        for label_name, prompt_list in prompts.items():
            for prompt in prompt_list:
                prompt_data.append({
                    'prompt': prompt,
                    'expected_label': label_name
                })

        total_prompts = len(prompt_data)
        total_calls = total_prompts * len(self.models)

        print(f"\n{'='*60}")
        print(f"🤖 COLLECTING RESPONSES FROM {len(self.models)} MODELS")
        print(f"{'='*60}")
        print(f"  Total prompts: {total_prompts}")
        print(f"  Models: {', '.join(self.models)}")
        print(f"  Total API calls: {total_calls}")
        print(f"{'='*60}\n")

        # Track per-model counts
        model_counts = {model: {'success': 0, 'error': 0} for model in self.models}

        # Collect responses with nested progress
        with tqdm(total=total_prompts, desc="Prompts", position=0) as prompt_pbar:
            for idx, prompt_info in enumerate(prompt_data, 1):
                prompt = prompt_info['prompt']
                expected_label = prompt_info['expected_label']

                print(f"\n{'─'*60}")
                print(f"📄 Prompt {idx}/{total_prompts} ({expected_label})")
                print(f"   \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
                print(f"{'─'*60}")

                for model_name in self.models:
                    try:
                        print(f"  ⏳ Querying {model_name}...", end=" ", flush=True)
                        response = self._query_model(model_name, prompt)

                        all_data.append({
                            'prompt': prompt,
                            'response': response,
                            'model': model_name,
                            'expected_label': expected_label,
                            'timestamp': datetime.now().isoformat()
                        })

                        model_counts[model_name]['success'] += 1
                        token_count = self._count_tokens(response)
                        print(f"✓ Success ({token_count} tokens)")

                        # Rate limiting
                        time.sleep(self.rate_delay)

                    except Exception as e:
                        model_counts[model_name]['error'] += 1
                        print(f"✗ Error: {str(e)[:800]}")  # Show full error (max 800 chars)

                        # Add error response
                        all_data.append({
                            'prompt': prompt,
                            'response': ERROR_RESPONSE,
                            'model': model_name,
                            'expected_label': expected_label,
                            'timestamp': datetime.now().isoformat()
                        })

                # Show running totals after each prompt
                print(f"  Running totals: ", end="")
                for model_name in self.models:
                    counts = model_counts[model_name]
                    print(f"{model_name}: {counts['success']}✓/{counts['error']}✗ | ", end="")
                print()

                prompt_pbar.update(1)

        df = pd.DataFrame(all_data)

        print(f"\n{'='*60}")
        print(f"✅ RESPONSE COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total responses: {len(df)}")
        for model_name in self.models:
            counts = model_counts[model_name]
            success_rate = (counts['success'] / total_prompts * 100) if total_prompts > 0 else 0
            print(f"  {model_name}: {counts['success']} success, {counts['error']} errors ({success_rate:.1f}% success rate)")
        print(f"{'='*60}\n")

        return df

    def _query_model(self, model_name: str, prompt: str) -> str:
        """
        Query a specific model with retry logic.

        Args:
            model_name: Name of model to query
            prompt: Prompt text

        Returns:
            Model response text
        """
        for attempt in range(self.max_retries):
            try:
                # WHY: Use config values instead of hardcoded model names
                if model_name == API_CONFIG['response_models']['claude']:
                    return self._query_claude(prompt)
                elif model_name == API_CONFIG['response_models']['gpt5']:
                    return self._query_gpt5(prompt)
                elif model_name == API_CONFIG['response_models']['gemini']:
                    return self._query_gemini(prompt)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    raise e

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        WHY: Token counting is critical for:
        - API cost tracking (APIs charge per token, not per character)
        - Understanding actual API usage vs limits
        - Professional best practice for LLM development

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def _query_claude(self, prompt: str) -> str:
        """
        Query Claude Sonnet 4.5.

        Temperature: Uses default 1.0 for fair comparison across models.
        Claude default: 1.0 (range: 0.0-1.0)
        """
        response = self.anthropic_client.messages.create(
            model=API_CONFIG['response_models']['claude'],
            max_tokens=self.max_tokens,
            temperature=API_CONFIG['temperature_response'],  # Default 1.0
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    def _query_gpt5(self, prompt: str) -> str:
        """
        Query GPT-5.

        Temperature: LOCKED at 1.0 (cannot be changed).
        GPT-5 is a reasoning model with architectural constraints:
        - Only supports temperature=1.0 (default)
        - Generates internal reasoning tokens that consume max_completion_tokens budget
        - Using "minimal" reasoning effort prevents token exhaustion
        """
        response = self.openai_client.chat.completions.create(
            model=API_CONFIG['response_models']['gpt5'],
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens,
            # temperature not included - GPT-5 only supports default 1.0
            reasoning_effort="minimal"  # Minimal reasoning = faster, no token exhaustion
        )
        content = response.choices[0].message.content
        # WHY: GPT-5 may return None if all tokens used for reasoning
        if content is None or content == "":
            raise ValueError(f"GPT-5 returned empty response. Reasoning tokens may have exhausted budget. Consider increasing max_completion_tokens or using 'minimal' reasoning_effort.")
        return content

    def _query_gemini(self, prompt: str) -> str:
        """
        Query Gemini 2.5 Flash.

        Temperature: Uses default 1.0 for fair comparison across models.
        Gemini default: 1.0 (range: 0.0-2.0)
        """
        generation_config = {
            'max_output_tokens': self.max_tokens,
            'temperature': API_CONFIG['temperature_response']  # Default 1.0
        }
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    def save_responses(self, df: pd.DataFrame, output_dir: str):
        """Save responses to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save by model
        for model_name in self.models:
            model_df = df[df['model'] == model_name]
            safe_name = model_name.replace('-', '_').replace('.', '_')
            filepath = os.path.join(output_dir, f"{safe_name}_responses.pkl")
            model_df.to_pickle(filepath)
            print(f"Saved {len(model_df)} responses to {filepath}")

        # Save combined
        combined_path = os.path.join(output_dir, "all_responses.pkl")
        df.to_pickle(combined_path)
        print(f"Saved combined responses to {combined_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
