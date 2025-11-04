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

        self.models = DATASET_CONFIG['models']
        self.max_tokens = API_CONFIG['max_tokens_response']
        self.rate_delay = API_CONFIG['rate_limit_delay']
        self.max_retries = API_CONFIG['max_retries']

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
                        print(f"✓ Success ({len(response)} chars)")

                        # Rate limiting
                        time.sleep(self.rate_delay)

                    except Exception as e:
                        model_counts[model_name]['error'] += 1
                        print(f"✗ Error: {str(e)[:80]}")

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

    def _query_claude(self, prompt: str) -> str:
        """Query Claude Sonnet 4.5."""
        response = self.anthropic_client.messages.create(
            model=API_CONFIG['response_models']['claude'],
            max_tokens=self.max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    def _query_gpt5(self, prompt: str) -> str:
        """Query GPT-5."""
        # WHY: GPT-5/o1/o3 models use max_completion_tokens instead of max_tokens
        response = self.openai_client.chat.completions.create(
            model=API_CONFIG['response_models']['gpt5'],
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _query_gemini(self, prompt: str) -> str:
        """Query Gemini 2.5 Flash."""
        response = self.gemini_model.generate_content(prompt)
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
