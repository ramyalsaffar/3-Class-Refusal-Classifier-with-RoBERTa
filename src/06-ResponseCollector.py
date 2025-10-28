# ResponseCollector Module
#-------------------------
# Collects responses from multiple LLM APIs (Claude, GPT-5, Gemini).
# All imports are in 00-Imports.py
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

        print(f"\nCollecting responses:")
        print(f"  Total prompts: {total_prompts}")
        print(f"  Models: {len(self.models)}")
        print(f"  Total API calls: {total_calls}")

        # Collect responses
        with tqdm(total=total_calls, desc="Collecting responses") as pbar:
            for prompt_info in prompt_data:
                prompt = prompt_info['prompt']
                expected_label = prompt_info['expected_label']

                for model_name in self.models:
                    try:
                        response = self._query_model(model_name, prompt)

                        all_data.append({
                            'prompt': prompt,
                            'response': response,
                            'model': model_name,
                            'expected_label': expected_label,
                            'timestamp': datetime.now().isoformat()
                        })

                        # Rate limiting
                        time.sleep(self.rate_delay)

                    except Exception as e:
                        print(f"\nError with {model_name} on prompt: {prompt[:50]}...")
                        print(f"Error: {e}")
                        # Add error response
                        all_data.append({
                            'prompt': prompt,
                            'response': ERROR_RESPONSE,
                            'model': model_name,
                            'expected_label': expected_label,
                            'timestamp': datetime.now().isoformat()
                        })

                    pbar.update(1)

        df = pd.DataFrame(all_data)
        print(f"\nCollected {len(df)} total responses")
        print(f"  Claude: {len(df[df['model'] == 'claude-sonnet-4.5'])}")
        print(f"  GPT-5: {len(df[df['model'] == 'gpt-5'])}")
        print(f"  Gemini: {len(df[df['model'] == 'gemini-2.5-flash'])}")

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
                if model_name == "claude-sonnet-4.5":
                    return self._query_claude(prompt)
                elif model_name == "gpt-5":
                    return self._query_gpt5(prompt)
                elif model_name == "gemini-2.5-flash":
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
        response = self.openai_client.chat.completions.create(
            model=API_CONFIG['response_models']['gpt5'],
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens
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
            filepath = os.path.join(output_dir, f"{safe_name}_responses.json")
            model_df.to_json(filepath, orient='records', indent=2)
            print(f"Saved {len(model_df)} responses to {filepath}")

        # Save combined
        combined_path = os.path.join(output_dir, "all_responses.csv")
        df.to_csv(combined_path, index=False)
        print(f"Saved combined responses to {combined_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
