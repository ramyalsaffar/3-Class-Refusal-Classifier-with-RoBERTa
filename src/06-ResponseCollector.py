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

        # NEW: Parallel processing and checkpointing support
        self.parallel_workers = API_CONFIG.get('parallel_workers', 5)
        self.use_async = API_CONFIG.get('use_async', True)
        self.checkpoint_every = CHECKPOINT_CONFIG['collection_checkpoint_every']
        self.checkpoint_manager = None  # Initialized when needed

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

    def collect_all_responses_with_checkpoints(self, prompts: Dict[str, List[str]]) -> pd.DataFrame:
        """
        NEW: Collect responses with parallel processing and checkpointing.

        Features:
        - Parallel API calls using ThreadPoolExecutor
        - Checkpoint every N responses for error recovery
        - Resume from last checkpoint if interrupted
        - Configurable worker count (local vs AWS)

        Args:
            prompts: Dictionary with keys: 'hard_refusal', 'soft_refusal', 'no_refusal'

        Returns:
            DataFrame with columns: [prompt, response, model, expected_label, timestamp]
        """
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=CHECKPOINT_CONFIG['collection_checkpoint_dir'],
            checkpoint_prefix='collection',
            verbose=CHECKPOINT_CONFIG['verbose']
        )

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

        print(f"\nCollecting responses (with checkpointing & parallel processing):")
        print(f"  Total prompts: {total_prompts}")
        print(f"  Models: {len(self.models)}")
        print(f"  Total API calls: {total_calls}")
        print(f"  Parallel workers: {self.parallel_workers}")
        print(f"  Checkpoint every: {self.checkpoint_every} responses")

        # Check for existing checkpoint
        checkpoint_data = None
        if CHECKPOINT_CONFIG['collection_resume_enabled']:
            checkpoint_data = self.checkpoint_manager.load_latest_checkpoint(
                max_age_hours=CHECKPOINT_CONFIG['max_checkpoint_age_hours']
            )

        # Resume from checkpoint or start fresh
        if checkpoint_data:
            print(f"\n📂 Resuming from checkpoint...")
            all_data = checkpoint_data['data'].to_dict('records')
            completed_count = checkpoint_data['last_index']
            print(f"   Already completed: {completed_count}/{total_calls} responses")
        else:
            print(f"\n🆕 Starting fresh collection...")
            all_data = []
            completed_count = 0

        # Create task list (remaining tasks only)
        tasks = []
        task_index = 0
        for prompt_info in prompt_data:
            for model_name in self.models:
                if task_index >= completed_count:
                    tasks.append({
                        'task_id': task_index,
                        'prompt': prompt_info['prompt'],
                        'expected_label': prompt_info['expected_label'],
                        'model_name': model_name
                    })
                task_index += 1

        print(f"   Remaining: {len(tasks)} API calls")

        # Parallel processing with ThreadPoolExecutor
        completed_since_checkpoint = 0
        lock = threading.Lock()  # Thread-safe updates

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._collect_single_response_safe, task): task
                for task in tasks
            }

            # Process completed tasks
            with tqdm(total=len(tasks), desc="Collecting responses") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]

                    try:
                        result = future.result()

                        # Thread-safe append
                        with lock:
                            all_data.append(result)
                            completed_since_checkpoint += 1

                            # Checkpoint periodically
                            if completed_since_checkpoint >= self.checkpoint_every:
                                current_df = pd.DataFrame(all_data)
                                self.checkpoint_manager.save_checkpoint(
                                    data=current_df,
                                    last_index=len(all_data),
                                    metadata={'total_calls': total_calls}
                                )
                                completed_since_checkpoint = 0

                    except Exception as e:
                        print(f"\n❌ Task {task['task_id']} failed: {e}")

                    pbar.update(1)

        # Final checkpoint
        final_df = pd.DataFrame(all_data)
        self.checkpoint_manager.save_checkpoint(
            data=final_df,
            last_index=len(all_data),
            metadata={'total_calls': total_calls, 'status': 'complete'}
        )

        # Cleanup old checkpoints
        if CHECKPOINT_CONFIG['auto_cleanup']:
            self.checkpoint_manager.cleanup_checkpoints(
                keep_last_n=CHECKPOINT_CONFIG['keep_last_n']
            )

        print(f"\n✓ Collection complete!")
        print(f"  Total responses: {len(final_df)}")
        for model_name in self.models:
            count = len(final_df[final_df['model'] == model_name])
            print(f"  {model_name}: {count}")

        return final_df

    def _collect_single_response_safe(self, task: Dict) -> Dict:
        """
        Safely collect a single response with error handling.

        Args:
            task: Dict with keys: task_id, prompt, expected_label, model_name

        Returns:
            Response dict
        """
        try:
            response = self._query_model(task['model_name'], task['prompt'])

            return {
                'prompt': task['prompt'],
                'response': response,
                'model': task['model_name'],
                'expected_label': task['expected_label'],
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"\n⚠️  Error with {task['model_name']}: {str(e)[:100]}")
            return {
                'prompt': task['prompt'],
                'response': ERROR_RESPONSE,
                'model': task['model_name'],
                'expected_label': task['expected_label'],
                'timestamp': datetime.now().isoformat()
            }

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
