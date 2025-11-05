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

    def collect_all_responses_with_checkpoints(self, prompts: Dict[str, List[str]],
                                               resume_from_checkpoint: bool = False) -> pd.DataFrame:
        """
        Collect responses from all models with parallel processing and checkpointing.

        NEW METHOD (Phase 2/3): Adds error recovery and 5-10x speedup via parallel API calls.

        Args:
            prompts: Dictionary with keys: 'hard_refusal', 'soft_refusal', 'no_refusal'
            resume_from_checkpoint: If True, resume from existing checkpoint

        Returns:
            DataFrame with columns: [prompt, response, model, expected_label, timestamp]
        """
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=data_checkpoints_path,
            operation_name='response_collection',
            checkpoint_every=CHECKPOINT_CONFIG['collection_checkpoint_every'],
            auto_cleanup=CHECKPOINT_CONFIG['auto_cleanup'],
            keep_last_n=CHECKPOINT_CONFIG['keep_last_n']
        )

        # Get parallel processing config
        self.parallel_workers = API_CONFIG['parallel_workers']
        self.checkpoint_every = API_CONFIG['collection_batch_size']

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
        print(f"🤖 COLLECTING RESPONSES (PARALLEL + CHECKPOINTED)")
        print(f"{'='*60}")
        print(f"  Total prompts: {total_prompts}")
        print(f"  Models: {', '.join(self.models)}")
        print(f"  Total API calls: {total_calls}")
        print(f"  Parallel workers: {self.parallel_workers}")
        print(f"  Checkpoint every: {self.checkpoint_every} responses")
        print(f"{'='*60}\n")

        # Check for existing checkpoint
        checkpoint_data = None
        if resume_from_checkpoint:
            checkpoint_data = self.checkpoint_manager.load_latest_checkpoint(
                max_age_hours=CHECKPOINT_CONFIG['max_checkpoint_age_hours']
            )

        # Resume from checkpoint or start fresh
        if checkpoint_data:
            responses_df = checkpoint_data['data'].copy()
            completed_count = checkpoint_data['last_index']
            print(f"✅ Resuming from checkpoint: {completed_count} responses already collected\n")
        else:
            responses_df = pd.DataFrame()
            completed_count = 0

        # Create tasks for all prompt-model combinations
        tasks = []
        for prompt_info in prompt_data:
            for model_name in self.models:
                # Skip if already completed (from checkpoint)
                task_id = f"{prompt_info['prompt'][:50]}_{model_name}"
                if checkpoint_data and task_id in checkpoint_data.get('metadata', {}).get('completed_tasks', set()):
                    continue

                tasks.append({
                    'prompt': prompt_info['prompt'],
                    'expected_label': prompt_info['expected_label'],
                    'model': model_name,
                    'task_id': task_id
                })

        # Track completed tasks
        completed_tasks = checkpoint_data.get('metadata', {}).get('completed_tasks', set()) if checkpoint_data else set()
        completed_since_checkpoint = 0

        # Parallel processing with ThreadPoolExecutor
        lock = threading.Lock()  # For thread-safe DataFrame updates

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._collect_single_response_safe, task): task
                for task in tasks
            }

            # Process completed tasks
            with tqdm(total=len(tasks), desc="Collecting responses", initial=completed_count) as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    result = future.result()

                    if result:
                        # Thread-safe DataFrame update
                        with lock:
                            new_row = pd.DataFrame([{
                                'prompt': result['prompt'],
                                'response': result['response'],
                                'model': result['model'],
                                'expected_label': result['expected_label'],
                                'timestamp': result['timestamp']
                            }])
                            responses_df = pd.concat([responses_df, new_row], ignore_index=True)
                            completed_tasks.add(task['task_id'])
                            completed_count += 1
                            completed_since_checkpoint += 1

                        # Checkpoint periodically
                        if completed_since_checkpoint >= self.checkpoint_every:
                            with lock:
                                self.checkpoint_manager.save_checkpoint(
                                    data=responses_df,
                                    last_index=completed_count,
                                    metadata={'completed_tasks': completed_tasks}
                                )
                                completed_since_checkpoint = 0

                    pbar.update(1)

        # Final checkpoint
        if completed_since_checkpoint > 0:
            self.checkpoint_manager.save_checkpoint(
                data=responses_df,
                last_index=completed_count,
                metadata={'completed_tasks': completed_tasks}
            )

        print(f"\n{'='*60}")
        print(f"✅ RESPONSE COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total responses: {len(responses_df)}")
        print(f"{'='*60}\n")

        return responses_df

    def _collect_single_response_safe(self, task: Dict) -> Optional[Dict]:
        """
        Thread-safe wrapper for collecting a single response.

        Args:
            task: Dictionary with 'prompt', 'expected_label', 'model', 'task_id'

        Returns:
            Result dictionary or None if error
        """
        try:
            response = self._query_model(task['model'], task['prompt'])
            return {
                'prompt': task['prompt'],
                'response': response,
                'model': task['model'],
                'expected_label': task['expected_label'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"\n❌ Error collecting response: {task['model']} - {str(e)[:100]}")
            return {
                'prompt': task['prompt'],
                'response': ERROR_RESPONSE,
                'model': task['model'],
                'expected_label': task['expected_label'],
                'timestamp': datetime.now().isoformat()
            }

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

        WHY: Using genai.GenerationConfig object ensures proper configuration enforcement.
        Dictionary format may not always be applied correctly by the Gemini SDK.
        """
        generation_config = genai.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=API_CONFIG['temperature_response']  # Default 1.0
        )
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    def save_responses(self, df: pd.DataFrame, output_dir: str):
        """Save responses to files with timestamps."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = get_timestamp()

        # Save by model
        for model_name in self.models:
            model_df = df[df['model'] == model_name]
            safe_name = model_name.replace('-', '_').replace('.', '_')
            filepath = os.path.join(output_dir, f"{safe_name}_responses_{timestamp}.pkl")
            model_df.to_pickle(filepath)
            print(f"Saved {len(model_df)} responses to {filepath}")

        # Save combined
        combined_path = os.path.join(output_dir, f"all_responses_{timestamp}.pkl")
        df.to_pickle(combined_path)
        print(f"Saved combined responses to {combined_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
