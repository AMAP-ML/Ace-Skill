"""API-based inference script (inference-only entry point)."""

import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.result_utils import save_results, print_summary
from engine.api_processors import process_single_sample

from api_utils import (
    create_base_parser,
    setup_common,
    run_single_rollout,
    retrieve_experiences_for_sample,
    compute_and_save_sample_summary,
    check_sample_completed,
    prepare_sample_args,
    get_sample_metadata,
    compute_dataset_summary,
    execute_pipeline_parallel_processing,
)


def main(args):
    """
    Main function to run the API-based inference process (inference-only, no experience generation).
    """
    result = setup_common(args)
    if result is None:
        return
    data, sampling_params, BASE_SYSTEM_PROMPT, experience_retriever = result

    all_results = []

    def process_single_sample_with_experience(sample, sample_idx):
        """Process a single sample with experience retrieval (no experience generation)."""
        sample_args = prepare_sample_args(args)

        question_id, sample_dir = get_sample_metadata(sample, sample_idx, sample_args.output_dir)
        os.makedirs(sample_dir, exist_ok=True)

        retrieve_experiences_for_sample(
            sample, sample_args, experience_retriever, BASE_SYSTEM_PROMPT,
            question_id, sample_dir,
            update_global_prompts=False
        )

        completed_info = check_sample_completed(sample_dir, sample_args)
        if completed_info:
            print(f"[Sample {sample_idx + 1}/{len(data)}] {question_id} - Already completed, skipping...")
            return {
                'sample': sample,
                'sample_idx': sample_idx,
                'question_id': question_id,
                'sample_dir': sample_dir,
                'sample_rollout_results': completed_info.get('sample_rollout_results', [])
            }

        sample_rollout_results = []

        if sample_args.rollouts_per_sample > 1:
            print(f"\n[Sample {sample_idx + 1}/{len(data)}] {question_id} - Running {sample_args.rollouts_per_sample} rollouts...")
            max_workers = min(sample_args.num_workers, sample_args.rollouts_per_sample)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rollout_idx = {
                    executor.submit(run_single_rollout, sample, sample_args, sampling_params, rollout_idx): rollout_idx
                    for rollout_idx in range(sample_args.rollouts_per_sample)
                }
                for future in as_completed(future_to_rollout_idx):
                    rollout_idx = future_to_rollout_idx[future]
                    result = future.result()
                    if result:
                        result['_rollout_idx'] = rollout_idx
                        sample_rollout_results.append(result)

            compute_and_save_sample_summary(
                question_id, sample_rollout_results, sample_args.rollouts_per_sample, sample_dir
            )

        else:
            result = process_single_sample(sample, sample_args, sampling_params, rollout_idx=None)
            if result:
                sample_rollout_results.append(result)

        return {
            'sample': sample,
            'sample_idx': sample_idx,
            'question_id': question_id,
            'sample_dir': sample_dir,
            'sample_rollout_results': sample_rollout_results
        }

    if args.num_workers > 1 and args.rollouts_per_sample > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")
        print(f"  Rollouts per sample: {args.rollouts_per_sample}")
        total_rollouts = len(data) * args.rollouts_per_sample
        print(f"  Total rollouts: {total_rollouts} (no batch limit)")

        sample_results = {}
        for sample_idx, sample in enumerate(data):
            question_id, sample_dir = get_sample_metadata(sample, sample_idx, args.output_dir)
            sample_results[sample_idx] = {
                'results': [],
                'sample': sample,
                'question_id': question_id,
                'sample_dir': sample_dir
            }

        execute_pipeline_parallel_processing(
            samples=data,
            sample_indices=list(range(len(data))),
            sample_results_dict=sample_results,
            args=args,
            experience_retriever=experience_retriever,
            base_system_prompt=BASE_SYSTEM_PROMPT,
            sampling_params=sampling_params,
            run_single_rollout_func=run_single_rollout,
            progress_desc="Processing rollouts",
            all_results_list=all_results
        )

    elif args.num_workers > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_single_sample_with_experience, sample, sample_idx): sample_idx
                for sample_idx, sample in enumerate(data)
            }

            for future in tqdm(as_completed(futures), total=len(data), desc="Processing samples"):
                sample_idx = futures[future]
                try:
                    sample_info = future.result()
                    if sample_info and sample_info.get('sample_rollout_results'):
                        all_results.extend(sample_info['sample_rollout_results'])
                except Exception as e:
                    print(f"  Error processing sample {sample_idx}: {e}")

    else:
        for sample_idx, sample in enumerate(tqdm(data, desc="Processing samples")):
            sample_info = process_single_sample_with_experience(sample, sample_idx)
            if sample_info and sample_info.get('sample_rollout_results'):
                all_results.extend(sample_info['sample_rollout_results'])

    save_results(all_results, args.output_dir)
    print_summary(all_results, args.output_dir)
    if args.rollouts_per_sample > 1:
        print(f"\n{'='*80}")
        print(f"Aggregating pass@k and average@k metrics across all samples")
        print(f"{'='*80}\n")
        compute_dataset_summary(data, args)

    print("API-based inference finished.")


if __name__ == "__main__":
    parser = create_base_parser(description="API-based inference script.")
    args = parser.parse_args()
    main(args)
