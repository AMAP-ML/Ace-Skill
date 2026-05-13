"""API-based training script with experience generation and skill refinement."""

import os
import json
import threading
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.result_utils import save_results, print_summary
from engine.api_processors import process_single_sample

from ace_skill import (
    ExperienceLLM,
    summarize_rollouts,
    intra_sample_experiences,
    batch_merge as exp_batch_merge,
    generate_skill_for_sample,
    merge_skills,
    refine_experience_library,
    refine_skill_document,
    create_sampler,
    compute_sample_reward,
)

from api_utils import (
    create_base_parser,
    setup_common,
    run_single_rollout,
    get_used_original_experiences,
    retrieve_experiences_for_sample,
    compute_and_save_sample_summary,
    check_sample_completed,
    reload_experiences,
    prepare_sample_args,
    get_sample_metadata,
    compute_dataset_summary,
    execute_pipeline_parallel_processing,
    _cluster_skill_path,
)


def current_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_experience_for_sample(sample_info, args):
    """
    Generate experience for a single sample (extracted from process_single_sample_with_experience).
    This function can be called in parallel for multiple samples.
    
    Args:
        sample_info: Dict containing:
            - 'sample': Data sample
            - 'sample_idx': Sample index
            - 'question_id': Question ID
            - 'sample_dir': Sample directory path
            - 'sample_rollout_results': List of rollout results
        args: Command line arguments
        
    Returns:
        Dict with:
            - 'sample_id': Question ID
            - 'success': bool
            - 'experience_ops': List of experience operations (normalized)
            - 'error': str or None
    """
    question_id = sample_info['question_id']
    sample_dir = sample_info['sample_dir']
    sample_rollout_results = sample_info['sample_rollout_results']
    
    result = {
        'sample_id': question_id,
        'success': False,
        'experience_ops': [],
        'skill_content': '',
        'error': None
    }
    
    try:
        llm = ExperienceLLM()
        
        traj_paths = []
        for rollout_idx in range(len(sample_rollout_results)):
            if args.rollouts_per_sample > 1:
                rollout_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}")
            else:
                rollout_dir = sample_dir
            
            traj_path = os.path.join(rollout_dir, 'traj.jsonl')
            if os.path.exists(traj_path):
                traj_paths.append(traj_path)
        
        if not traj_paths:
            result['error'] = "No trajectory files found"
            print(f"  Warning: No trajectory files found for {question_id}")
            return result
        
        merged_summaries = summarize_rollouts(traj_paths, llm, sample_dir=sample_dir)
        
        if not merged_summaries:
            has_any_turns = False
            for traj_path in traj_paths:
                try:
                    with open(traj_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                rec = json.loads(line)
                                if 'turn_idx' in rec and rec.get('turn_idx') is not None:
                                    has_any_turns = True
                                    break
                            except (json.JSONDecodeError, Exception):
                                continue
                    if has_any_turns:
                        break
                except Exception:
                    continue
            
            if not has_any_turns:
                result['error'] = "No turns in trajectories (empty trajectories)"
                print(f"  Warning: Skipping experience generation for {question_id} - trajectories contain no turns (only initial_prompt)")
            else:
                result['error'] = "Failed to generate summary"
                print(f"  Warning: Failed to generate summary for {question_id}")
            return result
        
        question = merged_summaries.get('question', '') or sample_rollout_results[0].get('prompt', '')
        ground_truth = merged_summaries.get('ground_truth', '') or sample_rollout_results[0].get('ground_truth', '')
        system_prompt_text = merged_summaries.get('system_prompt', '')
        
        summaries_only = {k: v for k, v in merged_summaries.items() 
                         if k not in ['question', 'ground_truth', 'system_prompt']}
        
        used_experiences = get_used_original_experiences(sample_dir)
        
        raw_ops = intra_sample_experiences(
            question, ground_truth, summaries_only, llm,
            max_ops=getattr(args, 'experience_max_ops', 2),
            debug_dir=sample_dir,
            system_prompt=system_prompt_text,
            used_experiences=used_experiences
        )
        
        norm_ops = []
        if isinstance(raw_ops, str):
            try:
                raw_ops = json.loads(raw_ops)
            except Exception:
                pass
        
        if isinstance(raw_ops, dict) and 'experiences' in raw_ops:
            for _, v in raw_ops['experiences'].items():
                if isinstance(v, str) and v.strip():
                    norm_ops.append({"experience": v.strip()})
        elif isinstance(raw_ops, list):
            for o in raw_ops:
                if isinstance(o, dict):
                    exp_txt = o.get('experience') or o.get('exp')
                    if isinstance(exp_txt, str) and exp_txt.strip():
                        norm_ops.append({"experience": exp_txt.strip()})
        
        if norm_ops:
            with open(os.path.join(sample_dir, 'exp_items.json'), 'w', encoding='utf-8') as f:
                json.dump(norm_ops, f, ensure_ascii=False, indent=2)
        
        if getattr(args, 'skill_enable', False):
            skill_result = generate_skill_for_sample(sample_info, llm, args, ground_truth=ground_truth)
            if skill_result['success']:
                result['skill_content'] = skill_result['skill_content']
            else:
                print(f"  Warning: Skill generation failed for {question_id}: {skill_result.get('error')}")

        result['success'] = True
        result['experience_ops'] = norm_ops
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def save_knowledge_snapshot(args, batch_idx):
    """
    Save a snapshot of current experience library and skill document before batch processing.
    
    Args:
        args: Command line arguments
        batch_idx: Current batch index (0-based)
    """
    import shutil
    
    snapshot_dir = os.path.join(args.output_dir, "snapshots", f"batch_{batch_idx:03d}")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    metadata = {
        "batch_idx": batch_idx,
        "timestamp": datetime.now().isoformat(),
    }
    
    snapshot_exp_path = os.path.join(snapshot_dir, "experiences.json")
    if args.experience_library and os.path.exists(args.experience_library):
        shutil.copy2(args.experience_library, snapshot_exp_path)
    if os.path.exists(snapshot_exp_path):
        try:
            with open(snapshot_exp_path, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)
            if isinstance(exp_data, list):
                metadata["experience_count"] = len(exp_data)
            elif isinstance(exp_data, dict):
                inner = exp_data.get("experiences", {})
                metadata["experience_count"] = len(inner) if isinstance(inner, (list, dict)) else 0
            else:
                metadata["experience_count"] = 0
        except Exception:
            metadata["experience_count"] = -1
    else:
        metadata["experience_count"] = 0
    
    if getattr(args, 'skill_library', None) and os.path.exists(args.skill_library):
        shutil.copy2(args.skill_library, os.path.join(snapshot_dir, "SKILL.md"))
        try:
            with open(args.skill_library, 'r', encoding='utf-8') as f:
                skill_content = f.read()
            metadata["skill_word_count"] = len(skill_content.split())
        except Exception:
            metadata["skill_word_count"] = -1
    else:
        metadata["skill_word_count"] = 0

    if getattr(args, 'skill_library', None):
        import glob as glob_mod
        skill_dir = os.path.dirname(args.skill_library)
        base_name = os.path.splitext(os.path.basename(args.skill_library))[0]
        ext = os.path.splitext(args.skill_library)[1]
        pattern = os.path.join(skill_dir, f"{base_name}_cluster_*{ext}")
        cluster_skill_files = glob_mod.glob(pattern)
        cluster_word_counts = {}
        for csf in cluster_skill_files:
            shutil.copy2(csf, os.path.join(snapshot_dir, os.path.basename(csf)))
            try:
                with open(csf, 'r', encoding='utf-8') as f:
                    cluster_word_counts[os.path.basename(csf)] = len(f.read().split())
            except Exception:
                cluster_word_counts[os.path.basename(csf)] = -1
        metadata["cluster_skill_files"] = len(cluster_skill_files)
        metadata["cluster_skill_word_counts"] = cluster_word_counts
    
    with open(os.path.join(snapshot_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"  [Snapshot] Saved batch_{batch_idx:03d} snapshot (exp: {metadata['experience_count']}, skill: {metadata['skill_word_count']} words)")


def _merge_experiences_clustered(all_ops, args, manager, experience_limit, is_final):
    """Merge experience ops per-cluster using the ClusteredExperienceManager."""
    from collections import defaultdict
    ops_by_cluster = defaultdict(list)
    for op in all_ops:
        cid = op.pop('_cluster_id', -1)
        ops_by_cluster[cid].append(op)

    total_merge_time = 0.0
    for cid, cluster_ops in sorted(ops_by_cluster.items()):
        if cid == -1:
            print(f"  [Cluster] Warning: {len(cluster_ops)} ops have no cluster assignment, skipping")
            continue

        existing = manager.load_cluster_library(cid)
        cluster_retriever = manager.retrievers.get(cid)
        merge_start = time.time()
        merged = exp_batch_merge(
            existing, cluster_ops, ExperienceLLM(),
            debug_dir=None,
            experience_limit=experience_limit,
            retriever=cluster_retriever,
        )
        merge_time = time.time() - merge_start
        total_merge_time += merge_time

        if getattr(args, 'experience_refine', False):
            if is_final or len(merged) > experience_limit:
                refine_start = time.time()
                merged = refine_experience_library(merged, ExperienceLLM(), debug_dir=args.output_dir)
                print(f"  [Timing] Cluster {cid} refine{'(final)' if is_final else ''}: "
                      f"{time.time() - refine_start:.1f}s")

        manager.save_cluster_library(cid, merged)
        manager.update_cluster_experiences(cid, merged)
        print(f"  [Cluster {cid}] Merged {len(cluster_ops)} ops (final size: {len(merged)}, {merge_time:.1f}s)")

    print(f"  [Timing] Total cluster merge: {total_merge_time:.1f}s")


def _merge_skills_clustered(all_skill_contents, args, is_final):
    """Merge skill contents per-cluster into separate skill files."""
    from collections import defaultdict

    skills_by_cluster = defaultdict(list)
    for entry in all_skill_contents:
        cid = entry.get('cluster_id', -1)
        skills_by_cluster[cid].append(entry['content'])

    total_merge_time = 0.0
    skill_llm = ExperienceLLM()
    skill_max_length = getattr(args, 'skill_max_length', 1000)

    for cid, contents in sorted(skills_by_cluster.items()):
        if cid == -1:
            print(f"  [Skill Cluster] Warning: {len(contents)} skills have no cluster assignment, skipping")
            continue

        cluster_path = _cluster_skill_path(args.skill_library, cid)
        existing_content = ""
        if os.path.exists(cluster_path):
            with open(cluster_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()

        merge_start = time.time()
        merged = merge_skills(existing_content, contents, skill_llm, args)

        if getattr(args, 'skill_refine', False):
            word_count = len(merged.split())
            if is_final or word_count > skill_max_length:
                refine_start = time.time()
                merged = refine_skill_document(
                    merged, skill_llm, skill_path=cluster_path,
                    word_threshold=skill_max_length, force_refine=is_final,
                )
                print(f"  [Timing] Skill cluster {cid} refine{'(final)' if is_final else ''}: "
                      f"{time.time() - refine_start:.1f}s")

        os.makedirs(os.path.dirname(cluster_path), exist_ok=True)
        with open(cluster_path, 'w', encoding='utf-8') as f:
            f.write(merged)

        merge_time = time.time() - merge_start
        total_merge_time += merge_time
        print(f"  [Skill Cluster {cid}] Merged {len(contents)} skills ({merge_time:.1f}s)")

    print(f"  [Timing] Total skill cluster merge: {total_merge_time:.1f}s")


def process_large_batch_experiences(samples_info, args, batch_idx=0, is_final=False):
    """
    Process experience generation for a large batch of samples in parallel,
    then merge all experiences into the library.
    
    Args:
        samples_info: List of sample_info dicts (from generate_experience_for_sample)
        args: Command line arguments
        batch_idx: Current batch index for snapshot naming
        is_final: If True, force refine regardless of threshold (for final batch)
        
    Returns:
        Number of successfully processed samples
    """
    if not samples_info:
        return 0
    start_time = time.time()
    print(f"\n[Large Batch {batch_idx}] Start time: {current_timestamp_str()}")
    print(f"\n[Large Batch {batch_idx}] Processing experience generation for {len(samples_info)} samples...")
    
    save_knowledge_snapshot(args, batch_idx)
    
    all_experience_ops = []
    all_skill_contents = []
    successful_samples = 0

    gen_start = time.time()
    num_tasks = len(samples_info)
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        exp_futures = {
            executor.submit(generate_experience_for_sample, sample_info, args): sample_info
            for sample_info in samples_info
        }
        for future in as_completed(exp_futures):
            sample_info = exp_futures[future]
            try:
                result = future.result()
                if result['success']:
                    ops = result['experience_ops']
                    cluster_id = sample_info.get('sample', {}).get('_cluster_id', -1)
                    for op in ops:
                        op['_cluster_id'] = cluster_id
                    all_experience_ops.extend(ops)
                    if result.get('skill_content'):
                        all_skill_contents.append({
                            'content': result['skill_content'],
                            'cluster_id': cluster_id,
                        })
                    successful_samples += 1
                else:
                    print(f"  Warning: Generation failed for {result['sample_id']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  Warning: Generation failed for {sample_info['question_id']}: {e}")
    
    gen_time = time.time() - gen_start
    print(f"  [Timing] Experience generation: {gen_time:.1f}s")
    
    if all_experience_ops and getattr(args, 'experience_library_update', False) and args.experience_library:
        try:
            if not hasattr(process_large_batch_experiences, '_experience_lock'):
                process_large_batch_experiences._experience_lock = threading.Lock()
            
            with process_large_batch_experiences._experience_lock:
                retriever = getattr(args, '_experience_retriever', None)
                experience_limit = getattr(args, 'experience_max_items', 16)

                if not hasattr(retriever, 'cluster_ids'):
                    raise RuntimeError(
                        "Cluster-only pipeline requires a ClusteredExperienceManager "
                        "as the experience retriever; got "
                        f"{type(retriever).__name__ if retriever is not None else 'None'}."
                    )
                _merge_experiences_clustered(
                    all_experience_ops, args, retriever, experience_limit, is_final,
                )
        except Exception as e:
            print(f"  Warning: Failed to merge experiences into library: {e}")
            
    if all_skill_contents and getattr(args, 'skill_library', None):
        try:
            if not hasattr(process_large_batch_experiences, '_skill_lock'):
                process_large_batch_experiences._skill_lock = threading.Lock()
            
            with process_large_batch_experiences._skill_lock:
                _merge_skills_clustered(all_skill_contents, args, is_final)
        except Exception as e:
            print(f"  Warning: Failed to merge skills into library: {e}")
    
    total_time = time.time() - start_time
    print(f"[Large Batch] Completed: {successful_samples}/{len(samples_info)} samples processed successfully (total: {total_time:.1f}s)")
    return successful_samples


def main(args):
    """
    Main function to run the API-based training process with experience generation.
    """
    # Validate and set experience batch parameters (train-specific)
    if getattr(args, 'experience_online_generate', False):
        small_batch = args.rollouts_per_sample
        large_batch = getattr(args, 'experience_large_batch', None) or small_batch
        args.experience_large_batch = large_batch

        if large_batch % small_batch != 0:
            raise ValueError(
                f"experience_large_batch ({large_batch}) must be a multiple of "
                f"rollouts_per_sample ({small_batch})"
            )

        args.experience_samples_per_large_batch = large_batch // small_batch

    # Shared setup
    result = setup_common(args)
    if result is None:
        return
    data, sampling_params, BASE_SYSTEM_PROMPT, experience_retriever = result
    # Attach retriever to args so process_large_batch_experiences can pass it
    # to batch_merge for embedding cache pre-population.
    args._experience_retriever = experience_retriever

    # Print train-specific batch info after shared setup
    if getattr(args, 'experience_online_generate', False):
        print(f"\n{'='*80}")
        print(f"Experience Generation Batching:")
        print(f"  Small batch (rollouts per sample): {args.rollouts_per_sample}")
        print(f"  Large batch (rollouts per batch): {args.experience_large_batch}")
        print(f"  Samples per large batch: {args.experience_samples_per_large_batch}")
        print(f"{'='*80}\n")

    # Initialize weighted sampler (None == original sequential order)
    weighted_sampler = None
    sampler_state_path = None
    max_sampling_samples = None
    sampling_config_path = getattr(args, 'sampling_config', None)
    if sampling_config_path:
        sampler_state_path = os.path.join(args.output_dir, 'sampler_state.json')
        weighted_sampler = create_sampler(
            n_samples=len(data),
            config_path=sampling_config_path,
            state_path=sampler_state_path,
            seed=args.seed_base,
        )
        max_sampling_samples = getattr(args, 'max_sampling_samples', None)
        diag = weighted_sampler.get_diagnostics()
        print(f"\n{'='*80}")
        print(f"Weighted Sampling Enabled:")
        print(f"  Config : {sampling_config_path}")
        print(f"  Algorithm : {diag['algorithm']}")
        print(f"  Samples : {len(data)}")
        if max_sampling_samples is not None:
            print(f"  Max samples : {max_sampling_samples}")
        else:
            print(f"  Max samples : {len(data)} (one epoch)")
        if 'rho' in diag:
            print(f"  rho={diag['rho']}, gamma={diag['gamma']}, epsilon={diag['epsilon']}")
        print(f"  Iter output   : Enabled (sample dirs prefixed with iter_XXX/)")
        print(f"{'='*80}\n")

    samples_processed = 0

    all_results = []

    # Large batch management for experience generation
    large_batch_queue = []
    large_batch_idx = 0
    use_experience_batching = getattr(args, 'experience_online_generate', False)

    def process_single_sample_with_experience(sample, sample_idx, iter_tag=None):
        """Process a single sample including experience generation."""
        sample_args = prepare_sample_args(args)
        if iter_tag is not None:
            sample_args.output_dir = os.path.join(sample_args.output_dir, iter_tag)

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

    # ------------------------------------------------------------------
    # Unified termination helper for the training loop.
    # When weighted_sampler is active:
    #   - if max_sampling_samples is set, stop after that many samples
    #     (allows multi-epoch if > len(data), early-stop if < len(data))
    #   - otherwise, run exactly one epoch (all samples once)
    # When weighted_sampler is None: iterate sequentially until all done.
    # ------------------------------------------------------------------
    def _should_continue(seq_idx: int) -> bool:
        if weighted_sampler is not None:
            limit = max_sampling_samples if max_sampling_samples is not None else len(data)
            return samples_processed < limit
        return seq_idx < len(data)

    if args.num_workers > 1 and args.rollouts_per_sample > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")
        print(f"  Rollouts per sample: {args.rollouts_per_sample}")
        max_concurrent_samples = args.num_workers // args.rollouts_per_sample

        if use_experience_batching:
            print(f"  Max concurrent samples per batch: {max_concurrent_samples} (limited by {args.num_workers} workers)")

            sample_idx = 0
            sampling_step = 0
            batch_num = 0
            while _should_continue(sample_idx):

                if weighted_sampler is not None:
                    want = max_concurrent_samples
                    if max_sampling_samples is not None:
                        want = min(want, max_sampling_samples - samples_processed)
                    batch_indices = weighted_sampler.sample_batch(want, sampling_step)
                    batch_samples = [data[i] for i in batch_indices]
                else:
                    batch_size = min(max_concurrent_samples, len(data) - sample_idx)
                    batch_indices = list(range(sample_idx, sample_idx + batch_size))
                    batch_samples = data[sample_idx:sample_idx + batch_size]

                batch_num += 1
                iter_output_dir = args.output_dir
                if weighted_sampler is not None:
                    iter_output_dir = os.path.join(args.output_dir, f"iter_{sampling_step:03d}")
                print(f"\n[Batch] Start time: {current_timestamp_str()}")
                print(f"[Batch] Processing {len(batch_indices)} samples (batch {batch_num})...")

                batch_sample_results = {}
                for local_idx, sample in enumerate(batch_samples):
                    actual_sample_idx = batch_indices[local_idx]
                    question_id, sample_dir = get_sample_metadata(sample, actual_sample_idx, iter_output_dir)

                    batch_sample_results[actual_sample_idx] = {
                        'results': [],
                        'sample': sample,
                        'question_id': question_id,
                        'sample_dir': sample_dir
                    }

                execute_pipeline_parallel_processing(
                    samples=batch_samples,
                    sample_indices=batch_indices,
                    sample_results_dict=batch_sample_results,
                    args=args,
                    experience_retriever=experience_retriever,
                    base_system_prompt=BASE_SYSTEM_PROMPT,
                    sampling_params=sampling_params,
                    run_single_rollout_func=run_single_rollout,
                    progress_desc=f"Batch {batch_num}",
                    all_results_list=all_results
                )

                batch_sample_infos = []
                for actual_sample_idx in batch_indices:
                    if actual_sample_idx in batch_sample_results:
                        sample_info = batch_sample_results[actual_sample_idx]
                        sample_info_dict = {
                            'sample': sample_info['sample'],
                            'sample_idx': actual_sample_idx,
                            'question_id': sample_info['question_id'],
                            'sample_dir': sample_info['sample_dir'],
                            'sample_rollout_results': sample_info['results']
                        }
                        batch_sample_infos.append(sample_info_dict)

                if weighted_sampler is not None:
                    for info in batch_sample_infos:
                        reward = compute_sample_reward(info)
                        weighted_sampler.update([info['sample_idx']], [reward], sampling_step)
                    sampling_step += 1

                samples_processed += len(batch_indices)

                large_batch_queue.extend(batch_sample_infos)

                if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                    print(f"\n[Large Batch] Triggering experience generation for {len(large_batch_queue)} samples...")
                    process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                    large_batch_idx += 1
                    large_batch_queue.clear()
                    reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)

                if weighted_sampler is None:
                    sample_idx += batch_size

        else:
            # No experience batching
            if weighted_sampler is not None:
                # Weighted sampling needs iterative batch processing for feedback
                sampling_step = 0
                batch_num = 0
                while _should_continue(0):
                    want = max_concurrent_samples
                    if max_sampling_samples is not None:
                        want = min(want, max_sampling_samples - samples_processed)
                    batch_indices = weighted_sampler.sample_batch(want, sampling_step)
                    batch_samples = [data[i] for i in batch_indices]

                    batch_num += 1
                    iter_output_dir = os.path.join(args.output_dir, f"iter_{sampling_step:03d}")
                    print(f"\n[Batch] Start time: {current_timestamp_str()}")
                    print(f"[Batch] Processing {len(batch_indices)} weighted-sampled samples (batch {batch_num})...")

                    batch_sample_results = {}
                    for local_idx, sample in enumerate(batch_samples):
                        actual_sample_idx = batch_indices[local_idx]
                        question_id, sample_dir = get_sample_metadata(sample, actual_sample_idx, iter_output_dir)
                        batch_sample_results[actual_sample_idx] = {
                            'results': [],
                            'sample': sample,
                            'question_id': question_id,
                            'sample_dir': sample_dir
                        }

                    execute_pipeline_parallel_processing(
                        samples=batch_samples,
                        sample_indices=batch_indices,
                        sample_results_dict=batch_sample_results,
                        args=args,
                        experience_retriever=experience_retriever,
                        base_system_prompt=BASE_SYSTEM_PROMPT,
                        sampling_params=sampling_params,
                        run_single_rollout_func=run_single_rollout,
                        progress_desc=f"Batch {batch_num}",
                        all_results_list=all_results
                    )

                    for actual_sample_idx in batch_indices:
                        if actual_sample_idx in batch_sample_results:
                            info = batch_sample_results[actual_sample_idx]
                            reward = compute_sample_reward({
                                'sample_rollout_results': info['results']
                            })
                            weighted_sampler.update([actual_sample_idx], [reward], sampling_step)

                    samples_processed += len(batch_indices)
                    sampling_step += 1

            else:
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
                samples_processed += len(data)

    elif args.num_workers > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")

        if use_experience_batching:
            max_concurrent_samples = args.num_workers
            batch_size = min(args.experience_samples_per_large_batch, max_concurrent_samples)

            sample_idx = 0
            sampling_step = 0
            batch_num = 0
            while _should_continue(sample_idx):

                if weighted_sampler is not None:
                    want = batch_size
                    if max_sampling_samples is not None:
                        want = min(want, max_sampling_samples - samples_processed)
                    batch_indices = weighted_sampler.sample_batch(want, sampling_step)
                    batch_samples = [data[i] for i in batch_indices]
                else:
                    actual_batch_size = min(batch_size, len(data) - sample_idx)
                    batch_indices = list(range(sample_idx, sample_idx + actual_batch_size))
                    batch_samples = data[sample_idx:sample_idx + actual_batch_size]

                batch_num += 1
                iter_tag = f"iter_{sampling_step:03d}" if weighted_sampler is not None else None
                print(f"\n[Batch] Start time: {current_timestamp_str()}")
                print(f"[Batch] Processing {len(batch_indices)} samples (batch {batch_num})...")
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = {
                        executor.submit(process_single_sample_with_experience, sample, actual_idx, iter_tag): actual_idx
                        for actual_idx, sample in zip(batch_indices, batch_samples)
                    }

                    batch_sample_infos = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num}"):
                        actual_idx = futures[future]
                        try:
                            sample_info = future.result()
                            if sample_info and sample_info.get('sample_rollout_results'):
                                all_results.extend(sample_info['sample_rollout_results'])
                                batch_sample_infos.append(sample_info)
                        except Exception as e:
                            print(f"  Error processing sample {actual_idx}: {e}")

                if weighted_sampler is not None:
                    for info in batch_sample_infos:
                        reward = compute_sample_reward(info)
                        weighted_sampler.update([info['sample_idx']], [reward], sampling_step)
                    sampling_step += 1

                samples_processed += len(batch_indices)

                if batch_sample_infos:
                    large_batch_queue.extend(batch_sample_infos)

                    if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                        print(f"\n[Large Batch] Triggering experience generation for {len(large_batch_queue)} samples...")
                        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                        large_batch_idx += 1
                        large_batch_queue.clear()
                        reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)

                if weighted_sampler is None:
                    sample_idx += actual_batch_size

            if large_batch_queue:
                print(f"\n[Final Batch] Processing remaining {len(large_batch_queue)} samples...")
                process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx, is_final=True)
                large_batch_idx += 1
                large_batch_queue.clear()

        else:
            # No experience batching
            if weighted_sampler is not None:
                sampling_step = 0
                batch_num = 0
                while _should_continue(0):
                    want = args.num_workers
                    if max_sampling_samples is not None:
                        want = min(want, max_sampling_samples - samples_processed)
                    batch_indices = weighted_sampler.sample_batch(want, sampling_step)
                    batch_samples = [data[i] for i in batch_indices]

                    batch_num += 1
                    iter_tag = f"iter_{sampling_step:03d}"
                    print(f"\n[Batch] Start time: {current_timestamp_str()}")
                    print(f"[Batch] Processing {len(batch_indices)} weighted-sampled samples (batch {batch_num})...")

                    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                        futures = {
                            executor.submit(process_single_sample_with_experience, sample, actual_idx, iter_tag): actual_idx
                            for actual_idx, sample in zip(batch_indices, batch_samples)
                        }
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num}"):
                            actual_idx = futures[future]
                            try:
                                sample_info = future.result()
                                if sample_info and sample_info.get('sample_rollout_results'):
                                    all_results.extend(sample_info['sample_rollout_results'])
                                    reward = compute_sample_reward(sample_info)
                                    weighted_sampler.update([sample_info['sample_idx']], [reward], sampling_step)
                            except Exception as e:
                                print(f"  Error processing sample {actual_idx}: {e}")

                    samples_processed += len(batch_indices)
                    sampling_step += 1

            else:
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
                samples_processed += len(data)

    else:
        # Sequential mode (num_workers == 1)
        if weighted_sampler is not None:
            sampling_step = 0
            total_target = max_sampling_samples if max_sampling_samples is not None else len(data)
            pbar = tqdm(total=total_target, desc="Processing samples (weighted)")
            while _should_continue(0):
                indices = weighted_sampler.sample_batch(1, sampling_step)
                sample_idx = indices[0]
                sample = data[sample_idx]

                sample_info = process_single_sample_with_experience(sample, sample_idx, iter_tag=f"iter_{sampling_step:03d}")
                if sample_info and sample_info.get('sample_rollout_results'):
                    all_results.extend(sample_info['sample_rollout_results'])
                    reward = compute_sample_reward(sample_info)
                    weighted_sampler.update(indices, [reward], sampling_step)
                    if use_experience_batching:
                        large_batch_queue.append(sample_info)
                        if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                            process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                            large_batch_idx += 1
                            large_batch_queue.clear()
                            reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)
                sampling_step += 1
                samples_processed += 1
                pbar.update(1)
            pbar.close()
        else:
            for sample_idx, sample in enumerate(tqdm(data, desc="Processing samples")):
                sample_info = process_single_sample_with_experience(sample, sample_idx)
                if sample_info and sample_info.get('sample_rollout_results'):
                    all_results.extend(sample_info['sample_rollout_results'])
                    if use_experience_batching:
                        large_batch_queue.append(sample_info)
                        if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                            process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                            large_batch_idx += 1
                            large_batch_queue.clear()
                            reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)
            samples_processed += len(data)

    # Process remaining samples in the queue (if any)
    if use_experience_batching and large_batch_queue:
        print(f"\n[Final Batch] Processing remaining {len(large_batch_queue)} samples...")
        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx, is_final=True)
        large_batch_idx += 1
        large_batch_queue.clear()

    # Persist sampler state for resumption
    if weighted_sampler is not None and sampler_state_path:
        weighted_sampler.save_state(sampler_state_path)
        diag = weighted_sampler.get_diagnostics()
        print(f"\n[WeightedSampler] State saved to {sampler_state_path}")
        if diag['algorithm'] == 'sqrt_bias':
            print(f"  Weight stats: mean={diag['weight_mean']:.4f}, "
                  f"std={diag['weight_std']:.4f}, "
                  f"min={diag['weight_min']:.4f}, "
                  f"max={diag['weight_max']:.4f}")
        elif diag['algorithm'] == 'uniform':
            print(f"  Algorithm: uniform, "
                  f"n_samples={diag['n_samples']}, "
                  f"remaining={diag['remaining']}")
        else:
            raise NotImplementedError(
                f"Unsupported algorithm '{diag['algorithm']}' for diagnostics output"
            )

    save_results(all_results, args.output_dir)
    print_summary(all_results, args.output_dir)
    if args.rollouts_per_sample > 1:
        print(f"\n{'='*80}")
        print(f"Aggregating pass@k and average@k metrics across all samples")
        print(f"{'='*80}\n")
        compute_dataset_summary(data, args)

    print("API-based inference finished.")


if __name__ == "__main__":
    parser = create_base_parser(description="API-based training script with experience generation.")

    # Train-specific arguments
    parser.add_argument("--experience-online-generate", action='store_true',
                       help="Generate per-sample experiences after inference")
    parser.add_argument("--experience-library-update", action='store_true',
                       help="Merge per-sample experiences back into the library")
    parser.add_argument("--experience-max-ops", type=int, default=2,
                       help="Max operations per sample during critique")
    parser.add_argument("--experience-large-batch", type=int, default=None,
                        help="Large batch size for experience generation (number of rollouts to trigger batch processing). "
                             "Must be a multiple of rollouts_per_sample. If None, equals rollouts_per_sample (no batching).")
    parser.add_argument("--experience-refine", action='store_true',
                       help="Enable experience library refinement after merge (uses --experience-max-items as threshold)")
    parser.add_argument("--skill-refine", action='store_true',
                       help="Enable skill refinement after merge (trim and consolidate)")
    parser.add_argument("--skill-max-length", type=int, default=1000,
                       help="Word count threshold to trigger skill refinement (default: 1000)")
    parser.add_argument("--sampling-config", type=str, default=None,
                       help="Path to weighted sampling config YAML (e.g. configs/run1.yaml). "
                            "If not specified, samples are processed in original order.")
    parser.add_argument("--max-sampling-samples", type=int, default=None,
                       help="Maximum number of samples to process when weighted sampling is enabled. "
                            "Allows early stop (< dataset size) or multi-epoch training (> dataset size). "
                            "Only effective when --sampling-config is set.")

    args = parser.parse_args()
    main(args)
