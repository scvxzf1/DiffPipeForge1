import argparse
import sys
import os
import json
import time
import traceback
import queue

# Add current dir to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from diffusion_nodes.start import Train
    from diffusion_nodes.general_config import GeneralConfig
    from diffusion_nodes.GeneralDatasetConfig import GeneralDatasetConfig
    from diffusion_nodes.model_config import ModelConfig
    from diffusion_nodes.advanced_train_config import AdvancedTrainConfig
except ImportError:
    print(f"Error importing nodes. Sys.path: {sys.path}")
    import traceback
    traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True, help='Action to perform')
    parser.add_argument('--full_config', help='Full configuration JSON string')
    parser.add_argument('--dataset_config', help='JSON string')
    parser.add_argument('--train_config', help='JSON string')
    parser.add_argument('--config_path', help='Path to save config')
    parser.add_argument('--resume_from_checkpoint', default="")

    args, unknown = parser.parse_known_args()
    
    if args.action == 'start_training':
        try:
            full_config = json.loads(args.full_config) if args.full_config else {}
            
            # --- 1. Dataset Config ---
            print("[Bridge] Generating Dataset Config...")
            dataset_node = GeneralDatasetConfig()
            dataset_inputs = full_config.get('dataset', {})
            # Map frontend keys to backend arguments
            dsc_args = {
                "input_path": dataset_inputs.get('input_path'), # handle dict/str
                "resolutions": dataset_inputs.get('resolutions', '[512]'),
                "enable_ar_bucket": dataset_inputs.get('enable_ar_bucket', True),
                "min_ar": dataset_inputs.get('min_ar', 0.5),
                "max_ar": dataset_inputs.get('max_ar', 2.0),
                "num_ar_buckets": dataset_inputs.get('num_ar_buckets', 7),
                "num_repeats": dataset_inputs.get('num_repeats', 1),
                "frame_buckets": dataset_inputs.get('frame_buckets'),
                "ar_buckets": dataset_inputs.get('ar_buckets')
            }
            # generate_config returns Tuple(config_content_str, )
            dataset_config_content_tuple = dataset_node.generate_config(**dsc_args)
            dataset_config_content = dataset_config_content_tuple[0]
            
            # --- 2. Model Config ---
            print("[Bridge] Generating Model Config...")
            model_node = ModelConfig()
            model_inputs = full_config.get('model', {})
            mc_args = {
                "model_path": model_inputs.get('model_path'),
                "model_type": model_inputs.get('model_type', 'diffusion'),
                "dtype": model_inputs.get('dtype', 'bfloat16'),
                "diffusion_transformer_dtype": model_inputs.get('diffusion_transformer_dtype', 'bfloat16'),
                "timestep_sample_method": model_inputs.get('timestep_sample_method', 'logit_normal')
            }
            model_config_tuple = model_node.generate_model_config(**mc_args)
            model_config_dict = model_config_tuple[0]
            
            # --- 3. Advanced Config (Optional) ---
            advanced_config_dict = None
            if 'advanced' in full_config:
                print("[Bridge] Generating Advanced Config...")
                adv_node = AdvancedTrainConfig()
                adv_inputs = full_config['advanced']
                # INPUT_TYPES keys match
                adv_tuple = adv_node.generate_advanced_config(**adv_inputs)
                advanced_config_json = adv_tuple[0]
                advanced_config_dict = json.loads(advanced_config_json)

            # --- 4. General Config (Training Config) ---
            print("[Bridge] Generating General Config...")
            general_node = GeneralConfig()
            train_inputs = full_config.get('training', {})
            
            # Prepare arguments for GeneralConfig.generate_settings
            gc_args = {
                "optimizer_config": train_inputs.get('optimizer_config', "{}"), # TODO: Optimizer
                "model_config": model_config_dict,
                "dataset_config": dataset_config_content, # Pass the CONTENT as expected by generate_settings logic?
                "output_folder_name": train_inputs.get('output_folder_name', 'mylora'),
                "epochs": train_inputs.get('epochs', 50),
                "micro_batch_size_per_gpu": train_inputs.get('micro_batch_size_per_gpu', 1),
                "pipeline_stages": train_inputs.get('pipeline_stages', 1),
                "gradient_accumulation_steps": train_inputs.get('gradient_accumulation_steps', 4),
                "gradient_clipping": train_inputs.get('gradient_clipping', 1.0),
                "warmup_steps": train_inputs.get('warmup_steps', 500),
                "blocks_to_swap": train_inputs.get('blocks_to_swap', 20),
                "activation_checkpointing": train_inputs.get('activation_checkpointing', True),
                "save_dtype": train_inputs.get('save_dtype', 'bfloat16'),
                "partition_method": train_inputs.get('partition_method', 'parameters'),
                
                # Optionals
                "save_every_n_epochs": train_inputs.get('save_every_n_epochs', 1),
                "advanced_config": advanced_config_dict,
                "adapter_config": None # TODO: Adapter support
            }
            
            # generate_settings returns (train_config_toml, abs_output_dir, config_save_path)
            train_config_tuple = general_node.generate_settings(**gc_args)
            final_config_path = train_config_tuple[2]
            
            print(f"[Bridge] Config generated at: {final_config_path}")
            
            # --- 5. Start Training ---
            print("[Bridge] Starting Training Process...")
            trainer = Train()
            start_args = full_config.get('start_args', {})
            
            status_code, message = trainer.start_training(
                dataset_config=dataset_config_content, # Passed again?
                train_config=train_config_tuple[0], # The TOML string
                config_path=final_config_path, # The Path
                resume_from_checkpoint=start_args.get('resume_from_checkpoint', ""),
                reset_dataloader=start_args.get('reset_dataloader', False),
                regenerate_cache=start_args.get('regenerate_cache', False),
                trust_cache=start_args.get('trust_cache', False),
                cache_only=start_args.get('cache_only', False),
                i_know_what_i_am_doing=start_args.get('i_know_what_i_am_doing', False),
                dump_dataset=start_args.get('dump_dataset', ""),
                reset_optimizer_params=start_args.get('reset_optimizer_params', False)
            )
            
            print(f"__JSON_START__{{\"status\": \"{status_code}\", \"message\": \"Training Initiated\"}}__JSON_END__", flush=True)
            
            if status_code == "TRAINING_STARTED":
                while True:
                    time.sleep(0.5)
                    status_tuple = trainer.get_training_status()
                    if isinstance(status_tuple, tuple):
                        curr_status, curr_msg = status_tuple
                    else:
                         curr_status, curr_msg = "UNKNOWN", str(status_tuple)

                    if curr_status in ["COMPLETED", "FAILED", "STOPPED", "ERROR", "NOT_RUNNING", "NOT_STARTED"]:
                        print(f"__JSON_START__{{\"status\": \"{curr_status}\", \"final_message\": \"Process finished\"}}__JSON_END__", flush=True)
                        break
                        
        except Exception as e:
            err_msg = str(e).replace('"', '\\"').replace('\n', ' ')
            print(f"__JSON_START__{{\"status\": \"ERROR\", \"message\": \"{err_msg}\"}}__JSON_END__", flush=True)
            traceback.print_exc()

    elif args.action == 'check_requirements':
        # ... (same as before) ...
        print(f"__JSON_START__{{\"success\": true}}__JSON_END__", flush=True)

if __name__ == '__main__':
    main()
