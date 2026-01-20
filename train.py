#!/usr/bin/env python3
"""
Gravitational Wave Detection Deep Learning Training Pipeline.

This script automates the end-to-end process for training a deep learning model 
to detect gravitational waves. It includes:
- Cloning required repositories
- Setting up the dataset directory structure
- Generating synthetic gravitational wave data
- Generating waveforms from the data
- Training the neural network model

The script integrates with TransformerLab for automatic tracking of:
- Artifacts (generated datasets, waveforms)
- Checkpoints (model checkpoints during training)
- Metrics and logs
"""

import os
import sys
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

from lab import lab


def run_command(command, description, cwd=None):
    """
    Execute a shell command with logging to TransformerLab.
    
    Args:
        command: Command to execute (string)
        description: Description of the command for logging
        cwd: Working directory for the command
    
    Returns:
        bool: True if command succeeded, False otherwise
    """
    lab.log(f"▶️  {description}")
    lab.log(f"   Command: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                lab.log(f"   {output.strip()}")
        
        # Read any remaining stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            for line in stderr_output.splitlines():
                lab.log(f"   {line}")
        
        returncode = process.poll()
        
        if returncode != 0:
            lab.log(f"❌ Command failed with return code {returncode}")
            return False
        
        lab.log(f"✅ {description} completed")
        return True
        
    except Exception as e:
        lab.log(f"❌ Error executing command: {e}")
        return False


def setup_repositories(base_dir):
    """
    Clone the required repositories.
    
    Args:
        base_dir: Base directory for the project
    
    Returns:
        dict: Paths to the cloned repositories
    """
    lab.log("\n" + "="*60)
    lab.log("STEP 1: Setting up repositories")
    lab.log("="*60)
    
    os.makedirs(base_dir, exist_ok=True)
    
    challenge_dir = os.path.join(base_dir, "ml-mock-data-challenge-1")
    gw_dir = os.path.join(base_dir, "gw-detection-deep-learning")
    
    # Clone ML Mock Data Challenge repository
    if not os.path.exists(challenge_dir):
        cmd = f"git clone https://github.com/gwastro/ml-mock-data-challenge-1.git {challenge_dir}"
        if not run_command(cmd, "Cloning ML Mock Data Challenge repository", base_dir):
            lab.error("Failed to clone ML Mock Data Challenge repository")
            return None
    else:
        lab.log(f"Repository already exists at {challenge_dir}")
    
    # Clone GW Detection Deep Learning repository
    if not os.path.exists(gw_dir):
        cmd = f"git clone https://github.com/vivinousi/gw-detection-deep-learning.git {gw_dir}"
        if not run_command(cmd, "Cloning GW Detection Deep Learning repository", base_dir):
            lab.error("Failed to clone GW Detection Deep Learning repository")
            return None
    else:
        lab.log(f"Repository already exists at {gw_dir}")
    
    return {
        "challenge_dir": challenge_dir,
        "gw_dir": gw_dir,
        "base_dir": base_dir
    }


def setup_dataset_structure(gw_dir):
    """
    Create the dataset-4 directory structure.
    
    Args:
        gw_dir: Path to GW detection repository
    
    Returns:
        str: Path to the dataset directory
    """
    lab.log("\n" + "="*60)
    lab.log("STEP 2: Setting up dataset directory structure")
    lab.log("="*60)
    
    dataset_dir = os.path.join(gw_dir, "dataset-4", "v2")
    os.makedirs(dataset_dir, exist_ok=True)
    lab.log(f"Created dataset directory: {dataset_dir}")
    
    return dataset_dir


def generate_datasets(challenge_dir, dataset_dir, config):
    """
    Generate training, validation, and test datasets.
    
    Args:
        challenge_dir: Path to ML Mock Data Challenge repository
        dataset_dir: Path to output dataset directory
        config: Configuration dictionary with dataset parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    lab.log("\n" + "="*60)
    lab.log("STEP 3: Generating datasets")
    lab.log("="*60)
    
    data_set = config.get("data_set", "4")
    train_samples = config.get("train_samples", "2000")
    val_samples = config.get("val_samples", "1000")
    test_samples = config.get("test_samples", "1000")
    
    generate_data_script = os.path.join(challenge_dir, "generate_data.py")
    
    datasets = [
        {
            "name": "Training Set",
            "samples": train_samples,
            "seed": "0",
            "prefix": "train_injections_s24w61w_1",
        },
        {
            "name": "Validation Set",
            "samples": val_samples,
            "seed": "1",
            "prefix": "val_injections_s24w6d1_1",
        },
        {
            "name": "Test Set",
            "samples": test_samples,
            "seed": "2",
            "prefix": "test_injections_s24w6d1_1",
        },
    ]
    
    for dataset in datasets:
        injection_file = os.path.join(dataset_dir, f"{dataset['prefix']}.hdf")
        background_file = os.path.join(dataset_dir, dataset['prefix'].replace('injections', 'background') + ".hdf")
        foreground_file = os.path.join(dataset_dir, dataset['prefix'].replace('injections', 'foreground') + ".hdf")
        
        cmd = (
            f"python3 {generate_data_script} "
            f"-d {data_set} -s {dataset['samples']} "
            f"-i {injection_file} "
            f"-b {background_file} "
            f"-f {foreground_file} "
            f"--verbose --force --seed {dataset['seed']}"
        )
        
        if not run_command(cmd, f"Generating {dataset['name']}", challenge_dir):
            lab.error(f"Failed to generate {dataset['name']}")
            return False
        
        # Save generated file as artifact
        for file_path in [injection_file, background_file, foreground_file]:
            if os.path.exists(file_path):
                try:
                    artifact_path = lab.save_artifact(file_path, os.path.basename(file_path))
                    lab.log(f"Saved artifact: {artifact_path}")
                except Exception as e:
                    lab.log(f"Note: Could not save artifact {file_path}: {e}")
    
    lab.log("✅ All datasets generated successfully")
    return True


def generate_waveforms(challenge_dir, dataset_dir, gw_dir, config):
    """
    Generate waveforms from the datasets.
    
    Args:
        challenge_dir: Path to ML Mock Data Challenge repository
        dataset_dir: Path to dataset directory
        gw_dir: Path to GW detection repository
        config: Configuration dictionary
    
    Returns:
        bool: True if successful, False otherwise
    """
    lab.log("\n" + "="*60)
    lab.log("STEP 4: Generating waveforms")
    lab.log("="*60)
    
    slice_duration = config.get("slice_duration", "1.25")
    waveform_gen_script = os.path.join(challenge_dir, "utils", "waveform_gen.py")
    
    # Add the challenge dir to PYTHONPATH so modules can be imported
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{challenge_dir}:{env.get('PYTHONPATH', '')}"
    
    # Fix hardcoded dataset-2 -> dataset-4
    waveform_config = os.path.join(gw_dir, "utils", "waveform_gen.py")
    if os.path.exists(waveform_config):
        lab.log("Fixing dataset references in waveform_gen.py...")
        try:
            with open(waveform_config, 'r') as f:
                content = f.read()
            content = content.replace("dataset-2", "dataset-4")
            with open(waveform_config, 'w') as f:
                f.write(content)
            lab.log("✅ Updated dataset references")
        except Exception as e:
            lab.log(f"Note: Could not update waveform_gen.py: {e}")
    
    # Generate validation waveforms
    cmd = (
        f"python3 {waveform_gen_script} "
        f"--background-file {os.path.join(dataset_dir, 'val_background_s24w6d1_1.hdf')} "
        f"--injection-file {os.path.join(dataset_dir, 'val_injections_s24w6d1_1.hdf')} "
        f"--output-file {os.path.join(dataset_dir, 'val_injections_s24w6d1_1.25s.npy')} "
        f"--slice-duration {slice_duration}"
    )
    
    if not run_command(cmd, "Generating validation waveforms", gw_dir):
        lab.error("Failed to generate validation waveforms")
        return False
    
    # Generate training waveforms
    cmd = (
        f"python3 {waveform_gen_script} "
        f"--background-file {os.path.join(dataset_dir, 'train_background_s24w61w_1.hdf')} "
        f"--injection-file {os.path.join(dataset_dir, 'train_injections_s24w61w_1.hdf')} "
        f"--output-file {os.path.join(dataset_dir, 'train_injections_s24w61w_1.25s_all.npy')} "
        f"--slice-duration {slice_duration}"
    )
    
    if not run_command(cmd, "Generating training waveforms", gw_dir):
        lab.error("Failed to generate training waveforms")
        return False
    
    # Save waveform artifacts
    for waveform_file in [
        "val_injections_s24w6d1_1.25s.npy",
        "train_injections_s24w61w_1.25s_all.npy"
    ]:
        file_path = os.path.join(dataset_dir, waveform_file)
        if os.path.exists(file_path):
            try:
                artifact_path = lab.save_artifact(file_path, waveform_file)
                lab.log(f"Saved waveform artifact: {artifact_path}")
            except Exception as e:
                lab.log(f"Note: Could not save waveform artifact {file_path}: {e}")
    
    lab.log("✅ All waveforms generated successfully")
    return True


def train_model(gw_dir, dataset_dir, config):
    """
    Train the gravitational wave detection model.
    
    Args:
        gw_dir: Path to GW detection repository
        dataset_dir: Path to dataset directory
        config: Configuration dictionary
    
    Returns:
        bool: True if successful, False otherwise
    """
    lab.log("\n" + "="*60)
    lab.log("STEP 5: Training the model")
    lab.log("="*60)
    
    output_dir = os.path.join(gw_dir, "runs", "train_output_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    slice_dur = config.get("slice_dur", "4.25")
    slice_stride = config.get("slice_stride", "2")
    learning_rate = config.get("learning_rate", "1e-3")
    lr_milestones = config.get("lr_milestones", "5,10")
    gamma = config.get("gamma", "0.5")
    epochs = config.get("epochs", "15")
    warmup_epochs = config.get("warmup_epochs", "1")
    p_augment = config.get("p_augment", "0.2")
    batch_size = config.get("batch_size", "1")
    train_device = config.get("train_device", "cuda:0")
    
    train_script = os.path.join(gw_dir, "train.py")
    
    cmd = (
        f"python3 {train_script} "
        f"--output-dir {output_dir} "
        f"--slice-dur {slice_dur} "
        f"--slice-stride {slice_stride} "
        f"--learning-rate {learning_rate} "
        f"--lr-milestones {lr_milestones} "
        f"--gamma {gamma} "
        f"--epochs {epochs} "
        f"--warmup-epochs {warmup_epochs} "
        f"--p-augment {p_augment} "
        f"--train-device {train_device} "
        f"--data-dir {dataset_dir} "
        f"--batch-size {batch_size} "
        f"--verbose"
    )
    
    if not run_command(cmd, "Training the model", gw_dir):
        lab.error("Model training failed")
        return False
    
    lab.log("✅ Model training completed successfully")
    
    # Save training checkpoints and outputs as artifacts
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                try:
                    artifact_path = lab.save_artifact(item_path, f"output/{item}")
                    lab.log(f"Saved output artifact: {artifact_path}")
                except Exception as e:
                    lab.log(f"Note: Could not save output artifact {item}: {e}")
    
    return True


def generate_summary_report(base_dir, gw_dir):
    """
    Generate a summary report of the training run.
    
    Args:
        base_dir: Base project directory
        gw_dir: Path to GW detection repository
    """
    lab.log("\n" + "="*60)
    lab.log("Generating Summary Report")
    lab.log("="*60)
    
    summary = {
        "task": "Gravitational Wave Detection Training",
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "base_directory": base_dir,
        "gw_repository": gw_dir,
        "output_directory": os.path.join(gw_dir, "runs", "train_output_dir"),
        "checkpoint_directory": os.path.join(gw_dir, "checkpoints"),
    }
    
    # Create summary artifact
    summary_file = os.path.join(gw_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    try:
        artifact_path = lab.save_artifact(summary_file, "training_summary.json")
        lab.log(f"Saved summary: {artifact_path}")
    except Exception as e:
        lab.log(f"Note: Could not save summary artifact: {e}")
    
    lab.log(json.dumps(summary, indent=2))


def main():
    """Main training pipeline orchestrator."""
    
    try:
        # Initialize lab
        lab.init()
        
        # Get configuration
        config = lab.get_config()
        
        lab.log("="*60)
        lab.log("GRAVITATIONAL WAVE DETECTION TRAINING PIPELINE")
        lab.log("="*60)
        lab.log(f"Started at: {datetime.now()}")
        
        # Set up base directory
        base_dir = os.path.expanduser("~/gw-project")
        
        lab.update_progress(5)
        
        # Step 1: Setup repositories
        repos = setup_repositories(base_dir)
        if not repos:
            lab.error("Repository setup failed")
            return False
        
        lab.update_progress(15)
        
        # Step 2: Setup dataset structure
        dataset_dir = setup_dataset_structure(repos["gw_dir"])
        
        lab.update_progress(25)
        
        # Step 3: Generate datasets
        if not generate_datasets(repos["challenge_dir"], dataset_dir, config):
            lab.error("Dataset generation failed")
            return False
        
        lab.update_progress(50)
        
        # Step 4: Generate waveforms
        if not generate_waveforms(repos["challenge_dir"], dataset_dir, repos["gw_dir"], config):
            lab.error("Waveform generation failed")
            return False
        
        lab.update_progress(75)
        
        # Step 5: Train the model
        if not train_model(repos["gw_dir"], dataset_dir, config):
            lab.error("Model training failed")
            return False
        
        lab.update_progress(95)
        
        # Generate summary report
        generate_summary_report(base_dir, repos["gw_dir"])
        
        lab.update_progress(100)
        
        lab.log("\n" + "="*60)
        lab.log("✅ PIPELINE COMPLETED SUCCESSFULLY")
        lab.log("="*60)
        lab.log(f"Ended at: {datetime.now()}")
        
        return True
        
    except Exception as e:
        lab.error(f"Pipeline failed with error: {e}")
        lab.log(f"Error traceback: {str(e)}", level="error")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
