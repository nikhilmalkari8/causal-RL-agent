import numpy as np
#!/usr/bin/env python3
"""
Complete Phase 1 Launcher Script
Runs the full systematic training and evaluation pipeline
"""

import sys
import os
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ FAILED")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False
    
    return True

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking Environment Setup...")
    
    required_files = [
        'envs/enhanced_causal_env.py',
        'models/enhanced_transformer_policy.py',
        'models/baseline_models.py',
        'agents/enhanced_ppo_agent.py',
        'language/instruction_processor.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    
    # Check Python packages
    try:
        import torch
        import numpy
        import matplotlib
        import gymnasium
        print("✅ Required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'plots', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def main():
    """Main execution pipeline"""
    print("🚀 PHASE 1: COMPLETE TRAINING AND EVALUATION PIPELINE")
    print("=" * 70)
    print("This script will:")
    print("1. Verify environment setup")
    print("2. Test environment functionality") 
    print("3. Run systematic training of all models")
    print("4. Generate comprehensive comparison results")
    print("=" * 70)
    
    # Step 0: Setup
    print("\n📋 STEP 0: ENVIRONMENT SETUP")
    if not check_environment():
        print("❌ Environment setup failed. Please fix issues and retry.")
        return False
    
    create_directories()
    
    # Step 1: Environment verification
    print("\n🧪 STEP 1: ENVIRONMENT VERIFICATION")
    if not run_command("python environment_test.py", "Testing environment functionality"):
        print("⚠️ Environment test failed, but continuing...")
        # Don't exit - some tests might fail but environment could still work
    
    # Step 2: Systematic training
    print("\n🏗️ STEP 2: SYSTEMATIC MODEL TRAINING")
    print("This will take significant time (30-60 minutes)...")
    
    start_time = time.time()
    
    if not run_command("python systematic_training.py", "Training all models systematically"):
        print("❌ Training failed. Check logs for details.")
        return False
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # Convert to minutes
    
    print(f"✅ Training completed in {training_time:.1f} minutes")
    
    # Step 3: Results analysis
    print("\n📊 STEP 3: RESULTS ANALYSIS")
    
    # Check if results were generated
    results_dir = "results"
    if os.path.exists(results_dir) and os.listdir(results_dir):
        latest_files = []
        for file in os.listdir(results_dir):
            if file.endswith('.json') or file.endswith('.txt'):
                latest_files.append(file)
        
        if latest_files:
            print("✅ Results generated:")
            for file in sorted(latest_files)[-3:]:  # Show latest 3 files
                print(f"   📄 {file}")
        else:
            print("⚠️ No result files found")
    else:
        print("⚠️ Results directory empty")
    
    # Check if plots were generated
    plots_dir = "plots"
    if os.path.exists(plots_dir) and os.listdir(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        if plot_files:
            print("✅ Plots generated:")
            for file in sorted(plot_files)[-3:]:  # Show latest 3 plots
                print(f"   📊 {file}")
        else:
            print("⚠️ No plot files found")
    else:
        print("⚠️ Plots directory empty")
    
    # Step 4: Summary
    print("\n🎊 PHASE 1 PIPELINE COMPLETE!")
    print("=" * 70)
    print("📁 Check the following directories for results:")
    print("   - results/: JSON data and text reports")  
    print("   - plots/: Comparison charts and training curves")
    print("   - models/: Trained model checkpoints")
    print("")
    print("📋 Next steps:")
    print("   1. Review the summary report in results/")
    print("   2. Examine comparison plots in plots/")
    print("   3. If results are satisfactory, proceed to Phase 2")
    print("   4. If results need improvement, analyze and retrain")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)