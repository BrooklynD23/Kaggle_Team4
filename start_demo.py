#!/usr/bin/env python
"""
One-click demo launcher for Student Success Prediction.

Usage:
    python start_demo.py
    python start_demo.py --no-retrain    # Skip retrain prompt
    python start_demo.py --skip-deps     # Skip dependency installation

This script will:
1. Install all Python dependencies (requirements.txt)
2. Check if trained models exist
3. Prompt to retrain if models found, or auto-train if none
4. Regenerate visualization assets
5. Install Node.js dependencies for the UI
6. Start the FastAPI backend and React frontend
"""

import argparse
import shutil
import subprocess
import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
UI_DIR = PROJECT_ROOT / "app" / "ui"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def get_python_cmd():
    """Get the best Python command for the system."""
    # If in a venv, always use the current executable
    if sys.prefix != sys.base_prefix:
        return [sys.executable]
        
    # On Windows, try 'py' launcher first (handles multiple Python versions)
    if sys.platform == "win32":
        if shutil.which("py"):
            return ["py"]
    # Fall back to current Python executable
    return [sys.executable]


def check_command(cmd: str) -> bool:
    """Return True if command is available on PATH."""
    return shutil.which(cmd) is not None


def check_uvicorn_available():
    """Check if uvicorn is importable."""
    python_cmd = get_python_cmd()
    result = subprocess.run(
        python_cmd + ["-c", "import uvicorn"],
        capture_output=True
    )
    return result.returncode == 0


def install_python_deps(skip: bool = False):
    """Install Python dependencies from requirements.txt."""
    if skip:
        print("\n📦 Skipping Python dependency installation.")
        return
    
    print("\n📦 Installing Python dependencies...")
    python_cmd = get_python_cmd()
    
    try:
        result = subprocess.run(
            python_cmd + ["-m", "pip", "install", "-q", "-r", str(REQUIREMENTS_FILE)],
            cwd=PROJECT_ROOT,
            timeout=300  # 5 minute timeout
        )
        if result.returncode == 0:
            print("   ✔ Python dependencies installed.")
        else:
            print("   ⚠️  pip install returned non-zero, but continuing...")
    except subprocess.TimeoutExpired:
        print("   ⚠️  pip install timed out, but continuing...")
    except Exception as e:
        print(f"   ⚠️  pip install failed ({e}), but continuing...")
    
    # Verify uvicorn is available
    if not check_uvicorn_available():
        print("   ⚠️  uvicorn not found. Attempting direct install...")
        subprocess.run(
            python_cmd + ["-m", "pip", "install", "uvicorn", "fastapi"],
            cwd=PROJECT_ROOT
        )


def find_models() -> list:
    """Return list of saved model files."""
    if not MODEL_DIR.exists():
        return []
    return list(MODEL_DIR.glob("*.joblib"))


def run_pipeline():
    """Execute the training pipeline."""
    print("\n🚀 Starting training pipeline...")
    python_cmd = get_python_cmd()
    result = subprocess.run(
        python_cmd + ["run_pipeline.py"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("❌ Pipeline failed. Check logs above.")
        sys.exit(1)
    print("✅ Pipeline completed successfully.")


def show_model_comparison():
    """Run the model comparison script."""
    print("\n📊 Running model comparison...")
    python_cmd = get_python_cmd()
    
    # Check if rich is installed (it should be, but just in case)
    try:
        subprocess.run(
            python_cmd + ["-c", "import rich"], 
            capture_output=True, 
            check=True
        )
    except subprocess.CalledProcessError:
        print("⚠️  'rich' library not found. Installing...")
        subprocess.run(python_cmd + ["-m", "pip", "install", "rich", "-q"])

    result = subprocess.run(
        python_cmd + ["compare_models.py"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("⚠️  Model comparison failed (non-critical).")



def regenerate_visuals():
    """Run src.evaluation.visuals to refresh plots."""
    print("\n🎨 Regenerating visualization assets...")
    python_cmd = get_python_cmd()
    result = subprocess.run(
        python_cmd + ["-m", "src.evaluation.visuals"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("⚠️  Visuals generation encountered an issue (non-critical).")
    else:
        print("✅ Visuals regenerated.")


def install_ui_deps():
    """Install UI dependencies via npm."""
    node_modules = UI_DIR / "node_modules"
    if node_modules.exists():
        print("\n📦 UI dependencies already installed.")
        return

    if not check_command("npm"):
        print("\n⚠️  npm not found. Please install Node.js to run the UI.")
        print("   Download from https://nodejs.org/")
        sys.exit(1)

    print("\n📦 Installing UI dependencies (npm install)...")
    result = subprocess.run(
        ["npm", "install"], 
        cwd=UI_DIR, 
        shell=(sys.platform == "win32")
    )
    if result.returncode != 0:
        print("❌ npm install failed.")
        sys.exit(1)
    print("   ✔ UI dependencies installed.")


def start_services():
    """Launch FastAPI backend and Vite dev server."""
    print("\n⚡ Starting services...")
    
    python_cmd = get_python_cmd()
    api_proc = None
    ui_proc = None
    
    try:
        # Start FastAPI backend
        print("   Starting FastAPI backend...")
        api_proc = subprocess.Popen(
            python_cmd + ["-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=PROJECT_ROOT,
            # Stream output to console for debugging
            stdout=None,
            stderr=None,
        )
        
        # Wait a moment and check if API started
        time.sleep(3)
        if api_proc.poll() is not None:
            # Process exited - show error
            output = api_proc.stdout.read() if api_proc.stdout else ""
            print(f"   ❌ Backend failed to start:\n{output}")
            sys.exit(1)
        
        print("   ✔ FastAPI backend running at http://127.0.0.1:8000")
        
        # Start Vite dev server
        print("   ✔ Starting React UI at http://localhost:5173")
        print("\n" + "=" * 50)
        print("🌐 Dashboard: http://localhost:5173")
        print("📡 API Docs:  http://127.0.0.1:8000/docs")
        print("=" * 50)
        print("Press Ctrl+C to stop the demo.")
        print("=" * 50 + "\n")
        
        # Run frontend in foreground
        ui_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=UI_DIR,
            shell=(sys.platform == "win32")
        )
        
        # Wait for either process to exit
        while True:
            # Check if UI exited
            if ui_proc.poll() is not None:
                break
            # Check if API exited
            if api_proc.poll() is not None:
                print("\n⚠️  Backend process exited unexpectedly.")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
    finally:
        # Cleanup
        if ui_proc and ui_proc.poll() is None:
            ui_proc.terminate()
            try:
                ui_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_proc.kill()
        
        if api_proc and api_proc.poll() is None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_proc.kill()
        
        print("👋 Demo stopped.")


def main():
    parser = argparse.ArgumentParser(description="Student Success Prediction Demo Launcher")
    parser.add_argument("--no-retrain", action="store_true", help="Skip retrain prompt")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎓 Student Success Prediction — Demo Launcher")
    print("=" * 50)

    # Step 1: Install Python deps
    install_python_deps(skip=args.skip_deps)

    # Step 2: Check for models
    models = find_models()

    if models:
        print(f"\n📂 Found {len(models)} saved model(s) in {MODEL_DIR.relative_to(PROJECT_ROOT)}")
        for m in models[-3:]:  # show last 3
            print(f"   • {m.name}")
        
        if not args.no_retrain:
            try:
                answer = input("\nDo you want to retrain the model? [y/N]: ").strip().lower()
                if answer in ("y", "yes"):
                    run_pipeline()
            except EOFError:
                # Non-interactive mode, skip retrain
                print("\n   (Non-interactive mode, skipping retrain)")
        else:
            print("\n   (Skipping retrain prompt)")
    else:
        print("\n⚠️  No saved models found. Training is required.")
        run_pipeline()

    # Step 2.5: Show model comparison
    show_model_comparison()

    # Step 2.6: Run Model Interpretation
    run_interpretation()

    # Step 3: Regenerate visuals (including story plots)
    regenerate_visuals()
    run_story_plots()

    # Step 4: Install UI deps
    install_ui_deps()

    # Step 5: Launch backend + frontend
    start_services()


def run_interpretation():
    """Run the model interpretation script."""
    print("\n🔍 Running model interpretation...")
    python_cmd = get_python_cmd()
    result = subprocess.run(
        python_cmd + ["run_interpretation.py"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("⚠️  Interpretation failed (non-critical).")


def run_story_plots():
    """Run the story plotting script."""
    print("\n📊 Generating storytelling plots...")
    python_cmd = get_python_cmd()
    result = subprocess.run(
        python_cmd + ["generate_story_plots.py"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("⚠️  Story plots generation failed (non-critical).")



if __name__ == "__main__":
    main()
