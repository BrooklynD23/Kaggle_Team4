"""
Script to run the complete training pipeline with progress tracking.
Hardware-friendly version with reduced parallelism and batch processing.
"""
from src.train_pipeline import train_student_success_model, PipelineConfig, TrainingPipeline
import sys
import time
import gc
import os

# Try to import rich for nice UI, fall back to tqdm
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better UI...")
    os.system("pip install rich -q")
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        from rich.panel import Panel
        from rich.table import Table
        from rich.live import Live
        from rich import print as rprint
        RICH_AVAILABLE = True
    except ImportError:
        from tqdm import tqdm
        RICH_AVAILABLE = False


class HardwareFriendlyConfig(PipelineConfig):
    """
    Configuration optimized for lower memory usage and stability.
    Reduces parallelism and batch sizes.
    """
    # Reduce hyperparameter tuning iterations (biggest memory saver)
    TUNE_HYPERPARAMETERS = True
    TUNING_N_ITER = 20  # Reduced from 50
    TUNING_CV_FOLDS = 3  # Keep small
    
    # Cross-validation
    CV_FOLDS = 3  # Reduced from 5
    
    # SMOTE - use smaller k_neighbors
    USE_SMOTE = True


class ProgressTracker:
    """Track and display pipeline progress with rich UI."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.stages = [
            ("Data Loading & Feature Engineering", 5),
            ("SMOTE Resampling", 5),
            ("Baseline Models Training", 10),
            ("Random Forest Tuning", 15),
            ("XGBoost Tuning", 20),
            ("LightGBM Tuning", 20),
            ("Ensemble Training", 15),
            ("Threshold Optimization", 5),
            ("Final Evaluation", 5),
        ]
        self.current_stage = 0
        self.results = {}
        
    def print_header(self):
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                "[bold cyan]🎓 Student Success Prediction Pipeline[/bold cyan]\n"
                "[dim]Hardware-friendly mode with progress tracking[/dim]",
                border_style="cyan"
            ))
        else:
            print("=" * 60)
            print("🎓 Student Success Prediction Pipeline")
            print("Hardware-friendly mode with progress tracking")
            print("=" * 60)
    
    def print_config(self, config):
        if RICH_AVAILABLE:
            table = Table(title="Configuration", show_header=True, header_style="bold magenta")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Feature Engineering", str(config.USE_FEATURE_ENGINEERING))
            table.add_row("SMOTE", str(config.USE_SMOTE))
            table.add_row("Hyperparameter Tuning", str(config.TUNE_HYPERPARAMETERS))
            table.add_row("Tuning Iterations", str(config.TUNING_N_ITER))
            table.add_row("CV Folds", str(config.CV_FOLDS))
            table.add_row("Threshold Optimization", str(config.OPTIMIZE_THRESHOLDS))
            self.console.print(table)
        else:
            print(f"Feature Engineering: {config.USE_FEATURE_ENGINEERING}")
            print(f"SMOTE: {config.USE_SMOTE}")
            print(f"Tuning Iterations: {config.TUNING_N_ITER}")
    
    def stage_complete(self, stage_name, metrics=None):
        """Mark a stage as complete and show results."""
        self.current_stage += 1
        if metrics:
            self.results[stage_name] = metrics
        
        # Force garbage collection between stages to free memory
        gc.collect()
        
        # Small pause to let system stabilize
        time.sleep(0.5)
    
    def print_results(self, results):
        """Print final results in a nice table."""
        if RICH_AVAILABLE:
            table = Table(title="🏆 Final Results", show_header=True, header_style="bold green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Model", results.get("model_name", "N/A"))
            table.add_row("Test Macro F1", f"{results['test_macro_f1']:.4f}")
            table.add_row("Test Weighted F1", f"{results['test_weighted_f1']:.4f}")
            table.add_row("Test Accuracy", f"{results['test_accuracy']:.4f}")
            table.add_row("Dropout F1", f"{results['test_per_class_f1'][0]:.4f}")
            table.add_row("Enrolled F1", f"{results['test_per_class_f1'][1]:.4f}")
            table.add_row("Graduate F1", f"{results['test_per_class_f1'][2]:.4f}")
            self.console.print(table)
        else:
            print("\n" + "=" * 60)
            print("🏆 FINAL RESULTS")
            print("=" * 60)
            print(f"Model: {results.get('model_name', 'N/A')}")
            print(f"Test Macro F1: {results['test_macro_f1']:.4f}")
            print(f"Test Accuracy: {results['test_accuracy']:.4f}")


def run_pipeline_with_progress():
    """Run the pipeline with progress tracking and memory management."""
    
    tracker = ProgressTracker()
    tracker.print_header()
    
    # Use hardware-friendly config
    config = HardwareFriendlyConfig()
    tracker.print_config(config)
    
    # Reduce n_jobs globally to prevent memory issues
    os.environ['LOKY_MAX_CPU_COUNT'] = '2'  # Limit parallel jobs
    
    if RICH_AVAILABLE:
        console = Console()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            
            # Overall progress
            overall_task = progress.add_task("[cyan]Overall Progress", total=100)
            
            # Create pipeline
            pipeline = TrainingPipeline(config)
            
            # Stage 1: Data Loading
            stage_task = progress.add_task("[green]Stage 1: Data Loading & Feature Engineering...", total=100)
            progress.update(stage_task, completed=0)
            pipeline.load_and_prepare('dataset.csv')
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=10)
            tracker.stage_complete("Data Loading")
            
            # Stage 2: SMOTE
            stage_task = progress.add_task("[green]Stage 2: SMOTE Resampling...", total=100)
            progress.update(stage_task, completed=0)
            pipeline.apply_smote()
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=15)
            tracker.stage_complete("SMOTE")
            
            # Stage 3: Baselines
            stage_task = progress.add_task("[green]Stage 3: Training Baselines...", total=100)
            progress.update(stage_task, completed=0)
            pipeline.train_baselines()
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=25)
            tracker.stage_complete("Baselines")
            
            # Stage 4: Tree Models (most intensive)
            stage_task = progress.add_task("[yellow]Stage 4: Training & Tuning Tree Models...", total=100)
            progress.update(stage_task, completed=0)
            console.print("[dim]This stage may take 10-20 minutes with hyperparameter tuning...[/dim]")
            pipeline.train_tree_models(tune=config.TUNE_HYPERPARAMETERS)
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=65)
            tracker.stage_complete("Tree Models")
            gc.collect()  # Free memory after intensive stage
            
            # Stage 5: Ensembles
            stage_task = progress.add_task("[green]Stage 5: Training Ensembles...", total=100)
            progress.update(stage_task, completed=0)
            pipeline.train_ensembles()
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=80)
            tracker.stage_complete("Ensembles")
            
            # Stage 6: Model Selection
            stage_task = progress.add_task("[green]Stage 6: Selecting Best Model...", total=100)
            progress.update(stage_task, completed=0)
            pipeline.select_best_model()
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=85)
            tracker.stage_complete("Selection")
            
            # Stage 7: Threshold Optimization
            stage_task = progress.add_task("[green]Stage 7: Optimizing Thresholds...", total=100)
            progress.update(stage_task, completed=0)
            pipeline.optimize_thresholds()
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=92)
            tracker.stage_complete("Thresholds")
            
            # Stage 8: Final Evaluation
            stage_task = progress.add_task("[green]Stage 8: Final Evaluation...", total=100)
            progress.update(stage_task, completed=0)
            results = pipeline.final_evaluation()
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=98)
            tracker.stage_complete("Evaluation")
            
            # Save and Report
            stage_task = progress.add_task("[green]Saving Model & Generating Report...", total=100)
            progress.update(stage_task, completed=0)
            pipeline.save_model()
            pipeline.generate_report()
            progress.update(stage_task, completed=100)
            progress.update(overall_task, completed=100)
        
        # Print final results
        console.print("\n")
        tracker.print_results(results)
        
        console.print(Panel.fit(
            "[bold green]✅ Pipeline Complete![/bold green]\n"
            f"[dim]Model saved to models/saved_models/[/dim]\n"
            f"[dim]Report saved to REPORT.md[/dim]",
            border_style="green"
        ))
        
    else:
        # Fallback to simple progress with tqdm
        from tqdm import tqdm
        
        pipeline = TrainingPipeline(config)
        
        stages = [
            ("Loading Data", lambda: pipeline.load_and_prepare('dataset.csv')),
            ("SMOTE Resampling", lambda: pipeline.apply_smote()),
            ("Training Baselines", lambda: pipeline.train_baselines()),
            ("Training Tree Models", lambda: pipeline.train_tree_models(tune=config.TUNE_HYPERPARAMETERS)),
            ("Training Ensembles", lambda: pipeline.train_ensembles()),
            ("Selecting Best Model", lambda: pipeline.select_best_model()),
            ("Optimizing Thresholds", lambda: pipeline.optimize_thresholds()),
            ("Final Evaluation", lambda: pipeline.final_evaluation()),
        ]
        
        for name, func in tqdm(stages, desc="Pipeline Progress"):
            print(f"\n>>> {name}...")
            result = func()
            gc.collect()
            time.sleep(0.5)
        
        results = result
        pipeline.save_model()
        pipeline.generate_report()
        tracker.print_results(results)
    
    return results


def run_quick_pipeline():
    """Run a quick version without hyperparameter tuning for testing."""
    
    print("=" * 60)
    print("🚀 Quick Pipeline (No Hyperparameter Tuning)")
    print("=" * 60)
    
    config = HardwareFriendlyConfig()
    config.TUNE_HYPERPARAMETERS = False  # Skip tuning for speed
    config.TUNING_N_ITER = 10
    
    pipeline = TrainingPipeline(config)
    
    print("\n[1/8] Loading data...")
    pipeline.load_and_prepare('dataset.csv')
    gc.collect()
    
    print("\n[2/8] Applying SMOTE...")
    pipeline.apply_smote()
    gc.collect()
    
    print("\n[3/8] Training baselines...")
    pipeline.train_baselines()
    gc.collect()
    
    print("\n[4/8] Training tree models...")
    pipeline.train_tree_models(tune=False)
    gc.collect()
    
    print("\n[5/8] Training ensembles...")
    pipeline.train_ensembles()
    gc.collect()
    
    print("\n[6/8] Selecting best model...")
    pipeline.select_best_model()
    
    print("\n[7/8] Optimizing thresholds...")
    pipeline.optimize_thresholds()
    
    print("\n[8/8] Final evaluation...")
    results = pipeline.final_evaluation()
    
    pipeline.save_model()
    pipeline.generate_report()
    
    print("\n" + "=" * 60)
    print("✅ QUICK PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model: {results['model_name']}")
    print(f"Test Macro F1: {results['test_macro_f1']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Student Success Prediction Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run quick version without tuning")
    parser.add_argument("--tuning-iter", type=int, default=20, help="Number of tuning iterations")
    args = parser.parse_args()
    
    try:
        if args.quick:
            results = run_quick_pipeline()
        else:
            results = run_pipeline_with_progress()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
