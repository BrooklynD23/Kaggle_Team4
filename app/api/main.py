import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, root_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
PLOTS_DIR = ARTIFACT_DIR / "plots"
LOG_DIR = ARTIFACT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ArtifactWatcher:
    def __init__(self, path: Path):
        self.path = path
        self.payload: Optional[Any] = None
        self.mtime: Optional[float] = None

    def refresh(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.payload = None
            self.mtime = None
            return

        if self.mtime == stat.st_mtime:
            return

        with self.path.open("r", encoding="utf-8") as fh:
            self.payload = json.load(fh)
        self.mtime = stat.st_mtime

    async def watch(self, interval: int = 10) -> None:
        while True:
            try:
                await asyncio.to_thread(self.refresh)
            except Exception as exc:  # pragma: no cover - log only
                print(f"[ArtifactWatcher] Failed to refresh {self.path}: {exc}")
            await asyncio.sleep(interval)


class ModelRegistry:
    def __init__(self, directory: Path):
        self.directory = directory
        self.bundle: Optional[Dict[str, Any]] = None
        self.latest_path: Optional[Path] = None
        self.mtime: Optional[float] = None

    def _load_model(self, model_path: Path) -> Dict[str, Any]:
        """Load a model from a specific path."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        payload = joblib.load(model_path)
        stat = model_path.stat()
        
        # Handle raw model files (legacy support)
        if not isinstance(payload, dict):
            feature_names = getattr(payload, "feature_names_in_", [])
            if hasattr(feature_names, "tolist"):
                feature_names = feature_names.tolist()
                
            return {
                "model": payload,
                "model_name": model_path.stem,
                "feature_names": feature_names,
                "threshold_optimizer": None,
                "class_names": ["Dropout", "Enrolled", "Graduate"],  # Default fallback
                "target_col": "Target",
                "path": model_path,
                "filename": model_path.name,
                "loaded_at": datetime.utcnow().isoformat(),
                "file_size": stat.st_size,
                "file_mtime": datetime.utcfromtimestamp(stat.st_mtime).isoformat()
            }

        return {
            "model": payload["model"],
            "model_name": payload.get("model_name", model_path.stem),
            "feature_names": payload.get("feature_names") or [],
            "threshold_optimizer": payload.get("threshold_optimizer"),
            "class_names": payload.get("config", {}).get("class_names", []),
            "target_col": payload.get("config", {}).get("target_col"),
            "path": model_path,
            "filename": model_path.name,
            "loaded_at": datetime.utcnow().isoformat(),
            "file_size": stat.st_size,
            "file_mtime": datetime.utcfromtimestamp(stat.st_mtime).isoformat()
        }

    def _load_latest(self) -> None:
        if not self.directory.exists():
            return

        candidates = list(self.directory.glob("*.joblib"))
        if not candidates:
            return

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        if self.latest_path == latest:
            return

        self.bundle = self._load_model(latest)
        self.latest_path = latest
        self.mtime = latest.stat().st_mtime

    def load_specific_model(self, filename: str) -> Dict[str, Any]:
        """Load a specific model by filename."""
        model_path = self.directory / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {filename}")
        
        self.bundle = self._load_model(model_path)
        self.latest_path = model_path
        self.mtime = model_path.stat().st_mtime
        return self.bundle

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata."""
        if not self.directory.exists():
            return []
        
        models = []
        for model_path in sorted(self.directory.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                stat = model_path.stat()
                # Try to load just metadata without the full model
                payload = joblib.load(model_path)
                
                model_name = model_path.stem
                if isinstance(payload, dict):
                    model_name = payload.get("model_name", model_path.stem)

                models.append({
                    "filename": model_path.name,
                    "model_name": model_name,
                    "file_size": stat.st_size,
                    "file_mtime": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
                    "is_current": self.latest_path == model_path if self.latest_path else False
                })
            except Exception as e:
                # If we can't load it, still include basic info
                stat = model_path.stat()
                models.append({
                    "filename": model_path.name,
                    "model_name": model_path.stem,
                    "file_size": stat.st_size,
                    "file_mtime": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
                    "is_current": False,
                    "error": str(e)
                })
        
        return models

    async def maybe_reload(self) -> None:
        await asyncio.to_thread(self._load_latest)

    async def watch(self, interval: int = 30) -> None:
        while True:
            try:
                await self.maybe_reload()
            except Exception as exc:  # pragma: no cover - log only
                print(f"[ModelRegistry] Failed to reload: {exc}")
            await asyncio.sleep(interval)

    def get_bundle(self) -> Optional[Dict[str, Any]]:
        return self.bundle


class PredictRequest(BaseModel):
    features: Optional[Dict[str, float]] = None
    values: Optional[List[float]] = None

    @root_validator(pre=True)
    def ensure_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("features") and not values.get("values"):
            raise ValueError("Provide either 'features' dict or 'values' list.")
        return values


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    raw_scores: List[float]
    feature_count: int
    missing_features: int
    timestamp: str


class MetricsResponse(BaseModel):
    generated_at: str
    macro_f1: float
    weighted_f1: Optional[float]
    accuracy: float
    per_class_f1: Dict[str, float]
    plots: Dict[str, str]
    artifact_path: str
    phases: Optional[List[Dict[str, Any]]]
    model_comparison: Optional[List[Dict[str, Any]]]


class HealthResponse(BaseModel):
    status: str
    model_name: Optional[str]
    model_loaded_at: Optional[str]
    model_path: Optional[str]
    metrics_updated_at: Optional[str]
    artifact_available: bool
    pipeline_job: Dict[str, Any]


class ModelInfo(BaseModel):
    filename: str
    model_name: str
    file_size: int
    file_mtime: str
    is_current: bool
    error: Optional[str] = None


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    current_model: Optional[str]


class SwitchModelRequest(BaseModel):
    filename: str


class RunResponse(BaseModel):
    status: str
    started_at: str
    log_file: str


app = FastAPI(
    title="Student Success API",
    description="Serves live metrics, insights, and predictions for the Student Success project.",
    version="1.0.0"
)
app.mount("/artifacts", StaticFiles(directory=ARTIFACT_DIR), name="artifacts")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_plot_paths() -> Dict[str, str]:
    if not PLOTS_DIR.exists():
        return {}
    mapping = {}
    for path in PLOTS_DIR.glob("*.png"):
        mapping[path.stem] = str(path.relative_to(PROJECT_ROOT))
    return mapping


def _build_feature_vector(
    payload: PredictRequest,
    feature_names: List[str]
) -> Tuple[np.ndarray, int]:
    if payload.features:
        missing = 0
        vector = []
        for name in feature_names:
            if name in payload.features:
                vector.append(payload.features[name])
            else:
                vector.append(0.0)
                missing += 1
        return np.array([vector], dtype=float), missing

    values = payload.values or []
    if len(values) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} values, received {len(values)}."
        )
    return np.array([values], dtype=float), 0


def _apply_thresholds(proba: np.ndarray, optimizer: Any) -> np.ndarray:
    if optimizer is None:
        return proba.argmax(axis=1)
    try:
        return optimizer.predict(proba)
    except Exception:
        return proba.argmax(axis=1)


async def _run_pipeline_job(state: Dict[str, Any], log_path: Path) -> None:
    cmd = ["py", "run_pipeline.py"]

    state["status"] = "running"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=log_file
        )
        state["pid"] = process.pid
        return_code = await process.wait()
    state["finished_at"] = datetime.utcnow().isoformat()
    state["status"] = "succeeded" if return_code == 0 else "failed"
    state["return_code"] = return_code


@app.on_event("startup")
async def on_startup() -> None:
    app.state.project_root = PROJECT_ROOT
    app.state.model_registry = ModelRegistry(MODEL_DIR)
    await app.state.model_registry.maybe_reload()

    app.state.metrics_cache = ArtifactWatcher(ARTIFACT_DIR / "latest_run.json")
    app.state.feature_cache = ArtifactWatcher(ARTIFACT_DIR / "feature_importance.json")
    app.state.metrics_cache.refresh()
    app.state.feature_cache.refresh()

    app.state.pipeline_job = {"status": "idle"}

    app.state.watch_tasks = [
        asyncio.create_task(app.state.metrics_cache.watch()),
        asyncio.create_task(app.state.feature_cache.watch()),
        asyncio.create_task(app.state.model_registry.watch())
    ]


@app.on_event("shutdown")
async def on_shutdown() -> None:
    tasks = getattr(app.state, "watch_tasks", [])
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    bundle = app.state.model_registry.get_bundle()
    metrics_mtime = app.state.metrics_cache.mtime
    return HealthResponse(
        status="ok" if bundle else "model-missing",
        model_name=bundle["model_name"] if bundle else None,
        model_loaded_at=bundle["loaded_at"] if bundle else None,
        model_path=str(bundle["path"].relative_to(PROJECT_ROOT)) if bundle else None,
        metrics_updated_at=datetime.utcfromtimestamp(metrics_mtime).isoformat() if metrics_mtime else None,
        artifact_available=app.state.metrics_cache.payload is not None,
        pipeline_job=app.state.pipeline_job
    )


@app.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """List all available saved models."""
    models_list = await asyncio.to_thread(app.state.model_registry.list_models)
    current_bundle = app.state.model_registry.get_bundle()
    current_filename = current_bundle["filename"] if current_bundle else None
    
    return ModelsListResponse(
        models=[ModelInfo(**m) for m in models_list],
        current_model=current_filename
    )


@app.post("/models/switch")
async def switch_model(req: SwitchModelRequest) -> Dict[str, Any]:
    """Switch to a different saved model."""
    try:
        bundle = await asyncio.to_thread(
            app.state.model_registry.load_specific_model,
            req.filename
        )
        return {
            "status": "success",
            "model_name": bundle["model_name"],
            "filename": bundle["filename"],
            "loaded_at": bundle["loaded_at"],
            "message": f"Switched to model: {bundle['model_name']}"
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/metrics/latest", response_model=MetricsResponse)
async def latest_metrics() -> MetricsResponse:
    payload = app.state.metrics_cache.payload
    if not payload:
        raise HTTPException(status_code=503, detail="latest_run.json not available. Run the pipeline first.")

    test_results = payload.get("test_results") or {}
    class_names = payload.get("class_names") or ["0", "1", "2"]
    per_class = test_results.get("test_per_class_f1", [])
    per_class_map = {name: float(val) for name, val in zip(class_names, per_class)}

    return MetricsResponse(
        generated_at=payload.get("generated_at"),
        macro_f1=float(test_results.get("test_macro_f1", 0)),
        weighted_f1=float(test_results.get("test_weighted_f1", 0)),
        accuracy=float(test_results.get("test_accuracy", 0)),
        per_class_f1=per_class_map,
        plots=_list_plot_paths(),
        artifact_path=str((ARTIFACT_DIR / "latest_run.json").relative_to(PROJECT_ROOT)),

        phases=payload.get("phases", []),
        model_comparison=payload.get("model_comparison", [])
    )


@app.get("/insights/feature-importance")
async def feature_insights() -> Dict[str, Any]:
    payload = app.state.feature_cache.payload
    if payload is None:
        raise HTTPException(status_code=503, detail="feature_importance.json not available.")
    return {
        "updated_at": datetime.utcfromtimestamp(app.state.feature_cache.mtime).isoformat() if app.state.feature_cache.mtime else None,
        "features": payload
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    bundle = app.state.model_registry.get_bundle()
    if not bundle:
        raise HTTPException(status_code=503, detail="No trained model found.")

    feature_names = bundle["feature_names"]
    vector, missing = _build_feature_vector(req, feature_names)

    model = bundle["model"]
    proba = model.predict_proba(vector)
    pred_indices = _apply_thresholds(proba, bundle["threshold_optimizer"])
    class_names = bundle["class_names"] or [str(i) for i in range(proba.shape[1])]
    probs = proba[0].tolist()

    return PredictResponse(
        predicted_class=class_names[int(pred_indices[0])],
        probabilities={name: float(val) for name, val in zip(class_names, probs)},
        raw_scores=[float(x) for x in probs],
        feature_count=len(feature_names),
        missing_features=missing,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/runs", response_model=RunResponse, status_code=202)
async def trigger_run() -> RunResponse:
    job = app.state.pipeline_job
    if job.get("status") == "running":
        raise HTTPException(status_code=409, detail="Pipeline run already in progress.")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"pipeline_{ts}.log"
    relative_log = str(log_path.relative_to(PROJECT_ROOT))
    job.update({
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "log_file": relative_log
    })

    asyncio.create_task(_run_pipeline_job(job, log_path))
    return RunResponse(
        status="started",
        started_at=job["started_at"],
        log_file=relative_log
    )

