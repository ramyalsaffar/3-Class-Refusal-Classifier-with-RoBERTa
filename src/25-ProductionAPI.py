# ProductionAPI Module
#---------------------
# FastAPI server for real-time inference with A/B testing support.
# Logs all predictions to PostgreSQL for monitoring.
# NOTE: This is a standalone production script with its own imports.
# Requires: pip install fastapi uvicorn pydantic
###############################################################################

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


# Request/Response models
class ClassifyRequest(BaseModel):
    prompt: str
    response: str
    metadata: dict = None


class ClassifyResponse(BaseModel):
    label: int
    label_name: str
    confidence: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float
    total_predictions: int


# Initialize FastAPI app
app = FastAPI(
    title="Refusal Classifier API",
    description="Production API for 3-class refusal classification with A/B testing",
    version="1.0.0"
)


# Global state
class APIState:
    """Global state for the API."""
    def __init__(self):
        self.data_manager = None
        self.active_model = None
        self.challenger_model = None
        self.active_version = None
        self.challenger_version = None
        self.tokenizer = None
        self.start_time = time.time()
        self.prediction_count = 0
        self.ab_test_active = False
        self.challenger_traffic = 0.0  # Percentage of traffic to challenger


state = APIState()


@app.on_event("startup")
async def startup_event():
    """Initialize models and database connection on startup."""
    print("\n" + "="*60)
    print("STARTING PRODUCTION API")
    print("="*60)

    # Initialize database
    state.data_manager = DataManager()

    # Load active model
    state.active_version = state.data_manager.get_active_model_version()
    if not state.active_version:
        print("⚠️  No active model found in database, using default")
        state.active_version = "v1.0.0"

    # Load model
    model_path = os.path.join(models_path, f"{state.active_version}.pt")
    if os.path.exists(model_path):
        state.active_model = RefusalClassifier().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state.active_model.load_state_dict(checkpoint['model_state_dict'])
        state.active_model.eval()
        print(f"✓ Loaded active model: {state.active_version}")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Please train a model first or specify correct path")
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load tokenizer
    state.tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

    # Check for challenger model (A/B testing)
    cursor = state.data_manager.conn.cursor()
    cursor.execute("""
        SELECT version, traffic_percentage FROM model_versions
        WHERE is_challenger = TRUE
        LIMIT 1
    """)
    result = cursor.fetchone()
    cursor.close()

    if result:
        challenger_version, traffic = result
        challenger_path = os.path.join(models_path, f"{challenger_version}.pt")

        if os.path.exists(challenger_path):
            state.challenger_model = RefusalClassifier().to(DEVICE)
            checkpoint = torch.load(challenger_path, map_location=DEVICE)
            state.challenger_model.load_state_dict(checkpoint['model_state_dict'])
            state.challenger_model.eval()
            state.challenger_version = challenger_version
            state.challenger_traffic = traffic
            state.ab_test_active = True
            print(f"✓ A/B Test Active: {traffic*100:.1f}% traffic to challenger {challenger_version}")
        else:
            print(f"⚠️  Challenger model file not found: {challenger_path}")

    print(f"\n✓ API Ready - {DEVICE}")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    if state.data_manager:
        state.data_manager.close()
    print("\n✓ API Shutdown")


def log_prediction_async(prompt: str, response: str, prediction: int,
                        confidence: float, model_version: str,
                        latency_ms: float, metadata: dict):
    """Log prediction to database (runs in background)."""
    try:
        state.data_manager.log_prediction(
            prompt=prompt,
            response=response,
            prediction=prediction,
            confidence=confidence,
            model_version=model_version,
            latency_ms=latency_ms,
            metadata=metadata
        )
    except Exception as e:
        print(f"⚠️  Failed to log prediction: {e}")


def select_model_for_request() -> tuple:
    """
    Select which model to use for this request (A/B testing).

    Returns:
        Tuple of (model, version)
    """
    if not state.ab_test_active or state.challenger_traffic == 0.0:
        return state.active_model, state.active_version

    # Random selection based on traffic percentage
    if np.random.random() < state.challenger_traffic:
        return state.challenger_model, state.challenger_version
    else:
        return state.active_model, state.active_version


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest, background_tasks: BackgroundTasks):
    """
    Classify a response as refusal type.

    Args:
        request: ClassifyRequest with prompt and response

    Returns:
        ClassifyResponse with label, confidence, and metadata
    """
    start_time = time.time()

    # Select model (A/B testing logic)
    model, version = select_model_for_request()

    # Tokenize
    encoding = state.tokenizer(
        request.response,
        max_length=MODEL_CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    prediction = predicted.item()
    conf = confidence.item()
    latency_ms = (time.time() - start_time) * 1000

    # Log to database (background task)
    background_tasks.add_task(
        log_prediction_async,
        prompt=request.prompt,
        response=request.response,
        prediction=prediction,
        confidence=conf,
        model_version=version,
        latency_ms=latency_ms,
        metadata=request.metadata
    )

    # Increment counter
    state.prediction_count += 1

    return ClassifyResponse(
        label=prediction,
        label_name=CLASS_NAMES[prediction],
        confidence=conf,
        model_version=version,
        latency_ms=latency_ms
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns:
        API health status and metrics
    """
    uptime = time.time() - state.start_time

    return HealthResponse(
        status="healthy",
        model_version=state.active_version,
        uptime_seconds=uptime,
        total_predictions=state.prediction_count
    )


@app.get("/metrics")
async def metrics():
    """
    Get API metrics and statistics.

    Returns:
        Dictionary with various metrics
    """
    uptime = time.time() - state.start_time

    # Get recent predictions
    recent_df = state.data_manager.get_recent_predictions(hours=24)

    metrics = {
        "uptime_seconds": uptime,
        "total_predictions": state.prediction_count,
        "predictions_last_24h": len(recent_df),
        "active_model": state.active_version,
        "device": str(DEVICE),
        "ab_test_active": state.ab_test_active
    }

    if state.ab_test_active:
        metrics["challenger_model"] = state.challenger_version
        metrics["challenger_traffic_percentage"] = state.challenger_traffic * 100

    if len(recent_df) > 0:
        metrics["avg_confidence_24h"] = float(recent_df['confidence'].mean())
        metrics["predictions_by_class"] = {
            CLASS_NAMES[i]: int((recent_df['prediction'] == i).sum())
            for i in range(3)
        }

    return JSONResponse(content=metrics)


@app.get("/ab-test-status")
async def ab_test_status():
    """
    Get A/B test status.

    Returns:
        A/B test information
    """
    if not state.ab_test_active:
        return {
            "active": False,
            "message": "No A/B test currently running"
        }

    # Get performance comparison
    active_df = state.data_manager.get_recent_predictions(
        hours=24,
        model_version=state.active_version
    )
    challenger_df = state.data_manager.get_recent_predictions(
        hours=24,
        model_version=state.challenger_version
    )

    return {
        "active": True,
        "active_model": state.active_version,
        "challenger_model": state.challenger_version,
        "challenger_traffic": state.challenger_traffic * 100,
        "predictions": {
            "active": len(active_df),
            "challenger": len(challenger_df)
        },
        "avg_confidence": {
            "active": float(active_df['confidence'].mean()) if len(active_df) > 0 else 0,
            "challenger": float(challenger_df['confidence'].mean()) if len(challenger_df) > 0 else 0
        }
    }


@app.post("/admin/promote-challenger")
async def promote_challenger(api_key: str):
    """
    Promote challenger to active (admin only).

    Args:
        api_key: Admin API key for authentication

    Returns:
        Success message
    """
    # Simple API key check (in production, use proper authentication)
    if api_key != PRODUCTION_CONFIG.get('admin_api_key', 'admin-key-here'):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not state.ab_test_active:
        raise HTTPException(status_code=400, detail="No A/B test active")

    # Swap models
    state.active_model = state.challenger_model
    old_version = state.active_version
    state.active_version = state.challenger_version

    # Update database
    cursor = state.data_manager.conn.cursor()

    # Deactivate old
    cursor.execute("""
        UPDATE model_versions
        SET is_active = FALSE
        WHERE version = %s
    """, (old_version,))

    # Activate new
    cursor.execute("""
        UPDATE model_versions
        SET is_active = TRUE, is_challenger = FALSE, traffic_percentage = 1.0
        WHERE version = %s
    """, (state.active_version,))

    state.data_manager.conn.commit()
    cursor.close()

    # Disable A/B test
    state.challenger_model = None
    state.challenger_version = None
    state.ab_test_active = False
    state.challenger_traffic = 0.0

    print(f"✓ Promoted {state.active_version} to active model")

    return {
        "success": True,
        "message": f"Challenger {state.active_version} promoted to active",
        "previous_version": old_version
    }


@app.post("/admin/rollback")
async def rollback(api_key: str):
    """
    Rollback A/B test (stop challenger) (admin only).

    Args:
        api_key: Admin API key

    Returns:
        Success message
    """
    if api_key != PRODUCTION_CONFIG.get('admin_api_key', 'admin-key-here'):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not state.ab_test_active:
        raise HTTPException(status_code=400, detail="No A/B test active")

    # Disable challenger
    cursor = state.data_manager.conn.cursor()
    cursor.execute("""
        UPDATE model_versions
        SET is_challenger = FALSE, traffic_percentage = 0.0
        WHERE version = %s
    """, (state.challenger_version,))
    state.data_manager.conn.commit()
    cursor.close()

    rolled_back_version = state.challenger_version
    state.challenger_model = None
    state.challenger_version = None
    state.ab_test_active = False
    state.challenger_traffic = 0.0

    print(f"✓ Rolled back A/B test (removed {rolled_back_version})")

    return {
        "success": True,
        "message": f"A/B test rolled back, removed challenger {rolled_back_version}",
        "active_model": state.active_version
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    uvicorn.run(app, host=host, port=port)


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 30, 2025
@author: ramyalsaffar
"""
