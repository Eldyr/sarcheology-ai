"""Starter FastAPI entrypoint."""
from fastapi import FastAPI
app = FastAPI(title="SARchaeology AI")
@app.get("/health")
def health():
    return {"status": "ok", "stage": "scaffold"}
