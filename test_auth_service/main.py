from fastapi import FastAPI, Request, HTTPException
from jose import jwt
from datetime import datetime, timedelta
import uuid

app = FastAPI()

# Hardcoded secret key for JWT signing
SECRET_KEY = "your-256-bit-secret"
ALGORITHM = "HS256"

@app.post("/auth")
async def authenticate(request: Request):
    # Extract headers
    app_id = request.headers.get("X-Auth-AppID")
    signature = request.headers.get("X-Auth-Signature")
    timestamp_str = request.headers.get("X-Auth-Timestamp")

    # Validate presence of headers
    if not all([app_id, signature, timestamp_str]):
        raise HTTPException(status_code=400, detail="Missing required authentication headers")

    try:
        timestamp_val = int(timestamp_str)
        # If timestamp is likely in milliseconds (common in Java/JS), convert to seconds
        if timestamp_val > 4102444800: # Timestamp for Jan 1, 2100 in seconds
             timestamp_val //= 1000
        timestamp = timestamp_val
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    # For now, we are not validating the signature or using the payload for JWT generation.
    # We'll just assume the payload structure from connectchain/utils/token_util.py
    try:
        payload_data = await request.json()
        # Example of accessing payload data (not used in JWT generation for this dummy service)
        # scope = payload_data.get("scope")
        # additional_claims = payload_data.get("additional_claims")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")


    # Define JWT claims
    issued_at = datetime.fromtimestamp(timestamp)
    expiration_time = issued_at + timedelta(hours=1)

    jwt_claims = {
        "sub": app_id,
        "iat": timestamp,
        "exp": int(expiration_time.timestamp()),
        "jti": uuid.uuid4().hex,
        # Include received payload data if needed in the future, for example:
        # "scope": scope,
        # "originator_source": additional_claims.get("originator_source") if additional_claims else None
    }

    # Generate JWT token
    try:
        token = jwt.encode(jwt_claims, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating token: {str(e)}")

    return {"authorization_token": token}

if __name__ == "__main__":
    import uvicorn
    # It's good practice to make host and port configurable,
    # but for this dummy service, we'll hardcode them.
    # Note: Running directly like this is for development/testing.
    # For production, you'd use a proper ASGI server like Gunicorn with Uvicorn workers.
    uvicorn.run(app, host="0.0.0.0", port=8000)
