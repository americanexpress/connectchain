# Test Authentication Service

This directory contains a simple FastAPI dummy authentication service for `connectchain`.
It's designed to demonstrate how `connectchain`'s `token_util` can interact with an OAuth 2.0 / JWT-based authentication provider.

## How to Run

1.  **Install Dependencies:**
    Make sure you have Python installed. It's recommended to use a virtual environment.
    Install the necessary dependencies:
    ```bash
    pip install -r requirements-dev.txt
    ```
    (Note: `requirements-dev.txt` is located in the root of the repository.)

2.  **Start the Authentication Service:**
    Navigate to the `test_auth_service` directory and run the Uvicorn server:
    ```bash
    cd test_auth_service
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    The service will be available at `http://localhost:8000`.

3.  **Configure `connectchain`:**
    Ensure your `connectchain/config/example.config.yml` (or your custom config file) has the `eas.url` pointing to this service:
    ```yaml
    eas:
      url: http://localhost:8000/auth
      # ... other eas settings like id_key, secret_key
    ```

4.  **Set Environment Variables:**
    The `connectchain/utils/token_util.py` script (and by extension, `connectchain` when used with an LLM requiring auth) expects certain environment variables to be set for authentication:
    *   `CONFIG_PATH`: Path to your connectchain configuration YAML file.
    *   The consumer ID environment variable name (e.g., `CONSUMER_ID1` if `eas.id_key` is `CONSUMER_ID1` in the config). Set this to any ID (e.g., `test_user`).
    *   The consumer secret environment variable name (e.g., `CONSUMER_SECRET1` if `eas.secret_key` is `CONSUMER_SECRET1` in the config). Set this to any secret (e.g., `test_secret`).

    Example:
    ```bash
    export CONFIG_PATH=connectchain/config/example.config.yml
    export CONSUMER_ID1=test_user 
    export CONSUMER_SECRET1=test_secret
    ```
    (Adjust `CONSUMER_ID1` and `CONSUMER_SECRET1` if your `id_key` and `secret_key` in the config are different.)

5.  **Test Token Retrieval:**
    Run the `token_util.py` script from the root of the repository:
    ```bash
    python connectchain/utils/token_util.py
    ```
    If successful, it will print a Bearer token (a JWT) to the console.

## How it Works

- The FastAPI app in `main.py` defines an `/auth` endpoint.
- When `connectchain`'s `TokenUtil` needs a token, it makes a POST request to this `/auth` endpoint.
- The request includes headers like `X-Auth-AppID`, `X-Auth-Signature`, and `X-Auth-Timestamp`.
- The dummy service uses `X-Auth-AppID` as the subject (`sub`) and `X-Auth-Timestamp` as the issued-at time (`iat`) in the JWT.
- It generates a JWT signed with a hardcoded secret and returns it in the format `{"authorization_token": "YOUR_JWT_HERE"}`.
- `TokenUtil` then prepends "Bearer " to this token.

This setup allows developers to test `connectchain`'s token acquisition flow without needing a live enterprise authentication service.
