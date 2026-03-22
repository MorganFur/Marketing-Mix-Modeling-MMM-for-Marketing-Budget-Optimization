from pathlib import Path
import importlib.util
import json
import warnings


BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "mmm_app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("mmm_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    module = load_app_module()
    client = module.app.test_client()

    sample_payload = {
        "GOOGLE_PAID_SEARCH_SPEND": 24,
        "GOOGLE_SHOPPING_SPEND": 27,
        "GOOGLE_PMAX_SPEND": 0,
        "META_FACEBOOK_SPEND": 39,
        "META_INSTAGRAM_SPEND": 0,
        "EMAIL_CLICKS": 54,
        "ORGANIC_SEARCH_CLICKS": 190,
        "DIRECT_CLICKS": 127,
        "BRANDED_SEARCH_CLICKS": 39,
        "year": 2022,
        "month": 6,
        "day_of_week": 3,
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LinearRegression was fitted with feature names",
        )
        response = client.post("/predict", json=sample_payload)
    body = response.get_json()

    print(f"status_code={response.status_code}")
    print(json.dumps(body, indent=2, ensure_ascii=False))

    if response.status_code != 200:
        raise SystemExit("Smoke test failed: /predict did not return HTTP 200.")

    predicted_revenue = body.get("predicted_revenue")
    if not isinstance(predicted_revenue, (int, float)):
        raise SystemExit("Smoke test failed: predicted_revenue is missing or invalid.")

    print("Smoke test passed.")

    batch_payload = [
        sample_payload,
        {
            "GOOGLE_PAID_SEARCH_SPEND": 1000,
            "GOOGLE_SHOPPING_SPEND": 800,
            "GOOGLE_PMAX_SPEND": 300,
            "META_FACEBOOK_SPEND": 900,
            "META_INSTAGRAM_SPEND": 200,
            "EMAIL_CLICKS": 500,
            "ORGANIC_SEARCH_CLICKS": 1200,
            "DIRECT_CLICKS": 700,
            "BRANDED_SEARCH_CLICKS": 350,
            "year": 2024,
            "month": 5,
            "day_of_week": 2,
        },
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LinearRegression was fitted with feature names",
        )
        batch_response = client.post("/predict", json=batch_payload)
    batch_body = batch_response.get_json()

    print(f"batch_status_code={batch_response.status_code}")
    print(json.dumps(batch_body, indent=2, ensure_ascii=False))

    if batch_response.status_code != 200:
        raise SystemExit("Smoke test failed: batch /predict did not return HTTP 200.")

    predictions = batch_body.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(batch_payload):
        raise SystemExit("Smoke test failed: batch predictions are missing or invalid.")

    if not all(isinstance(item.get("predicted_revenue"), (int, float)) for item in predictions):
        raise SystemExit("Smoke test failed: one or more batch predictions are invalid.")

    print("Batch smoke test passed.")


if __name__ == "__main__":
    main()
