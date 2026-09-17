from fastapi.testclient import TestClient

from dataos.api.app import create_app
from dataos.llm.deterministic import DeterministicLLMClient
from dataos.semantics.dictionary import SemanticDictionary


def _client(tmp_path, *, dictionary: SemanticDictionary | None = None) -> TestClient:
    app = create_app(
        store_root=tmp_path / "dataos_store",
        llm_client=DeterministicLLMClient(),
        semantic_dictionary=dictionary or SemanticDictionary(),
    )
    return TestClient(app)


def test_health_reports_the_deterministic_stub_backend(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["llm_backend"] == "caller-supplied"


def test_upload_dataset_returns_a_profile_and_parse_report(tmp_path, fixtures_dir):
    client = _client(tmp_path)

    with open(fixtures_dir / "orders_basic.csv", "rb") as f:
        response = client.post(
            "/api/datasets",
            files={"file": ("orders.csv", f, "text/csv")},
            data={"name": "orders"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["name"] == "orders"
    assert body["dataset"]["profile"]["row_count"] == 5
    assert body["parse_report"]["rows_rejected"] == 0


def test_full_analyst_flow_upload_to_release_and_explain(tmp_path, fixtures_dir, sample_semantic_dictionary):
    client = _client(tmp_path, dictionary=sample_semantic_dictionary)

    with open(fixtures_dir / "orders_basic.csv", "rb") as f:
        upload = client.post(
            "/api/datasets", files={"file": ("orders.csv", f, "text/csv")}, data={"name": "orders"}
        )
    dataset_id = upload.json()["dataset"]["dataset_id"]

    created = client.post(
        "/api/requirements", json={"objective": "show order count", "dataset_ids": [dataset_id]}
    )
    assert created.status_code == 200
    contract_id = created.json()["contract_id"]
    contract = created.json()["contract"]
    assert contract["clarifications"] == []
    assert contract["metrics"][0]["definition_status"] == "GOVERNED"

    approved = client.post(f"/api/requirements/{contract_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["contract"]["status"] == "APPROVED"

    planned = client.post(f"/api/requirements/{contract_id}/plan")
    assert planned.status_code == 200
    plan_body = planned.json()
    assert plan_body["steps"]
    assert plan_body["data_quality"]["fitness"] in ("PASS", "CONDITIONAL")

    started = client.post(f"/api/requirements/{contract_id}/start")
    assert started.status_code == 200
    run_id = started.json()["run_id"]

    run_status = client.get(f"/api/runs/{run_id}")
    assert run_status.status_code == 200
    assert run_status.json()["run"]["state"] == "VERIFYING"

    released = client.post(f"/api/runs/{run_id}/release")
    assert released.status_code == 200
    assert released.json()["run"]["state"] == "RELEASED"

    explanation = client.get(f"/api/runs/{run_id}/explain")
    assert explanation.status_code == 200
    assert explanation.json()["explanation"]["findings"]

    visualization = client.get(f"/api/runs/{run_id}/visualize")
    assert visualization.status_code == 200

    result = client.get(f"/api/runs/{run_id}/result")
    assert result.status_code == 200
    assert result.json()["row_count"] == 1


def test_approve_is_refused_when_the_contract_is_ungoverned(tmp_path, fixtures_dir):
    client = _client(tmp_path)

    with open(fixtures_dir / "orders_basic.csv", "rb") as f:
        upload = client.post(
            "/api/datasets", files={"file": ("orders.csv", f, "text/csv")}, data={"name": "orders"}
        )
    dataset_id = upload.json()["dataset"]["dataset_id"]

    created = client.post(
        "/api/requirements",
        json={"objective": "show total mystery_metric_xyz by region", "dataset_ids": [dataset_id]},
    )
    contract_id = created.json()["contract_id"]

    approved = client.post(f"/api/requirements/{contract_id}/approve")

    assert approved.status_code == 409


def test_start_is_blocked_until_a_required_cleaning_rule_is_approved(tmp_path, fixtures_dir, sample_semantic_dictionary):
    client = _client(tmp_path, dictionary=sample_semantic_dictionary)

    with open(fixtures_dir / "orders_dirty_region.csv", "rb") as f:
        upload = client.post(
            "/api/datasets", files={"file": ("orders.csv", f, "text/csv")}, data={"name": "orders"}
        )
    dataset_id = upload.json()["dataset"]["dataset_id"]

    created = client.post(
        "/api/requirements", json={"objective": "show order count by region", "dataset_ids": [dataset_id]}
    )
    contract_id = created.json()["contract_id"]
    client.post(f"/api/requirements/{contract_id}/approve")

    planned = client.post(f"/api/requirements/{contract_id}/plan")
    assert planned.status_code == 200
    plan_body = planned.json()
    assert plan_body["data_quality"]["fitness"] == "CONDITIONAL"
    rule = next(r for r in plan_body["cleaning_plan"]["rules"] if r["rule_id"] == "text_clean:orders.region")
    assert rule["approval_required"] is True
    assert plan_body["blocking"] is True

    blocked_start = client.post(f"/api/requirements/{contract_id}/start")
    assert blocked_start.status_code == 409
    assert rule["rule_id"] in blocked_start.json()["detail"]["rule_ids"]

    approved = client.post(
        f"/api/requirements/{contract_id}/cleaning/approve", json={"rule_ids": [rule["rule_id"]]}
    )
    assert approved.status_code == 200
    approved_body = approved.json()
    assert approved_body["blocking"] is False
    assert approved_body["approved_cleaning_rule_ids"] == [rule["rule_id"]]
    assert approved_body["steps"][0]["operation_id"] == "text_clean"

    started = client.post(f"/api/requirements/{contract_id}/start")
    assert started.status_code == 200
    run_id = started.json()["run_id"]

    released = client.post(f"/api/runs/{run_id}/release")
    assert released.status_code == 200
    assert released.json()["run"]["state"] == "RELEASED"

    result = client.get(f"/api/runs/{run_id}/result")
    assert result.status_code == 200
    # 5 distinct-looking regions collapse to 4 real ones once cleaned.
    assert result.json()["row_count"] == 4


def test_get_unknown_dataset_is_a_404(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/datasets/does-not-exist")

    assert response.status_code == 404
