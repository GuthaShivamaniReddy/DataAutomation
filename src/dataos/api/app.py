"""FastAPI application for the analyst workflow: upload a dataset,
describe a requirement in plain English, resolve clarifications,
approve, plan, run, release, and see the explained/visualized result.

Not part of the AI Prompt Library or Blueprint spec - this is the HTTP
surface over the already-tested compiler/orchestrator pipeline. Every
endpoint below is thin: it loads/stores plumbing via `SessionStore` and
otherwise calls straight into `WorkflowOrchestrator`/`RequirementCompiler`/
`WorkflowPlanner`, never reimplementing any decision those already make.

Single local user, no auth - matches this app's current scope. State
lives in `SessionStore` (datasets/contracts) plus the existing
`RunStore`/`ArtifactStore` (runs) under `.dataos_store/`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataos.api.store import ContractRecord, DatasetRecord, SessionStore
from dataos.compiler.analytics_strategy_agent import AnalyticsStrategyResult
from dataos.compiler.cleaning_strategy_agent import CleaningPlan
from dataos.compiler.data_quality_assessor import DataQualityReport
from dataos.compiler.explanation_agent import ExplanationAgent
from dataos.compiler.requirement_compiler import RequirementCompiler
from dataos.compiler.schema_mapping_agent import SchemaMappingResult
from dataos.compiler.workflow_planner import WorkflowPlanner
from dataos.contracts.requirement_contract import RequirementContract, RequirementStatus
from dataos.errors import ErrorCode, PlatformError
from dataos.ingestion.parsers import ParseReport, parse_csv, parse_parquet, parse_xlsx
from dataos.ingestion.profiling import DatasetProfile
from dataos.ingestion.snapshot import ingest_file
from dataos.llm.client import LLMClient
from dataos.llm.deterministic import DeterministicLLMClient
from dataos.registry.operations.aggregate import AggregateOperation
from dataos.registry.operations.cast import CastOperation
from dataos.registry.operations.deduplicate import DeduplicateOperation
from dataos.registry.operations.derive import DeriveOperation
from dataos.registry.operations.export import ExportOperation
from dataos.registry.operations.join import JoinOperation
from dataos.registry.operations.pivot import PivotOperation
from dataos.registry.operations.select_filter import SelectFilterOperation
from dataos.registry.operations.text_clean import TextCleanOperation
from dataos.registry.operations.unpivot import UnpivotOperation
from dataos.registry.operations.validate_schema import ValidateSchemaOperation
from dataos.registry.registry import OperationRegistry
from dataos.semantics.dictionary import SemanticDictionary
from dataos.workflow.artifact_store import ArtifactStore
from dataos.workflow.dsl import WorkflowStep
from dataos.workflow.orchestrator import WorkflowOrchestrator
from dataos.workflow.store import RunStore

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DEFAULT_STORE_ROOT = Path(os.environ.get("DATAOS_STORE_ROOT", ".dataos_store"))


def _build_registry() -> OperationRegistry:
    """Every deterministic operation this codebase has, including
    `join` - `orchestrator.run()` now executes `BinaryOperation` steps
    too (see its own docstring)."""
    registry = OperationRegistry()
    for op_cls in (
        SelectFilterOperation,
        CastOperation,
        DeduplicateOperation,
        DeriveOperation,
        AggregateOperation,
        ExportOperation,
        ValidateSchemaOperation,
        JoinOperation,
        PivotOperation,
        UnpivotOperation,
        TextCleanOperation,
    ):
        registry.register(op_cls())
    return registry


def _default_llm_client() -> tuple[LLMClient, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from dataos.llm.anthropic_client import AnthropicLLMClient

            return AnthropicLLMClient(api_key=api_key), "anthropic"
        except PlatformError:
            pass
    return DeterministicLLMClient(), "deterministic-stub"


def _default_semantic_dictionary() -> SemanticDictionary:
    """Section 5's governed semantic layer starts empty by default - a
    generic starter set (recognized_revenue/gross_revenue/order_count/
    region) can be loaded via DATAOS_SEMANTIC_DICTIONARY_PATH for a demo
    that does not immediately hit "UNGOVERNED" on every metric term."""
    path = os.environ.get("DATAOS_SEMANTIC_DICTIONARY_PATH")
    if path:
        return SemanticDictionary.load_from_json(path)
    return SemanticDictionary()


class CreateRequirementRequest(BaseModel):
    objective: str
    dataset_ids: list[str]


class ReviseRequirementRequest(BaseModel):
    additional_context: str


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    row_count: int
    column_count: int
    created_at: str


class UploadDatasetResponse(BaseModel):
    dataset: DatasetRecord
    parse_report: ParseReport


class PlanReviewResponse(BaseModel):
    workflow_id: str
    steps: list[WorkflowStep]
    schema_mapping: SchemaMappingResult
    data_quality: DataQualityReport
    cleaning_plan: CleaningPlan
    approved_cleaning_rule_ids: list[str]
    analytics_strategy: AnalyticsStrategyResult
    blocking: bool


class ApproveCleaningRulesRequest(BaseModel):
    rule_ids: list[str]


def _unapproved_required_cleaning_rules(cleaning_plan: CleaningPlan, approved_rule_ids: list[str]) -> list[str]:
    """A rule the agent marked `approval_required` (every lossy fix) still
    blocks the run until a human has approved that exact rule id - see
    `cleaning_strategy_agent.py`'s own "never automatically apply a lossy
    rule" discipline."""
    approved = set(approved_rule_ids)
    return [r.rule_id for r in cleaning_plan.rules if r.approval_required and r.rule_id not in approved]


def _plan_blocking(
    schema_mapping: SchemaMappingResult,
    data_quality: DataQualityReport,
    cleaning_plan: CleaningPlan,
    approved_cleaning_rule_ids: list[str],
) -> bool:
    return (
        bool(schema_mapping.blocking_items)
        or data_quality.fitness == "FAIL"
        or bool(cleaning_plan.blocking_items)
        or bool(_unapproved_required_cleaning_rules(cleaning_plan, approved_cleaning_rule_ids))
    )


def create_app(
    *,
    store_root: Path | str | None = None,
    llm_client: LLMClient | None = None,
    semantic_dictionary: SemanticDictionary | None = None,
) -> FastAPI:
    root = Path(store_root) if store_root is not None else _DEFAULT_STORE_ROOT
    session_store = SessionStore(root=root / "api")
    registry = _build_registry()

    resolved_llm_client, llm_backend_name = (llm_client, "caller-supplied") if llm_client is not None else _default_llm_client()
    dictionary = semantic_dictionary if semantic_dictionary is not None else _default_semantic_dictionary()

    compiler = RequirementCompiler(resolved_llm_client, dictionary)
    planner = WorkflowPlanner(resolved_llm_client, registry)
    explainer = ExplanationAgent(resolved_llm_client)

    def _new_orchestrator() -> WorkflowOrchestrator:
        """A fresh `RunStore`/`ArtifactStore` per call, both cheap,
        correct re-opens of the same on-disk SQLite file/artifact
        directory: `sqlite3` connections are not safe to share across
        threads, and a sync or async FastAPI endpoint is not guaranteed
        to run on the same thread `create_app()` itself ran on (this is
        exactly what a shared, request-scoped connection would get
        wrong under ASGI). `registry`/`planner`/`explainer` hold no I/O
        handles, so they stay shared."""
        return WorkflowOrchestrator(
            registry, RunStore(root / "runs.db"), ArtifactStore(root / "artifacts"), planner, explainer=explainer
        )

    app = FastAPI(title="DataOS Analyst Workbench", version="0.1.0")

    @app.exception_handler(PlatformError)
    async def _platform_error_handler(_, exc: PlatformError) -> JSONResponse:
        status_code = 404 if exc.code == ErrorCode.SCHEMA_MISSING else 422
        return JSONResponse(status_code=status_code, content=exc.to_dict())

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "llm_backend": llm_backend_name}

    # ---- datasets ----

    @app.post("/api/datasets", response_model=UploadDatasetResponse)
    async def upload_dataset(
        file: UploadFile = File(...),
        name: str | None = Form(None),
        mode: Literal["strict", "permissive"] = Form("permissive"),
    ) -> UploadDatasetResponse:
        suffix = Path(file.filename or "upload").suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            version = ingest_file(tmp_path, storage_root=root / "raw")
            version = version.model_copy(update={"original_filename": file.filename or version.original_filename})

            if suffix == ".csv":
                df, parse_report = parse_csv(tmp_path, mode=mode)
            elif suffix in (".xlsx", ".xls"):
                df, parse_report = parse_xlsx(tmp_path, mode=mode)
            elif suffix == ".parquet":
                df, parse_report = parse_parquet(tmp_path, mode=mode)
            else:
                raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix or '(none)'}")
        finally:
            tmp_path.unlink(missing_ok=True)

        dataset_name = name or Path(file.filename or "dataset").stem
        record = session_store.add_dataset(name=dataset_name, df=df, version=version)
        return UploadDatasetResponse(dataset=record, parse_report=parse_report)

    @app.get("/api/datasets", response_model=list[DatasetSummary])
    async def list_datasets() -> list[DatasetSummary]:
        return [
            DatasetSummary(
                dataset_id=d.dataset_id, name=d.name, row_count=d.profile.row_count,
                column_count=d.profile.column_count, created_at=d.created_at,
            )
            for d in session_store.list_datasets()
        ]

    @app.get("/api/datasets/{dataset_id}", response_model=DatasetRecord)
    async def get_dataset(dataset_id: str) -> DatasetRecord:
        return session_store.require_dataset(dataset_id)

    @app.get("/api/datasets/{dataset_id}/sample")
    async def sample_dataset(dataset_id: str, limit: int = 50) -> list[dict]:
        df = session_store.load_dataset_frame(dataset_id)
        return df.head(limit).to_dicts()

    # ---- requirements ----

    @app.post("/api/requirements", response_model=ContractRecord)
    async def create_requirement(request: CreateRequirementRequest) -> ContractRecord:
        datasets = [session_store.require_dataset(did) for did in request.dataset_ids]
        profile: DatasetProfile | None = datasets[0].profile if datasets else None

        output = compiler.compile(
            user_request=request.objective,
            sources=[d.name for d in datasets],
            profile=profile,
        )
        dataset_ids_by_source_name = {d.name: d.dataset_id for d in datasets}
        return session_store.add_contract(contract=output.contract, dataset_ids_by_source_name=dataset_ids_by_source_name)

    @app.get("/api/requirements", response_model=list[ContractRecord])
    async def list_requirements() -> list[ContractRecord]:
        return session_store.list_contracts()

    @app.get("/api/requirements/{contract_id}", response_model=ContractRecord)
    async def get_requirement(contract_id: str) -> ContractRecord:
        return session_store.require_contract(contract_id)

    @app.post("/api/requirements/{contract_id}/revise", response_model=ContractRecord)
    async def revise_requirement(contract_id: str, request: ReviseRequirementRequest) -> ContractRecord:
        record = session_store.require_contract(contract_id)
        augmented_objective = f"{record.contract.objective}\n\nAdditional clarification: {request.additional_context}"

        profiles = [
            session_store.require_dataset(did).profile for did in record.dataset_ids_by_source_name.values()
        ]
        output = compiler.compile(
            user_request=augmented_objective,
            sources=list(record.dataset_ids_by_source_name.keys()),
            profile=profiles[0] if profiles else None,
        )
        return session_store.update_contract(contract_id, contract=output.contract)

    @app.post("/api/requirements/{contract_id}/approve", response_model=ContractRecord)
    async def approve_requirement(contract_id: str) -> ContractRecord:
        record = session_store.require_contract(contract_id)
        if record.contract.clarifications:
            raise HTTPException(
                status_code=409,
                detail="cannot approve a contract with unresolved clarifications - call /revise first",
            )
        approved = RequirementContract.model_validate(
            {**record.contract.model_dump(), "status": RequirementStatus.APPROVED.value}
        )
        if not approved.ready_for_planning:
            raise HTTPException(
                status_code=409,
                detail={"message": "contract is still not ready for planning", "blocking_reasons": approved.blocking_reasons},
            )
        return session_store.update_contract(contract_id, contract=approved)

    @app.post("/api/requirements/{contract_id}/plan", response_model=PlanReviewResponse)
    async def plan_requirement(contract_id: str) -> PlanReviewResponse:
        record = session_store.require_contract(contract_id)
        if not record.contract.ready_for_planning:
            raise HTTPException(status_code=409, detail="contract must be APPROVED (with no clarifications) before planning")

        planner_output = planner.plan(contract=record.contract, contract_id=contract_id)
        workflow = planner_output.workflow

        profiles = {
            name: session_store.require_dataset(did).profile
            for name, did in record.dataset_ids_by_source_name.items()
        }
        orchestrator = _new_orchestrator()
        schema_mapping = orchestrator.map_schema(record.contract, profiles)
        data_quality = orchestrator.assess_data_quality(record.contract, profiles, schema_mapping=schema_mapping)
        cleaning_plan = orchestrator.propose_cleaning_rules(record.contract, quality_report=data_quality)
        analytics = orchestrator.select_analytics_strategy(record.contract)

        session_store.update_contract(contract_id, workflow=workflow, cleaning_plan=cleaning_plan)

        blocking = _plan_blocking(schema_mapping, data_quality, cleaning_plan, approved_cleaning_rule_ids=[])
        return PlanReviewResponse(
            workflow_id=contract_id,
            steps=workflow.steps,
            schema_mapping=schema_mapping,
            data_quality=data_quality,
            cleaning_plan=cleaning_plan,
            approved_cleaning_rule_ids=[],
            analytics_strategy=analytics,
            blocking=blocking,
        )

    @app.post("/api/requirements/{contract_id}/cleaning/approve", response_model=PlanReviewResponse)
    async def approve_cleaning_rules(contract_id: str, request: ApproveCleaningRulesRequest) -> PlanReviewResponse:
        record = session_store.require_contract(contract_id)
        if record.workflow is None or record.cleaning_plan is None:
            raise HTTPException(status_code=409, detail="call /plan before approving cleaning rules")

        profiles = {
            name: session_store.require_dataset(did).profile
            for name, did in record.dataset_ids_by_source_name.items()
        }
        approved_rule_ids = sorted(set(record.approved_cleaning_rule_ids) | set(request.rule_ids))
        orchestrator = _new_orchestrator()
        workflow = orchestrator.apply_cleaning_rules(
            record.workflow, record.cleaning_plan, approved_rule_ids, profiles=profiles
        )

        session_store.update_contract(
            contract_id, workflow=workflow, approved_cleaning_rule_ids=approved_rule_ids
        )

        schema_mapping = orchestrator.map_schema(record.contract, profiles)
        data_quality = orchestrator.assess_data_quality(record.contract, profiles, schema_mapping=schema_mapping)
        analytics = orchestrator.select_analytics_strategy(record.contract)
        blocking = _plan_blocking(schema_mapping, data_quality, record.cleaning_plan, approved_rule_ids)
        return PlanReviewResponse(
            workflow_id=contract_id,
            steps=workflow.steps,
            schema_mapping=schema_mapping,
            data_quality=data_quality,
            cleaning_plan=record.cleaning_plan,
            approved_cleaning_rule_ids=approved_rule_ids,
            analytics_strategy=analytics,
            blocking=blocking,
        )

    @app.post("/api/requirements/{contract_id}/start")
    async def start_requirement_run(contract_id: str) -> dict:
        record = session_store.require_contract(contract_id)
        if record.workflow is None:
            raise HTTPException(status_code=409, detail="call /plan before /start")
        if record.cleaning_plan is not None:
            unresolved = _unapproved_required_cleaning_rules(record.cleaning_plan, record.approved_cleaning_rule_ids)
            if unresolved:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "unresolved cleaning rule(s) require approval before this run can start",
                        "rule_ids": unresolved,
                    },
                )

        source_frames = {
            name: session_store.load_dataset_frame(did) for name, did in record.dataset_ids_by_source_name.items()
        }
        source_versions = {
            name: session_store.require_dataset(did).version for name, did in record.dataset_ids_by_source_name.items()
        }

        orchestrator = _new_orchestrator()
        run_id = orchestrator.start_run(
            record.workflow, source_frames=source_frames, source_versions=source_versions, contract=record.contract
        )
        session_store.update_contract(contract_id, run_id=run_id)
        run = orchestrator.run(run_id, record.workflow)
        return {"run_id": run_id, "state": run.state.value}

    # ---- runs ----

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        run_store = RunStore(root / "runs.db")
        run = run_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run with id '{run_id}'")
        steps = run_store.list_step_runs(run_id)
        return {"run": run.model_dump(mode="json"), "steps": [s.model_dump(mode="json") for s in steps]}

    @app.post("/api/runs/{run_id}/release")
    async def release_run(run_id: str) -> dict:
        record = session_store.get_contract_for_run(run_id)
        result = _new_orchestrator().release(run_id, record.workflow, record.contract)
        return result.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/explain")
    async def explain_run(run_id: str, audience: Literal["analyst", "operator", "executive", "customer", "api"] = "analyst") -> dict:
        record = session_store.get_contract_for_run(run_id)
        output = _new_orchestrator().explain(run_id, record.workflow, record.contract, audience=audience)
        return output.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/visualize")
    async def visualize_run(run_id: str) -> dict:
        record = session_store.get_contract_for_run(run_id)
        result = _new_orchestrator().plan_visualizations(run_id, record.workflow, record.contract)
        return result.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/incident")
    async def incident_run(run_id: str) -> dict:
        result = _new_orchestrator().diagnose_incident(run_id)
        return result.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/result")
    async def result_run(run_id: str, limit: int = 100, offset: int = 0) -> dict:
        record = session_store.get_contract_for_run(run_id)
        execution_order = record.workflow.execution_order()
        if not execution_order:
            raise HTTPException(status_code=404, detail="workflow has no steps to load a result from")
        terminal_step_id = execution_order[-1]
        df = ArtifactStore(root / "artifacts").load_by_ids(run_id, terminal_step_id)
        return {
            "step_id": terminal_step_id,
            "row_count": df.height,
            "column_count": df.width,
            "rows": df.slice(offset, limit).to_dicts(),
        }

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app


app = create_app()
