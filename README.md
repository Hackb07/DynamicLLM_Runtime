# DynamicLLM Runtime (DLLM)

## A Rust-Based Dynamic Execution Runtime for Large Language Models

Project Domain: Artificial Intelligence, Large Language Models, Runtime Systems, GPU Computing, and Agentic AI

Primary Programming Language: Rust

Core Technologies: Rust, Tokio, CUDA, LLM Inference Engines, Axum, Serde, and System Resource Monitoring

Project Type: AI Systems Engineering and Research & Development

Project Status: Complete — all roadmap phases implemented (v0.1.0)

## 1. Abstract

DynamicLLM Runtime (DLLM) is a Rust-based AI execution runtime designed to simplify the integration, orchestration, and resource-aware execution of Large Language Models (LLMs). The system aims to provide a unified interface for multiple model providers and inference backends while dynamically managing model selection, execution scheduling, and computational resources.

Traditional LLM applications often rely on provider-specific APIs and independent inference pipelines. These approaches can introduce integration complexity, inconsistent execution behavior, unnecessary model-loading overhead, and inefficient resource utilization. These challenges become more significant when multiple models must operate on hardware with limited system memory and GPU video memory (VRAM).

DynamicLLM Runtime addresses these problems through a modular architecture comprising a runtime orchestrator, model registry, inference adapters, intelligent routing engine, resource manager, and telemetry system. Rust serves as the primary language for implementing the runtime because of its memory safety, performance characteristics, strong type system, and support for concurrent execution.

The project also explores dynamic model lifecycle management, including model loading, unloading, caching, and resource-aware scheduling across supported inference environments. CUDA-compatible inference backends can provide GPU acceleration, while local inference engines enable execution of compatible quantized models.

The long-term objective is to establish a flexible and extensible runtime capable of adapting LLM execution to workload requirements, model capabilities, and available computational resources. The system aims to provide a foundation for scalable AI applications and agentic workflows while enabling measurable improvements in execution efficiency and resource utilization.

Keywords: Rust, DynamicLLM Runtime, Large Language Models, LLM Inference, CUDA, Runtime Orchestration, Model Routing, GPU Memory Management, Agentic AI.

## 2. Introduction

Large Language Models have become an essential component of modern AI applications, supporting natural language processing, code generation, document analysis, intelligent assistants, and autonomous AI agents. As these applications become more complex, efficiently managing model execution becomes increasingly important.

Existing LLM applications commonly communicate directly with model APIs or run inference through separate model-serving systems. Although this approach is effective for simple applications, it can become difficult to manage when several models, execution backends, or resource constraints are involved.

DynamicLLM Runtime introduces a dedicated execution layer between AI applications and inference engines. Instead of requiring applications to manage every model integration independently, the runtime provides standardized interfaces and coordinates execution through a common architecture.

Rust is used to develop the core runtime, including asynchronous task execution, request processing, model management, scheduling, and resource monitoring. The runtime can integrate with existing inference engines rather than implementing every model architecture and GPU operation from scratch.

The system is intended to support both hosted LLM APIs and compatible local inference backends. This flexibility enables developers to select an appropriate execution environment based on application requirements, hardware availability, latency constraints, and model capabilities.

A major research direction is the dynamic management of model residency and computational resources. By tracking model usage, loading costs, and memory availability, the runtime can investigate when models should be loaded, retained, or unloaded to reduce unnecessary overhead.

## 3. Problem Statement

Modern AI systems face several challenges when integrating and executing multiple LLMs.

### 3.1 Model Integration Complexity

Different model providers expose different APIs, authentication mechanisms, request formats, and response structures. Applications must often implement provider-specific logic, increasing development and maintenance effort.

### 3.2 Inefficient Resource Utilization

Large models can require substantial RAM and GPU VRAM. Loading multiple models simultaneously can exceed available memory, while repeatedly loading and unloading models may increase execution latency.

### 3.3 Static Model Selection

Applications frequently use a fixed model regardless of the task. However, simple classification, code generation, and complex reasoning may require different capabilities and resource budgets.

### 3.4 Model Switching Overhead

Switching between local models may require loading model weights, allocating memory, initializing inference engines, and preparing execution resources. These operations can create significant delays for workloads involving multiple models.

### 3.5 Limited Runtime Observability

Without a centralized runtime, it can be difficult to understand model-loading delays, inference latency, memory consumption, failed requests, and resource contention.

### 3.6 Agentic Workflow Complexity

Agentic AI applications require coordinated model calls, tool execution, intermediate state management, validation, and error recovery. These workflows benefit from a dedicated execution controller with predictable limits and consistent interfaces.

DynamicLLM Runtime aims to address these challenges through standardized model integration, intelligent routing, resource-aware scheduling, and centralized execution management.

## 4. Project Objectives

The primary objectives of DynamicLLM Runtime are:

1. Develop a high-performance LLM runtime using Rust.
2. Provide a unified interface for multiple inference providers and backends.
3. Implement a model registry for tracking model metadata and capabilities.
4. Develop a routing mechanism that selects suitable models for different tasks.
5. Investigate dynamic model loading, unloading, caching, and reuse.
6. Monitor CPU, RAM, GPU utilization, and VRAM where supported.
7. Implement asynchronous request execution using Tokio.
8. Integrate compatible GPU-accelerated and local inference engines.
9. Support agentic workflows involving multiple model calls and controlled tool execution.
10. Implement structured logging, tracing, error handling, and performance benchmarking.
11. Evaluate runtime performance against a direct-inference baseline.
12. Establish an extensible architecture for future research into predictive scheduling and advanced memory management.

## 5. System Architecture

The architecture separates the application interface, execution logic, model management, resource scheduling, and inference backends.

AI Applications and Agents

Applications, developer tools, and agentic workflows

API and Request Validation

Axum · Serde · Request schemas

Rust Runtime Orchestrator

Execution lifecycle · Cancellation · Error handling

Model Router

Task and capability selection

Resource Manager

Capacity and model lifecycle

Model Registry and Inference Adapters

Common interfaces · Provider-specific implementations

Cloud APIs

Local Engines

Agent Tools

Response Processing and Telemetry

Normalized output · Tracing · Metrics · Error reporting

Figure 1. Logical architecture of DynamicLLM Runtime.

### 5.1 API Layer

The API layer receives requests from external applications and validates their parameters. Axum can provide HTTP endpoints, while Serde supports serialization and deserialization of requests and responses.

The layer can handle request identifiers, generation settings, timeouts, authentication boundaries, and structured error responses.

### 5.2 Runtime Orchestrator

The runtime orchestrator is the central execution component. It coordinates model selection, resource preparation, inference, result processing, and telemetry.

Tokio provides asynchronous execution for I/O-intensive operations, concurrent requests, and communication with remote inference services.

### 5.3 Model Registry

The model registry maintains information about available models, including model identifiers, backend types, supported capabilities, context limits, quantization formats, and resource requirements.

It provides a central source of metadata for routing and model management.

### 5.4 Model Router

The model router identifies an eligible model according to the request's requirements and the configured routing policy. It can consider task type, model capabilities, latency requirements, availability, and resource constraints.

### 5.5 Resource Manager

The resource manager monitors resource availability and coordinates model lifecycle operations. It can work with the model registry to determine which models are loaded, in use, or eligible for eviction.

### 5.6 Inference Adapters

Inference adapters provide a common interface for different model providers and execution engines. They isolate provider-specific request formats, response formats, authentication, and backend-specific errors.

### 5.7 Telemetry System

The telemetry system collects execution metrics, logs model selection decisions, records failures, and provides information needed to evaluate runtime performance.

## 6. Dynamic Model Loading and Memory Management

Dynamic model management is one of the central research areas of the project.

Large language models may require substantial memory to store weights, intermediate tensors, and inference caches. On systems with limited GPU memory, efficient resource allocation is essential.

### 6.1 Memory Hierarchy

The runtime considers several hardware resources:

| Resource   | Purpose                                           | Constraint                            |
| ---------- | ------------------------------------------------- | ------------------------------------- |
| SSD/NVMe   | Persistent model storage                          | Storage bandwidth and loading latency |
| System RAM | Host-side model data and buffers                  | Available physical memory             |
| GPU VRAM   | GPU-resident weights, tensors, and caches         | Limited capacity                      |
| CPU        | Runtime orchestration and supported CPU inference | Compute throughput                    |
| GPU        | Parallel inference computation                    | Memory and compute contention         |

Data movement between these resources introduces overhead. Consequently, a model's performance depends not only on its size but also on its placement, backend, hardware, and workload.

### 6.2 Model Lifecycle Management

The model manager tracks model states such as:

- Registered
- Unloaded
- Loading
- Ready
- Busy
- Evicting
- Failed

The lifecycle manager should prevent conflicting load and unload operations and avoid evicting resources that are still needed by active inference requests.

### 6.3 Model Caching

The runtime can retain frequently used models to reduce repeated initialization costs.

Potential policies include:

- Least Recently Used (LRU): Evict the model that has been unused for the longest period.
- Least Frequently Used (LFU): Favor models with higher historical usage.
- Cost-aware eviction: Consider memory footprint, loading time, and expected reuse.
- Priority-aware eviction: Preserve models associated with higher-priority workloads.

The effectiveness of these policies should be measured against a baseline that loads models without an explicit residency policy.

### 6.4 Memory Mapping and Weight Streaming

Memory mapping and weight streaming can help manage model data in some environments, but their benefits depend on the model format and inference engine.

Memory mapping does not automatically eliminate RAM requirements, and moving model data between storage and VRAM can increase latency. Any implementation should rely on supported backend capabilities and verify the results experimentally.

### 6.5 Quantized Models

Quantization reduces the precision of model weights to decrease memory consumption and potentially improve inference performance.

DynamicLLM Runtime can use compatible quantized models through an inference backend that supports their formats. GGUF, for example, is commonly used with compatible llama.cpp-based inference engines.

The runtime should track format compatibility and account for differences in model quality, memory requirements, and execution speed.

## 7. Intelligent Model Routing

DynamicLLM Runtime aims to select an appropriate model for each request instead of relying exclusively on a fixed model.

### 7.1 Routing Criteria

The router may consider:

- Task category and complexity.
- Required model capabilities.
- Input length and context-window requirements.
- Tool-calling or structured-output support.
- Model availability and provider health.
- Estimated memory consumption.
- Latency targets.
- Configured cost or resource limits.

### 7.2 Routing Strategies

Rule-based routing: Uses explicit mappings between task types and models. It is suitable for the first implementation because decisions are easy to reproduce.

Capability-based routing: Filters models according to required features and compatibility.

Resource-aware routing: Selects among eligible models based on available resources and estimated execution requirements.

Fallback routing: Attempts another compatible backend when a request fails, subject to retry limits and safety policies.

Quality-aware routing: Uses benchmark results and measured task performance to rank eligible models.

### 7.3 Example

A request to summarize a short document may be routed to a smaller, faster model, while a complex programming task may require a model with stronger code-generation capabilities.

The router should first eliminate incompatible models and then select among eligible candidates using a configurable policy.

## 8. Agentic AI Integration

DynamicLLM Runtime can provide the execution foundation for agentic AI systems that perform multi-step tasks using LLMs and external tools.

### 8.1 Agent Components

- Task Controller: Maintains the task's current execution state.
- Model Router: Selects models for individual steps.
- Tool Registry: Describes available tools and their input/output contracts.
- Execution Controller: Enforces timeouts, step limits, and cancellation.
- State Manager: Stores relevant intermediate results.
- Result Validator: Checks generated outputs against task requirements.
- Audit Logger: Records model and tool execution outcomes.

### 8.2 Agent Execution Workflow

1. An application submits a task.
2. The runtime initializes its execution context.
3. The router selects a compatible model.
4. The model generates a response or requests a tool operation.
5. The runtime validates and executes authorized tool calls.
6. The tool result is returned to the model.
7. The execution controller determines whether additional steps are permitted.
8. The runtime validates the final output and records the execution metrics.

Tool access must be controlled through explicit permissions. Model-generated instructions should not automatically authorize file access, shell commands, network operations, or access to secrets.

## 9. Technology Stack

The following technologies form the implementation stack.

| Technology                           | Role                                                       |
| ------------------------------------ | ---------------------------------------------------------- |
| Rust                                 | Core runtime and execution management                      |
| Tokio                                | Asynchronous runtime and concurrency                       |
| Axum                                 | HTTP API service                                           |
| Serde                                | Serialization and configuration                            |
| `async-trait` or native async traits | Common asynchronous backend interfaces                     |
| CUDA                                 | GPU acceleration through compatible inference integrations |
| llama.cpp                            | Candidate local inference engine                           |
| GGUF                                 | Supported model artifact format for compatible engines     |
| Candle                               | Rust-native model experimentation and inference            |
| `sysinfo`                            | Host resource monitoring                                   |
| `tracing`                            | Structured logging and diagnostics                         |
| Criterion                            | Benchmarking Rust components                               |
| Cargo                                | Dependency management, builds, and testing                 |

Not every component is mandatory for the initial release. The runtime should begin with a small, compatible set of dependencies and add other technologies when required.

CUDA integration depends on the selected inference engine and its supported bindings. Rust should coordinate execution and resource policies without unnecessarily reimplementing mature model inference systems.

## 10. Project Structure

```
dynamic-llm-runtime/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── QUICKSTART.md
├── config/
│   ├── models.json
│   └── runtime.json
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── config.rs
│   ├── engine.rs
│   ├── sdk.rs
│   ├── api/
│   │   ├── dashboard.rs
│   │   ├── handlers.rs
│   │   ├── mod.rs
│   │   └── routes.rs
│   ├── core/
│   │   ├── error.rs
│   │   ├── mod.rs
│   │   ├── orchestrator.rs
│   │   ├── runtime.rs
│   │   └── types.rs
│   ├── models/
│   │   ├── capabilities.rs
│   │   ├── lifecycle.rs
│   │   ├── manager.rs
│   │   ├── mod.rs
│   │   └── registry.rs
│   ├── adapters/
│   │   ├── base.rs
│   │   ├── http.rs
│   │   ├── llama_cpp.rs
│   │   ├── local.rs
│   │   ├── mod.rs
│   │   ├── openai.rs
│   │   └── worker.rs
│   ├── routing/
│   │   ├── mod.rs
│   │   ├── router.rs
│   │   └── strategy.rs
│   ├── scheduling/
│   │   ├── mod.rs
│   │   ├── predictor.rs
│   │   ├── queue.rs
│   │   ├── scheduler.rs
│   │   └── workers.rs
│   ├── memory/
│   │   ├── estimator.rs
│   │   ├── eviction.rs
│   │   ├── mod.rs
│   │   └── monitor.rs
│   ├── agents/
│   │   ├── controller.rs
│   │   ├── executor.rs
│   │   ├── mod.rs
│   │   ├── sessions.rs
│   │   ├── tools.rs
│   │   └── types.rs
│   └── telemetry/
│       └── mod.rs
├── tests/
├── benches/
│   └── runtime_bench.rs
└── examples/
    └── sdk_basic.rs
```

This matches the actual repository layout.

### Module Responsibilities

- `api/`: HTTP request handling, API responses, and ops dashboard.
- `core/`: Runtime orchestration and common error types.
- `models/`: Model metadata, registration, and lifecycle management.
- `adapters/`: Provider and inference-engine integrations.
- `routing/`: Model selection and fallback logic.
- `scheduling/`: Execution queues, prefetch prediction, and resource-aware scheduling.
- `memory/`: Memory estimates, residency tracking, and eviction policies.
- `agents/`: Multi-step execution and tool coordination.
- `telemetry/`: Metrics, logging, and tracing.
- `config.rs`: Runtime and model-file configuration loading.
- `engine.rs`: In-process inference engine support.
- `sdk.rs`: In-process Rust client mirroring the HTTP API.

## 11. Execution Lifecycle

The runtime follows a structured execution process.

### Step 1: Request Intake

The API receives the request, validates its fields, and creates an execution identifier.

### Step 2: Task Analysis

The runtime determines the task requirements and the capabilities needed to execute it.

### Step 3: Model Selection

The router selects a compatible model based on configured rules, model availability, and resource constraints.

### Step 4: Resource Preparation

The model manager checks whether the selected model is already available. If it is not loaded, the runtime verifies resource availability and requests initialization through the inference backend.

### Step 5: Inference

The appropriate adapter submits the normalized request to the selected backend and processes the result.

### Step 6: Result Validation

The runtime normalizes the response and validates structured outputs when required.

### Step 7: Agent Continuation

For multi-step tasks, the execution controller determines whether another model or tool operation is necessary and permitted.

### Step 8: Telemetry and Response

The runtime records relevant metrics and returns the result to the application.

## 12. Reliability and Security

### 12.1 Reliability

The runtime should implement:

- Bounded retries with backoff for transient errors.
- Request and execution timeouts.
- Cancellation and graceful shutdown.
- Safe model lifecycle transitions.
- Provider health checks.
- Structured error reporting.
- Concurrency limits and backpressure.
- Resource-aware admission control.

Retries should be designed carefully for operations that may have external side effects.

### 12.2 Security

The system should:

- Protect provider credentials.
- Validate request parameters and tool arguments.
- Apply authorization to sensitive operations.
- Restrict filesystem, shell, and network access.
- Enforce request and execution limits.
- Avoid logging sensitive prompts or credentials by default.
- Isolate untrusted tool outputs from privileged operations.

### 12.3 Observability

The telemetry layer should collect, where available:

- Request identifiers.
- Selected model and backend.
- Routing decisions.
- Queue duration.
- Model initialization time.
- Time to first token.
- Total execution latency.
- Token counts.
- Peak RAM and VRAM.
- Retry count.
- Execution status and error category.

These measurements help identify performance bottlenecks and establish whether proposed runtime optimizations are effective.

## 13. Performance Evaluation

Performance evaluation is essential for demonstrating the value of DynamicLLM Runtime.

The runtime should be compared with a direct-inference baseline using the same model, hardware, generation parameters, and workload.

| Metric                | Description                                            |
| --------------------- | ------------------------------------------------------ |
| Time to first token   | Delay before the first generated token                 |
| Total latency         | Time required to complete a request                    |
| Generation throughput | Tokens generated per second                            |
| Cold-start latency    | Time to initialize an unloaded model                   |
| Warm-request latency  | Time for a request using an already available model    |
| Peak VRAM             | Maximum GPU memory consumption                         |
| Peak RAM              | Maximum host memory consumption                        |
| Model cache hit rate  | Fraction of requests served using retained models      |
| Success rate          | Percentage of successful requests                      |
| Concurrent throughput | Completed requests under a defined concurrent workload |
| Switching overhead    | Additional cost when changing models                   |

### Testing Scenarios

1. Single-model inference.
2. Repeated requests to the same model.
3. Alternating requests across multiple models.
4. Cold-start versus warm-start execution.
5. Resource-constrained local inference.
6. Concurrent requests.
7. Provider failures and recovery.
8. Mixed workloads with different model requirements.

Every benchmark should document the model version, quantization, inference backend, hardware configuration, workload, and measurement method.

Performance improvements must be demonstrated through measured results rather than assumed from the use of Rust alone.

## 14. Implementation Roadmap (all phases delivered)

### Phase 1: Core Runtime

- Initialize the Rust project.
- Define request and response types.
- Implement the common inference interface.
- Build the execution orchestrator.
- Add unit tests and error handling.

Delivered: a working runtime with one inference backend.

### Phase 2: Model Integration

- Implement the model registry.
- Add additional inference adapters.
- Normalize model capabilities and response formats.
- Add integration tests.

Delivered: a unified interface for multiple supported models.

### Phase 3: Routing and Telemetry

- Implement rule-based model routing.
- Add structured logging and tracing.
- Record request latency and execution failures.
- Introduce timeouts and bounded retries.

Delivered: a measurable runtime with predictable model selection.

### Phase 4: Local Inference

- Integrate a compatible local inference engine.
- Implement model loading and unloading.
- Add model reuse and lifecycle synchronization.
- Test resource consumption.

Delivered: a runtime capable of managing compatible local inference workloads.

### Phase 5: Resource-Aware Scheduling

- Monitor system resource availability.
- Implement scheduling and admission-control policies.
- Introduce model eviction and reuse strategies.
- Benchmark multi-model execution.

Delivered: a runtime that adapts execution decisions to resource constraints.

### Phase 6: Agentic Execution

- Implement the task controller.
- Add a tool registry and execution permissions.
- Support multi-step execution state.
- Introduce step limits and result validation.

Delivered: a controlled runtime for agentic AI workflows.

### Phase 7: Advanced Research

- Investigate model prefetching.
- Evaluate memory-mapping and supported streaming approaches.
- Explore predictive scheduling.
- Compare advanced policies against baseline strategies.

Delivered: prefetching, memory-mapped managed servers, predictive scheduling, and benchmark harnesses for comparing strategies.

## 15. Technical Challenges

### 15.1 Backend Compatibility

Inference engines differ in supported model formats, hardware acceleration, streaming, and configuration. The runtime must maintain explicit adapter contracts and validate backend capabilities.

### 15.2 GPU Memory Constraints

Model weights, intermediate tensors, and inference caches compete for VRAM. The runtime must account for these requirements rather than considering model weights alone.

### 15.3 Concurrent Model Operations

Multiple requests may attempt to load, use, or unload the same model. Lifecycle synchronization and safe ownership tracking are necessary.

### 15.4 Scheduling Decisions

A model that uses less memory may not deliver the required output quality, and a faster model may not support the requested operation. Routing must account for capability and quality constraints alongside resource usage.

### 15.5 Memory Transfer Overhead

Moving data between storage, RAM, and VRAM can introduce latency. The runtime must benchmark supported memory-management techniques for each target environment.

### 15.6 Rust and Native Inference Integration

Rust can efficiently manage orchestration and concurrency, but native inference libraries may require FFI bindings, separate processes, or service-based integration. Compatibility and error handling must be evaluated for the chosen backend.

## 16. Expected Outcomes

The project delivers:

- A modular Rust-based LLM runtime.
- A unified interface for compatible inference backends.
- A model registry and lifecycle manager.
- Configurable model-routing policies.
- Resource-aware scheduling and model reuse.
- Local inference integration where supported.
- An extensible foundation for agentic AI workflows.
- Structured telemetry and performance benchmarking.
- Reproducible evaluations of model-loading and scheduling strategies.

All items above are implemented and covered by unit and integration tests (`cargo test`); see `QUICKSTART.md` for operating notes and known limitations.

## 17. Future Scope

### Predictive Model Scheduling

Shipped as frequency-based next-model prediction with warm-set and predictive prefetching (`prefetch_warm_models`, `predictive_prefetch_enabled`). Future work: richer demand models over historical workload data.

### Advanced Memory Management

Shipped: SSD-backed `--mmap` managed servers and in-process engine, admission budgets, and LRU/LFU/cost-aware/priority-aware eviction. Future work: asynchronous data transfers and partial GPU residency where backends support them.

### Multi-Agent Execution

Shipped: isolated sessions with independent task states, bounded global/per-agent concurrency, shared tool policies, and idle expiry (`remove_idle_sessions`). Future work: cross-session coordination primitives.

### Distributed Inference

Shipped as opt-in remote execution: configured workers serve inference and streaming with affinity plus least-loaded placement and local fallback. Future work: worker auto-discovery and latency-aware placement.

### Adaptive Model Routing

Shipped as `adaptive` / `quality_aware` strategies scoring measured latency and success rate with task-affinity bonuses. Future work: quality signals beyond latency and success rate.

### Runtime Dashboard

Shipped: `GET /dashboard` shows models, lifecycle, memory, scheduler, telemetry, and workers by polling `/v1/status` and `/v1/models`.

### Developer SDK

Shipped: the `sdk` module (`Client`, `InferBuilder`, `AgentTaskBuilder`) mirrors the HTTP API in-process; see `examples/sdk_basic.rs`.

## 18. Conclusion

DynamicLLM Runtime is a Rust-based execution platform for integrating and managing large language models across supported inference environments. Its architecture separates model integration, orchestration, routing, resource management, and telemetry to improve modularity and operational control.

Rust provides the foundation for safe concurrent execution and efficient runtime implementation. Existing inference engines and compatible GPU integrations provide the underlying model computation capabilities.

The project's central research direction is to make model execution more adaptable to available computational resources through model lifecycle management, resource-aware scheduling, and intelligent routing.

By progressing from a reliable core runtime toward advanced memory management and agentic execution, DynamicLLM Runtime provides a strong foundation for efficient, extensible, and measurable AI systems.

## 19. Project Summary

Project Name: DynamicLLM Runtime (DLLM)

One-line Description: A Rust-based AI execution runtime designed to unify LLM inference backends and explore intelligent model routing, resource-aware scheduling, dynamic model lifecycle management, and agentic execution.

Primary Language: Rust

Primary Research Areas: LLM inference, runtime systems, model residency, GPU memory management, and intelligent scheduling.

Core Engineering Goal: Provide a standardized and observable execution environment for large language models.

Long-Term Goal: Build an adaptable AI runtime that manages model execution according to workload requirements, backend capabilities, and available computational resources.

Documentation note: This document describes the system design and implementation. All roadmap phases are implemented; see `QUICKSTART.md` for run instructions and `cargo test` / `cargo bench` for verification.

## 20. Operating the runtime

Practical entry points (see `QUICKSTART.md` for the full guide):

- Run: `cargo run`, then use `http://127.0.0.1:8080` (or your configured
  `host:port`).
- Dashboard: `GET /dashboard` — a dependency-free page showing registered
  models, loaded/resident models, memory, scheduler/agent gates, and
  telemetry (latency, success/fail, per-model counts) by polling
  `/v1/status` and `/v1/models`.

HTTP endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/models` | Registered models |
| `GET` | `/v1/status` | Loaded models, memory, scheduler, telemetry |
| `POST` | `/v1/infer` | Non-streaming inference |
| `POST` | `/v1/infer/stream` | SSE streaming inference |
| `POST` | `/v1/agent/run` | Agentic task run (tool loop + audit) |
| `GET` | `/v1/agent/tools` | Tool definitions the agent executor can run |
| `GET` | `/dashboard` | Ops dashboard (static page) |

Rust SDK (in-process, no socket; see `examples/sdk_basic.rs`):

```rust
use dynamic_llm_runtime::sdk::{AgentTaskBuilder, Client, InferBuilder};

let client = Client::from_config_dir("config").await?;
let response = client.infer(InferBuilder::new("Say hi").build()).await?;
let agent = client
    .run_agent(AgentTaskBuilder::new("Compute 2+3").allow_tool("calc").build())
    .await?;
```

Recent `config/runtime.json` fields (all optional): `routing_latency_weight`,
`routing_success_weight`, `routing_affinity_bonus` (adaptive/quality-aware
routing), `prefetch_concurrency`, `prefetch_warm_models`,
`predictive_prefetch_enabled`, `predictor_history_limit` (prefetching),
`workers` (remote inference workers; empty = single-node),
`llamacpp_binary`, `llamacpp_ctx_size`, `llamacpp_extra_args`, `n_gpu_layers` (managed
local servers). Strategies: `fixed`, `rule_based`, `capability_based`,
`resource_aware`, `fallback`, `adaptive`, `quality_aware`.
