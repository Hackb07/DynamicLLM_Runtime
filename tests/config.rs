use dynamic_llm_runtime::config::{ModelsFile, RuntimeConfig};
use std::sync::Mutex;

// Process env vars are global; serialize every test that reads them.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn write(dir: &tempfile::TempDir, name: &str, contents: &str) {
    std::fs::write(dir.path().join(name), contents).unwrap();
}

#[test]
fn runtime_config_file_loading() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::remove_var("DLLM_PORT");
    std::env::remove_var("OPENAI_API_KEY");

    let dir = tempfile::tempdir().unwrap();
    write(
        &dir,
        "runtime.json",
        r#"{
            "host": "0.0.0.0", "port": 9090, "request_timeout_secs": 30,
            "max_retries": 1, "retry_backoff_ms": 50,
            "max_concurrent_requests": 4, "max_fallback_models": 1,
            "local_model_budget_mb": 1024, "routing_strategy": "resource_aware",
            "eviction_policy": "lfu", "fixed_model": null,
            "openai_base_url": "http://mock/v1", "openai_default_model": "m",
            "openai_api_key": null, "llamacpp_base_url": "http://mock:8081",
            "llamacpp_default_model": "l"
        }"#,
    );
    let config = RuntimeConfig::load(dir.path());
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 9090);
    assert_eq!(config.request_timeout_secs, 30);
    assert_eq!(config.routing_strategy, "resource_aware");

    // Missing file falls back to defaults.
    let empty = tempfile::tempdir().unwrap();
    let defaults = RuntimeConfig::load(empty.path());
    assert_eq!(defaults.port, 8080);

    // Invalid JSON falls back to defaults rather than crashing.
    write(&dir, "runtime.json", "{ not json");
    let fallback = RuntimeConfig::load(dir.path());
    assert_eq!(fallback.port, 8080);

    // Structurally valid but out-of-range values fall back to defaults.
    let full = r#"{
        "host": "0.0.0.0", "port": 9090, "request_timeout_secs": 30,
        "max_retries": 1, "retry_backoff_ms": 50,
        "max_concurrent_requests": 4, "max_fallback_models": 1,
        "local_model_budget_mb": 1024, "routing_strategy": "rule_based",
        "eviction_policy": "lru", "fixed_model": null,
        "openai_base_url": "http://mock/v1", "openai_default_model": "m",
        "openai_api_key": null, "llamacpp_base_url": "http://mock:8081",
        "llamacpp_default_model": "l"
    }"#;
    write(&dir, "runtime.json", &full.replace("9090", "0"));
    let validated = RuntimeConfig::load(dir.path());
    assert_eq!(validated.port, 8080);
}

#[test]
fn env_overrides_apply() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();

    std::env::set_var("DLLM_PORT", "7070");
    std::env::set_var("OPENAI_API_KEY", "env-secret");
    let config = RuntimeConfig::load(dir.path());
    assert_eq!(config.port, 7070);
    assert_eq!(config.openai_api_key(), Some("env-secret".to_string()));

    std::env::remove_var("DLLM_PORT");
    std::env::remove_var("OPENAI_API_KEY");
    let clean = RuntimeConfig::load(dir.path());
    assert_eq!(clean.port, 8080);
    assert_eq!(clean.openai_api_key(), None);
}

#[test]
fn managed_llamacpp_config_loading() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::remove_var("LLAMACPP_BINARY");
    std::env::remove_var("LLAMACPP_CTX_SIZE");

    let dir = tempfile::tempdir().unwrap();
    write(
        &dir,
        "runtime.json",
        r#"{
            "host": "127.0.0.1", "port": 8080, "request_timeout_secs": 120,
            "max_retries": 3, "retry_backoff_ms": 200,
            "max_concurrent_requests": 16, "max_fallback_models": 2,
            "local_model_budget_mb": 8192, "routing_strategy": "rule_based",
            "eviction_policy": "lru", "fixed_model": null,
            "openai_base_url": "https://api.openai.com/v1",
            "openai_default_model": "gpt-3.5-turbo", "openai_api_key": null,
            "llamacpp_base_url": "http://127.0.0.1:8081",
            "llamacpp_default_model": "llama-3.2-3b",
            "llamacpp_binary": "E:/tools/llama-server.exe",
            "llamacpp_ctx_size": 8192,
            "llamacpp_extra_args": ["--n-gpu-layers", "0"]
        }"#,
    );
    write(
        &dir,
        "models.json",
        r#"{"models": [{
            "id": "big-local", "name": "Big Local", "backend": "llama.cpp",
            "capabilities": {"supports_tools": false, "supports_streaming": true,
                "supports_structured_output": false, "task_affinity": ["general"]},
            "context_limit": 8192, "memory_required_mb": 4096,
            "quantization": "GGUF", "enabled": true,
            "gguf_path": "E:/models/large.gguf", "local_port": 8091
        }]}"#,
    );
    let config = RuntimeConfig::load(dir.path());
    assert_eq!(config.llamacpp_binary, "E:/tools/llama-server.exe");
    assert_eq!(config.llamacpp_ctx_size, 8192);
    assert_eq!(config.llamacpp_extra_args, vec!["--n-gpu-layers", "0"]);
    let file = ModelsFile::load(dir.path()).unwrap().expect("file present");
    assert_eq!(
        file.models[0].gguf_path.as_deref(),
        Some("E:/models/large.gguf")
    );
    assert_eq!(file.models[0].local_port, Some(8091));

    // Env overrides win over the file.
    std::env::set_var("LLAMACPP_BINARY", "/usr/local/bin/llama-server");
    std::env::set_var("LLAMACPP_CTX_SIZE", "2048");
    let overridden = RuntimeConfig::load(dir.path());
    assert_eq!(overridden.llamacpp_binary, "/usr/local/bin/llama-server");
    assert_eq!(overridden.llamacpp_ctx_size, 2048);
    std::env::remove_var("LLAMACPP_BINARY");
    std::env::remove_var("LLAMACPP_CTX_SIZE");

    // Files written before managed mode existed still parse.
    let legacy = tempfile::tempdir().unwrap();
    write(
        &legacy,
        "models.json",
        r#"{"models": [{
            "id": "m1", "name": "M1", "backend": "llama.cpp",
            "capabilities": {"supports_tools": false, "supports_streaming": true,
                "supports_structured_output": false, "task_affinity": ["qa"]},
            "context_limit": 1024, "memory_required_mb": 2048,
            "quantization": "GGUF", "enabled": true
        }]}"#,
    );
    let file = ModelsFile::load(legacy.path())
        .unwrap()
        .expect("file present");
    assert_eq!(file.models[0].gguf_path, None);
    assert_eq!(file.models[0].local_port, None);
    let defaults = RuntimeConfig::load(legacy.path());
    assert_eq!(defaults.llamacpp_binary, "llama-server");
    assert_eq!(defaults.llamacpp_ctx_size, 4096);
    assert!(defaults.llamacpp_extra_args.is_empty());
}

#[test]
fn models_file_loading() {
    let dir = tempfile::tempdir().unwrap();
    write(
        &dir,
        "models.json",
        r#"{"models": [{
            "id": "m1", "name": "M1", "backend": "openai",
            "capabilities": {"supports_tools": true, "supports_streaming": true,
                "supports_structured_output": false, "task_affinity": ["qa"]},
            "context_limit": 1024, "memory_required_mb": 0,
            "quantization": null, "enabled": true
        }]}"#,
    );
    let file = ModelsFile::load(dir.path()).unwrap().expect("file present");
    assert_eq!(file.models.len(), 1);
    assert_eq!(file.models[0].id, "m1");

    let empty = tempfile::tempdir().unwrap();
    assert!(ModelsFile::load(empty.path()).unwrap().is_none());

    write(&dir, "models.json", "{ broken");
    assert!(ModelsFile::load(dir.path()).is_err());
}
