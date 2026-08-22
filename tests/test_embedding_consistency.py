"""
Guard against the vector-store embedding-model mismatch bug.

The index is built by build_vector_db.py and queried by tools.py::get_vector_store.
If they ever diverge in how they construct embeddings, dimension mismatches will
cause the vector search to silently return nothing. These tests pin both to the
single source of truth in src/config.py (build_embeddings) and reject hardcoded
model names in the pipeline modules.
"""
import inspect
import textwrap

from src import build_vector_db, config, tools


def _module_source(module):
    return textwrap.dedent(inspect.getsource(module))


def _unnamed_model_refs(source):
    """Lines that pass a model literal into an embedding builder (drift risk)."""
    bad = []
    for line in source.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith('"""') or s.startswith("from") or s.startswith("import"):
            continue
        if "model_name=" in s and "EMBEDDING_MODEL" not in s:
            bad.append(s)
        if "model=" in s and "OllamaEmbeddings" in s and "EMBEDDING_MODEL" not in s:
            bad.append(s)
    return bad


def test_shared_constant_is_not_empty_or_dangling():
    assert isinstance(config.EMBEDDING_MODEL, str)
    assert config.EMBEDDING_MODEL.strip() != ""
    assert config.EMBEDDING_PROVIDER in {"ollama", "huggingface"}


def test_tools_builds_embeddings_via_config():
    """tools must obtain its embedder from config, not re-declare a model."""
    assert "build_embeddings()" in _module_source(tools)


def test_build_vector_db_builds_embeddings_via_config():
    source = _module_source(build_vector_db)
    assert "build_embeddings()" in source


def test_no_hardcoded_model_names_in_pipeline():
    """Neither index nor query path may hardcode an embedding model name."""
    for name, mod in (("tools", tools), ("build_vector_db", build_vector_db)):
        bad = _unnamed_model_refs(_module_source(mod))
        assert not bad, f"{name}.py has hardcoded embedding model refs: {bad}"