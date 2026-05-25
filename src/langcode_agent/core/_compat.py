"""Compatibility shims for locally installed LangChain/LangGraph packages."""


def patch_langchain_debug() -> None:
    try:
        import langchain  # type: ignore
    except Exception:
        return

    if not hasattr(langchain, "debug"):
        langchain.debug = False
    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
    if not hasattr(langchain, "llm_cache"):
        langchain.llm_cache = None
