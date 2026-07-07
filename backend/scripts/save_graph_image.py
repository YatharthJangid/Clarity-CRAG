"""Saves a PNG diagram of the compiled CRAG LangGraph to assets/graph.png.

Building/compiling a StateGraph needs no API keys - only *invoking* nodes
does - so this can run anytime.

Usage: python backend/scripts/save_graph_image.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.crag.graph import build_uncompiled_graph


def main():
    compiled = build_uncompiled_graph().compile()
    settings.assets_dir.mkdir(exist_ok=True)
    out_path = settings.assets_dir / "graph.png"
    out_path.write_bytes(compiled.get_graph().draw_mermaid_png())
    print(f"Saved graph diagram to {out_path}")


if __name__ == "__main__":
    main()
