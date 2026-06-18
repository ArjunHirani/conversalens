import sys
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.rag_engine import build_rag

def main():
    parser = argparse.ArgumentParser(description="Build RAG index from conversations CSV")
    parser.add_argument("--csv", default="data/conversations.csv")
    parser.add_argument("--out", default="data/rag_state.pkl")
    args = parser.parse_args()

    state = build_rag(args.csv, verbose=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(state, f)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
