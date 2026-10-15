import json
from pathlib import Path

from src.retrieval import Retriever


ROOT = Path(__file__).resolve().parent


def normalize(text):
    # Bỏ khoảng trắng thừa và chuyển thành chữ thường.
    return " ".join(text.lower().split())


def main():
    test_file = ROOT / "data" / "testset.json"
    questions = json.loads(test_file.read_text(encoding="utf-8"))

    retriever = Retriever()
    results = []

    for mode in ["dense", "hybrid"]:
        correct = 0
        total = 0

        print(f"\n--- {mode.upper()} ---")

        for item in questions:
            hits = retriever.search(item["question"], mode=mode, top_k=3)

            # Chỉ chấm tìm kiếm khi có đoạn bằng chứng.
            found = None
            if not item["no_answer"]:
                total += 1
                evidence = normalize(item["evidence"])
                found = any(
                    evidence in normalize(chunk.text)
                    for chunk, score in hits
                )
                if found:
                    correct += 1

                status = "TIM THAY" if found else "CHUA TIM THAY"
            else:
                status = "CHO KIEM TRA TU CHOI KHI GHEP AGENT"

            print(f"{item['id']}: {status}")

            results.append({
                "id": item["id"],
                "mode": mode,
                "question": item["question"],
                "no_answer": item["no_answer"],
                "evidence_found": found,
                "retrieved": [
                    {
                        "doc_id": chunk.source.doc_id,
                        "chunk_id": chunk.source.chunk_id,
                        "score": float(score),
                        "text": chunk.text
                    }
                    for chunk, score in hits
                ]
            })

        print(f"Tim thay bang chung trong top 3: {correct}/{total}")

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "retrieval_eval.json"
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("\nDa luu ket qua vao results/retrieval_eval.json")


if __name__ == "__main__":
    main()