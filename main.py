"""Command-line entry point for the Question Answering project."""
from __future__ import annotations

import argparse
import json
import sys

from qa_pipeline import QuestionAnsweringService
from utils import parse_batch_questions, read_context_file, save_results_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hugging Face question answering application")
    parser.add_argument("--question", help="A single question to answer")
    parser.add_argument("--batch-file", help="UTF-8 file containing one question per line")
    parser.add_argument("--context-file", default="sample_context.txt", help="Context text file")
    parser.add_argument("--model", help="Override the configured model name")
    parser.add_argument("--save-model", action="store_true", help="Save loaded pipeline artifacts")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        context = read_context_file(args.context_file)
        service = QuestionAnsweringService(args.model) if args.model else QuestionAnsweringService()
        if args.save_model:
            print(f"Saved model to: {service.save_model()}")
            return 0
        if args.batch_file:
            questions = parse_batch_questions(read_context_file(args.batch_file))
            results = service.answer_batch(questions, context)
        elif args.question:
            results = [service.answer(args.question, context)]
        else:
            raise ValueError("Pass --question or --batch-file, or run the Streamlit UI.")
        csv_path = save_results_csv(results)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"CSV saved to: {csv_path}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
