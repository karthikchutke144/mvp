"""Run and validate the participant solution.

This module is deliberately limited to submission checks.  Financial
reconstruction and recommendation logic remain in ``code/main.py``.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REQUIRED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def read_rows(path: Path) -> list[dict[str, str]]:
	with path.open(newline="", encoding="utf-8") as handle:
		return list(csv.DictReader(handle))


def validate_output(output_path: Path, requests_path: Path) -> list[str]:
	rows = read_rows(output_path)
	requests = {row["request_id"]: row for row in read_rows(requests_path)}
	errors: list[str] = []

	with output_path.open(newline="", encoding="utf-8") as handle:
		if handle.readline().strip().split(",") != REQUIRED_COLUMNS:
			errors.append("output.csv has incorrect columns")

	if len(rows) != len(requests):
		errors.append(f"expected {len(requests)} output rows, found {len(rows)}")

	seen: set[str] = set()
	for row in rows:
		request_id = row.get("request_id", "")
		request = requests.get(request_id)
		if request is None:
			errors.append(f"unknown request_id: {request_id}")
			continue
		seen.add(request_id)
		try:
			amount = float(row["amount_safe_to_pay"])
			requested = float(request["requested_amount"])
			if not 0 <= amount <= requested:
				errors.append(f"amount_safe_to_pay out of bounds for {request_id}")
		except (KeyError, ValueError):
			errors.append(f"invalid amount_safe_to_pay for {request_id}")
		if row.get("affordability_status") not in STATUSES:
			errors.append(f"invalid affordability_status for {request_id}")
		if row.get("recommended_payment_method") not in METHODS:
			errors.append(f"invalid recommended_payment_method for {request_id}")
		if row.get("recommended_payment_method") == "not_recommended" and row.get("payment_plan") != "none":
			errors.append(f"rejected request has a payment plan: {request_id}")

	missing = set(requests) - seen
	for request_id in sorted(missing):
		errors.append(f"missing output row for {request_id}")
	return errors


def main() -> int:
	repository_root = Path(__file__).resolve().parents[2]
	solution = repository_root / "code" / "main.py"
	dataset_dir = repository_root / "dataset"
	output = dataset_dir / "output.csv"
	requests = dataset_dir / "requests.csv"

	completed = subprocess.run([sys.executable, str(solution)], cwd=repository_root, check=False)
	if completed.returncode:
		return completed.returncode

	errors = validate_output(output, requests)
	if errors:
		for error in errors:
			print(f"ERROR: {error}")
		return 1

	print(f"Evaluation passed: {len(read_rows(output))} predictions validated.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
