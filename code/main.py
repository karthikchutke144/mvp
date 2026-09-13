"""Generate deterministic Buy or Wait? recommendations.

The program intentionally uses only the participant-facing CSV files.  It can
be run from any working directory with ``python code/main.py``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
	"request_id",
	"amount_safe_to_pay",
	"affordability_status",
	"recommended_payment_method",
	"payment_plan",
	"earliest_date_for_full_payment",
	"spending_changes_needed",
	"decision_explanation",
]
ACTIVE_STATUSES = {"settled", "pending", "scheduled"}
IGNORED_STATUSES = {"failed", "cancelled", "unrealized"}
STATUS_PRIORITY = {"cancelled": 5, "settled": 4, "pending": 3, "scheduled": 2, "estimated": 1}
CURRENCY_PATTERN = r"(?:INR|IDR|ZAR|USD|EUR)\s*([\d,]+(?:\.\d+)?)"
DATE_PATTERN = r"(\d{4}-\d{2}-\d{2})"


def split_values(value: object) -> set[str]:
	if pd.isna(value):
		return set()
	return {part.strip() for part in str(value).split("|") if part.strip()}


def number(value: object, default: float = 0.0) -> float:
	if pd.isna(value) or value == "":
		return default
	return float(value)


def money(value: float) -> str:
	rounded = round(max(0.0, value), 2)
	return f"{rounded:.2f}".rstrip("0").rstrip(".")


def date_text(value: date) -> str:
	return value.isoformat()


@dataclass(frozen=True)
class Profile:
	currency: str
	balance: float
	minimum: float
	methods: set[str]
	max_installment_months: int | None


class DecisionEngine:
	def __init__(self, dataset_dir: Path) -> None:
		self.dataset_dir = dataset_dir
		self.profiles = pd.read_csv(dataset_dir / "financial_profiles.csv").set_index("user_id")
		self.messages = pd.read_csv(dataset_dir / "messages.csv").fillna("")
		self.images = pd.read_csv(dataset_dir / "images.csv").fillna("")
		self.rates = pd.read_csv(dataset_dir / "exchange_rates.csv")
		self.events = pd.read_csv(dataset_dir / "financial_events.csv").drop_duplicates("event_id")
		self.requests = pd.read_csv(dataset_dir / "requests.csv")
		self.options = pd.read_csv(dataset_dir / "request_payment_options.csv")
		self.events = self.resolve_event_evidence(self.events)
		self.events["event_date"] = pd.to_datetime(self.events["event_date"]).dt.date
		self.events["settlement_date"] = pd.to_datetime(
			self.events["settlement_date"], errors="coerce"
		).dt.date
		self.events = self.deduplicate_events(self.events)

	def resolve_event_evidence(self, events: pd.DataFrame) -> pd.DataFrame:
		"""Resolve image amounts and apply only explicit message amendments."""
		events = events.copy()
		missing = events["amount"].isna() | events["amount"].eq("")
		if missing.any():
			try:
				from image_extraction import resolve_missing_amounts
				events = resolve_missing_amounts(
					events,
					self.images,
					str(self.dataset_dir / "media" / "images"),
				)
			except (ImportError, OSError, RuntimeError, ValueError):
				# An unreadable image must not become a fabricated zero amount.
				pass
		events.loc[events["amount"].eq(""), "amount"] = float("nan")

		for index, event in events.iterrows():
			related = self.messages[self.messages["related_event_id"].eq(event["event_id"])]
			for _, message in related.sort_values("sent_at").iterrows():
				text = str(message["message_text"])
				if re.search(r"\b(cancelled|canceled|reversed|voided)\b", text, re.I):
					events.at[index, "status"] = "cancelled"
					break
				amount_match = re.search(CURRENCY_PATTERN, text, re.I)
				if amount_match and re.search(r"\b(amended|updated|changed|now)\b", text, re.I):
					events.at[index, "amount"] = float(amount_match.group(1).replace(",", ""))
		return events

	def deduplicate_events(self, events: pd.DataFrame) -> pd.DataFrame:
		"""Keep the strongest record for each transaction representation."""
		events = events.copy()
		events["_status_priority"] = events["status"].map(STATUS_PRIORITY).fillna(0)
		events["_transaction_key"] = events.apply(
			lambda row: row["linked_event_id"]
			if not pd.isna(row["linked_event_id"]) and str(row["linked_event_id"]).strip()
			else "|".join(str(row[field]) for field in (
				"user_id", "direction", "amount", "currency", "event_date", "settlement_date", "category"
			)),
			axis=1,
		)
		events = events.sort_values(["_transaction_key", "_status_priority", "event_id"])
		return events.drop_duplicates("_transaction_key", keep="last").drop(
			columns=["_status_priority", "_transaction_key"], errors="ignore"
		)

	def convert_to_home(self, amount: float, currency: str, home_currency: str, event_date: date) -> float:
		if currency == home_currency:
			return amount
		day_rates = self.rates[self.rates["rate_date"].eq(event_date.isoformat())]
		for _, rate in day_rates.iterrows():
			if rate["from_currency"] == currency and rate["to_currency"] == home_currency:
				return amount * number(rate["rate"])
			if rate["from_currency"] == home_currency and rate["to_currency"] == currency:
				return amount / number(rate["rate"])
		# Use a supplied two-leg route when no direct pair exists on that date.
		for _, first in day_rates[day_rates["from_currency"].eq(currency)].iterrows():
			for _, second in day_rates[day_rates["from_currency"].eq(first["to_currency"])].iterrows():
				if second["to_currency"] == home_currency:
					return amount * number(first["rate"]) * number(second["rate"])
		return amount

	def confirmed_message_movements(self, user_id: str, home_currency: str, start: date, end: date) -> dict[date, float]:
		"""Add only explicitly confirmed, dated salary credits from messages."""
		movements: dict[date, float] = {}
		messages = self.messages[self.messages["user_id"].eq(user_id)]
		for _, message in messages.iterrows():
			text = str(message["message_text"])
			if not re.search(r"\b(confirmed salary|salary .*confirmed|confirmed credit)\b", text, re.I):
				continue
			date_match = re.search(DATE_PATTERN, text)
			amount_match = re.search(CURRENCY_PATTERN, text, re.I)
			if not date_match or not amount_match:
				continue
			credit_date = date.fromisoformat(date_match.group(1))
			if not start <= credit_date <= end:
				continue
			currency_match = re.search(r"(INR|IDR|ZAR|USD|EUR)", amount_match.group(0), re.I)
			currency = currency_match.group(1).upper()
			converted = self.convert_to_home(float(amount_match.group(1).replace(",", "")), currency, home_currency, credit_date)
			movements[credit_date] = movements.get(credit_date, 0.0) + converted
		return movements

	def profile_for(self, user_id: str) -> Profile:
		row = self.profiles.loc[user_id]
		months = None if pd.isna(row["max_installment_months"]) else int(row["max_installment_months"])
		return Profile(
			currency=str(row["home_currency"]),
			balance=number(row["current_available_balance"]),
			minimum=number(row["minimum_balance_to_keep"]),
			methods=split_values(row["payment_methods_user_will_consider"]),
			max_installment_months=months,
		)

	def user_events(self, user_id: str) -> pd.DataFrame:
		events = self.events[self.events["user_id"].eq(user_id)].copy()
		return events[~events["status"].isin(IGNORED_STATUSES)]

	def forecast(self, user_id: str, start: date) -> dict[date, float]:
		"""Return net known cash movement by date for the next 90 days."""
		end = start + timedelta(days=90)
		events = self.user_events(user_id)
		profile = self.profile_for(user_id)
		movements: dict[date, float] = self.confirmed_message_movements(user_id, profile.currency, start, end)
		for _, event in events.iterrows():
			event_date = event["settlement_date"] or event["event_date"]
			if event_date is None or not start <= event_date <= end:
				continue
			# Pending credits are not available; pending debits are reserved.
			if event["status"] == "pending" and event["direction"] == "credit":
				continue
			amount = number(event["amount"])
			if not amount:
				continue
			amount = self.convert_to_home(amount, str(event["currency"]), profile.currency, event_date)
			signed_amount = amount if event["direction"] == "credit" else -amount
			movements[event_date] = movements.get(event_date, 0.0) + signed_amount
		return movements

	def balance_path(self, profile: Profile, start: date, movements: dict[date, float]) -> dict[date, float]:
		balance = profile.balance
		path: dict[date, float] = {}
		for offset in range(91):
			current = start + timedelta(days=offset)
			balance += movements.get(current, 0.0) if offset else 0.0
			path[current] = balance
		return path

	def safe_capacity(self, profile: Profile, start: date, movements: dict[date, float], payment_date: date) -> float:
		path = self.balance_path(profile, start, movements)
		end = start + timedelta(days=90)
		available = math.inf
		for current in path:
			if current >= payment_date:
				available = min(available, path[current] - profile.minimum)
		return max(0.0, available)

	def plan_is_safe(
		self,
		profile: Profile,
		start: date,
		movements: dict[date, float],
		payments: list[tuple[date, float]],
	) -> bool:
		path = self.balance_path(profile, start, movements)
		for payment_date, amount in payments:
			if payment_date not in path or payment_date < start:
				return False
			for current in path:
				if current >= payment_date:
					path[current] -= amount
		return min(path.values()) >= profile.minimum - 0.01

	def formatted_plan(self, payments: list[tuple[date, float]]) -> str:
		return "|".join(f"{date_text(day)}:{money(amount)}" for day, amount in payments)

	def installment_plan(
		self,
		request: pd.Series,
		profile: Profile,
		movements: dict[date, float],
	) -> tuple[str, list[tuple[date, float]]] | None:
		if "installments" not in profile.methods:
			return None
		options = self.options[self.options["request_id"].eq(request["request_id"])]
		candidates: list[tuple[float, date, int, str, list[tuple[date, float]]]] = []
		for _, option in options.iterrows():
			count = int(number(option["number_of_payments"]))
			if option["payment_method"] != "installments" or count < 1:
				continue
			if profile.max_installment_months is not None and count > profile.max_installment_months:
				continue
			first = pd.to_datetime(option["first_payment_date"]).date()
			frequency = int(number(option["payment_frequency_days"], 0))
			payments = [
				(first + timedelta(days=frequency * index), number(option["payment_amount"]))
				for index in range(count)
			]
			if payments[-1][0] > pd.to_datetime(request["desired_completion_date"]).date():
				continue
			request_date = pd.to_datetime(request["request_date"]).date()
			if self.plan_is_safe(profile, request_date, movements, payments):
				candidates.append((
					number(option["total_payable_amount"]),
					payments[0][0],
					count,
					str(option["payment_option_id"]),
					payments,
				))
		if not candidates:
			return None
		_, _, _, _, payments = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
		return self.formatted_plan(payments), payments

	def full_payment_option(self, request: pd.Series, profile: Profile) -> tuple[date, float] | None:
		if "full_payment" not in profile.methods:
			return None
		options = self.options[
			self.options["request_id"].eq(request["request_id"])
			& self.options["payment_method"].eq("full_payment")
		]
		if options.empty:
			return None
		requested = number(request["requested_amount"])
		options = options[options["payment_amount"].map(number).sub(requested).abs() <= 0.01]
		if options.empty:
			return None
		option = options.sort_values("payment_option_id").iloc[0]
		return pd.to_datetime(option["first_payment_date"]).date(), number(option["payment_amount"])

	def decide(self, request: pd.Series) -> dict[str, object]:
		user_id = str(request["user_id"])
		profile = self.profile_for(user_id)
		start = pd.to_datetime(request["request_date"]).date()
		deadline = pd.to_datetime(request["desired_completion_date"]).date()
		amount = number(request["requested_amount"])
		movements = self.forecast(user_id, start)
		safe_today = min(amount, self.safe_capacity(profile, start, movements, start))
		safe_today = round(max(0.0, safe_today), 2)

		earliest: date | None = None
		for offset in range(91):
			candidate = start + timedelta(days=offset)
			if candidate <= deadline and self.safe_capacity(profile, start, movements, candidate) >= amount - 0.01:
				earliest = candidate
				break

		full_option = self.full_payment_option(request, profile)
		if full_option and full_option[0] == start:
			full_payments = [full_option]
			if self.plan_is_safe(profile, start, movements, full_payments):
				return self.result(request, safe_today, "affordable_now", "full_payment", full_payments, start, profile)

		installment = self.installment_plan(request, profile, movements)
		if installment:
			plan, payments = installment
			return self.result(request, safe_today, "affordable_with_plan", "installments", payments, earliest, profile)

		if (
			bool(request["allows_partial_payment"])
			and "partial_payment" in profile.methods
			and 0 < safe_today < amount
			and earliest is not None
		):
			payments = [(start, safe_today), (earliest, round(amount - safe_today, 2))]
			if self.plan_is_safe(profile, start, movements, payments):
				return self.result(request, safe_today, "affordable_with_plan", "partial_payment", payments, earliest, profile)

		if earliest is not None and "full_payment" in profile.methods:
			payments = [(earliest, amount)]
			return self.result(request, safe_today, "affordable_later", "wait", payments, earliest, profile)

		return self.result(request, safe_today, "not_affordable", "not_recommended", [], None, profile)

	def result(
		self,
		request: pd.Series,
		safe_today: float,
		status: str,
		method: str,
		payments: list[tuple[date, float]],
		earliest: date | None,
		profile: Profile,
	) -> dict[str, object]:
		currency = profile.currency
		if method == "not_recommended":
			explanation = f"Do not make this payment by {request['desired_completion_date']}; the {currency} {money(profile.minimum)} minimum is not protected."
		elif method == "wait":
			explanation = f"Wait until {date_text(payments[0][0])}, when the full {currency} {money(number(request['requested_amount']))} payment is safe."
		elif method == "installments":
			explanation = f"Use {len(payments)} installments of {currency} {money(payments[0][1])}; the minimum balance remains protected."
		elif method == "partial_payment":
			explanation = f"Pay {currency} {money(payments[0][1])} now and the remainder on {date_text(payments[1][0])}."
		else:
			explanation = f"Pay {currency} {money(number(request['requested_amount']))} today while keeping the minimum balance protected."
		return {
			"request_id": request["request_id"],
			"amount_safe_to_pay": safe_today,
			"affordability_status": status,
			"recommended_payment_method": method,
			"payment_plan": self.formatted_plan(payments) if payments else "none",
			"earliest_date_for_full_payment": date_text(earliest) if earliest else "",
			"spending_changes_needed": "none",
			"decision_explanation": explanation,
		}

	def run(self, output_path: Path) -> None:
		results = [self.decide(request) for _, request in self.requests.iterrows()]
		output = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
		output.to_csv(output_path, index=False)


def main() -> None:
	repository_root = Path(__file__).resolve().parents[1]
	dataset_dir = repository_root / "dataset"
	DecisionEngine(dataset_dir).run(dataset_dir / "output.csv")


if __name__ == "__main__":
	main()
