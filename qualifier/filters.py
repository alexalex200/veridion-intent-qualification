"""Hard gates: cheap, deterministic pass/fail checks applied before any
scoring. A company is only rejected when a field is present AND clearly
violates the constraint - missing data never causes a rejection, it's
recorded as "unverified" so the caller can see reduced confidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Company
from .query_parser import ParsedQuery


@dataclass
class GateResult:
    passed: bool
    hard_fail_reasons: list = field(default_factory=list)
    unverified: list = field(default_factory=list)
    matched_region: bool = False


def evaluate_gates(company: Company, parsed: ParsedQuery) -> GateResult:
    fails: list = []
    unverified: list = []
    matched_region = False

    if parsed.region_specified:
        cc = company.country_code
        if cc is None:
            unverified.append("country unknown")
        elif cc in parsed.country_codes:
            matched_region = True
        else:
            fails.append(f"country '{cc}' not in requested region")

    if parsed.employee_min is not None or parsed.employee_max is not None:
        if company.employee_count is None:
            unverified.append("employee_count unknown")
        else:
            if parsed.employee_min is not None and company.employee_count < parsed.employee_min:
                fails.append(
                    f"employee_count {company.employee_count} < min {parsed.employee_min}"
                )
            if parsed.employee_max is not None and company.employee_count > parsed.employee_max:
                fails.append(
                    f"employee_count {company.employee_count} > max {parsed.employee_max}"
                )

    if parsed.revenue_min is not None or parsed.revenue_max is not None:
        if company.revenue is None:
            unverified.append("revenue unknown")
        else:
            if parsed.revenue_min is not None and company.revenue < parsed.revenue_min:
                fails.append(f"revenue {company.revenue} < min {parsed.revenue_min}")
            if parsed.revenue_max is not None and company.revenue > parsed.revenue_max:
                fails.append(f"revenue {company.revenue} > max {parsed.revenue_max}")

    if parsed.founded_min is not None or parsed.founded_max is not None:
        if company.year_founded is None:
            unverified.append("year_founded unknown")
        else:
            if parsed.founded_min is not None and company.year_founded < parsed.founded_min:
                fails.append(f"year_founded {company.year_founded} < min {parsed.founded_min}")
            if parsed.founded_max is not None and company.year_founded > parsed.founded_max:
                fails.append(f"year_founded {company.year_founded} > max {parsed.founded_max}")

    if parsed.is_public is not None:
        if company.is_public is None:
            unverified.append("is_public unknown")
        elif company.is_public != parsed.is_public:
            fails.append(f"is_public {company.is_public} != required {parsed.is_public}")

    return GateResult(
        passed=len(fails) == 0,
        hard_fail_reasons=fails,
        unverified=unverified,
        matched_region=matched_region,
    )
