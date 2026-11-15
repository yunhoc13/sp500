"""Which XBRL tags feed each financial item.

Companies name the same item differently, so each item lists tags in
priority order: the first tag a company reports for that fiscal year wins.
kind: "duration" = covers the whole fiscal year (income / cash flow items)
      "instant"  = value on the fiscal-year-end date (balance sheet items)
"""

CONCEPTS = {
    "revenue": ("duration", "USD", [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
    ]),
    "cogs": ("duration", "USD", [
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    ]),
    "gross_profit": ("duration", "USD", ["GrossProfit"]),
    "op_income": ("duration", "USD", ["OperatingIncomeLoss"]),
    "da": ("duration", "USD", [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ]),
    # fallback when no combined D&A tag exists: depreciation + amortization
    "depreciation": ("duration", "USD", ["Depreciation", "DepreciationNonproduction"]),
    "amortization": ("duration", "USD", ["AmortizationOfIntangibleAssets"]),
    # fallback for operating income: pre-tax income + interest expense (EBIT)
    "pretax_income": ("duration", "USD", [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ]),
    "interest_expense": ("duration", "USD", [
        "InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
        "InterestPaidNet",
    ]),
    "net_income": ("duration", "USD", ["NetIncomeLoss", "ProfitLoss"]),
    "eps": ("duration", "USD/shares", [
        "EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted",
        "EarningsPerShareBasic",
    ]),
    "shares": ("duration", "shares", [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ]),
    "equity": ("instant", "USD", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    "cash": ("instant", "USD", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ]),
    # Debt: a total if the company reports one, otherwise noncurrent + current.
    "debt_total": ("instant", "USD", [
        "LongTermDebt",
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    ]),
    "debt_noncurrent": ("instant", "USD", [
        "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
    ]),
    "debt_current": ("instant", "USD", [
        "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "DebtCurrent",
    ]),
    "short_term_borrowings": ("instant", "USD", [
        "ShortTermBorrowings", "CommercialPaper",
    ]),
}

ALL_TAGS = sorted({t for _, _, tags in CONCEPTS.values() for t in tags})
