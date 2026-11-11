#!/usr/bin/env python3
"""E2 — measure the prompt-token cost of "format in prose" vs "format in grammar".

Counts real tokens with the actual Qwen2.5-Coder tokenizer (tokenizer.json),
not an estimate. Reproduce:

    curl -sSL -o /tmp/opencode/qwen_tok.json \
      https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/resolve/main/tokenizer.json
    .venv/bin/python code/e2_token_measure.py /tmp/opencode/qwen_tok.json

Why this matters: every token of "please respond in JSON with these fields..."
plus every few-shot example is billed on EVERY call. Constrained decoding
(Outlines / json_schema response_format / GBNF) moves that specification out of
the prompt and into the decoder, where it costs zero prompt tokens.
"""
import json
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/qwen_tok.json")


def n(s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False).ids)


# ---------------------------------------------------------------- case 1: extraction
# A. What a typical "prompt engineered" app sends: role + format spec + 2 examples.
A_SYSTEM = """You are a document extraction engine for an accounts-payable system.

Extract the invoice fields from the user's text and return them as a single JSON
object. Follow these rules exactly:

1. Respond with JSON only. Do not add any prose, explanation, or markdown fences.
2. The JSON object must contain exactly these keys, in this order:
   - "invoice_number": string, the invoice identifier, digits and dashes only
   - "vendor": string, the supplier's legal name
   - "invoice_date": string, ISO-8601 date (YYYY-MM-DD)
   - "currency": string, three-letter ISO-4217 code
   - "line_items": array of objects, each with "description" (string),
     "quantity" (number), "unit_price" (number), "amount" (number)
   - "subtotal": number, sum of line item amounts
   - "tax": number, total tax, 0 if absent
   - "total": number, the amount due
3. If a field is missing from the text, use null. Never invent values.
4. Numbers must be JSON numbers, not strings. No thousands separators.
5. Dates must be normalised to ISO-8601 even if the source uses another format.

Example input:
Invoice #A-1042 from Acme Industrial Supply, dated 3 March 2024.
2 x Widget, 19.99 each. 1 x Freight, 45.00. Tax 8.60. Total due 93.58 USD.

Example output:
{"invoice_number":"A-1042","vendor":"Acme Industrial Supply","invoice_date":"2024-03-03","currency":"USD","line_items":[{"description":"Widget","quantity":2,"unit_price":19.99,"amount":39.98},{"description":"Freight","quantity":1,"unit_price":45.0,"amount":45.0}],"subtotal":84.98,"tax":8.6,"total":93.58}

Example input:
Bill 7781, Northwind Traders, 2024-11-02. Consulting 10 hrs @ 120.00. VAT 240.00.
Total 1440.00 EUR.

Example output:
{"invoice_number":"7781","vendor":"Northwind Traders","invoice_date":"2024-11-02","currency":"EUR","line_items":[{"description":"Consulting","quantity":10,"unit_price":120.0,"amount":1200.0}],"subtotal":1200.0,"tax":240.0,"total":1440.0}
"""

# B. Same task with the schema enforced by the decoder (Outlines / json_schema).
#    The schema is sent once as a machine-readable definition, not as prose rules,
#    and no few-shot examples are needed because invalid output is unrepresentable.
B_SYSTEM = "Extract the invoice fields."
B_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "invoice_number": {"type": ["string", "null"]},
        "vendor": {"type": ["string", "null"]},
        "invoice_date": {"type": ["string", "null"], "format": "date"},
        "currency": {"type": ["string", "null"]},
        "line_items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "quantity": {"type": "number"},
                "unit_price": {"type": "number"},
                "amount": {"type": "number"},
            },
            "required": ["description", "quantity", "unit_price", "amount"],
        }},
        "subtotal": {"type": ["number", "null"]},
        "tax": {"type": ["number", "null"]},
        "total": {"type": ["number", "null"]},
    },
    "required": ["invoice_number", "vendor", "invoice_date", "currency",
                 "line_items", "subtotal", "tax", "total"],
}, separators=(",", ":"))

USER = ("Invoice INV-2291 dated 14/02/2025 from Brightline Logistics Ltd. "
        "3 pallets handling @ 75.00, 1 express surcharge @ 120.00, VAT 20%. "
        "Total 390.00 GBP.")

# ---------------------------------------------------------------- case 2: classification
C_SYSTEM = """You are a support-ticket router. Classify the ticket into exactly one
of these categories and reply with the category name only, lowercase, no punctuation:

billing, refund, shipping, account_access, product_defect, feature_request,
cancellation, legal, spam, other

If the ticket mentions money owed or an invoice, choose billing. If the customer
asks for money back, choose refund even if they also mention billing. If the
ticket mentions a lawyer, chargeback threat, or GDPR, choose legal. If two
categories apply, prefer the more specific one, in this order:
legal > refund > billing > account_access > product_defect > shipping >
cancellation > feature_request > other.
"""

D_SYSTEM = "Route the ticket."  # enum enforced by grammar
LABELS = ["billing", "refund", "shipping", "account_access", "product_defect",
          "feature_request", "cancellation", "legal", "spam", "other"]

# ---------------------------------------------------------------- case 3: tool defs
# A "tool definition" is billed as input tokens on every call in the OpenAI and
# Anthropic APIs. Ten verbose tools is a fixed per-call tax.
TOOL_TEMPLATE = {
    "name": "search_documents",
    "description": ("Search the internal document store and return the most relevant "
                    "passages. Use this when the user asks a question whose answer "
                    "might be in company policy, contracts, or the engineering wiki. "
                    "Do not use for general knowledge questions."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "top_k": {"type": "integer", "description": "Number of passages to return, 1-20."},
            "collection": {"type": "string", "enum": ["policy", "contracts", "wiki"]},
        },
        "required": ["query"],
    },
}

rows = []
a = n(A_SYSTEM) + n(USER)
b = n(B_SYSTEM) + n(B_SCHEMA) + n(USER)
rows.append(("extraction: prose spec + 2 few-shot", a, a))
rows.append(("extraction: schema-constrained", b, b))

c = n(C_SYSTEM) + n("My invoice was charged twice, I want my money back.")
d = n(D_SYSTEM) + n("My invoice was charged twice, I want my money back.")
rows.append(("classification: prose label rules", c, c))
rows.append(("classification: enum-constrained", d, d))

t = n(json.dumps(TOOL_TEMPLATE))
rows.append(("10 tool definitions (json, verbose)", 10 * t, 10 * t))

print(f"{'case':46s} {'tokens':>7s}  {'per 1M calls':>12s}")
for name, v, _ in rows:
    print(f"{name:46s} {v:7d}  {v:12,d}")

print()
print(f"extraction saving:     {a} -> {b} tokens = {100*(a-b)/a:.1f}% fewer input tokens")
print(f"classification saving: {c} -> {d} tokens = {100*(c-d)/c:.1f}% fewer input tokens")
print(f"one verbose tool def:  {t} tokens; 10 tools = {10*t} tokens/call")
print()
# Break-even arithmetic for prompt caching, using live Anthropic Sonnet 5.5 prices.
base_in, cache_write, cache_read = 2.00, 2.50, 0.10   # $/MTok, 5m TTL
prefix = 4000                                          # cached prefix tokens
saved_per_call = prefix * (base_in - cache_read) / 1e6
cost_to_write = prefix * (cache_write - base_in) / 1e6
print(f"caching a {prefix}-token prefix on Sonnet 5.5: "
      f"write costs +${cost_to_write:.5f}, each hit saves ${saved_per_call:.5f}, "
      f"break-even at {cost_to_write/saved_per_call:.2f} calls")
