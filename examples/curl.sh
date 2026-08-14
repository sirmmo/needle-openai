#!/usr/bin/env bash
# Every endpoint, over plain curl.
set -euo pipefail

BASE="${NEEDLE_BASE_URL:-http://127.0.0.1:8000}"
AUTH=()
if [[ -n "${NEEDLE_API_KEY:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${NEEDLE_API_KEY}")
fi

say() { printf '\n=== %s ===\n' "$1"; }

say "health"
curl -s "${BASE}/health" | python3 -m json.tool

say "models"
curl -s "${AUTH[@]}" "${BASE}/v1/models" | python3 -m json.tool

say "chat completion with a tool"
curl -s "${AUTH[@]}" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "needle-2",
    "messages": [{"role": "user", "content": "what is it like in Lagos right now?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }' | python3 -m json.tool

say "structured output via response_format"
curl -s "${AUTH[@]}" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "needle-2",
    "messages": [{"role": "user", "content": "Invoice from Acme Corp, $1,200.00, due 2026-09-01"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "Invoice",
        "schema": {
          "type": "object",
          "properties": {
            "vendor": {"type": "string"},
            "total": {"type": "number"},
            "due_date": {"type": "string"}
          },
          "required": ["vendor", "total", "due_date"]
        }
      }
    }
  }' | python3 -m json.tool

say "streaming (raw SSE)"
curl -sN "${AUTH[@]}" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "needle-2",
    "stream": true,
    "messages": [{"role": "user", "content": "weather in Tokyo?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
  }'

say "needle-native extraction"
curl -s "${AUTH[@]}" "${BASE}/v1/needle/extract" \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Invoice from Acme Corp, $1,200.00, due 2026-09-01",
    "name": "Invoice",
    "schema": {
      "type": "object",
      "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "due_date": {"type": "string"}
      },
      "required": ["vendor", "total", "due_date"]
    }
  }' | python3 -m json.tool

say "needle-native passthrough (raw engine response)"
curl -s "${AUTH[@]}" "${BASE}/v1/needle/complete" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "what is it like in Lagos right now?",
    "tools": [{"name": "get_weather", "description": "Get the current weather for a city.",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
      "required": ["city"]}}]
  }' | python3 -m json.tool
