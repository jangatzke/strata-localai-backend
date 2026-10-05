"""Five real LocalAI completions with API-only tool history. Never execute calls."""
import json
import os
import urllib.request
from pathlib import Path
def main():
    base = os.environ.get('LOCALAI_BASE_URL', 'http://127.0.0.1:8081/v1').rstrip('/')
    model = os.environ.get('LOCALAI_MODEL', 'qwen3.8-flash-next-strata')
    out = Path(os.environ['LIVE_HISTORY_RESULTS']) if os.environ.get('LIVE_HISTORY_RESULTS') else None
    tools = [{'type': 'function', 'function': {'name': 'write_file', 'description': 'API-only regression tool. The harness never executes returned calls.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']}}}]
    messages = []
    results = []
    for step in range(1, 6):
        expected = {'path': '/unused-api-only-step-' + str(step) + '.md', 'content': 'History step ' + str(step) + ': ä😀日本語.'}
        messages.append({'role': 'user', 'content': 'Return exactly one write_file call using these exact values for this NEW step: ' + json.dumps(expected, ensure_ascii=False)})
        body = {'model': model, 'stream': False, 'temperature': 0, 'max_tokens': 4096, 'tools': tools, 'messages': messages}
        req = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + os.environ['LOCALAI_API_KEY']} if os.environ.get('LOCALAI_API_KEY') else {})})
        with urllib.request.urlopen(req, timeout=360) as response:
            obj = json.load(response)
        choice = obj['choices'][0]
        message = choice['message']
        calls = message.get('tool_calls', [])
        exact = len(calls) == 1 and calls[0]['function']['name'] == 'write_file' and json.loads(calls[0]['function']['arguments']) == expected
        record = {'step': step, 'call_count': len(calls), 'arguments_exact': exact, 'finish_reason': choice.get('finish_reason'), 'tools_executed': 0}
        results.append(record)
        if out:
            out.write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
        assert exact and choice.get('finish_reason') == 'tool_calls', 'Multi-turn regression failed'
        messages.append({'role': 'assistant', 'content': message.get('content') or '', 'tool_calls': calls})
        messages.append({'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'API-only harness: this tool call was intentionally NOT executed. Continue with the next requested step.'})
    assert len(results) == 5 and all(r['arguments_exact'] for r in results)
    print('PASS: five consecutive exact API-only tool-history turns; no tools executed.')


if __name__ == "__main__":
    main()
