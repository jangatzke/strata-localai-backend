"""Live LocalAI SSE smoke test: one structured tool call, no raw tool markup."""
import json
import os
import urllib.request

base_url = os.environ.get('LOCALAI_BASE_URL', 'http://127.0.0.1:8081/v1').rstrip('/')
model = os.environ.get('LOCALAI_MODEL', 'qwen3.8-flash-next-strata')
request_body = {
    'model': model,
    'stream': True,
    'max_tokens': 180,
    'messages': [
        {'role': 'system', 'content': 'You are Zoo. Use one tool to record the plan.'},
        {'role': 'user', 'content': 'Call update_todo_list with todos "[ ] Build frontend".'},
    ],
    'tools': [{'type': 'function', 'function': {
        'name': 'update_todo_list', 'description': 'Update the todo checklist',
        'parameters': {'type': 'object', 'properties': {
            'todos': {'type': 'string', 'description': 'Full markdown checklist'}},
            'required': ['todos']},
    }}],
}
req = urllib.request.Request(base_url + '/chat/completions',
                             json.dumps(request_body).encode(),
                             {'Content-Type': 'application/json'})
content, calls, finishes = [], [], []
with urllib.request.urlopen(req, timeout=420) as response:
    for line in response:
        if not line.startswith(b'data: '):
            continue
        data = line[len(b'data: '):].strip()
        if data == b'[DONE]':
            break
        event = json.loads(data)
        if 'error' in event:
            raise RuntimeError(event['error'])
        for choice in event.get('choices', []):
            delta = choice.get('delta', {})
            content.append(delta.get('content') or '')
            calls.extend(delta.get('tool_calls') or [])
            if choice.get('finish_reason'):
                finishes.append(choice['finish_reason'])
text = ''.join(content)
assert not any(tag in text for tag in ('<tool_call>', '<function=', '<parameter=')), text
assert calls and 'tool_calls' in finishes, (calls, finishes)
assert any((call.get('function') or {}).get('name') == 'update_todo_list' for call in calls)
print('PASS: structured update_todo_list tool call; no raw markup in SSE content')
