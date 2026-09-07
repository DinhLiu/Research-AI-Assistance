"""UI boundaries and four-stage orchestration without live provider calls."""
import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from research_assistant.ui.settings import load_settings, save_settings, schema, validate
from research_assistant.ui.server import Application, make_handler
from research_assistant.ui.worker import run_pipeline


def defaults(tmp_path):
    env, configs = load_settings(tmp_path / '.env')
    return env, configs


def test_env_roundtrip_preserves_unrelated_values_and_hides_secrets(tmp_path):
    path = tmp_path / '.env'
    path.write_text('# Custom settings\nUNRELATED=keep-me\n')
    app = Application(tmp_path)
    env, configs = defaults(tmp_path)
    env['GEMINI_API_KEY'] = 'test-secret-not-real'
    configs['retrieval']['year_from'] = '2020'
    configs['retrieval']['categories'] = 'cs.AI, cs.CL'
    save_settings(env, configs, path)
    loaded_env, loaded_configs = load_settings(path)
    _, parsed = validate(loaded_env, loaded_configs)
    assert parsed['retrieval'].year_from == 2020
    assert parsed['retrieval'].categories == ('cs.AI', 'cs.CL')
    assert 'UNRELATED=keep-me' in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert 'test-secret' not in json.dumps(app.settings())
    merged, _ = app.merge({'env': {'GEMINI_API_KEY': ''}})
    assert merged['GEMINI_API_KEY'] == 'test-secret-not-real'
    cleared, _ = app.merge({'clear_secrets': ['GEMINI_API_KEY']})
    assert cleared['GEMINI_API_KEY'] == ''


@pytest.mark.parametrize('stage,key,value', [
    ('retrieval', 'top_k', '0'), ('retrieval', 'year_from', 'n/a'),
    ('extraction', 'llm_timeout_s', 'nan'), ('extraction', 'llm_concurrency', '-1'),
    ('writing', 'language', 'xx'), ('retrieval', 'use_rerank', 'maybe'),
])
def test_invalid_configuration_rejected(tmp_path, stage, key, value):
    env, configs = defaults(tmp_path)
    configs[stage][key] = value
    with pytest.raises(ValueError):
        validate(env, configs)


def test_preflight_checks_artifacts_and_quotas(tmp_path):
    env, configs = defaults(tmp_path)
    configs['retrieval']['artifacts_dir'] = str(tmp_path)
    with pytest.raises(ValueError, match='SPECTER2'):
        validate(env, configs, preflight=True)
    (tmp_path / 'manifest.json').write_text('{}')
    env.update(GEMINI_API_KEY='test-key', LLM_RPM='', LLM_TPM='', LLM_RPD='')
    with pytest.raises(ValueError, match='LLM_RPM'):
        validate(env, configs, preflight=True)


class Result(SimpleNamespace):
    def model_dump_json(self, **kwargs):
        return '{}'


def test_pipeline_transfers_outputs_and_keeps_partial_review(tmp_path):
    env, raw = defaults(tmp_path)
    _, configs = validate(env, raw)
    retrieved = Result(papers=[1])
    extracted = Result(records=[Result(status='ok')])
    synthesized = Result(assignments=[1], paper_manifest=[1], execution=Result(summary_status='complete'))
    written = Result(generation_status='partial', execution=Result(stop_reason='budget_exhausted'))
    calls = []
    def call(expected, result):
        def run(input, config):
            assert input is expected
            calls.append(input)
            return result
        return run
    topic = 'A research topic'
    services = (call(topic, retrieved), call(retrieved, extracted), call(extracted, synthesized),
                call(synthesized, written), lambda result: '# Review')
    states = []
    run_pipeline(topic, configs, tmp_path, states.append, services)
    assert len(calls) == 4
    assert states[-1]['status'] == 'completed_with_warnings'
    assert states[-1]['stages'] == ['complete'] * 4
    assert len(states[-1]['files']) == 5
    assert (tmp_path / 'review.md').read_text() == '# Review'


def test_pipeline_stops_after_empty_retrieval(tmp_path):
    env, raw = defaults(tmp_path)
    _, configs = validate(env, raw)
    def unexpected(*args):
        pytest.fail('Later stages must not run after an empty retrieval')
    states = []
    run_pipeline('topic', configs, tmp_path, states.append,
                 (lambda *args: Result(papers=[]), unexpected, unexpected, unexpected, unexpected))
    assert states[-1]['status'] == 'failed'
    assert states[-1]['stages'] == ['failed', 'skipped', 'skipped', 'skipped']


def test_http_access_guard_and_static_ui(tmp_path):
    app = Application(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url) as response:
            page = response.read()
            assert 'Bắt đầu một nghiên cứu'.encode() in page
            assert b'data-language="vi"' in page
            assert b'data-language="en"' in page
        with urlopen(url + '/app.js') as response:
            javascript = response.read()
            assert b'Start a research project.' in javascript
            assert b'research-assistant-language' in javascript
        with urlopen(url + '/i18n.css') as response:
            assert b'.language-switch' in response.read()
        with urlopen(url + '/api/settings') as response:
            settings = json.load(response)
            assert len(settings['schema']) == 4
        for headers in ({'Origin': 'https://external.example'}, {'Host': 'evil.example'}):
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(url + '/api/settings', headers=headers))
            assert exc.value.code == 403
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(url + '/api/cancel', data=b'{}', method='POST'))
        assert exc.value.code == 403
        with urlopen(Request(url + '/api/cancel', data=b'{}', method='POST',
                            headers={'X-UI-Token': settings['token']})) as response:
            assert json.load(response)['ok']
        with pytest.raises(ValueError):
            app.directory('../../.env')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
