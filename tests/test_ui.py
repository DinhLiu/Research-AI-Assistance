"""UI boundaries and four-stage orchestration without live provider calls."""
import json
import logging
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
from research_assistant.ui.worker import RedactingFormatter


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


def test_pipeline_transfers_outputs_and_keeps_partial_review(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger='research_assistant.pipeline')
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
    assert 'Stage 1/4 started | name=retrieval' in caplog.text
    assert 'Stage 4/4 completed | name=writing' in caplog.text


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


def test_log_tail_is_bounded_and_credentials_are_redacted(tmp_path):
    app = Application(tmp_path)
    run_id = 'a' * 32
    directory = app.runs / run_id
    directory.mkdir()
    (directory / 'run.log').write_text('old\n' + ('x' * 80) + '\nlatest\n')
    tail = app.log_tail(run_id, max_bytes=20)
    assert 'Earlier log entries omitted' in tail
    assert tail.endswith('latest\n')

    formatter = RedactingFormatter('%(levelname)s %(message)s', ['test-secret'])
    record = logging.LogRecord('test', logging.ERROR, __file__, 1,
                               'request failed: test-secret', (), None)
    rendered = formatter.format(record)
    assert 'test-secret' not in rendered
    assert '[REDACTED]' in rendered


def test_http_access_guard_and_static_ui(tmp_path):
    app = Application(tmp_path)
    run_id = 'b' * 32
    run_dir = app.runs / run_id
    run_dir.mkdir()
    (run_dir / 'run.log').write_text('INFO pipeline started\n')
    (run_dir / 'status.json').write_text(json.dumps({
        'status': 'completed', 'topic': 'logging test', 'stages': ['complete'] * 4,
        'summaries': [''] * 4, 'warnings': [], 'files': ['run.log'], 'error': None,
    }))
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url) as response:
            page = response.read()
            assert b'Start a research project.' in page
            assert b'data-language="vi"' in page
            assert b'data-language="en"' in page
        with urlopen(url + '/app.js') as response:
            javascript = response.read()
            assert b'research-assistant-language' in javascript
        with urlopen(url + '/i18n.js') as response:
            assert b'Start a research project.' in response.read()
        with urlopen(url + '/i18n.css') as response:
            assert b'.language-switch' in response.read()
        with urlopen(url + f'/api/runs/{run_id}/log') as response:
            assert response.read() == b'INFO pipeline started\n'
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


def test_stage_profiles_roundtrip_and_secret_masking(tmp_path):
    app = Application(tmp_path)
    env, configs = defaults(tmp_path)
    env.update(RA_RETRIEVAL_LLM_PROVIDER='groq',
               RA_RETRIEVAL_GROQ_API_KEY='private-stage-key',
               RA_RETRIEVAL_GROQ_MODEL='custom-small-model')
    save_settings(env, configs, tmp_path / '.env')
    assert 'private-stage-key' not in json.dumps(app.settings())
    assert app.settings()['secrets']['RA_RETRIEVAL_GROQ_API_KEY']
    merged, _ = app.merge({'env': {'RA_RETRIEVAL_GROQ_API_KEY': ''}})
    assert merged['RA_RETRIEVAL_GROQ_API_KEY'] == 'private-stage-key'
    assert merged['RA_RETRIEVAL_GROQ_MODEL'] == 'custom-small-model'
    merged, _ = app.merge({'clear_secrets': ['RA_RETRIEVAL_GROQ_API_KEY']})
    assert merged['RA_RETRIEVAL_GROQ_API_KEY'] == ''


def test_stage_profile_routes_client_and_restores_environment(monkeypatch):
    import os
    from research_assistant.ui.worker import stage_profile
    from research_assistant.llm.client import expansion_provider, extraction_provider, model_for_provider, env_key
    for provider in ('GROQ', 'GEMINI', 'OPENAI'):
        monkeypatch.setenv(f'{provider}_API_KEY', 'shared-key')
    monkeypatch.setenv('OPENAI_MODEL', 'shared-model')
    monkeypatch.setenv('LLM_RPM', '10')
    monkeypatch.setenv('RA_RETRIEVAL_LLM_PROVIDER', 'openai')
    monkeypatch.setenv('RA_RETRIEVAL_OPENAI_API_KEY', 'stage-key')
    monkeypatch.setenv('RA_RETRIEVAL_OPENAI_MODEL', 'stage-model')
    monkeypatch.setenv('RA_RETRIEVAL_OPENAI_LLM_RPM', '50')
    with pytest.raises(RuntimeError):
        with stage_profile('retrieval'):
            assert expansion_provider() == extraction_provider('gemini') == 'openai'
            assert model_for_provider('openai') == 'stage-model'
            assert env_key('OPENAI_API_KEY') == 'stage-key'
            assert os.environ['LLM_RPM'] == '50'
            raise RuntimeError('simulate stage failure')
    assert env_key('OPENAI_API_KEY') == 'shared-key'
    assert model_for_provider('openai') == 'shared-model'
    assert os.environ['LLM_RPM'] == '10'
    assert expansion_provider() == 'groq'


def test_selected_provider_requires_its_own_key(tmp_path):
    from research_assistant.ui.settings import ENV_DEFAULTS
    env = dict(ENV_DEFAULTS)
    _, configs = defaults(tmp_path)
    configs['retrieval']['artifacts_dir'] = str(tmp_path)
    (tmp_path / 'manifest.json').write_text('{}')
    env.update(GEMINI_API_KEY='another-provider', RA_RETRIEVAL_LLM_PROVIDER='groq')
    with pytest.raises(ValueError, match='retrieval: enter an API key for groq'):
        validate(env, configs, preflight=True)


@pytest.mark.parametrize('key,value', [
    ('RA_WRITING_LLM_PROVIDER', 'unknown'),
    ('RA_WRITING_OPENAI_LLM_TPM', '-1'),
    ('RA_WRITING_GROQ_LLM_RPM', '0'),
])
def test_invalid_stage_profile(tmp_path, key, value):
    env, configs = defaults(tmp_path)
    env[key] = value
    with pytest.raises(ValueError):
        validate(env, configs)
