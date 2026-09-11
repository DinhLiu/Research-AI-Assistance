'use strict';

const $ = id => document.getElementById(id);


let language = getStoredLanguage();
let token, definition, currentRun, timer, previewed, lastState, lastRuns = [];
let busy = false;
const fields = new Map();

function getStoredLanguage() {
  try {
    const saved = localStorage.getItem('research-assistant-language');
    if (saved === 'vi' || saved === 'en') return saved;
  } catch (_) {}
  return navigator.language.toLowerCase().startsWith('vi') ? 'vi' : 'en';
}

function t(key, values = {}) {
  let value = messages[language][key] ?? messages.en[key] ?? key;
  for (const [name, replacement] of Object.entries(values)) value = value.replace(`{${name}}`, replacement);
  return value;
}

function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function applyLanguage(nextLanguage, remember = true) {
  language = nextLanguage === 'en' ? 'en' : 'vi';
  document.documentElement.lang = language;
  document.title = t('pageTitle');
  document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  document.querySelectorAll('[data-i18n-aria-label]').forEach(node => { node.setAttribute('aria-label', t(node.dataset.i18nAriaLabel)); });
  const switcher = document.querySelector('.language-switch');
  switcher.setAttribute('aria-label', t('languageAria'));
  switcher.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.language === language)));
  if (remember) {
    try { localStorage.setItem('research-assistant-language', language); } catch (_) {}
  }
  document.querySelectorAll('[data-stage-index]').forEach(node => {
    node.textContent = `Stage ${Number(node.dataset.stageIndex) + 1} · ${t(`stage.${node.dataset.stageIndex}`)}`;
  });
  document.querySelectorAll('input[data-secret-saved]').forEach(input => {
    input.placeholder = input.dataset.secretSaved === 'true' ? t('savedSecret') : t('enterApiKey');
  });
  if (lastState) draw(lastState);
  if (definition) renderHistory(lastRuns);
}

function translateMessage(value) {
  if (!value) return value;
  let translated = value;
  for (const [vietnamese, english] of backendTranslations) {
    translated = translated.split(language === 'vi' ? english : vietnamese)
      .join(language === 'vi' ? vietnamese : english);
  }
  return translated;
}

function notice(text, error = false) {
  $('notice').textContent = translateMessage(text);
  $('notice').className = error ? 'error' : '';
  $('notice').hidden = !text;
}

async function api(path, data) {
  const response = await fetch(path, {
    method: data === undefined ? 'GET' : 'POST',
    headers: data === undefined ? {} : {'Content-Type': 'application/json', 'X-UI-Token': token},
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || t('connectionFailed'));
  return body;
}

function translatedLabel(name) {
  const key = `field.${name}`;
  return messages[language][key] ? t(key) : name;
}

function inputField(stage, field, value, parent) {
  const key = `${stage}.${field.name}`;
  const wrap = element('div', undefined, 'field');
  const label = element('label');
  label.htmlFor = key;
  const labelText = element('span', translatedLabel(field.name));
  if (messages.vi[`field.${field.name}`]) labelText.dataset.i18n = `field.${field.name}`;
  let input;
  const choices = field.name === 'language' ? ['vi', 'en'] : field.name === 'prefer_provider' ? ['gemini', 'openai', 'groq'] : field.name === 'device' ? ['auto', 'cpu', 'cuda', 'mps'] : null;
  if (choices) {
    input = element('select');
    for (const choice of choices) {
      const choiceKey = `reviewLanguage.${choice}`;
      const option = element('option', messages.vi[choiceKey] ? t(choiceKey) : choice);
      if (messages.vi[choiceKey]) option.dataset.i18n = choiceKey;
      option.value = choice;
      input.append(option);
    }
  } else {
    input = element('input');
    input.type = field.kind === 'boolean' ? 'checkbox' : ['integer', 'number'].includes(field.kind) ? 'number' : 'text';
    if (input.type === 'number') { input.step = field.kind === 'integer' ? '1' : 'any'; input.min = '0'; }
  }
  input.id = key;
  input.dataset.kind = field.kind;
  if (field.kind === 'boolean') input.checked = [true, 'True', 'true', '1'].includes(value);
  else input.value = value ?? '';
  if (field.kind === 'boolean') label.append(input, labelText);
  else label.append(labelText);
  wrap.append(label);
  if (field.kind !== 'boolean') wrap.append(input);
  if (field.nullable) { input.placeholder = t('noLimit'); input.dataset.i18nPlaceholder = 'noLimit'; }
  if (stage !== 'env') wrap.append(element('small', `RA_${stage}_${field.name}`.toUpperCase()));
  if (field.name === 'artifacts_dir') wrap.classList.add('field-wide');
  fields.set(key, input);
  parent.append(wrap);
  return wrap;
}

async function load() {
  definition = await api('/api/settings');
  token = definition.token;
  const stageNames = Object.keys(definition.schema);
  const profileGroups = [];
  const quick = new Set(['retrieval.top_k', 'writing.language', 'writing.target_words', 'retrieval.artifacts_dir']);
  for (const [stage, stageSchema] of Object.entries(definition.schema)) {
    const index = stageNames.indexOf(stage);
    const details = element('details');
    const summary = element('summary', `Stage ${index + 1} · ${t(`stage.${index}`)}`);
    summary.dataset.stageIndex = String(index);
    details.append(summary);
    $('stage-fields').append(details);
    details.open = false;
    const providerKey = `RA_${stage.toUpperCase()}_LLM_PROVIDER`;
    const pickerWrap = element('div', undefined, 'field');
    const pickerLabel = element('label', t('stageProvider'));
    pickerLabel.dataset.i18n = 'stageProvider';
    pickerLabel.htmlFor = `env.${providerKey}`;
    const picker = element('select');
    picker.id = `env.${providerKey}`;
    for (const [value, label] of [['groq', 'Groq'], ['openai', 'OpenAI'], ['gemini', 'Gemini']]) {
      const option = element('option', label); option.value = value; picker.append(option);
    }
    const preferred = definition.configs[stage].prefer_provider || 'groq';
    const available = [preferred, 'gemini', 'openai', 'groq'].find(p => definition.secrets[`${p.toUpperCase()}_API_KEY`]);
    picker.value = definition.env[providerKey] || available || preferred;
    fields.set(`env.${providerKey}`, picker);
    pickerWrap.append(pickerLabel, picker);
    details.append(pickerWrap);
    const hint = element('p', t('stageProfileHint'), 'hint');
    hint.dataset.i18n = 'stageProfileHint'; details.append(hint);
    const panels = [];
    for (const provider of ['groq', 'openai', 'gemini']) {
      const panel = element('div');
      panel.hidden = picker.value !== provider;
      details.append(panel); panels.push([provider, panel]);
      const names = ['API_KEY', 'MODEL', ...(provider === 'openai' ? ['BASE_URL'] : []), 'LLM_RPM', 'LLM_TPM', 'LLM_RPD', 'LLM_QUOTA_GROUP'];
      profileGroups.push([provider, names.map(name => `RA_${stage.toUpperCase()}_${provider.toUpperCase()}_${name}`), panel]);
    }
    picker.onchange = () => panels.forEach(([provider, panel]) => { panel.hidden = picker.value !== provider; });
    for (const field of stageSchema.filter(field => field.name !== 'prefer_provider')) inputField(stage, field, definition.configs[stage][field.name], quick.has(`${stage}.${field.name}`) ? $('quick-config') : details);
  }
  const groups = [
    ['group.quota', ['LLM_RPM', 'LLM_TPM', 'LLM_RPD']], ['Gemini', ['GEMINI_API_KEY', 'GEMINI_MODEL']],
    ['group.openai', ['OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_BASE_URL']], ['Groq', ['GROQ_API_KEY', 'GROQ_MODEL']],
    ['group.other', ['SEMANTIC_SCHOLAR_API_KEY', 'LLM_QUOTA_GROUP', 'LLM_STATE_DIR']],
  ];
  for (const [titleKey, keys, panel] of [...groups, ...profileGroups]) {
    const group = element('details');
    group.open = titleKey === 'group.quota' || keys.some(key => definition.secrets[key]);
    const summary = element('summary', messages.vi[titleKey] ? t(titleKey) : titleKey);
    if (messages.vi[titleKey]) summary.dataset.i18n = titleKey;
    group.append(summary);
    (panel || $('env-fields')).append(group);
    if (panel) { group.open = true; summary.hidden = true; }
    for (const key of keys) {
      const secret = key.endsWith('API_KEY');
      const wrap = inputField('env', {name: key, kind: 'text'}, secret ? '' : definition.env[key], group);
      const input = fields.get(`env.${key}`);
      if (panel) {
        const shortName = key.replace(/^RA_[A-Z]+_(GROQ|OPENAI|GEMINI)_/, '');
        wrap.querySelector('label span').textContent = shortName;
        if (!secret) { input.placeholder = t('inheritShared'); input.dataset.i18nPlaceholder = 'inheritShared'; }
      }
      if (secret) {
        input.type = 'password';
        input.autocomplete = 'new-password';
        input.placeholder = definition.secrets[key] ? t('savedSecret') : t('enterApiKey');
        input.dataset.secretSaved = definition.secrets[key] ? 'true' : 'false';
        const clear = element('input');
        clear.type = 'checkbox';
        clear.id = `clear-${key}`;
        const clearLabel = element('label', undefined, 'key-clear');
        const clearText = element('span', t('clearSavedKey'));
        clearText.dataset.i18n = 'clearSavedKey';
        clearLabel.append(clear, clearText);
        wrap.append(clearLabel);
      }
      if (['LLM_RPM', 'LLM_TPM', 'LLM_RPD'].includes(key)) {
        const placeholderKey = {LLM_RPM: 'requestsMinute', LLM_TPM: 'tokensMinute', LLM_RPD: 'requestsDay'}[key];
        input.inputMode = 'numeric';
        input.placeholder = t(placeholderKey);
        input.dataset.i18nPlaceholder = placeholderKey;
      }
    }
  }
  applyLanguage(language, false);
  $('provider-settings').open = !Object.entries(definition.secrets)
    .some(([key, saved]) => saved && /(?:GEMINI|OPENAI|GROQ)_API_KEY$/.test(key));
  await history(true);
}

function payload() {
  const data = {topic: $('topic').value, env: {}, configs: {}, clear_secrets: []};
  for (const [key, input] of fields) {
    const [stage, name] = key.split('.');
    const value = input.dataset.kind === 'boolean' ? input.checked : input.value;
    if (stage === 'env') {
      data.env[name] = value;
      if ($(`clear-${name}`)?.checked) data.clear_secrets.push(name);
    } else {
      data.configs[stage] ??= {};
      data.configs[stage][name] = value;
    }
  }
  return data;
}

function draw(state = {stages: ['pending', 'pending', 'pending', 'pending'], summaries: [], files: [], warnings: []}) {
  lastState = state;
  $('stages').replaceChildren();
  state.stages.forEach((status, index) => {
    const item = element('li', undefined, `stage ${status}`);
    const content = element('div');
    content.append(element('h3', t(`stage.${index}`)), element('p', translateMessage(state.summaries[index]) || t(`stageDescription.${index}`)));
    item.append(element('span', status === 'complete' ? '✓' : String(index + 1).padStart(2, '0'), 'stage-number'), content, element('span', t(`status.${status}`), 'state-label'));
    $('stages').append(item);
  });
  const count = state.stages.filter(status => status === 'complete').length;
  $('progress').style.width = `${count * 25}%`;
  document.querySelector('.progress-track').setAttribute('aria-valuenow', String(count));
  $('progress-label').textContent = t('progress', {count});
  $('run-status').textContent = t(`status.${state.status || 'ready'}`);
  $('run-status').dataset.status = state.status || 'ready';
  $('run-topic').textContent = state.topic || t('idleDescription');
  busy = state.status === 'running';
  $('run').disabled = busy;
  $('cancel').hidden = !busy;
  $('warnings').replaceChildren();
  for (const warning of [...(state.warnings || []), ...(state.error ? [state.error] : [])]) $('warnings').append(element('p', translateMessage(warning), 'warning'));
  $('downloads').replaceChildren();
  for (const file of state.files || []) {
    const link = element('a', `↓ ${file}`);
    link.href = `/api/runs/${currentRun}/files/${file}`;
    link.download = file;
    $('downloads').append(link);
  }
  $('empty-files').hidden = (state.files || []).length > 0;
  $('empty-result').hidden = previewed === currentRun && Boolean(currentRun);
}

async function refresh() {
  if (!currentRun) return;
  const state = await api(`/api/runs/${currentRun}`);
  draw(state);
  await refreshLog();
  if (state.files.includes('review.md') && previewed !== currentRun) {
    const response = await fetch(`/api/runs/${currentRun}/files/review.md`);
    if (!response.ok) throw new Error(t('reviewReadFailed'));
    $('review').textContent = await response.text();
    $('review').hidden = false;
    if ($('copy-review')) $('copy-review').hidden = false;
    previewed = currentRun;
    $('empty-result').hidden = true;
  }
  if (busy) timer = setTimeout(poll, 2000);
  else await history();
}

async function refreshLog() {
  if (!currentRun) return;
  const response = await fetch(`/api/runs/${currentRun}/log`, {cache: 'no-store'});
  if (!response.ok) return;
  const content = await response.text();
  const output = $('run-log');
  const wasNearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 40;
  output.textContent = content;
  output.hidden = !content;
  $('empty-log').hidden = Boolean(content);
  $('log-download').href = `/api/runs/${currentRun}/files/run.log`;
  $('log-download').hidden = false;
  if (wasNearBottom) output.scrollTop = output.scrollHeight;
}

async function poll() {
  try { await refresh(); }
  catch (error) {
    notice(t('reconnecting', {error: translateMessage(error.message)}), true);
    timer = setTimeout(poll, 5000);
  }
}

async function selectRun(id) {
  clearTimeout(timer);
  currentRun = id;
  renderHistory(lastRuns);
  previewed = null;
  $('review').hidden = true;
  if ($('copy-review')) $('copy-review').hidden = true;
  $('run-log').hidden = true;
  $('empty-log').hidden = false;
  $('log-download').hidden = true;
  await refresh();
}

function renderHistory(runs) {
  $('history').replaceChildren();
  if (!runs.length) $('history').append(element('p', t('noRuns'), 'hint'));
  const active = runs.find(run => run.status === 'running');
  for (const run of runs) {
    const button = element('button', undefined, 'history-item');
    button.type = 'button';
    button.setAttribute('aria-current', String(run.run_id === currentRun));
    button.disabled = Boolean(active) && active.run_id !== run.run_id;
    button.append(element('span', run.topic), element('small', t(`status.${run.status}`)));
    button.onclick = () => selectRun(run.run_id).catch(error => notice(error.message, true));
    $('history').append(button);
  }
}

async function history(restore = false) {
  lastRuns = await api('/api/runs');
  renderHistory(lastRuns);
  if (restore && lastRuns.length) {
    const active = lastRuns.find(run => run.status === 'running');
    await selectRun((active || lastRuns[0]).run_id);
  }
}

function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.setAttribute('aria-controls', `pane-${tab.dataset.tab}`);
    tab.tabIndex = tab.classList.contains('active') ? 0 : -1;
    $(`pane-${tab.dataset.tab}`).tabIndex = 0;
    tab.addEventListener('keydown', event => {
      const list = [...tabs];
      const index = list.indexOf(tab);
      const next = event.key === 'ArrowRight' ? (index + 1) % list.length
        : event.key === 'ArrowLeft' ? (index - 1 + list.length) % list.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? list.length - 1 : null;
      if (next === null) return;
      event.preventDefault();
      list[next].click();
      list[next].focus();
    });
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => {
        const active = t.dataset.tab === target;
        t.classList.toggle('active', active);
        t.setAttribute('aria-selected', String(active));
        t.tabIndex = active ? 0 : -1;
      });
      document.querySelectorAll('.tab-pane').forEach(pane => {
        const active = pane.id === `pane-${target}`;
        pane.classList.toggle('active', active);
        pane.hidden = !active;
      });
    });
  });

  if ($('copy-review')) {
    $('copy-review').onclick = async () => {
      const text = $('review').textContent;
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        $('copy-review').textContent = t('copied');
        setTimeout(() => { $('copy-review').textContent = t('copyReview'); }, 2000);
      } catch (_) { notice(t('copyFailed'), true); }
    };
  }
}

let theme = getStoredTheme();

function getStoredTheme() {
  try {
    const saved = localStorage.getItem('research-assistant-theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch (_) {}
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function applyTheme(nextTheme, remember = true) {
  theme = nextTheme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  const switcher = document.querySelector('.theme-switch');
  if (switcher) {
    switcher.querySelectorAll('button').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.themeBtn === theme));
    });
  }
  if (remember) {
    try { localStorage.setItem('research-assistant-theme', theme); } catch (_) {}
  }
}

document.querySelectorAll('[data-language]').forEach(button => { button.onclick = () => applyLanguage(button.dataset.language); });
document.querySelectorAll('[data-theme-btn]').forEach(button => { button.onclick = () => applyTheme(button.dataset.themeBtn); });

// Native validation must be able to focus fields inside collapsed settings.
$('research-form').addEventListener('invalid', event => {
  let parent = event.target.parentElement;
  while (parent && parent !== event.currentTarget) {
    if (parent.tagName === 'DETAILS') parent.open = true;
    parent = parent.parentElement;
  }
}, true);

$('research-form').onsubmit = async event => {
  event.preventDefault();
  $('run').disabled = true;
  $('run').setAttribute('aria-busy', 'true');
  notice(t('checkingConfig'));
  try {
    const result = await api('/api/runs', payload());
    notice(t('pipelineStarted'));
    await selectRun(result.run_id);
  } catch (error) {
    notice(error.message, true);
    $('run').disabled = busy;
  }
  finally { $('run').removeAttribute('aria-busy'); }
};

$('save').onclick = async () => {
  const button = $('save');
  button.disabled = true;
  button.setAttribute('aria-busy', 'true');
  try {
    await api('/api/settings', payload());
    const updated = await api('/api/settings');
    for (const key of Object.keys(updated.secrets)) {
      const input = fields.get(`env.${key}`);
      input.value = '';
      input.dataset.secretSaved = updated.secrets[key] ? 'true' : 'false';
      input.placeholder = updated.secrets[key] ? t('savedSecret') : t('enterApiKey');
      $(`clear-${key}`).checked = false;
    }
    notice(t('configSaved'));
  } catch (error) { notice(error.message, true); }
  finally { button.disabled = false; button.removeAttribute('aria-busy'); }
};

$('cancel').onclick = async () => {
  try {
    $('cancel').disabled = true;
    await api('/api/cancel', {});
    clearTimeout(timer);
    await refresh();
  } catch (error) { notice(error.message, true); }
  finally { $('cancel').disabled = false; }
};

$('refresh').onclick = () => history(true).catch(error => notice(error.message, true));
$('log-refresh').onclick = () => refreshLog().catch(error => notice(error.message, true));

applyLanguage(language, false);
applyTheme(theme, false);
draw();
setupTabs();
$('run').disabled = true;
load().then(() => { $('run').disabled = busy; $('save').disabled = false; })
  .catch(error => notice(t('settingsLoadFailed', {error: translateMessage(error.message)}), true));
