'use strict';

const $ = id => document.getElementById(id);

const messages = {
  vi: {
    pageTitle: 'Research Assistant · Không gian nghiên cứu',
    languageAria: 'Ngôn ngữ giao diện', localWorkspace: 'Không gian làm việc cục bộ',
    eyebrow: 'TỪ CHỦ ĐỀ ĐẾN LITERATURE REVIEW', heading: 'Bắt đầu một nghiên cứu.',
    intro: 'Nhập chủ đề, thiết lập cấu hình và để pipeline thực hiện cả 4 giai đoạn.',
    researchSetup: 'Thiết lập nghiên cứu', researchTopic: 'Chủ đề cần tìm',
    topicPlaceholder: 'Ví dụ: Retrieval-augmented generation for scientific literature review',
    topicHint: 'Mô tả rõ phương pháp, lĩnh vực hoặc vấn đề bạn muốn khảo sát.',
    envConfig: 'API & cấu hình .env', enterOnce: 'Nhập một lần',
    envHint: 'Nhập key của ít nhất một nhà cung cấp và quota thực tế. Key đã lưu được giữ nguyên khi ô nhập để trống.',
    stageConfig: 'Cấu hình từng stage', advanced: 'Nâng cao',
    stageHint: 'Tên biến được giữ giống cấu hình Python. Đường dẫn tương đối tính từ thư mục dự án.',
    runPipeline: 'Chạy toàn bộ pipeline', saveConfig: 'Lưu cấu hình .env',
    runHint: 'Nút chạy dùng cấu hình hiện tại. Chọn lưu để dùng lại trong các lần sau.',
    outputLabel: 'Theo dõi và kết quả', pipelineProgress: 'Tiến trình pipeline',
    idleDescription: 'Kết quả từng giai đoạn sẽ xuất hiện tại đây.',
    progress: '{count} / 4 giai đoạn hoàn tất', cancelPipeline: 'Dừng pipeline',
    researchResults: 'Kết quả nghiên cứu', outputFiles: 'TỆP ĐẦU RA',
    reviewHere: 'Bản review của bạn sẽ ở đây',
    reviewDescription: 'Pipeline tìm bài báo, trích xuất bằng chứng, tổng hợp và viết bản review có trích dẫn.',
    reviewAria: 'Nội dung review Markdown', researchRuns: 'Các lượt nghiên cứu', refresh: 'Làm mới',
    noRuns: 'Chưa có lượt chạy nào.',
    footer: 'Dữ liệu và API key được lưu trên máy của bạn. Các stage có thể gọi dịch vụ bên ngoài theo cấu hình.',
    noLimit: 'Không giới hạn', savedSecret: 'Đã lưu · để trống để giữ nguyên', enterApiKey: 'Nhập API key',
    clearSavedKey: 'Xóa key đã lưu', requestsMinute: 'Requests / phút', tokensMinute: 'Tokens / phút',
    requestsDay: 'Requests / ngày', checkingConfig: 'Đang kiểm tra cấu hình…',
    pipelineStarted: 'Pipeline đã bắt đầu. Bạn có thể theo dõi các giai đoạn ở bên phải.',
    configSaved: 'Đã lưu cấu hình vào .env.', connectionFailed: 'Không thể kết nối',
    reviewReadFailed: 'Không đọc được bản review', reconnecting: 'Mất kết nối: {error}. Đang thử lại…',
    settingsLoadFailed: 'Không tải được cấu hình: {error}. Hãy tải lại trang.',
    'group.quota': 'Quota LLM (bắt buộc khi dùng LLM)', 'group.openai': 'OpenAI / API tương thích',
    'group.other': 'Tùy chọn khác',
    'field.top_k': 'Số bài báo', 'field.language': 'Ngôn ngữ bản review',
    'field.target_words': 'Số từ mục tiêu', 'field.artifacts_dir': 'Thư mục dữ liệu SPECTER2',
    'field.use_llm': 'Dùng LLM để viết review', 'field.summarize': 'Dùng LLM để tổng hợp',
    'reviewLanguage.vi': 'Tiếng Việt', 'reviewLanguage.en': 'English',
    'stage.0': 'Tìm kiếm bài báo', 'stage.1': 'Trích xuất bằng chứng',
    'stage.2': 'Tổng hợp nghiên cứu', 'stage.3': 'Viết literature review',
    'stageDescription.0': 'SPECTER2 · tìm kiếm và xếp hạng',
    'stageDescription.1': 'Đọc nguồn · trích xuất thông tin có căn cứ',
    'stageDescription.2': 'Phân nhóm phương pháp · đối chiếu kết quả',
    'stageDescription.3': 'Bản review và danh mục trích dẫn',
    'status.pending': 'Đang chờ', 'status.running': 'Đang chạy', 'status.complete': 'Hoàn tất',
    'status.failed': 'Thất bại', 'status.skipped': 'Bỏ qua', 'status.cancelled': 'Đã dừng',
    'status.completed': 'Hoàn tất', 'status.completed_with_warnings': 'Có lưu ý', 'status.ready': 'Sẵn sàng',
  },
  en: {
    pageTitle: 'Research Assistant · Research workspace',
    languageAria: 'Interface language', localWorkspace: 'Local workspace',
    eyebrow: 'FROM TOPIC TO LITERATURE REVIEW', heading: 'Start a research project.',
    intro: 'Enter a topic, adjust the configuration, and run all four pipeline stages.',
    researchSetup: 'Research setup', researchTopic: 'Research topic',
    topicPlaceholder: 'Example: Retrieval-augmented generation for scientific literature review',
    topicHint: 'Describe the method, field, or problem you want to investigate.',
    envConfig: 'API & .env configuration', enterOnce: 'Enter once',
    envHint: 'Enter at least one provider key and its actual quota. Leave a saved key blank to keep it unchanged.',
    stageConfig: 'Stage configuration', advanced: 'Advanced',
    stageHint: 'Variable names match the Python configuration. Relative paths start from the project directory.',
    runPipeline: 'Run the full pipeline', saveConfig: 'Save .env configuration',
    runHint: 'Run uses the values currently shown. Save them if you want to reuse them later.',
    outputLabel: 'Progress and results', pipelineProgress: 'Pipeline progress',
    idleDescription: 'Results from each stage will appear here.',
    progress: '{count} / 4 stages complete', cancelPipeline: 'Stop pipeline',
    researchResults: 'Research results', outputFiles: 'OUTPUT FILES',
    reviewHere: 'Your review will appear here',
    reviewDescription: 'The pipeline finds papers, extracts evidence, synthesizes findings, and writes a cited review.',
    reviewAria: 'Markdown review content', researchRuns: 'Research runs', refresh: 'Refresh',
    noRuns: 'No research runs yet.',
    footer: 'Your data and API keys are stored on this computer. Stages may call external services according to your configuration.',
    noLimit: 'No limit', savedSecret: 'Saved · leave blank to keep unchanged', enterApiKey: 'Enter API key',
    clearSavedKey: 'Delete saved key', requestsMinute: 'Requests / minute', tokensMinute: 'Tokens / minute',
    requestsDay: 'Requests / day', checkingConfig: 'Checking configuration…',
    pipelineStarted: 'The pipeline has started. You can follow each stage on the right.',
    configSaved: 'Configuration saved to .env.', connectionFailed: 'Unable to connect',
    reviewReadFailed: 'Unable to read the review', reconnecting: 'Connection lost: {error}. Retrying…',
    settingsLoadFailed: 'Unable to load settings: {error}. Reload the page.',
    'group.quota': 'LLM quota (required when using an LLM)', 'group.openai': 'OpenAI / compatible API',
    'group.other': 'Other options',
    'field.top_k': 'Number of papers', 'field.language': 'Review language',
    'field.target_words': 'Target word count', 'field.artifacts_dir': 'SPECTER2 data directory',
    'field.use_llm': 'Use an LLM to write the review', 'field.summarize': 'Use an LLM for synthesis',
    'reviewLanguage.vi': 'Vietnamese', 'reviewLanguage.en': 'English',
    'stage.0': 'Find papers', 'stage.1': 'Extract evidence',
    'stage.2': 'Synthesize research', 'stage.3': 'Write literature review',
    'stageDescription.0': 'SPECTER2 · search and ranking',
    'stageDescription.1': 'Read sources · extract grounded evidence',
    'stageDescription.2': 'Cluster methods · compare findings',
    'stageDescription.3': 'Review and bibliography',
    'status.pending': 'Pending', 'status.running': 'Running', 'status.complete': 'Complete',
    'status.failed': 'Failed', 'status.skipped': 'Skipped', 'status.cancelled': 'Stopped',
    'status.completed': 'Complete', 'status.completed_with_warnings': 'Needs attention', 'status.ready': 'Ready',
  },
};

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
  let value = messages[language][key] ?? messages.vi[key] ?? key;
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
  if (language === 'vi' || !value) return value;
  const exact = {
    'Người dùng đã dừng pipeline.': 'The user stopped the pipeline.',
    'Một số bài bị bỏ qua ở stage 2; xem JSON để biết lý do.': 'Some papers were skipped in stage 2; see the JSON output for details.',
    'Không tìm thấy bài báo. Hãy đổi chủ đề hoặc nới bộ lọc.': 'No papers were found. Try another topic or broaden the filters.',
    'Không trích xuất được bài nào để tổng hợp.': 'No papers could be extracted for synthesis.',
    'Stage 3 không có bài đủ điều kiện để viết review.': 'Stage 3 found no eligible papers for the review.',
    'Stage 4 không tạo được bản review hợp lệ.': 'Stage 4 could not create a valid review.',
    'Pipeline đang chạy. Hãy chờ hoàn tất hoặc dừng lượt hiện tại.': 'A pipeline is already running. Wait for it to finish or stop the current run.',
    'Nhập chủ đề từ 1 đến 2000 ký tự': 'Enter a topic between 1 and 2,000 characters.',
    'Nhập ít nhất một API key để dùng LLM, hoặc tắt summarize và use_llm.': 'Enter at least one API key to use an LLM, or disable summarize and use_llm.',
    'Nhập đủ LLM_RPM, LLM_TPM, LLM_RPD theo quota của bạn.': 'Enter LLM_RPM, LLM_TPM, and LLM_RPD according to your quota.',
  };
  if (exact[value]) return exact[value];
  let match = value.match(/^(\d+) bài báo$/);
  if (match) return `${match[1]} papers`;
  match = value.match(/^(\d+)\/(\d+) bài được trích xuất$/);
  if (match) return `${match[1]}/${match[2]} papers extracted`;
  match = value.match(/^(\d+) nhóm · (.+)$/);
  if (match) return `${match[1]} clusters · ${match[2]}`;
  match = value.match(/^Tổng hợp: (.+)$/);
  if (match) return `Synthesis: ${match[1]}`;
  match = value.match(/^Bản review: (.+)$/);
  if (match) return `Review: ${match[1]}`;
  return value;
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
  fields.set(key, input);
  parent.append(wrap);
  return wrap;
}

async function load() {
  definition = await api('/api/settings');
  token = definition.token;
  const stageNames = Object.keys(definition.schema);
  const quick = new Set(['retrieval.top_k', 'writing.language', 'writing.target_words', 'retrieval.artifacts_dir']);
  for (const [stage, stageSchema] of Object.entries(definition.schema)) {
    const index = stageNames.indexOf(stage);
    const details = element('details');
    const summary = element('summary', `Stage ${index + 1} · ${t(`stage.${index}`)}`);
    summary.dataset.stageIndex = String(index);
    details.append(summary);
    $('stage-fields').append(details);
    for (const field of stageSchema) inputField(stage, field, definition.configs[stage][field.name], quick.has(`${stage}.${field.name}`) ? $('quick-config') : details);
  }
  const groups = [
    ['group.quota', ['LLM_RPM', 'LLM_TPM', 'LLM_RPD']], ['Gemini', ['GEMINI_API_KEY', 'GEMINI_MODEL']],
    ['group.openai', ['OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_BASE_URL']], ['Groq', ['GROQ_API_KEY', 'GROQ_MODEL']],
    ['group.other', ['SEMANTIC_SCHOLAR_API_KEY', 'LLM_QUOTA_GROUP', 'LLM_STATE_DIR']],
  ];
  for (const [titleKey, keys] of groups) {
    const group = element('details');
    group.open = titleKey === 'group.quota' || keys.some(key => definition.secrets[key]);
    const summary = element('summary', messages.vi[titleKey] ? t(titleKey) : titleKey);
    if (messages.vi[titleKey]) summary.dataset.i18n = titleKey;
    group.append(summary);
    $('env-fields').append(group);
    for (const key of keys) {
      const secret = key.endsWith('API_KEY');
      const wrap = inputField('env', {name: key, kind: 'text'}, secret ? '' : definition.env[key], group);
      const input = fields.get(`env.${key}`);
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
  $('progress-label').textContent = t('progress', {count});
  $('run-status').textContent = t(`status.${state.status || 'ready'}`);
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
  $('empty-result').hidden = (state.files || []).length > 0;
}

async function refresh() {
  if (!currentRun) return;
  const state = await api(`/api/runs/${currentRun}`);
  draw(state);
  if (state.files.includes('review.md') && previewed !== currentRun) {
    const response = await fetch(`/api/runs/${currentRun}/files/review.md`);
    if (!response.ok) throw new Error(t('reviewReadFailed'));
    $('review').textContent = await response.text();
    $('review').hidden = false;
    previewed = currentRun;
  }
  if (busy) timer = setTimeout(poll, 2000);
  else await history();
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
  previewed = null;
  $('review').hidden = true;
  await refresh();
}

function renderHistory(runs) {
  $('history').replaceChildren();
  if (!runs.length) $('history').append(element('p', t('noRuns'), 'hint'));
  const active = runs.find(run => run.status === 'running');
  for (const run of runs) {
    const button = element('button', undefined, 'history-item');
    button.type = 'button';
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

document.querySelectorAll('[data-language]').forEach(button => { button.onclick = () => applyLanguage(button.dataset.language); });

$('research-form').onsubmit = async event => {
  event.preventDefault();
  $('run').disabled = true;
  notice(t('checkingConfig'));
  try {
    const result = await api('/api/runs', payload());
    notice(t('pipelineStarted'));
    await selectRun(result.run_id);
  } catch (error) {
    notice(error.message, true);
    $('run').disabled = busy;
  }
};

$('save').onclick = async () => {
  const button = $('save');
  button.disabled = true;
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
  finally { button.disabled = false; }
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

applyLanguage(language, false);
draw();
$('run').disabled = true;
load().then(() => { $('run').disabled = busy; })
  .catch(error => notice(t('settingsLoadFailed', {error: translateMessage(error.message)}), true));
