const messages = {
  vi: {
    skipToSetup: 'Đến thiết lập nghiên cứu', viewResults: 'Xem tiến trình và kết quả ↓',
    quickParameters: 'THÔNG SỐ BẢN TỔNG QUAN',
    filesEmpty: 'Tệp kết quả sẽ xuất hiện khi từng giai đoạn hoàn tất.',
    copyFailed: 'Không thể sao chép tự động. Hãy chọn nội dung bản tổng quan để sao chép.',
    pageTitle: 'Research Assistant · Không gian nghiên cứu',
    languageAria: 'Ngôn ngữ giao diện', localWorkspace: 'Không gian làm việc cục bộ',
    eyebrow: 'TỪ CHỦ ĐỀ ĐẾN LITERATURE REVIEW', heading: 'Bắt đầu một nghiên cứu.',
    intro: 'Nhập chủ đề, thiết lập cấu hình và để pipeline thực hiện cả 4 giai đoạn.',
    researchSetup: 'Thiết lập nghiên cứu', researchTopic: 'Chủ đề cần tìm',
    topicPlaceholder: 'Ví dụ: Retrieval-augmented generation for scientific literature review',
    topicHint: 'Mô tả rõ phương pháp, lĩnh vực hoặc vấn đề bạn muốn khảo sát.',
    envConfig: 'Kết nối nhà cung cấp', enterOnce: 'Thiết lập lần đầu',
    envHint: 'Nhập key của ít nhất một nhà cung cấp và quota thực tế. Key đã lưu được giữ nguyên khi ô nhập để trống.',
    stageConfig: 'Cấu hình từng stage', advanced: 'Nâng cao',
    stageHint: 'Tên biến được giữ giống cấu hình Python. Đường dẫn tương đối tính từ thư mục dự án.',
    runPipeline: 'Bắt đầu nghiên cứu', saveConfig: 'Lưu thiết lập',
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
    runLog: 'Nhật ký lượt chạy', refreshLog: 'Làm mới log', downloadLog: 'Tải log',
    logEmpty: 'Log sẽ xuất hiện sau khi pipeline bắt đầu.', logAria: 'Nhật ký chạy pipeline',
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
    tabReview: 'Bản tổng quan', tabLog: 'Nhật ký xử lý',
    tabArtifacts: 'Tệp đầu ra', tabHistory: 'Lịch sử nghiên cứu',
    copyReview: 'Sao chép Review', copied: 'Đã sao chép!',
    themeLight: 'Sáng', themeDark: 'Tối', themeAria: 'Giao diện sáng/tối',
  },
  en: {
    skipToSetup: 'Skip to research setup', viewResults: 'View progress & results ↓',
    quickParameters: 'REVIEW PARAMETERS',
    filesEmpty: 'Output files will appear as each stage finishes.',
    copyFailed: 'Unable to copy automatically. Select the review text to copy it manually.',
    pageTitle: 'Research Assistant · Research workspace',
    languageAria: 'Interface language', localWorkspace: 'Local workspace',
    eyebrow: 'FROM TOPIC TO LITERATURE REVIEW', heading: 'Start a research project.',
    intro: 'Enter a topic, adjust the configuration, and run all four pipeline stages.',
    researchSetup: 'Research setup', researchTopic: 'Research topic',
    topicPlaceholder: 'Example: Retrieval-augmented generation for scientific literature review',
    topicHint: 'Describe the method, field, or problem you want to investigate.',
    envConfig: 'Provider connections', enterOnce: 'First-time setup',
    envHint: 'Enter at least one provider key and its actual quota. Leave a saved key blank to keep it unchanged.',
    stageConfig: 'Stage configuration', advanced: 'Advanced',
    stageHint: 'Variable names match the Python configuration. Relative paths start from the project directory.',
    runPipeline: 'Start research', saveConfig: 'Save settings',
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
    runLog: 'Run log', refreshLog: 'Refresh log', downloadLog: 'Download log',
    logEmpty: 'Log entries will appear after a pipeline starts.', logAria: 'Pipeline run log',
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
    tabReview: 'Literature Review', tabLog: 'Terminal Log',
    tabArtifacts: 'Output Files', tabHistory: 'Research History',
    copyReview: 'Copy Review', copied: 'Copied!',
    themeLight: 'Light', themeDark: 'Dark', themeAria: 'Color theme',
  },
};

// Canonical backend messages are English; retain legacy Vietnamese run support.
const backendTranslations = [
  [
    "[Các dòng log cũ đã được lược bỏ khỏi phần xem trực tiếp. Tải run.log để xem log đầy đủ.]",
    "[Earlier log entries omitted from live view. Download run.log for the complete log.]"
  ],
  [
    "Tiến trình đã dừng ngoài dự kiến hoặc ứng dụng đã khởi động lại. Có thể chạy lại bằng cấu hình đã lưu.",
    "The process stopped unexpectedly or the application restarted. You can start a new run using the saved configuration."
  ],
  [
    "Nhập ít nhất một API key để dùng LLM, hoặc tắt summarize và use_llm.",
    "Enter at least one API key to use an LLM, or disable summarize and use_llm."
  ],
  [
    "Pipeline đang chạy. Hãy chờ hoàn tất hoặc dừng lượt hiện tại.",
    "A pipeline is already running. Wait for it to finish or stop the current run."
  ],
  [
    "Không tìm thấy bài báo. Hãy đổi chủ đề hoặc nới bộ lọc.",
    "No papers were found. Try another topic or broaden the filters."
  ],
  [
    "Một số bài bị bỏ qua ở stage 2; xem JSON để biết lý do.",
    "Some papers were skipped in stage 2; see the JSON output for details."
  ],
  [
    "Nhập đủ LLM_RPM, LLM_TPM, LLM_RPD theo quota của bạn.",
    "Enter LLM_RPM, LLM_TPM, and LLM_RPD according to your quota."
  ],
  [
    "Stage 3 không có bài đủ điều kiện để viết review.",
    "Stage 3 found no eligible papers for the review."
  ],
  [
    "top_k không được lớn hơn candidate_pool_size",
    "top_k must not exceed candidate_pool_size"
  ],
  [
    "Năm bắt đầu không được lớn hơn năm kết thúc",
    "The start year must not exceed the end year"
  ],
  [
    "Không trích xuất được bài nào để tổng hợp.",
    "No papers could be extracted for synthesis."
  ],
  [
    "Provider phải là gemini, openai hoặc groq",
    "Provider must be gemini, openai, or groq"
  ],
  [
    "Stage 4 không tạo được bản review hợp lệ.",
    "Stage 4 could not create a valid review."
  ],
  [
    "Chỉ truy cập từ ứng dụng trên máy này.",
    "Access is restricted to the application on this computer."
  ],
  [
    "Phiên không hợp lệ. Hãy tải lại trang.",
    "Invalid session. Reload the page."
  ],
  [
    "device: chọn auto, cpu, cuda hoặc mps",
    "device: choose auto, cpu, cuda, or mps"
  ],
  [
    "Nhập chủ đề từ 1 đến 2000 ký tự",
    "Enter a topic between 1 and 2,000 characters."
  ],
  [
    "Người dùng đã dừng pipeline.",
    "The user stopped the pipeline."
  ],
  [
    "Cấu hình phải là một object",
    "Configuration must be an object"
  ],
  [
    "Cấu hình không được hỗ trợ",
    "Unsupported configuration"
  ],
  [
    "Dữ liệu quá lớn hoặc trống",
    "Request data is too large or empty"
  ],
  [
    "Không tìm thấy lượt chạy",
    "Run not found"
  ],
  [
    "cần số hữu hạn không âm",
    "expected a finite non-negative number"
  ],
  [
    "Thiếu dữ liệu SPECTER2:",
    "Missing SPECTER2 data:"
  ],
  [
    "cần giá trị true/false",
    "expected true or false"
  ],
  [
    "Lượt chạy không hợp lệ",
    "Invalid run"
  ],
  [
    "cấu hình không hợp lệ",
    "invalid configuration"
  ],
  [
    "giá trị không hợp lệ",
    "invalid value"
  ],
  [
    "API key không hợp lệ",
    "Invalid API key"
  ],
  [
    "Dữ liệu không hợp lệ",
    "Invalid request data"
  ],
  [
    "cần số nguyên dương",
    "expected a positive integer"
  ],
  [
    "bài được trích xuất",
    "papers extracted"
  ],
  [
    "Không tìm thấy tệp",
    "File not found"
  ],
  [
    "phải lớn hơn 0",
    "must be greater than zero"
  ],
  [
    "Không tìm thấy",
    "Not found"
  ],
  [
    "cần đường dẫn",
    "a path is required"
  ],
  [
    "cần nhập số",
    "expected a number"
  ],
  [
    "Bản review:",
    "Review:"
  ],
  [
    "xem kết quả",
    "see results"
  ],
  [
    "Tổng hợp:",
    "Synthesis:"
  ],
  [
    "bài báo",
    "papers"
  ],
  [
    "nhóm ·",
    "clusters ·"
  ]
];

Object.assign(messages.vi, {
  stageProvider: 'Nhà cung cấp AI cho giai đoạn này',
  stageProfileHint: 'Chọn nhà cung cấp rồi nhập model, API key và quota riêng. Ô trống dùng cấu hình chung của nhà cung cấp đã chọn. Không tự chuyển sang nhà cung cấp khác.',
  inheritShared: 'Dùng cấu hình chung',
});
Object.assign(messages.en, {
  stageProvider: 'AI provider for this stage',
  stageProfileHint: 'Choose a provider and enter a model, API key and quota for this stage. Blank fields inherit shared settings for the selected provider. No fallback to another provider.',
  inheritShared: 'Use shared settings',
});
