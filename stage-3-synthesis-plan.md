# Stage 3 Synthesis — revised implementation plan

Status: design proposal; implementation and empirical evaluation pending.

## Objective and boundaries

Consume a validated ExtractionResult snapshot, group papers by methodological similarity, and optionally produce traceable synthesis. Do not reopen sources, retrieve additional papers, or write the Stage 4 review. CPU-only clustering and an offline path remain mandatory. TF-IDF is the baseline, not a demonstrated optimum.

The output describes the supplied extraction snapshot. It cannot establish the absence of prior work, the completeness of retrieval, or the completeness of extraction. Do not hardcode arXiv/year/category restrictions unless supplied as provenance.

## Findings against the current repository

1. Critical: extraction/fingerprint.py hashes pipeline configuration, model and policy, not records. extraction/pipeline.py puts that value into ExtractionResult.fingerprint. Reusing it as corpus identity can return another corpus's synthesis.
2. Critical: valid paper IDs establish reference integrity, not semantic support. Dropping evidence and retaining only paper IDs leaves no claim-level support contract for Stage 4.
3. Critical: extraction/types.py represents datasets and metrics as Mention lists. A mention does not establish that a proposed method was evaluated on that dataset. An absent mention does not establish non-coverage. A method-by-dataset Cartesian product would invent relations.
4. High: silhouette optimization always selects a partition from the candidate partitions, even if none is useful. The original search cannot select one cluster or represent a mostly singleton corpus properly. Low silhouette can mean ambiguity, not a unique method.
5. High: the proposed gold mixes selection mechanism, scheduling, optimization objective, and paper role. Dynamic pruning can also be score-based; a library can describe multiple families. Eight Stage 2 annotation records are a smoke fixture, not independent clustering ground truth.
6. High: llm/client.py selects an available provider but does not implement a runtime provider cascade. Its HTTP helper retries up to four times internally. One logical completion is not one network request.
7. Medium: extraction confidence is a heuristic derived from source type and retained units, not a calibrated probability of correctness. It should be displayed as provenance, not used as scientific certainty.
8. Medium: the package installs torch/FAISS as mandatory dependencies. Offline synthesis need not load them, but that is distinct from providing a lightweight installation.

## Processing stages

Validate snapshot → normalize identities → create evidence registry and paper cards → build features → cluster or abstain → freeze assignments → build descriptive coverage → optional LLM narration → validate individual claims → export/cache.

Keep deterministic artifacts usable when narration fails. Treat schema/configuration errors differently from recoverable provider failures.

## Input and identity contract

- Validate ExtractionResult and retain all input records in an inventory with disposition and reason.
- Normalize IDs using arxiv_ids.py. Preserve version; use paper_key = canonical ID + version internally.
- Deduplicate identical records. For multiple versions, select the highest numeric version with an explicit selection report. Conflicting records for the same ID/version are input errors, not silently overwritten.
- Cluster only accepted statuses with nonblank method text and usable features. Distinguish skipped, missing_method, empty_features, duplicate, and superseded_version.
- No usable papers produces a valid empty result with a reason. One usable paper produces a singleton. Missing keywords alone is not grounds for exclusion.
- Preserve source_kind and heuristic extraction confidence. Abstract-derived cards may support provisional groups, but missing fields remain unknown.
- Related-work exclusion cannot be enforced retroactively by Stage 3. Carry an upstream policy manifest when available; otherwise mark it unknown. Stage 2 should eventually export that manifest alongside its fingerprint.

## Feature baseline

Use two independently normalized feature channels: method-text TF-IDF and normalized keyword-phrase TF-IDF. Concatenate sqrt(1-w) times text with sqrt(w) times keywords, then normalize each row. Start with w=0.5 as a development baseline; freeze the selected weight after development evaluation. A missing channel uses the available channel.

Method channel: word unigrams/bigrams, min_df=1, sublinear_tf=True, configurable max_features. Keywords remain phrases rather than being split into generic words. Normalize Unicode, case and whitespace; use a versioned, narrow alias map. Avoid aggressive aliases that merge distinct methods.

Do not inject the literal KEYWORDS marker into features. Do not include dataset/metric names as independent feature channels. Text may still contain topic or benchmark vocabulary; record this as a known limitation and test it with adversarial fixtures. Empty vocabulary and zero rows must be handled before cosine distance.

Fit vocabulary in a canonical paper order. Record channel weights, vectorizer settings and normalizer version. Select descriptive labels by keyword document frequency and contrast against other clusters; use deterministic tie-breaking. Labels are descriptions, not verified taxonomic categories.

## Clustering and abstention

Use a small dense cosine distance matrix, sanitized for numerical noise and with an exact zero diagonal. AgglomerativeClustering must explicitly use average linkage with precomputed distances; do not inherit Ward defaults.

Recommended MVP: cut the hierarchy at a configurable distance threshold. This naturally admits one group, multiple groups, and singletons. Tune the threshold on development data and record it; there is no justified universal numeric threshold yet. If no development annotations are available, expose the threshold and label results provisional rather than claiming validated automatic selection.

Retain silhouette-based k search only as an evaluation baseline. If retained in production, it needs explicit one-cluster and abstention candidates with independent quality rules; silhouette alone cannot score those candidates.

- Singleton means no accepted merge under the selected representation and threshold, not scientific novelty.
- Low silhouette means a diagnostic ambiguity warning. Do not automatically remove the paper or merge all ambiguous papers into other.
- Keep unassigned records separately when features are unusable; they are not a semantic cluster.
- For n=2 or n=3 use the same distance-cut rule. For identical vectors permit one cluster. For mutually dissimilar vectors permit singletons.
- Calculate silhouette only when its label-count precondition holds; otherwise store null and a reason. Do not manufacture a score of zero.
- Report size, within-cluster distance and nearest alternative distance. These are geometric diagnostics, not probabilities.
- Primary membership is exclusive for a simple Stage 4 interface. Secondary method tags may overlap; do not force paper role or scheduling into the same taxonomy.
- Cluster IDs hash sorted member paper keys and the clustering version. They are stable for the same snapshot, not guaranteed across corpus updates.

## Evidence and synthesis contract

Build an immutable EvidenceRegistry from existing claims/mentions. Each unit has an ID derived from paper_key, field path, and content digest, plus original text and evidence locations. No source reopening is needed.

PaperCard includes short field text and evidence-unit IDs. Preserve a full local registry; optionally include bounded source quotes for important method/result/limitation units in the LLM request. Truncation must be explicit and must never sever a displayed statement from its reference.

LLM output contains cluster_id and claim objects only; it does not return assignments. The application owns membership. Reject unknown cluster IDs and unexpected fields. A claim contains text, support_refs, subject_paper_keys, and validation status.

The deterministic validator checks that references exist, belong to the permitted papers, match the permitted field kinds, and cover every paper explicitly named in the statement. It also requires at least two supported subjects for a shared claim and references for both sides of a difference. Claims about all members need support for every member; otherwise phrase them as a supported subset.

These checks establish reference integrity and structural grounding, not entailment. Mark accepted free-text synthesis as structurally_validated; reserve semantically_verified for a separate human or evaluated support check. Strict evidence mode can emit extracted statements with template wording instead of unverified paraphrases.

Validate claims independently and keep valid claims if another fails. A bounded repair may address invalid items once; retain rejection reasons and counts. Never accept arbitrary prose citations that bypass support_refs. Treat extracted text as untrusted data, with clear prompt delimiters and no tool permissions.

## Coverage, gaps and comparisons

MVP coverage describes observations: a dataset was mentioned, a metric was mentioned, or a limitation was extracted. Do not infer an evaluated method-dataset pair from co-occurrence. Explicit relations may be recorded only when an existing evidence unit supports them.

Use observed / unknown / explicitly_not_evaluated states. The last requires an explicit source statement; empty lists map to unknown. Keep extraction omissions, corpus-level observations, explicit limitations, and research hypotheses distinct.

Replace automatic GapClaim with GapCandidate: kind, statement, scope, supporting refs, inspected paper keys, unknown paper keys, and verification_needed. Default automatic candidates should be documented limitations or statements that a relationship was not observed in this extraction snapshot. Do not claim that papers did not cover a relationship merely because it is missing from JSON.

No ranking of numerical results unless dataset/split, metric and direction, units, model, protocol and relevant compute/pruning budget are comparable and supported. Existing results are free text; therefore MVP comparisons are qualitative with comparability=unknown when metadata is missing. Structured experiment extraction is a later upstream extension.

## Public output

SynthesisResult includes schema_version, topic, corpus_digest, upstream_pipeline_fingerprint, effective_config, input_inventory, paper_manifest, assignments, unassigned, evidence_registry, descriptive_coverage, summaries, comparisons, gap_candidates, diagnostics and execution.

Summary status is one of not_requested, unavailable, complete, partial or failed. Empty summaries must not ambiguously mean no findings. execution reports effective provider/model, logical calls, HTTP attempts if observable, failure reasons and current-run duration.

Stage 4 resolves support_refs against the registry and paper manifest, verifies the corpus digest/version, and preserves the scope and validation status. It must not convert unknown coverage into a scientific gap or structurally validated text into verified fact.

## Caching

Canonicalize input content and hash IDs, versions, statuses, all consumed fields and evidence. Use a separate corpus digest; upstream fingerprint is provenance only. Canonicalize the record order, retain duplicate/conflict policy, and exclude volatile runtime metrics.

Use two cache layers:

- Cluster key: corpus digest + selection/normalization/features/clustering config and algorithm/schema versions.
- Narration key: cluster key + evidence/card digest + topic/scope + prompt/validator/card versions + effective provider/model + generation and truncation settings.

Offline output must not satisfy a requested successful narration cache entry. Failed narration must remain retryable. Store effective provider on fallback, not only requested provider. Validate cache schema, keys and references when reading; ignore corrupt entries with diagnostics. Write atomically via a temporary file and rename. Separate artifact-generation metadata from current cache-hit timing.

## LLM budget and provider behavior

Default summarize=False. With narration enabled, target one completion and allow at most one repair within an explicit logical-call cap. Missing key returns unavailable and deterministic output.

Estimate complete prompt size, including evidence, schema and reserved output budget. Use deterministic per-field/per-paper bounds, preserve method evidence first, and report omissions. If still oversized, skip narration with a budget reason in MVP; do not silently exceed the call budget or drop papers from assignments.

Inject complete_fn for tests, matching Stage 2's testability pattern. Pass provider explicitly. Reuse the shared client only after defining how transport attempts and deadlines are bounded: its existing four-attempt backoff otherwise makes the claimed one-to-two-call budget misleading. Runtime provider fallback is optional and must consume the same budget; key-based provider selection is not runtime fallback.

Do not promise byte-identical LLM text from temperature=0. Deterministic membership and validated cached output provide reproducibility at the artifact level.

## Modules and CLI

Use synthesis/{types,cards,features,cluster,coverage,prompt,validate,fingerprint,pipeline,cli,__main__}.py. Keep evidence-registry construction with cards initially; split only if complexity warrants it. Provide synthesize(extracted, config, *, complete_fn=None), a package export, synthesize-papers entry point and an example.

CLI: --summarize, --strict-ok, --distance-threshold, --no-cache, --json-out, --verbose. Use explicit status selection rather than an implicit ordering of status strings. Log to stderr; JSON output remains clean. Invalid input/config exits nonzero. Provider failure produces a valid partial result; an optional --require-summary makes it nonzero for automation that requires prose.

Add a tested scikit-learn dependency constraint and resolve the environment lock if the project uses one. Verify that synthesis never instantiates retrieval models or indexes. A lightweight dependency extra can be a separate packaging improvement.

## Evaluation and acceptance

Create synthesis annotations independently of Stage 2 mention labels. Define primary methodological axis, paper role, optional overlapping tags and uncertain pairs. Human review should resolve or explicitly retain ambiguities. Keep the eight supplied papers as a smoke set; add held-out topics and lexical/semantic contrast cases before reporting generalization.

Compare keywords-only, method-only and weighted features against one-cluster and singleton baselines. Evaluate both clustering and accepted-paper coverage: ARI on the agreed exclusive subset, pairwise precision/recall, singleton/unassigned rate, and sensitivity to input ordering and small feature changes. Purity alone rewards singleton partitions. Do not assume a feature combination must win.

Test gates separately from content quality: reference rejection, supported claim precision by human review, surviving claim coverage, scope compliance and forbidden unsupported numeric comparisons. Citation keep rate alone can be high for false statements or deceptively low after conservative rejection; report its numerator/denominator and null for no claims.

Required offline tests:

1. Empty input, all skipped, blank method, missing keywords, empty vocabulary, identical vectors, n=1/2/3 and all dissimilar papers.
2. Same-ID duplicates, conflicting same-version records, multiple versions and reordered inputs.
3. Every selected paper appears exactly once in assignments or unassigned; excluded records have reasons; IDs are stable under reordering.
4. Unknown references, cross-cluster references, named unsupported subjects, invented assignments and mixed valid/invalid claims.
5. Missing dataset remains unknown; mentions do not fabricate evaluation relations; incompatible metrics cannot produce a winner.
6. Same upstream fingerprint with different records causes cache misses; changed evidence invalidates narration; off/on summary modes remain distinct; corrupt cache is recoverable.
7. Missing key, timeout, malformed JSON, repair exhaustion and oversized prompt preserve deterministic output and respect budgets.
8. Offline CLI works without network, GPU/model/index initialization; JSON round-trips; Stage 4-style reference resolution succeeds.

Quality thresholds must be agreed after development calibration and before held-out evaluation. Structural invariants above are mandatory regardless of ARI. No empirical accuracy or latency claim is established by this design review.

## Delivery order

1. Freeze identity, evidence, scope, output states and cache contracts with tests.
2. Implement offline cards/features/clustering, diagnostics and CLI.
3. Implement descriptive coverage and claim/reference validation before LLM integration.
4. Add bounded narration, partial recovery and separate narration cache.
5. Calibrate clustering on independent development labels; evaluate held-out examples and finalize defaults.

Keep embeddings, multilingual normalization beyond the observed corpus, overlapping clustering, additional retrieval, structured experiment extraction, and Stage 4 prose generation outside the MVP. Reconsider a small method-text encoder only if measured lexical failure modes justify it; the cost of SPECTER2 alone does not rule out every embedding approach.

## Technical references

- Repository: research_assistant/extraction/{types,fingerprint,pipeline,validate}.py; research_assistant/llm/client.py; research_assistant/arxiv_ids.py; evaluation/extraction_gold.json; pyproject.toml.
- Agglomerative API and linkage constraints: https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- Silhouette implementation and valid label counts: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/cluster/_unsupervised.py
