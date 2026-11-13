package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	defaultVoyageEndpoint = "https://api.voyageai.com/v1"

	// voyageDefaultDimensions is the native output size of every supported
	// Voyage model. Matryoshka models accept a smaller (or larger) size
	// through embedder.dimensions.
	voyageDefaultDimensions = 1024

	// voyageMaxInputs is the documented maximum number of inputs per request.
	voyageMaxInputs = 1000

	// voyageContextChunkSize is the chunk_size sent with every contextualized
	// document request, in tokens.
	voyageContextChunkSize = 32_000

	// voyageBatchBudgetPercent is the share of a model's per-request token
	// cap that a request may use according to the local estimate, which can
	// undercount code.
	voyageBatchBudgetPercent = 80
)

// voyageKind tells which Voyage endpoint a model is served by.
type voyageKind int

const (
	voyageKindText voyageKind = iota
	voyageKindContextualized
)

// voyageModel describes one named Voyage model. There is deliberately no
// default model: embedder.model must be set to one of voyageModels.
type voyageModel struct {
	kind voyageKind
	// dimensions lists the accepted output_dimension values.
	dimensions []int
	// contextTokens is the model's per-input context window in tokens.
	contextTokens int
	// maxBatchTokens is the documented token cap of one request. 0 means the
	// documentation states none, so no cap is invented.
	maxBatchTokens int
}

var (
	voyageMatryoshka = []int{256, 512, 1024, 2048}
	voyageFixed1024  = []int{1024}
)

var voyageModels = map[string]voyageModel{
	"voyage-4-large":   {voyageKindText, voyageMatryoshka, 32000, 120_000},
	"voyage-4":         {voyageKindText, voyageMatryoshka, 32000, 320_000},
	"voyage-4-lite":    {voyageKindText, voyageMatryoshka, 32000, 1_000_000},
	"voyage-code-4":    {voyageKindText, voyageMatryoshka, 32000, 0},
	"voyage-3-large":   {voyageKindText, voyageMatryoshka, 32000, 120_000},
	"voyage-3.5":       {voyageKindText, voyageMatryoshka, 32000, 320_000},
	"voyage-3.5-lite":  {voyageKindText, voyageMatryoshka, 32000, 1_000_000},
	"voyage-code-3":    {voyageKindText, voyageMatryoshka, 32000, 120_000},
	"voyage-finance-2": {voyageKindText, voyageFixed1024, 32000, 120_000},
	"voyage-law-2":     {voyageKindText, voyageFixed1024, 16000, 120_000},
	"voyage-context-4": {voyageKindContextualized, voyageMatryoshka, 32000, 120_000},
	"voyage-context-3": {voyageKindContextualized, voyageMatryoshka, 32000, 120_000},
}

// VoyageModelNames returns the supported Voyage model names, sorted.
func VoyageModelNames() []string {
	names := make([]string, 0, len(voyageModels))
	for name := range voyageModels {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// VoyageEmbedder implements the Embedder interface for the Voyage AI API.
//
// Search queries (Embed) are sent with input_type "query" and indexed chunks
// (EmbedBatch) with input_type "document". Text models use POST /embeddings.
// The voyage-context-* models use POST /contextualizedembeddings: documents
// are sent as one string each with auto-chunking enabled, queries as a single
// one-element inner list without auto-chunking.
type VoyageEmbedder struct {
	endpoint    string
	model       string
	spec        voyageModel
	apiKey      string
	dimensions  *int
	client      *http.Client
	retryPolicy RetryPolicy
}

type voyageEmbedRequest struct {
	Model           string   `json:"model"`
	Input           []string `json:"input"`
	InputType       string   `json:"input_type"`
	OutputDimension *int     `json:"output_dimension,omitempty"`
}

type voyageEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

type voyageContextRequest struct {
	Model              string `json:"model"`
	Inputs             any    `json:"inputs"`
	InputType          string `json:"input_type"`
	OutputDimension    *int   `json:"output_dimension,omitempty"`
	EnableAutoChunking bool   `json:"enable_auto_chunking,omitempty"`
	ChunkSize          int    `json:"chunk_size,omitempty"`
}

type voyageContextResponse struct {
	Data []struct {
		Index int `json:"index"`
		Data  []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	} `json:"data"`
}

// voyageErrorResponse is the error body of the Voyage API: {"detail": "..."},
// sometimes with an "error_code".
type voyageErrorResponse struct {
	Detail    json.RawMessage `json:"detail"`
	ErrorCode string          `json:"error_code"`
}

type VoyageOption func(*VoyageEmbedder)

func WithVoyageEndpoint(endpoint string) VoyageOption {
	return func(e *VoyageEmbedder) {
		if endpoint != "" {
			e.endpoint = strings.TrimRight(endpoint, "/")
		}
	}
}

// WithVoyageModel sets the model. It is required: there is no default.
func WithVoyageModel(model string) VoyageOption {
	return func(e *VoyageEmbedder) {
		e.model = strings.TrimSpace(model)
	}
}

func WithVoyageKey(key string) VoyageOption {
	return func(e *VoyageEmbedder) {
		e.apiKey = key
	}
}

// WithVoyageDimensions requests a specific output_dimension. Without it the
// parameter is not sent and the model returns its native size.
func WithVoyageDimensions(dimensions int) VoyageOption {
	return func(e *VoyageEmbedder) {
		e.dimensions = &dimensions
	}
}

// WithVoyageTimeout overrides the HTTP client timeout. Values <= 0 are ignored.
func WithVoyageTimeout(d time.Duration) VoyageOption {
	return func(e *VoyageEmbedder) {
		if d > 0 && e.client != nil {
			e.client.Timeout = d
		}
	}
}

// WithVoyageRetryPolicy overrides the backoff used for HTTP 429 and 5xx.
func WithVoyageRetryPolicy(policy RetryPolicy) VoyageOption {
	return func(e *VoyageEmbedder) {
		e.retryPolicy = policy
	}
}

// WithVoyageMaxRetries caps the retries for HTTP 429 and 5xx. Values <= 0
// are ignored, preserving the default.
func WithVoyageMaxRetries(n int) VoyageOption {
	return func(e *VoyageEmbedder) {
		if n > 0 {
			e.retryPolicy.MaxAttempts = n
		}
	}
}

func NewVoyageEmbedder(opts ...VoyageOption) (*VoyageEmbedder, error) {
	e := &VoyageEmbedder{
		endpoint:    defaultVoyageEndpoint,
		client:      &http.Client{Timeout: 60 * time.Second},
		retryPolicy: DefaultRetryPolicy(),
	}
	for _, opt := range opts {
		opt(e)
	}

	if e.model == "" {
		return nil, fmt.Errorf("voyage requires embedder.model: there is no default model, set it to one of %s", strings.Join(VoyageModelNames(), ", "))
	}
	spec, ok := voyageModels[e.model]
	if !ok {
		return nil, fmt.Errorf("unknown voyage model %q: supported models are %s", e.model, strings.Join(VoyageModelNames(), ", "))
	}
	e.spec = spec

	if e.dimensions != nil && !containsInt(spec.dimensions, *e.dimensions) {
		return nil, fmt.Errorf("voyage model %s does not support embedder.dimensions %d: supported values are %s",
			e.model, *e.dimensions, joinInts(spec.dimensions))
	}

	// Fall back to the environment when no key is configured. The
	// environment key is only sent to the provider's official endpoint.
	if e.apiKey == "" {
		key, err := apiKeyFromEnv(e.endpoint, defaultVoyageEndpoint, "VOYAGE_API_KEY")
		if err != nil {
			return nil, err
		}
		e.apiKey = key
	}
	if e.apiKey == "" {
		return nil, fmt.Errorf("voyage API key not set: set embedder.api_key in the grepai config, or VOYAGE_API_KEY when using the official endpoint")
	}

	return e, nil
}

// Embed embeds a search query (input_type "query").
func (e *VoyageEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	var (
		vectors [][]float32
		err     error
	)
	if e.spec.kind == voyageKindContextualized {
		vectors, err = e.contextualizedRequest(ctx, []string{text}, "query", 0)
	} else {
		vectors, err = e.textRequest(ctx, []string{text}, "query")
	}
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedBatch embeds indexed chunks (input_type "document"). Requests are split
// to stay within the 1000-input limit and the model's per-request token cap.
func (e *VoyageEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for _, r := range e.splitForLimits(texts) {
		vectors, err := e.embedDocuments(ctx, texts[r[0]:r[1]], r[0])
		if err != nil {
			return nil, err
		}
		out = append(out, vectors...)
	}
	return out, nil
}

// embedDocuments sends one document request. When the API reports that the
// batch is too large it is halved and retried. offset is the position of
// texts[0] in the caller's slice, used to report which chunk is at fault.
func (e *VoyageEmbedder) embedDocuments(ctx context.Context, texts []string, offset int) ([][]float32, error) {
	var (
		vectors [][]float32
		err     error
	)
	if e.spec.kind == voyageKindContextualized {
		vectors, err = e.contextualizedRequest(ctx, texts, "document", offset)
	} else {
		vectors, err = e.textRequest(ctx, texts, "document")
	}
	var apiErr *voyageAPIError
	if err != nil && errors.As(err, &apiErr) && apiErr.batchTooLarge && len(texts) > 1 {
		mid := len(texts) / 2
		left, lerr := e.embedDocuments(ctx, texts[:mid], offset)
		if lerr != nil {
			return nil, lerr
		}
		right, rerr := e.embedDocuments(ctx, texts[mid:], offset+mid)
		if rerr != nil {
			return nil, rerr
		}
		return append(left, right...), nil
	}
	return vectors, err
}

// splitForLimits returns [start, end) ranges of texts that each respect the
// input-count limit and, when the model documents one, the token cap.
func (e *VoyageEmbedder) splitForLimits(texts []string) [][2]int {
	budget := 0
	if e.spec.maxBatchTokens > 0 {
		budget = e.spec.maxBatchTokens * voyageBatchBudgetPercent / 100
	}
	var ranges [][2]int
	start, tokens := 0, 0
	for i, text := range texts {
		// Inputs longer than the context window are truncated by the API,
		// so one input never counts for more than the window.
		t := min((len(text)+2)/3, e.spec.contextTokens)
		if i > start && (i-start >= voyageMaxInputs || (budget > 0 && tokens+t > budget)) {
			ranges = append(ranges, [2]int{start, i})
			start, tokens = i, 0
		}
		tokens += t
	}
	return append(ranges, [2]int{start, len(texts)})
}

func (e *VoyageEmbedder) textRequest(ctx context.Context, texts []string, inputType string) ([][]float32, error) {
	body, err := e.post(ctx, "/embeddings", voyageEmbedRequest{
		Model:           e.model,
		Input:           texts,
		InputType:       inputType,
		OutputDimension: e.dimensions,
	})
	if err != nil {
		return nil, err
	}

	var result voyageEmbedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to decode voyage response: %w", err)
	}
	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("voyage returned %d embeddings for %d inputs", len(result.Data), len(texts))
	}
	vectors := make([][]float32, len(texts))
	for _, item := range result.Data {
		if item.Index < 0 || item.Index >= len(texts) || vectors[item.Index] != nil {
			return nil, fmt.Errorf("voyage returned an invalid embedding index %d", item.Index)
		}
		if err := e.checkDimensions(item.Embedding); err != nil {
			return nil, err
		}
		vectors[item.Index] = item.Embedding
	}
	return vectors, nil
}

// contextualizedRequest calls /contextualizedembeddings. Documents are sent as
// a flat list with one string per document and auto-chunking enabled (the API
// only auto-chunks documents). Queries are sent as one inner list holding the
// query, without auto-chunking.
func (e *VoyageEmbedder) contextualizedRequest(ctx context.Context, texts []string, inputType string, offset int) ([][]float32, error) {
	req := voyageContextRequest{
		Model:           e.model,
		InputType:       inputType,
		OutputDimension: e.dimensions,
	}
	if inputType == "document" {
		req.Inputs = texts
		req.EnableAutoChunking = true
		req.ChunkSize = voyageContextChunkSize
	} else {
		req.Inputs = [][]string{texts}
	}

	body, err := e.post(ctx, "/contextualizedembeddings", req)
	if err != nil {
		return nil, err
	}

	var result voyageContextResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to decode voyage response: %w", err)
	}
	if inputType != "document" {
		if len(result.Data) != 1 {
			return nil, fmt.Errorf("voyage returned %d result lists for 1 query list", len(result.Data))
		}
		// One inner list holding the queries: one embedding per query.
		chunks := result.Data[0].Data
		if len(chunks) != len(texts) {
			return nil, fmt.Errorf("voyage returned %d embeddings for %d inputs", len(chunks), len(texts))
		}
		vectors := make([][]float32, len(texts))
		for _, c := range chunks {
			if c.Index < 0 || c.Index >= len(texts) || vectors[c.Index] != nil {
				return nil, fmt.Errorf("voyage returned an invalid embedding index %d", c.Index)
			}
			if err := e.checkDimensions(c.Embedding); err != nil {
				return nil, err
			}
			vectors[c.Index] = c.Embedding
		}
		return vectors, nil
	}

	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("voyage returned %d documents for %d inputs", len(result.Data), len(texts))
	}
	vectors := make([][]float32, len(texts))
	for _, doc := range result.Data {
		if doc.Index < 0 || doc.Index >= len(texts) || vectors[doc.Index] != nil {
			return nil, fmt.Errorf("voyage returned an invalid document index %d", doc.Index)
		}
		// grepai stores one vector per chunk it made itself. A document the
		// API split further cannot be mapped to one vector: ask the indexer
		// to re-chunk it into smaller pieces.
		if len(doc.Data) != 1 {
			return nil, NewContextLengthError(offset+doc.Index, 0, voyageContextChunkSize,
				fmt.Sprintf("voyage split the input into %d chunks, expected exactly 1", len(doc.Data)))
		}
		if err := e.checkDimensions(doc.Data[0].Embedding); err != nil {
			return nil, err
		}
		vectors[doc.Index] = doc.Data[0].Embedding
	}
	return vectors, nil
}

func (e *VoyageEmbedder) checkDimensions(vector []float32) error {
	if len(vector) != e.Dimensions() {
		return fmt.Errorf("voyage returned a %d-dimension vector, expected %d", len(vector), e.Dimensions())
	}
	return nil
}

// voyageAPIError is a non-200 answer from the Voyage API.
type voyageAPIError struct {
	status        int
	message       string
	batchTooLarge bool
}

func (e *voyageAPIError) Error() string {
	return fmt.Sprintf("voyage API error (status %d): %s", e.status, e.message)
}

// post sends a JSON request and returns the body of a 200 answer. HTTP 429 and
// 5xx answers are retried with exponential backoff.
func (e *VoyageEmbedder) post(ctx context.Context, path string, payload any) ([]byte, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint+path, bytes.NewReader(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+e.apiKey)

		resp, err := e.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to send request to Voyage: %w", err)
		}
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("failed to read response: %w", readErr)
		}
		if resp.StatusCode == http.StatusOK {
			return body, nil
		}

		apiErr := parseVoyageError(resp.StatusCode, body)
		if !IsRetryable(resp.StatusCode) || !e.retryPolicy.ShouldRetry(attempt) {
			return nil, apiErr
		}
		timer := time.NewTimer(e.retryPolicy.Calculate(attempt))
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

// parseVoyageError extracts the message of a Voyage error body, whose shape is
// {"detail": "<text>"}. Any other body is reported verbatim.
func parseVoyageError(status int, body []byte) *voyageAPIError {
	msg := strings.TrimSpace(string(body))
	var parsed voyageErrorResponse
	code := ""
	if json.Unmarshal(body, &parsed) == nil && len(parsed.Detail) > 0 {
		var detail string
		if json.Unmarshal(parsed.Detail, &detail) == nil {
			msg = detail
		} else {
			msg = string(parsed.Detail)
		}
		code = parsed.ErrorCode
	}
	return &voyageAPIError{
		status:  status,
		message: msg,
		batchTooLarge: status == http.StatusBadRequest &&
			(code == "TOO_MANY_TOKENS_IN_BATCH" || strings.Contains(msg, "batch size limit")),
	}
}

func (e *VoyageEmbedder) Dimensions() int {
	if e.dimensions == nil {
		return voyageDefaultDimensions
	}
	return *e.dimensions
}

func (e *VoyageEmbedder) Close() error {
	return nil
}

func containsInt(values []int, v int) bool {
	for _, x := range values {
		if x == v {
			return true
		}
	}
	return false
}

func joinInts(values []int) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmt.Sprint(v)
	}
	return strings.Join(parts, ", ")
}
