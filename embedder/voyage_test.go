package embedder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yoanbernabeu/grepai/config"
)

const voyageTestKey = "test-voyage-key"

func voyageVector(dim int) []float32 {
	return make([]float32, dim)
}

// newVoyageTestEmbedder returns an embedder pointed at a test server running
// handler. The fast retry policy keeps retry tests quick.
func newVoyageTestEmbedder(t *testing.T, model string, handler http.HandlerFunc, opts ...VoyageOption) *VoyageEmbedder {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	all := append([]VoyageOption{
		WithVoyageModel(model),
		WithVoyageKey(voyageTestKey),
		WithVoyageEndpoint(srv.URL),
		WithVoyageRetryPolicy(RetryPolicy{BaseDelay: time.Millisecond, Multiplier: 1, MaxDelay: time.Millisecond, MaxAttempts: 2}),
	}, opts...)
	e, err := NewVoyageEmbedder(all...)
	if err != nil {
		t.Fatalf("NewVoyageEmbedder: %v", err)
	}
	return e
}

// textEmbeddingsHandler answers /embeddings with one vector per input and
// records the decoded requests.
func textEmbeddingsHandler(dim int, requests *[]map[string]any) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		*requests = append(*requests, req)
		inputs := req["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i := range inputs {
			data[i] = map[string]any{"embedding": voyageVector(dim), "index": i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}
}

func TestNewVoyageEmbedder_RequiresExplicitModel(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	_, err := NewVoyageEmbedder()
	if err == nil || !strings.Contains(err.Error(), "no default model") {
		t.Fatalf("expected a no-default-model error, got %v", err)
	}
}

func TestNewVoyageEmbedder_RejectsUnknownAndForeignModels(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	for _, model := range []string{"voyage-nope", "text-embedding-3-small", "rerank-3"} {
		_, err := NewVoyageEmbedder(WithVoyageModel(model))
		if err == nil || !strings.Contains(err.Error(), "unknown voyage model") {
			t.Errorf("model %q: expected an unknown-model error, got %v", model, err)
		}
	}
}

func TestNewVoyageEmbedder_RejectsUnsupportedDimensions(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	tests := []struct {
		model string
		dim   int
		ok    bool
	}{
		{"voyage-4-lite", 256, true},
		{"voyage-4-lite", 2048, true},
		{"voyage-4-lite", 768, false},
		{"voyage-code-4", 512, true},
		{"voyage-context-4", 2048, true},
		{"voyage-finance-2", 1024, true},
		{"voyage-finance-2", 256, false},
		{"voyage-law-2", 512, false},
	}
	for _, tt := range tests {
		_, err := NewVoyageEmbedder(WithVoyageModel(tt.model), WithVoyageDimensions(tt.dim))
		if tt.ok && err != nil {
			t.Errorf("%s/%d: unexpected error %v", tt.model, tt.dim, err)
		}
		if !tt.ok && (err == nil || !strings.Contains(err.Error(), "does not support")) {
			t.Errorf("%s/%d: expected an unsupported-dimensions error, got %v", tt.model, tt.dim, err)
		}
	}
}

func TestNewVoyageEmbedder_RequiresAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	_, err := NewVoyageEmbedder(WithVoyageModel("voyage-4-lite"))
	if err == nil || !strings.Contains(err.Error(), "VOYAGE_API_KEY") {
		t.Fatalf("expected a missing-key error, got %v", err)
	}
}

func TestNewVoyageEmbedder_EnvKeyNotSentToCustomEndpoint(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-voyage-secret")
	_, err := NewVoyageEmbedder(WithVoyageModel("voyage-4-lite"), WithVoyageEndpoint("https://user:pw@proxy.example/v1?k=query-secret"))
	if err == nil {
		t.Fatal("expected the environment key to be refused for a custom endpoint")
	}
	for _, leak := range []string{"env-voyage-secret", "user", "pw", "query-secret"} {
		if strings.Contains(err.Error(), leak) {
			t.Errorf("error leaks %q: %v", leak, err)
		}
	}
}

func TestVoyageEmbedder_TextRequestShape(t *testing.T) {
	var requests []map[string]any
	e := newVoyageTestEmbedder(t, "voyage-4-lite", textEmbeddingsHandler(1024, &requests))

	if _, err := e.Embed(context.Background(), "query text"); err != nil {
		t.Fatal(err)
	}
	if _, err := e.EmbedBatch(context.Background(), []string{"a", "b"}); err != nil {
		t.Fatal(err)
	}

	if got := requests[0]["input_type"]; got != "query" {
		t.Errorf("Embed input_type = %v, want query", got)
	}
	if got := requests[1]["input_type"]; got != "document" {
		t.Errorf("EmbedBatch input_type = %v, want document", got)
	}
	for i, req := range requests {
		if _, present := req["output_dimension"]; present {
			t.Errorf("request %d sent output_dimension although none was requested", i)
		}
	}
}

func TestVoyageEmbedder_SendsOutputDimensionOnlyWhenRequested(t *testing.T) {
	var requests []map[string]any
	e := newVoyageTestEmbedder(t, "voyage-4", textEmbeddingsHandler(256, &requests), WithVoyageDimensions(256))
	vec, err := e.Embed(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != 256 || e.Dimensions() != 256 {
		t.Errorf("got %d-dim vector and Dimensions()=%d, want 256", len(vec), e.Dimensions())
	}
	if got := requests[0]["output_dimension"]; got != float64(256) {
		t.Errorf("output_dimension = %v, want 256", got)
	}
}

func TestVoyageEmbedder_ContextualizedRequestShape(t *testing.T) {
	var requests []map[string]any
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/contextualizedembeddings" {
			t.Errorf("path = %s, want /contextualizedembeddings", r.URL.Path)
		}
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		requests = append(requests, req)
		inputs := req["inputs"].([]any)
		var data []map[string]any
		if req["input_type"] == "query" {
			chunks := make([]map[string]any, len(inputs[0].([]any)))
			for i := range chunks {
				chunks[i] = map[string]any{"embedding": voyageVector(1024), "index": i}
			}
			data = []map[string]any{{"index": 0, "data": chunks}}
		} else {
			for i := range inputs {
				data = append(data, map[string]any{"index": i, "data": []map[string]any{
					{"embedding": voyageVector(1024), "index": 0, "text": "chunk"},
				}})
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}
	for _, model := range []string{"voyage-context-4", "voyage-context-3"} {
		requests = nil
		e := newVoyageTestEmbedder(t, model, handler)
		if _, err := e.EmbedBatch(context.Background(), []string{"doc one", "doc two"}); err != nil {
			t.Fatalf("%s document: %v", model, err)
		}
		if _, err := e.Embed(context.Background(), "the query"); err != nil {
			t.Fatalf("%s query: %v", model, err)
		}

		doc, query := requests[0], requests[1]
		if doc["input_type"] != "document" || doc["enable_auto_chunking"] != true || doc["chunk_size"] != float64(32000) {
			t.Errorf("%s document request = %v", model, doc)
		}
		// Auto-chunking takes one string per document, never nested lists.
		if got := doc["inputs"].([]any); got[0] != "doc one" || got[1] != "doc two" {
			t.Errorf("%s document inputs = %v, want a flat list of strings", model, got)
		}
		if query["input_type"] != "query" {
			t.Errorf("%s query input_type = %v", model, query["input_type"])
		}
		if _, present := query["enable_auto_chunking"]; present {
			t.Errorf("%s query request enables auto-chunking, which the API rejects", model)
		}
		if _, present := query["chunk_size"]; present {
			t.Errorf("%s query request sends chunk_size", model)
		}
		inner, ok := query["inputs"].([]any)[0].([]any)
		if !ok || len(inner) != 1 || inner[0] != "the query" {
			t.Errorf("%s query inputs = %v, want [[query]]", model, query["inputs"])
		}
	}
}

func TestVoyageEmbedder_ContextualizedSplitDocumentAsksForRechunk(t *testing.T) {
	e := newVoyageTestEmbedder(t, "voyage-context-4", func(w http.ResponseWriter, r *http.Request) {
		// The second document comes back as two chunks.
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{
			{"index": 0, "data": []map[string]any{{"embedding": voyageVector(1024), "index": 0}}},
			{"index": 1, "data": []map[string]any{
				{"embedding": voyageVector(1024), "index": 0},
				{"embedding": voyageVector(1024), "index": 1},
			}},
		}})
	})
	_, err := e.EmbedBatch(context.Background(), []string{"fits", "too long"})
	ctxErr := AsContextLengthError(err)
	if ctxErr == nil || ctxErr.ChunkIndex != 1 {
		t.Fatalf("expected a context length error for chunk 1, got %v", err)
	}
}

func TestVoyageEmbedder_ErrorDetailExtraction(t *testing.T) {
	e := newVoyageTestEmbedder(t, "voyage-4-lite", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"detail":"Value 'bogus' supplied for argument 'input_type' is not valid"}`))
	})
	_, err := e.Embed(context.Background(), "q")
	if err == nil || !strings.Contains(err.Error(), "status 400") || !strings.Contains(err.Error(), "'input_type' is not valid") {
		t.Fatalf("expected the detail text in the error, got %v", err)
	}
	if strings.Contains(err.Error(), `"detail"`) || strings.Contains(err.Error(), voyageTestKey) {
		t.Errorf("error carries raw JSON or the key: %v", err)
	}
}

func TestVoyageEmbedder_ErrorNonJSONBodyReportedVerbatim(t *testing.T) {
	e := newVoyageTestEmbedder(t, "voyage-4-lite", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("Unauthorized"))
	})
	_, err := e.Embed(context.Background(), "q")
	if err == nil || !strings.Contains(err.Error(), "Unauthorized") {
		t.Fatalf("expected the body in the error, got %v", err)
	}
}

func TestVoyageEmbedder_RetriesTransientErrorsOnly(t *testing.T) {
	var calls atomic.Int32
	var requests []map[string]any
	ok := textEmbeddingsHandler(1024, &requests)
	e := newVoyageTestEmbedder(t, "voyage-4-lite", func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"detail":"rate limited"}`))
			return
		}
		ok(w, r)
	})
	if _, err := e.Embed(context.Background(), "q"); err != nil {
		t.Fatalf("429 should be retried: %v", err)
	}
	if calls.Load() != 2 {
		t.Errorf("calls = %d, want 2", calls.Load())
	}

	calls.Store(0)
	bad := newVoyageTestEmbedder(t, "voyage-4-lite", func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"detail":"bad"}`))
	})
	if _, err := bad.Embed(context.Background(), "q"); err == nil {
		t.Fatal("expected an error")
	}
	if calls.Load() != 1 {
		t.Errorf("a 400 was retried: calls = %d", calls.Load())
	}
}

func TestVoyageEmbedder_RetryGivesUpAndHonorsContext(t *testing.T) {
	var calls atomic.Int32
	e := newVoyageTestEmbedder(t, "voyage-4-lite", func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"detail":"down"}`))
	})
	if _, err := e.Embed(context.Background(), "q"); err == nil || !strings.Contains(err.Error(), "status 503") {
		t.Fatalf("expected the final 503, got %v", err)
	}
	if calls.Load() != 3 { // first attempt plus MaxAttempts retries
		t.Errorf("calls = %d, want 3", calls.Load())
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.Embed(ctx, "q"); err == nil {
		t.Fatal("expected an error from a cancelled context")
	}
}

func TestVoyageEmbedder_SplitsBatchesAtInputLimit(t *testing.T) {
	var requests []map[string]any
	e := newVoyageTestEmbedder(t, "voyage-4-lite", textEmbeddingsHandler(1024, &requests))
	texts := make([]string, 2500)
	for i := range texts {
		texts[i] = "x"
	}
	vectors, err := e.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 2500 || len(requests) != 3 {
		t.Fatalf("vectors=%d requests=%d, want 2500 and 3", len(vectors), len(requests))
	}
	for i, req := range requests {
		if n := len(req["input"].([]any)); n > 1000 {
			t.Errorf("request %d has %d inputs, over the limit of 1000", i, n)
		}
	}
}

func TestVoyageEmbedder_SplitsBatchesAtTokenCap(t *testing.T) {
	// voyage-4-large caps a request at 120K tokens; 30K-token inputs
	// (about 90K characters by the local estimate) must not share a request.
	var requests []map[string]any
	e := newVoyageTestEmbedder(t, "voyage-4-large", textEmbeddingsHandler(1024, &requests))
	big := strings.Repeat("a", 90_000)
	if _, err := e.EmbedBatch(context.Background(), []string{big, big, big, big, big}); err != nil {
		t.Fatal(err)
	}
	if len(requests) < 2 {
		t.Fatalf("requests = %d, want the batch split by token cap", len(requests))
	}
	for i, req := range requests {
		if n := len(req["input"].([]any)); n > 3 {
			t.Errorf("request %d has %d inputs, over the 120K-token budget", i, n)
		}
	}
}

func TestVoyageEmbedder_NoTokenCapInventedForCode4(t *testing.T) {
	var requests []map[string]any
	e := newVoyageTestEmbedder(t, "voyage-code-4", textEmbeddingsHandler(1024, &requests))
	big := strings.Repeat("a", 90_000)
	if _, err := e.EmbedBatch(context.Background(), []string{big, big, big, big, big}); err != nil {
		t.Fatal(err)
	}
	if len(requests) != 1 {
		t.Errorf("requests = %d, want 1: voyage-code-4 documents no token cap", len(requests))
	}
}

func TestVoyageEmbedder_HalvesBatchTheAPIRejectsAsTooLarge(t *testing.T) {
	var calls atomic.Int32
	e := newVoyageTestEmbedder(t, "voyage-code-4", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		calls.Add(1)
		if len(req["input"].([]any)) > 2 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"detail":"The max allowed tokens per submitted batch is 120000.","error_code":"TOO_MANY_TOKENS_IN_BATCH"}`))
			return
		}
		inputs := req["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i := range inputs {
			data[i] = map[string]any{"embedding": voyageVector(1024), "index": i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	})
	vectors, err := e.EmbedBatch(context.Background(), []string{"a", "b", "c", "d", "e"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 5 {
		t.Fatalf("vectors = %d, want 5", len(vectors))
	}
}

func TestVoyageEmbedder_FailsClosedOnMalformedResponses(t *testing.T) {
	respond := func(body string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(body)) }
	}
	tests := []struct {
		name string
		body string
		want string
	}{
		{"fewer vectors than inputs", `{"data":[]}`, "0 embeddings for 1 inputs"},
		{"wrong vector size", `{"data":[{"embedding":[1],"index":0}]}`, "1-dimension vector"},
		{"out-of-range index", `{"data":[{"embedding":[],"index":5}]}`, "invalid embedding index"},
		{"not JSON", `nope`, "failed to decode"},
	}
	for _, tt := range tests {
		e := newVoyageTestEmbedder(t, "voyage-4-lite", respond(tt.body))
		if _, err := e.Embed(context.Background(), "q"); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: expected %q, got %v", tt.name, tt.want, err)
		}
	}
}

func TestNewFromConfig_VoyageRequiresModel(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "test-key")
	cfg := &config.Config{Embedder: config.EmbedderConfig{Provider: "voyage"}}
	if _, err := NewFromConfig(cfg); err == nil || !strings.Contains(err.Error(), "no default model") {
		t.Fatalf("expected a no-default-model error, got %v", err)
	}

	cfg.Embedder.Model = "voyage-code-4"
	cfg.Embedder.Endpoint = config.DefaultVoyageEndpoint
	emb, err := NewFromConfig(cfg)
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	if _, ok := emb.(*VoyageEmbedder); !ok {
		t.Fatalf("expected *VoyageEmbedder, got %T", emb)
	}
	if emb.Dimensions() != cfg.Embedder.GetDimensions() {
		t.Errorf("embedder dimensions %d differ from config dimensions %d", emb.Dimensions(), cfg.Embedder.GetDimensions())
	}
}

func TestParseVoyageError_RecognizesBatchLimits(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"token cap with code", 400, `{"detail":"Request to model 'voyage-4-large' failed. The max allowed tokens per submitted batch is 120000. Your batch has 205712 tokens after truncation.","error_code":"TOO_MANY_TOKENS_IN_BATCH"}`, true},
		{"token cap detail only", 400, `{"detail":"The max allowed tokens per submitted batch is 120000."}`, true},
		{"input count", 400, `{"detail":"The batch size limit is 1000. Your batch size is 1001."}`, true},
		{"other 400", 400, `{"detail":"Model nope is not supported."}`, false},
		{"not a 400", 500, `{"detail":"The batch size limit is 1000."}`, false},
	}
	for _, tt := range tests {
		if got := parseVoyageError(tt.status, []byte(tt.body)).batchTooLarge; got != tt.want {
			t.Errorf("%s: batchTooLarge = %v, want %v", tt.name, got, tt.want)
		}
	}
}
