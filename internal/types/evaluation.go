package types

import (
	"embed"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/yanyiwu/gojieba"
)

//go:embed dict/*.utf8
var jiebaDictFS embed.FS

// Jieba is the global Chinese segmentation instance, initialized lazily by
// GetJieba. The previous eager package-level init crashed the Windows desktop
// build: wails binding generation runs package init from an unexpected working
// directory, gojieba's default ./dict paths are missing, and the C++ exception
// raised across the cgo boundary aborts the process. GetJieba extracts the
// embedded dictionaries to a local cache directory and constructs Jieba with
// absolute paths, which is safe in any working directory.
var (
	jiebaOnce sync.Once
	Jieba     *gojieba.Jieba
)

// GetJieba returns the lazily initialized global Jieba instance.
func GetJieba() *gojieba.Jieba {
	jiebaOnce.Do(func() {
		Jieba = newJieba()
	})
	return Jieba
}

var jiebaDictFiles = []string{
	"jieba.dict.utf8",
	"hmm_model.utf8",
	"user.dict.utf8",
	"idf.utf8",
	"stop_words.utf8",
}

func newJieba() *gojieba.Jieba {
	dictDir := os.Getenv("JIEBA_DICT_DIR")
	if dictDir == "" {
		dictDir = extractEmbeddedDicts()
		if dictDir == "" {
			// Last resort: default behavior (requires ./dict in CWD).
			return gojieba.NewJieba()
		}
	}

	return gojieba.NewJieba(
		filepath.Join(dictDir, "jieba.dict.utf8"),
		filepath.Join(dictDir, "hmm_model.utf8"),
		filepath.Join(dictDir, "user.dict.utf8"),
		filepath.Join(dictDir, "idf.utf8"),
		filepath.Join(dictDir, "stop_words.utf8"),
	)
}

// extractEmbeddedDicts materializes the embedded dictionaries under the
// system temp directory and returns that directory.
func extractEmbeddedDicts() string {
	cacheDir := filepath.Join(os.TempDir(), "weknora-jieba-dict")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return ""
	}
	for _, name := range jiebaDictFiles {
		dst := filepath.Join(cacheDir, name)
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		data, err := jiebaDictFS.ReadFile("dict/" + name)
		if err != nil {
			return ""
		}
		if err := os.WriteFile(dst, data, 0o644); err != nil {
			return ""
		}
	}
	return cacheDir
}
// EvaluationStatue represents the status of an evaluation task
type EvaluationStatue int

const (
	EvaluationStatuePending EvaluationStatue = iota // Task is waiting to start
	EvaluationStatueRunning                         // Task is in progress
	EvaluationStatueSuccess                         // Task completed successfully
	EvaluationStatueFailed                          // Task failed
)

// EvaluationTask contains information about an evaluation task
type EvaluationTask struct {
	ID        string `json:"id"`         // Unique task ID
	TenantID  uint64 `json:"tenant_id"`  // Tenant/Organization ID
	DatasetID string `json:"dataset_id"` // Dataset ID for evaluation

	StartTime time.Time        `json:"start_time"`        // Task start time
	Status    EvaluationStatue `json:"status"`            // Current task status
	ErrMsg    string           `json:"err_msg,omitempty"` // Error message if failed

	Total    int `json:"total,omitempty"`    // Total items to evaluate
	Finished int `json:"finished,omitempty"` // Completed items count
}

// EvaluationDetail contains detailed evaluation information
type EvaluationDetail struct {
	Task   *EvaluationTask `json:"task"`             // Evaluation task info
	Params *ChatManage     `json:"params"`           // Evaluation parameters
	Metric *MetricResult   `json:"metric,omitempty"` // Evaluation metrics
}

// String returns JSON representation of EvaluationTask
func (e *EvaluationTask) String() string {
	b, _ := json.Marshal(e)
	return string(b)
}

// MetricInput contains input data for metric calculation
type MetricInput struct {
	RetrievalGT  [][]int // Ground truth for retrieval
	RetrievalIDs []int   // Retrieved IDs

	GeneratedTexts string // Generated text for evaluation
	GeneratedGT    string // Ground truth text for comparison
}

// MetricResult contains evaluation metrics
type MetricResult struct {
	RetrievalMetrics  RetrievalMetrics  `json:"retrieval_metrics"`  // Retrieval performance metrics
	GenerationMetrics GenerationMetrics `json:"generation_metrics"` // Text generation quality metrics
}

// RetrievalMetrics contains metrics for retrieval evaluation
type RetrievalMetrics struct {
	Precision float64 `json:"precision"` // Precision score
	Recall    float64 `json:"recall"`    // Recall score

	NDCG3  float64 `json:"ndcg3"`  // Normalized Discounted Cumulative Gain at 3
	NDCG10 float64 `json:"ndcg10"` // Normalized Discounted Cumulative Gain at 10
	MRR    float64 `json:"mrr"`    // Mean Reciprocal Rank
	MAP    float64 `json:"map"`    // Mean Average Precision
}

// GenerationMetrics contains metrics for text generation evaluation
type GenerationMetrics struct {
	BLEU1 float64 `json:"bleu1"` // BLEU-1 score
	BLEU2 float64 `json:"bleu2"` // BLEU-2 score
	BLEU4 float64 `json:"bleu4"` // BLEU-4 score

	ROUGE1 float64 `json:"rouge1"` // ROUGE-1 score
	ROUGE2 float64 `json:"rouge2"` // ROUGE-2 score
	ROUGEL float64 `json:"rougel"` // ROUGE-L score
}

// EvalState represents different stages of evaluation process
type EvalState int

const (
	StateBegin             EvalState = iota // Evaluation started
	StateAfterQaPairs                       // After loading QA pairs
	StateAfterDataset                       // After processing dataset
	StateAfterEmbedding                     // After generating embeddings
	StateAfterVectorSearch                  // After vector search
	StateAfterRerank                        // After reranking
	StateAfterComplete                      // After completion
	StateEnd                                // Evaluation ended
)
