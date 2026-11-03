# Báo Cáo Tiến Độ Hệ Thống Truy Hồi Văn Bản Pháp Luật Tiếng Việt

[English version](benchmark-report.en.md)

Trọng tâm: Hybrid Search (Dense + BM25) + BGE Reranker

---

## 1. So Sánh Các Cấu Hình Benchmark

**Tổng số câu hỏi:** 1.238

### Bảng so sánh

| Config | R@1 | R@3 | R@10 | MRR@10 | nDCG@10 | Norm P@10 | Tổng TGian | TB/câu hỏi | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Không reranker** | **0.7916** | 0.9249 | 0.9693 | **0.8607** | **0.8436** | 0.9142 | 954.65s | 771.12ms | **1.30 q/s** |
| Global rerank top 10, blend 0.5 | 0.7746 | 0.8853 | 0.9047 | 0.8293 | 0.8037 | 0.8464 | 2532.90s | 2045.96ms | 0.49 q/s |
| Global rerank top 30, blend 0.5 | 0.7746 | 0.8853 | 0.9079 | 0.8295 | 0.8050 | 0.8505 | 1456.48s | 1176.48ms | 0.85 q/s |
| Global rerank max_cand=3, blend 0.5 | 0.7771 | 0.8772 | 0.9241 | 0.8319 | 0.8111 | 0.8677 | 4365.45s | 3526.21ms | 0.28 q/s |
| **Reorder window top 3, blend 1.0** | 0.7835 | **0.9249** | **0.9693** | 0.8579 | 0.8428 | **0.9142** | 1433.04s | 1157.54ms | 0.86 q/s |

### Nhận xét chính

1. **Không reranker** vẫn là cấu hình tốt nhất: Recall@1 cao nhất (**0.7916**), MRR@10 và nDCG@10 cao nhất, độ trễ thấp nhất.
2. **Global rerank (mọi biến thể) đều gây hại**: R@3 giảm từ 0.9249 → ~0.88, R@10 giảm từ 0.9693 → ~0.90-0.92. Nguyên nhân: reranker xáo trộn toàn bộ danh sách, đưa item không liên quan lên top.
3. **Reorder window top 3 an toàn hơn**: giữ nguyên R@3 và R@10, không gây hại top-10. Tuy nhiên R@1 và MRR@10 vẫn thấp hơn baseline nhẹ (0.7835 vs 0.7916).
4. **Reranker không cải thiện ranking so với hybrid search** dù có tín hiệu phân biệt (gap rel-irr = +0.164).

---

## 2. Chi Tiết Lần Chạy Mới Nhất: Reorder Window Top 3

- **Thời điểm chạy:** 2026-07-27 10:15
- **Cấu hình:**
  - `mode: reorder_window`, `max_candidates: 3`, `top_n: 3`, `blend_weight: 1.0`
  - Conditional reranking: `false`
- **Tổng thời gian:** 1433.04s (~24 phút)
- **Trung bình mỗi câu hỏi:** 1157.54ms
- **Số lần gọi reranker:** 1.238 (attempted), 0 (skipped), 0 (failed)

### Kết quả Metrics

| Chỉ số | Giá trị |
|---|---|
| Recall@1 | **0.7835** (970/1238) |
| Recall@3 | **0.9249** (1145/1238) |
| Recall@10 | **0.9693** (1200/1238) |
| Precision@1 | 0.7835 |
| Precision@3 | 0.3250 |
| Precision@10 | 0.1090 |
| MRR@10 | **0.8579** |
| nDCG@10 | **0.8428** |
| R-Precision | 0.7326 |
| Norm Precision@10 | 0.9142 (trần lý thuyết: 0.1270) |

### So sánh Pre-Reranker vs Post-Reranker

| Metric | Pre-Reranker | Post-Reranker | Delta |
|---|---:|---:|---:|
| MRR@10 | 0.8607 | 0.8579 | **-0.0028** |
| nDCG@10 | 0.8436 | 0.8428 | **-0.0008** |
| Norm Precision@10 | 0.9142 | 0.9142 | **0.0000** |

Reranker làm giảm nhẹ chất lượng ranking ở top đầu.

### Phân tích Chi Tiết Reranker

**Chất lượng candidate (pool 50):**
- Avg candidate recall: **0.9621**
- Oracle Recall@10: **0.9621**
- 13 câu hỏi không có đáp án đúng trong candidate pool → không thể cứu được.

**Tín hiệu API score:**
- Relevant avg API score: **0.9356**
- Irrelevant avg API score: **0.7716**
- Gap (rel − irr): **+0.1640**

Reranker có khả năng phân biệt trung bình, nhưng không đủ ổn định per-query.

**Phân loại từng câu hỏi:**

| Loại | Số lượng | Ý nghĩa |
|---|---|---|
| unchanged_hit | 1004 | Hit@10 không đổi, trước và sau reranker |
| hurt_rank | 100 | Hit@10 giữ nhưng thứ hạng giảm |
| improved_rank | 96 | Hit@10 giữ và thứ hạng tăng |
| unchanged_miss | 25 | Miss@10 không đổi |
| missing_from_candidates | 13 | Không có đáp án trong pool 50 |

Hai chiều hurt_rank (100) và improved_rank (96) gần như cân bằng. Reranker không có lợi thế ròng.

---

## 3. Pipeline Hiện Tại

```
rag_pipeline/
├── __init__.py
├── config.yaml                       # All tunable parameters
│
├── cli/                              # CLI entry points
│   ├── main.py                       #   index / search
│   └── benchmark.py                  #   Full evaluation suite
│
├── config/                           # Configuration
│   └── config.py                     #   Dataclasses + YAML loader (env var resolution)
│
├── data/                             # Data loading
│   └── data_loader.py                #   JSON parser + benchmark ID mapping
│
├── indexing/                         # Indexing pipeline
│   ├── chunker.py                    #   Recursive semantic chunker + disk cache
│   ├── context_generator.py          #   Heuristic or LLM article summarizer
│   ├── embedder.py                   #   Local + API embedding backends
│   └── vector_store.py               #   ChromaDB persistent store
│
├── retrieval/                        # Retrieval pipeline
│   ├── bm25_retriever.py            #   BM25 + pyvi Vietnamese word segmentation
│   ├── query_rewriter.py            #   Vietnamese query normalization
│   ├── reranker.py                  #   BGE API reranker (global_score_sort / reorder_window)
│   └── retriever.py                 #   Hybrid: vector + BM25 + rerank
│
├── shared/                           # Shared types
│   └── pipeline_types.py            #   RetrievalResult dataclass
│
└── requirements.txt
```

Các thư mục dữ liệu / artifact bên ngoài:

| Thư mục | Vai trò |
|---|---|
| `data/` | Input JSON files (45 file `chu_de_*.json`) |
| `benchmarks/input/` | Benchmark question map + Correct ID mapping |
| `benchmarks/results/` | Benchmark run logs (`benchmark_runs.jsonl`, `benchmark_results.json`) |
| `artifacts/cache/` | Precomputed chunk cache (`chunks_cache.pkl`) |
| `artifacts/vector_store/` | ChromaDB persistent store (normal index) |
| `artifacts/vector_store_bench/` | ChromaDB store dùng cho benchmark |

---

## 4. Dataset

Dữ liệu văn bản pháp luật được lấy từ Bộ Pháp điển công khai của Việt Nam. [Cổng thông tin điện tử pháp điển của Bộ Tư pháp](https://phapdien.moj.gov.vn/Pages/chi-tiet-bo-phap-dien.aspx)

| Thuộc tính | Giá trị |
|---|---|
| Nguồn dữ liệu | 45 file JSON (`chu_de_1.json` … `chu_de_45.json`) |
| Định dạng | VBQPPL đã parse: topic → section → chapter → article |
| Tổng số điều | **76.459** |
| Số điều có benchmark ID | **±32.700** (khớp nội dung qua `Correct ID.json`) |
| Số câu hỏi benchmark | 1.238 (từ `1238_question_map_phap_dien.json`) |
| Số luật liên quan duy nhất | 1.325 |

Phân bố số luật đúng trên 1.238 câu hỏi:

| Số luật đúng/câu hỏi | Số câu hỏi |
|---|---|
| 1 | 993 |
| 2 | 190 |
| 3 | 34 |
| 4+ | 21 |
| **Trung bình** | **1,27** |

---

## 5. Chi Tiết Từng Bước Trong Pipeline

### 5.1 Nạp dữ liệu

| Component | Tool | Chi tiết |
|---|---|---|
| Data loader | `data_loader.py` | Làm phẳng hệ thống phân cấp topic → section → chapter → article |
| Mapping ID benchmark | Content matching | `benchmarks/input/Correct ID.json`: `article_text → numeric_id` |
| Tỉ lệ match ID | ±32.700 / 76.459 | Toàn bộ 1.325 ID benchmark được coverage |

### 5.2 Chunking

| Component | Tool | Chi tiết |
|---|---|---|
| Tokenizer | HuggingFace `AutoTokenizer` | Khớp với embedding model |
| Thuật toán chunking | `chunker.py` | Đệ quy Clause → Point → Paragraph → Sentence |
| Clause pattern | Regex `^\d+\.\s` | Các khoản đánh số |
| Point pattern | Regex `^[a-đ]\)\s` | Các điểm đánh chữ |
| Paragraph split | Double-newline boundary | Chia ngữ đoạn |
| Sentence split | Vietnamese sentence boundaries | Fallback cuối cùng |

**Tham số chunking:**

| Parameter | Giá trị | Mô tả |
|---|---|---|
| `max_tokens` | 1024 | Ngưỡng chia |
| `min_tokens` | 30 | Gộp chunk nhỏ với chunk liền kề |
| `overlap_tokens` | 50 | Token overlap tại ranh giới chunk |

**Các loại unit:** ARTICLE, CLAUSE, POINT, PARAGRAPH, SENTENCE

### 5.3 Context Generation

| Component | Tool | Chi tiết |
|---|---|---|
| Phương pháp | Heuristic | Tên điều + câu đầu tiên |
| LLM option | `gemma-4-31B-it` qua API | Không dùng (`use_llm: false`) |
| Tiền tố mỗi chunk | `[context] + [chunk_content]` | Được ghép trước khi embedding |

### 5.4 Embedding

| Component | Tool | Chi tiết |
|---|---|---|
| Backend | API công ty (`requests`) | `POST /v1/embeddings` |
| Model | **Vietnamese_Embedding** | 1024 chiều |
| Base URL | `https://mkp-api.fptcloud.com` | OpenAI-compatible |
| Xác thực | Bearer token | Biến môi trường `$FPT_API_KEY` |
| Max sequence length | 2.048 tokens | Input chunks ≤ 1024 |
| Batch size | 64 texts mỗi API call | |
| Retry | 3 lần, exponential backoff | Xử lý lỗi mạng tạm thời |

**Cô lập embedding:** Trường `benchmark_id` chỉ lưu trong metadata ChromaDB. Model embedding không bao giờ nhận được nó. Chỉ dùng ở bước đánh giá cuối cùng.

### 5.5 Vector Store

| Component | Tool | Chi tiết |
|---|---|---|
| Database | **ChromaDB** (persistent) | SQLite + binary segment files |
| Index type | HNSW | Approximate nearest neighbor |
| Distance metric | Cosine | Vectors được L2-normalized |
| Collection | `vbqppl_legal_chunks` | Lưu trong `artifacts/vector_store_bench/` |

### 5.6 BM25 Sparse Index

| Component | Tool | Chi tiết |
|---|---|---|
| Library | **rank-bm25** | Okapi BM25 |
| Tokenizer | **pyvi** | Vietnamese word segmentation |
| `k1` | 1.2 | Term frequency saturation |
| `b` | 0.75 | Document length normalization |
| Persistence | Pickle file | Lưu cạnh ChromaDB |
| Compound words | `đấu_giá`, `giấy_phép`, `xây_dựng` | pyvi xử lý đúng |

### 5.7 Hybrid Retrieval

#### Vector Search

| Parameter | Giá trị |
|---|---|
| Candidate pool | `candidate_pool_size=50` |
| Similarity metric | Cosine (1 − distance/2) |

#### BM25 Search (song song)

| Parameter | Giá trị |
|---|---|
| Candidate pool | `top_k=50` |
| Scoring | Okapi BM25 (`k1=1.2`, `b=0.75`) |

#### Score Fusion

| Parameter | Giá trị |
|---|---|
| Fusion method | Weighted sum |
| `fusion_weight` | 0.05 (5% BM25, 95% vector) |
| BM25-only candidates | Thêm với score `bm25_score × fusion_weight` |

#### BGE Reranker (tùy chọn)

| Parameter | Giá trị (hiện tại) |
|---|---|
| Mode | `reorder_window` hoặc `global_score_sort` |
| `max_candidates` | 3 (cửa sổ rerank) |
| `top_n` | 3 |
| `blend_weight` | 1.0 |
| Conditional | `false` |

### 5.8 Evaluation

| Component | Chi tiết |
|---|---|
| Metrics | Recall@1/3/10, Precision@1/3/10, MRR@10, nDCG@10, R-Precision, Normalized Precision@10 |
| Phân tích reranker | Oracle recall, API score gap, per-query categorization (hurt/improved/unchanged/missing) |
| Hit definition | Ít nhất 1 trong top-k chunk có `benchmark_id` khớp `relevant_laws` |
| Query embedding | Batch 1.238 query trong một API call |
| Số câu hỏi | 1.238 |

---

## 6. Các Phương Pháp / Thành Phần Đã Thử Hoặc Không Dùng

| Nhóm | Phương pháp | Trạng thái / nhận xét |
|---|---|---|
| **BGE reranker — global (top 10, blend 0.5)** | Rerank toàn bộ candidate, global sort | Đã thử — R@3 giảm 0.9249 → 0.8853, R@10 giảm 0.9693 → 0.9047. **Không dùng.** |
| **BGE reranker — global (top 30, blend 0.5)** | Rerank top 30, global sort | Đã thử — R@3 giảm tương tự. **Không dùng.** |
| **BGE reranker — global (max_cand=3, blend 0.5)** | Rerank top 3 nhưng global sort | Đã thử — R@3 giảm 0.9249 → 0.8772. **Không dùng.** |
| **BGE reranker — reorder window (top 3, blend 1.0)** | Chỉ sắp xếp lại top 3, giữ nguyên phần còn lại | Đã thử — An toàn (R@3 và R@10 không đổi) nhưng R@1 giảm nhẹ (0.7916 → 0.7835). **Không cải thiện.** |
| Query rewriting | Mở rộng từ viết tắt (dk → điều kiện) | `enabled=false` |
| Metadata filtering | Boost điểm theo trạng thái / topic / unit type | `enable_metadata_filtering=false` |
| LLM context generation | Gemma tóm tắt điều luật | `use_llm=false` |
| BM25 tuning | Grid search `k1`, `b`, `fusion_weight` | Đã thử, `fusion_weight=0.05` tốt nhất |
| LLM reranker (Gemma) | Gemma chọn best trong top 3 | Đã loại bỏ |
| MMR reranking | Diversity-based reranking | Đã loại bỏ |

---

## 7. Tham Số Chi Tiết

| Nhóm | Tham số | Giá trị |
|---|---|---|
| **Chunking** | `max_tokens` | 1024 |
| | `min_tokens` | 30 |
| | `overlap_tokens` | 50 |
| **Embedding** | Backend | API |
| | Model | `Vietnamese_Embedding` |
| | Dimension | 1.024 |
| | Max seq length | 2.048 |
| | Batch size API | 64 |
| | Retries | 3 (exponential backoff) |
| **Vector Store** | Database | ChromaDB (HNSW) |
| | Distance | Cosine |
| **BM25** | `k1` | 1.2 |
| | `b` | 0.75 |
| | `fusion_weight` | 0.05 |
| | `top_k` (candidate) | 50 |
| **Retrieval** | `candidate_pool_size` | 50 |
| | `top_k` | 10 |
| | `similarity_threshold` | 0.0 |
| **Context** | `use_llm` | false (heuristic) |
| **Query Rewriter** | `enabled` | false |

---

## 8. Kết Luận Hiện Tại

1. **Hybrid search (dense vector + BM25 weighted fusion) không reranker** vẫn là pipeline tốt nhất: R@1=0.7916, R@3=0.9249, R@10=0.9693, latency 771ms/query.
2. **Recall@10 = 0.9693** — retrieval tạo candidate rất mạnh, gần như luôn đưa được tài liệu liên quan vào top 10.
3. **Candidate oracle = 0.9621** — bottleneck là ranking, không phải độ phủ.
4. **Khoảng cách R@1 → R@10 (~0,18)** — vấn đề chính là thứ hạng trong top đầu, không phải độ phủ.
5. **Reranker (mọi biến thể) không cải thiện được kết quả** dù có tín hiệu trung bình tốt (gap rel-irr = +0.164). Số câu hỏi bị hurt (100) và improved (96) gần như cân bằng.
6. **Hướng đi tiếp theo:**
   - Thử nghiệm **conditional reranking**: chỉ rerank khi hybrid không tự tin (top1_top2_gap).
   - Thử nghiệm **reorder window với window lớn hơn** (top 5, top 10).
   - Thử nghiệm **blend_weight thấp hơn** (0.2-0.3) trong reorder window để giảm tác động tiêu cực.
   - Tìm reranker model mạnh hơn hoặc fine-tune trên dữ liệu pháp luật Việt Nam.
7. **Các thành phần khác** (query rewriting, metadata boost, LLM context) đã được implement sẵn nhưng chưa kích hoạt trong cấu hình chuẩn.
