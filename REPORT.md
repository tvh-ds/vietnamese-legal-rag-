# Báo Cáo Tiến Độ Hệ Thống Truy Hồi Văn Bản Pháp Luật Tiếng Việt

Trọng tâm: Hybrid Search và Reranker

---

## 1. Tóm Tắt Kết Quả

- Hybrid search (dense vector + BM25) là baseline mạnh nhất hiện tại.
- BGE reranker cải thiện ít
- Kết quả tốt nhất hiện tại:

| Phương pháp                | R@1              | R@3              |             R@10 |
| ----------------------------- | ---------------- | ---------------- | ---------------: |
| Hybrid search không reranker | **0.7924** | **0.9273** | **0.9717** |
| Hybrid search + BGE reranker  | **0.7981** | **0.9346** | **0.9725** |

---

## 2. Tổng Quan Pipeline Hiện Tại

| Bước             | Phương pháp đang dùng                                             | Ghi chú                                       |
| ------------------ | ---------------------------------------------------------------------- | ---------------------------------------------- |
| Nạp dữ liệu     | Đọc JSON pháp điển đã parse                                     | Dữ liệu văn bản pháp luật tiếng Việt   |
| Chunking           | Chia văn bản theo cấu trúc điều/khoản/điểm + giới hạn token | `max_tokens=512`, `overlap_tokens=50`     |
| Context generation | Context heuristic, không dùng LLM                                    | `context.enabled=true`, `use_llm=false`    |
| Embedding          | API embedding tiếng Việt                                             | Model`Vietnamese_Embedding`, 1024 chiều     |
| Vector store       | ChromaDB                                                               | Lưu vector embedding và thông tin chunk     |
| Dense retrieval    | Vector search theo cosine similarity                                   | `candidate_pool_size=50`                     |
| Sparse retrieval   | BM25                                                                   | `k1=1.2`, `b=0.75`, `top_k=50`           |
| Fusion             | Weighted score fusion                                                  | BM25 weight`0.05`                            |
| Reranking          | BGE reranker API, tùy cấu hình                                      | Dùng trong cấu hình tốt nhất có reranker |
| Evaluation         | Recall@1, Recall@3, Recall@10                                          | Benchmark trên 1.238 câu hỏi                |

---

## 3. Dataset

| Thuộc tính                   | Giá trị                                                  |
| ------------------------------ | ---------------------------------------------------------- |
| Nguồn dữ liệu               | 45 file JSON (`chu_de_1.json` … `chu_de_45.json`)     |
| Định dạng                   | VBQPPL đã parse: topic → section → chapter → article  |
| Tổng số điều               | **76.459**                                           |
| Số điều có benchmark ID    | **32.768** (khớp nội dung qua `Correct ID.json`) |
| Số câu hỏi benchmark        | 1.238 (từ`1238_question_map_phap_dien.json`)            |
| Số luật liên quan duy nhất | 1.325                                                      |

---

## 4. Chi Tiết Từng Bước Trong Pipeline

### 4.1 Nạp dữ liệu

| Component            | Tool               | Chi tiết                                                                |
| -------------------- | ------------------ | ------------------------------------------------------------------------ |
| Data loader          | `data_loader.py` | Làm phẳng hệ thống phân cấp topic → section → chapter → article |
| Mapping ID benchmark | Content matching   | `Correct ID.json`: `article_text → numeric_id`                      |
| Tỉ lệ match ID     | 32.768 / 76.459    | Toàn bộ 1.325 ID benchmark được coverage                            |

**Metadata được trích xuất mỗi điều:**

| Trường                     | Nguồn                                 |
| ---------------------------- | -------------------------------------- |
| `article_id` (UUID)        | Điều ID                              |
| `title`                    | Tên điều                            |
| `content`                  | Nội dung                              |
| `document_id`              | Regex từ Ghi chú                     |
| `document_type`            | Luật / Nghị định / Thông tư      |
| `effective_date`           | `metadata[Hiệu lực]`               |
| `status`                   | `metadata[Thi hành]`                |
| `topic_id`, `topic_name` | Hệ thống topic cha                   |
| `benchmark_id`             | Content-matched từ`Correct ID.json` |

### 4.2 Chunking

| Component             | Tool                           | Chi tiết                                         |
| --------------------- | ------------------------------ | ------------------------------------------------- |
| Tokenizer             | HuggingFace`AutoTokenizer`   | Khớp với embedding model                        |
| Thuật toán chunking | `chunker.py`                 | Đệ quy Clause → Point → Paragraph → Sentence |
| Clause pattern        | Regex`^\d+\.\s`              | Các khoản đánh số                            |
| Point pattern         | Regex`^[a-đ]\)\s`           | Các điểm đánh chữ                           |
| Paragraph split       | Double-newline boundary        | Chia ngữ đoạn                                  |
| Sentence split        | Vietnamese sentence boundaries | Fallback cuối cùng                              |

**Tham số chunking:**

| Parameter          | Giá trị | Mô tả                              |
| ------------------ | --------- | ------------------------------------ |
| `max_tokens`     | 512       | Ngưỡng chia                        |
| `min_tokens`     | 30        | Gộp chunk nhỏ với chunk liền kề |
| `overlap_tokens` | 50        | Token overlap tại ranh giới chunk  |

**Các loại unit:** ARTICLE, CLAUSE, POINT, PARAGRAPH, SENTENCE

### 4.3 Context Generation

| Component            | Tool                            | Chi tiết                           |
| -------------------- | ------------------------------- | ----------------------------------- |
| Phương pháp       | Heuristic                       | Tên điều + câu đầu tiên      |
| LLM option           | `gemma-4-31B-it` qua API      | Không dùng (`use_llm: false`)   |
| Tiền tố mỗi chunk | `[context] + [chunk_content]` | Được ghép trước khi embedding |

### 4.4 Embedding

| Component           | Tool                             | Chi tiết                           |
| ------------------- | -------------------------------- | ----------------------------------- |
| Backend             | API công ty (`requests`)      | `POST /v1/embeddings`             |
| Model               | **Vietnamese_Embedding**   | 1024 chiều                         |
| Base URL            | `https://mkp-api.fptcloud.com` | OpenAI-compatible                   |
| Xác thực          | Bearer token                     | Biến môi trường`$FPT_API_KEY` |
| Max sequence length | 2.048 tokens                     | Input chunks ≤ 512                 |
| Batch size          | 64 texts mỗi API call           |                                     |
| Retry               | 3 lần, exponential backoff      | Xử lý lỗi mạng tạm thời       |

**Cô lập embedding:** Trường `benchmark_id` chỉ lưu trong metadata ChromaDB. Model embedding không bao giờ nhận được nó. Chỉ dùng ở bước đánh giá cuối cùng.

### 4.5 Vector Store

| Component       | Tool                            | Chi tiết                      |
| --------------- | ------------------------------- | ------------------------------ |
| Database        | **ChromaDB** (persistent) | SQLite + binary segment files  |
| Index type      | HNSW                            | Approximate nearest neighbor   |
| Distance metric | Cosine                          | Vectors được L2-normalized  |
| Collection      | `bench_data`                  | Lưu trong`chroma_db_bench/` |

### 4.6 BM25 Sparse Index

| Component      | Tool                                            | Chi tiết                     |
| -------------- | ----------------------------------------------- | ----------------------------- |
| Library        | **rank-bm25**                             | Okapi BM25                    |
| Tokenizer      | **pyvi**                                  | Vietnamese word segmentation  |
| `k1`         | 1.2                                             | Term frequency saturation     |
| `b`          | 0.75                                            | Document length normalization |
| Persistence    | Pickle file                                     | Lưu cạnh ChromaDB           |
| Compound words | `đấu_giá`, `giấy_phép`, `xây_dựng` | pyvi xử lý đúng           |

### 4.7 Query Rewriting

| Component            | Tool                  | Chi tiết                                          |
| -------------------- | --------------------- | -------------------------------------------------- |
| Expansion viết tắt | `query_rewriter.py` | 80+ từ viết tắt (`dk` → `điều kiện`)    |
| Domain context       | Append phrase         | `"theo pháp luật hiện hành"` cho query ngắn |

**Ghi chú:** Query rewriting không nằm trong cấu hình tốt nhất hiện tại.

### 4.8 Hybrid Retrieval

#### Vector Search

| Parameter         | Giá trị                  |
| ----------------- | -------------------------- |
| Candidate pool    | `candidate_pool_size=50` |
| Similarity metric | Cosine (1 − distance/2)   |

#### BM25 Search (song song)

| Parameter      | Giá trị                           |
| -------------- | ----------------------------------- |
| Candidate pool | `top_k=50`                        |
| Scoring        | Okapi BM25 (`k1=1.2`, `b=0.75`) |

#### Score Fusion

| Parameter            | Giá trị                                       |
| -------------------- | ----------------------------------------------- |
| Fusion method        | Weighted sum                                    |
| `fusion_weight`    | 0.05 (5% BM25, 95% vector)                      |
| BM25-only candidates | Thêm với score`bm25_score × fusion_weight` |

### 4.9 Reranker

| Component          | Tool             | Chi tiết                           |
| ------------------ | ---------------- | ----------------------------------- |
| Model              | BGE reranker API | `bge-reranker-v2-m3`              |
| `max_candidates` | 10               | Số candidate gửi đến reranker   |
| `top_n`          | 10               | Số kết quả sau reranking         |
| `blend_weight`   | 0.05             | Blend điểm BGE với điểm hybrid |

### 4.10 Evaluation

| Component       | Chi tiết                                                                 |
| --------------- | ------------------------------------------------------------------------- |
| Metrics         | Recall@1, Recall@3, Recall@10                                             |
| Formula         | `hits / total_questions`                                                |
| Hit definition  | Ít nhất 1 trong top-k chunk có`benchmark_id` khớp `relevant_laws` |
| Query embedding | Batch 1.238 query trong một API call                                     |
| Số câu hỏi   | 1.238                                                                     |

---

## 5. Cấu Hình Hybrid Search Tốt Nhất Không Dùng Reranker

```yaml
bm25:
  k1: 1.2
  b: 0.75
  top_k: 50
  fusion_weight: 0.05

retrieval:
  top_k: 10
  candidate_pool_size: 50
  similarity_threshold: 0.0

reranker:
  enabled: false
```

**Kết quả:**

| Chỉ số  | Kết quả        |
| --------- | ---------------- |
| Recall@1  | **0.7924** |
| Recall@3  | **0.9273** |
| Recall@10 | **0.9717** |

Diễn giải:

- `R@3=0.9273` — đáp án đúng thường đã nằm trong top 3.
- `R@10=0.9717` — bước tạo candidate rất mạnh.
- Vấn đề chính còn lại: cải thiện thứ hạng trong nhóm top đầu, đặc biệt từ rank 2/3 lên rank 1.

---

## 6. Cấu Hình Tốt Nhất Có Reranker

Giữ nguyên cấu hình hybrid, thêm:

```yaml
reranker:
  enabled: true
  bge:
    model: "bge-reranker-v2-m3"
    max_candidates: 10
    top_n: 10
    blend_weight: 0.05
```

**Kết quả:**

| Chỉ số  | Hybrid only | Hybrid + BGE reranker | Chênh lệch |
| --------- | ----------: | --------------------: | -----------: |
| Recall@1  |      0.7924 |      **0.7981** |      +0.0057 |
| Recall@3  |      0.9273 |      **0.9346** |      +0.0073 |
| Recall@10 |      0.9717 |      **0.9725** |      +0.0008 |

Diễn giải:

- BGE reranker cải thiện nhẹ nhưng ổn định.
- Tác động rõ nhất ở R@1 và R@3.
- R@10 gần như không đổi — retrieval đã đưa đáp án đúng vào top 10 trong đa số trường hợp.

---

## 7. Nhận Xét Về Reranker

- BGE reranker hiệu quả nhất khi chỉ dùng như tín hiệu phụ.
- `blend_weight=0.05` là cấu hình tốt nhất hiện tại.
- Khi tăng `blend_weight` quá cao (≥ 0.5), hiệu quả giảm rõ rệt.
- Điểm BGE có thang nén, khoảng quan sát `0.03–0.43`.
- Nếu dùng điểm BGE quá mạnh sẽ làm mất tín hiệu tốt từ hybrid retrieval.

---

## 8. Các Phương Pháp Đã Thử Trong Retrieval Và Ranking

| Nhóm                    | Phương pháp đã thử                          | Kết quả / nhận xét                                    |
| ------------------------ | ------------------------------------------------- | --------------------------------------------------------- |
| Dense retrieval          | Vector search bằng Vietnamese embedding API      | Nền tảng chính của hệ thống                         |
| Sparse retrieval         | BM25                                              | Cải thiện exact match + thuật ngữ pháp lý           |
| Hybrid retrieval         | Weighted score fusion giữa dense và BM25        | Tốt nhất hiện tại:`fusion_weight=0.05`              |
| BM25 tuning              | Grid search`k1`, `b`, `fusion_weight`       | Tốt nhất:`k1=1.2`, `b=0.75`, `fusion_weight=0.05` |
| Candidate pool tuning    | Tăng pool lên 50                                | Cải thiện recall trước reranking                      |
| Query rewriting          | Chuẩn hóa / mở rộng query                     | Không nằm trong cấu hình tốt nhất                   |
| Contextual chunking      | Context heuristic vào chunk                      | Đang dùng trong pipeline hiện tại                     |
| Raw content test         | Thử dùng nội dung gốc không prefix context   | Tín hiệu tích cực nhỏ, chưa đủ thay đổi         |
| Query embedding batching | Embed 1.238 query một lần                       | Giảm API call, tăng tốc benchmark                      |
| BGE reranker             | Rerank top 10 bằng`bge-reranker-v2-m3`         | Cải thiện nhẹ R@1 và R@3                              |
| BGE blend tuning         | Thử nhiều`blend_weight` (0.01–1.0)           | Tốt nhất:`0.05`; cao hơn làm giảm kết quả        |
| Conditional reranking    | Chỉ gọi reranker khi điểm top đầu gần nhau | `top1_top2_gap=0.03`, giảm API call                    |
| RRF fusion               | Reciprocal Rank Fusion thay thế weighted score   | Đã implement, chưa phải kết quả tốt nhất          |
| MMR reranking            | Diversity-based reranking                         | Đã loại bỏ — không phù hợp mục tiêu R@1         |
| LLM reranker (Gemma)     | Gemma chọn best trong top 3                      | Đã implement, chưa benchmark chính thức              |

---

## 9. Tham Số Chi Tiết

| Nhóm                          | Tham số                 | Giá trị                                 |
| ------------------------------ | ------------------------ | ----------------------------------------- |
| **Chunking**             | `max_tokens`           | 512                                       |
|                                | `min_tokens`           | 30                                        |
|                                | `overlap_tokens`       | 50                                        |
| **Embedding**            | Backend                  | API                                       |
|                                | Model                    | `Vietnamese_Embedding`                  |
|                                | Dimension                | 1.024                                     |
|                                | Max seq length           | 2.048                                     |
|                                | Batch size API           | 64                                        |
|                                | Retries                  | 3 (exponential backoff)                   |
| **Vector Store**         | Database                 | ChromaDB (HNSW)                           |
|                                | Distance                 | Cosine                                    |
| **BM25**                 | `k1`                   | 1.2                                       |
|                                | `b`                    | 0.75                                      |
|                                | `fusion_weight`        | 0.05                                      |
|                                | `top_k` (candidate)    | 50                                        |
| **Retrieval**            | `candidate_pool_size`  | 50                                        |
|                                | `top_k`                | 10                                        |
|                                | `similarity_threshold` | 0.0                                       |
| **Reranker** (khi dùng) | Model                    | `bge-reranker-v2-m3`                    |
|                                | `max_candidates`       | 10                                        |
|                                | `top_n`                | 10                                        |
|                                | `blend_weight`         | 0.05                                      |
|                                | `timeout_sec`          | 120                                       |
|                                | Conditional              | `enabled=true`, `top1_top10_gap=0.08` |
| **Context**              | `use_llm`              | false (heuristic)                         |
| **Query Rewriter**       | `enabled`              | false                                     |

---

## 10. Kết Luận Hiện Tại

1. Hybrid search (dense + BM25) với `fusion_weight=0.05` là cấu hình baseline mạnh nhất và ổn định nhất.
2. BGE reranker có cải thiện nhỏ, nhưng mức cải thiện có giới hạn.
3. Phần retrieval đã rất mạnh ở R@10 (0.9717), nên hướng cải thiện chính là thứ hạng trong nhóm top 3 / top 10.
4. Cấu hình tốt nhất hiện tại:
   - **Không reranker:** R@1=0.7924, R@3=0.9273, R@10=0.9717
   - **Có BGE reranker:** R@1=0.7981, R@3=0.9346, R@10=0.9725
