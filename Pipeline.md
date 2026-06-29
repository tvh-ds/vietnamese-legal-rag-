# **Chunking Pipeline for Legal RAG System**


## **Pipeline Overview**

The chunking pipeline processes Vietnamese legal documents (VBQPPL) through two main branches:

- **Offline Indexing Pipeline:** Processes documents, performs chunking, generates embeddings, and builds the knowledge graph.
- **Online Retrieval Pipeline:** Handles user queries, retrieves relevant data, and generates responses.

---

## **1. Structure Parser**

The system receives parsed JSON files that follow a defined ontology. Documents have a hierarchical structure from the overall document down to individual articles. However, only units from **ARTICLE** level downward contain meaningful legal content for retrieval. Therefore, the chunking process starts at the **Article** level, which serves as the primary retrieval unit.

The parser extracts the following information from JSON data:

- **Document-level Metadata:** `documentId` (normalized `officialNumber`), `documentType`, `title`, `promulgationDate`, `effectiveDate`, `expirationDate`, `status` (ACTIVE, EXPIRED, etc.), `legal_effect_level` (1-15), `signerName`, `signerTitle`, `issuingPlace`, `scopeLevel`, `legislative_regime`.
- **Structural Metadata:** `unitID`, `unitType` (ARTICLE, SECTION, CHAPTER, etc.), `heading`, `content`, `parentUnitID`, `order`, `mapCode`.
- **Classification Metadata:** Information for assigning `LegalTopicNode` (45 topics) and `LegalClassificationNode` (5-level classification tree).
- **Reference Metadata:** `references` text for building graph relationships.

---

## **2. Relation Graph Builder**(not needed in this stage, implemented seperately somewhere else)

Parallel to chunking, all reference information between articles and document structures is converted into a knowledge graph.

### **Vertex Types Created:**

- **LegalDocumentNode:** Central node for each document.
- **LegalUnitNode:** Nodes for structural units (Article, Clause, Point).
- **IssuingAuthorityNode:** Node for issuing authority.
- **PersonNode:** Node for document signer.
- **LegalTopicNode:** Node for legal topics (45 topics from the Legal Code).
- **LegalClassificationNode:** Nodes for classification categories.
- **AdministrativeUnitNode:** Node for administrative units.
- **WordLegalNode:** Node for legal keywords.
- **PlaceholderNode:** Temporary node for referenced documents not yet in the system.

### **Edge Types Created:**

- **Document-Unit:** `CONTAINS` (Document → Unit, Unit → ChildUnit).
- **Document-Document:** `AMENDS`, `REPLACES`, `ABOLISHED_BY`, `SUSPENDED_BY`, `DETAILS` (with `survivorship_mode` STRONG/WEAK), `BASED_ON`, `MENTIONS`.
- **Document-Subject:** `ISSUED_BY` (with `is_jointly_issued`), `SIGNED_BY` (with `role` TM/KT/TL), `APPLIES_TO`.
- **Classification:** `CLASSIFIED_AS`, `CLASSIFIES`, `CONTAINS_CLASSIFICATION`.
- **Word:** `CONTAINS_WORD`.

This graph is used for GraphRAG and query expansion instead of embedding reference lists into chunk content.

---

## **3. Recursive Semantic Chunking**

The goal is to divide text into semantic chunks of appropriate size for embedding while preserving the logical structure and hierarchy of legal documents.

The process follows a recursive, structure-aware approach, only splitting further when the current unit exceeds the allowed size limit (e.g., 512 or 768 tokens).

### **Implementation Steps:**

1. **Step 1:** Check the size of the **ARTICLE**.
   - If the Article is below the threshold, keep it as one chunk.
   - If the Article exceeds the threshold, proceed to Step 2.

2. **Step 2:** Split the Article into **CLAUSES**. Each Clause is checked:
   - If a Clause is sufficiently small, stop.
   - If a Clause is still too large, proceed to Step 3.

3. **Step 3:** Split the Clause into **POINTS**.
   - If a Point is sufficiently small, use it.
   - If still above threshold, proceed to fallback options.

4. **Step 4: Late Chunking (for Articles without substructure)**
   - For Articles lacking Clause/Point structure and consisting only of paragraphs, the system applies **Late Chunking**.
   - The entire Article is fed into the embedding model once to obtain `contextualized token embeddings`.
   - Individual paragraphs are identified, and each paragraph's embedding is created by pooling corresponding tokens.
   - This method ensures each paragraph's embedding retains the context of the entire Article.

> **Note:** If a paragraph remains too long, the system uses Sentence-aware Chunking or Sliding Window Chunking as final fallback options.

---

## **4. Chunk Quality Validator**

After generating the final set of chunks, the system performs quality checks to ensure they meet standards.

### **Key Tasks:**

- **Size Validation:** Ensure each chunk is within minimum and maximum size thresholds.
- **Handling Short Chunks:** Merge overly small chunks with adjacent ones when possible.
- **Sentence Integrity:** Ensure chunks don't cut sentences in the middle.
- **Duplicate Removal:** Detect and remove duplicate content chunks.
- **Hierarchy Validation:** Ensure each chunk belongs to the correct Article with complete hierarchical information (e.g., `parentUnitID`).

---

## **5. Context Generation and Context Injection**

After the final chunk set is ready, the system generates a semantic summary (context) for the entire Article using a large language model (LLM).

### **Context Generation:**

- The context provides a semantic overview, not an extremely short summary. Example: "This article regulates the conditions for granting construction permits, exemption cases, and responsibilities of the management authority."
- Context length is approximately 60-120 tokens to retain sufficient information without causing interference.
- Context is generated once per Article.

### **Context Injection:**

- This context is prepended to the content of **each chunk** belonging to that Article before embedding.
- The resulting input for the Embedder becomes: `[Context] + [Chunk Content]`. This prevents the chunk's embedding from being isolated from the Article's overall context.

---

## **6. Metadata Attachment**

Metadata is kept completely separate from chunk content and attached as structured data fields.

### **Key Metadata Fields:**

- **Document Metadata:** `documentId`, `documentType`, `effectiveDate`, `status`, `legal_effect_level`, `scopeLevel`.
- **Unit Metadata:** `unitID`, `unitType`, `parentUnitID`, `order`, `mapCode`.
- **Classification Metadata:** List of related `LegalTopicNode` and `LegalClassificationNode`.
- **Graph Metadata:** `chunk_id`, `article_id`, `hierarchy_info`.

This metadata is stored in the Vector Database and used for filtering and classification but does **not** participate in the embedding process.

---

## **7. Embedder**

Each preprocessed chunk is converted into a vector embedding. The input string for the embedding model is the combination of the generated **Context** and the **Chunk Content**.

The resulting **Vector** is stored in the Vector Database alongside corresponding metadata.

---

## **8. Vector Database**

The Vector Database stores the following information for each chunk:

- `embedding_vector`: Semantic vector representation of the chunk.
- `content`: Original chunk content (may include context).
- `metadata`: All metadata attached in Step 6.
- `chunk_id`: Unique chunk identifier.
- `article_id`: Parent Article ID.
- `hierarchy_info`: Hierarchy and parent-child relationship information.

This database supports both semantic search (vector search) and metadata filtering.

---

## **9. Query Rewriter**

When users submit queries, the input is normalized to improve retrieval effectiveness. Example:
- User input: `dk cap phep xay dung`
- Query Rewriter normalizes to: `Điều kiện cấp giấy phép xây dựng theo pháp luật hiện hành`

This step improves matching with content in the database.

---

## **10. Hybrid Retrieval**

The retrieval process combines three main components:

- **Vector Search:** Semantic search based on embeddings, finding the most relevant chunks.
- **Metadata Filtering:** Apply filters based on metadata such as `documentType`, `status` (validity), `hierarchy` (Article/Clause/Point level), and `LegalTopic` to eliminate irrelevant or invalid chunks.
- **Graph Expansion:** After finding relevant chunks, the system uses the Knowledge Graph to expand results. For example, from a found Article, the system can query related Articles through edges like `REFERENCES`, `BASED_ON`, or `DETAILS` to gather additional context that vector search might miss.

---

## **11. Context Builder and Reranker**

Chunks from different sources (vector search, graph expansion) are merged. A Reranker model reassesses the relevance of each chunk to the original query. The highest-scoring and most diverse chunks are selected to build the final context for the generation model.

---

## **12. LLM Generator**

The optimized context (after reranking) is passed to a large language model (LLM). The LLM uses the entire context to synthesize the response, cite relevant Articles, Clauses, Points, and generate the final answer for the user.

---
