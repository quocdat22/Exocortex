# Chunking Strategies Benchmark Report

## 1. Summary of Benchmark Results

| Strategy           |   Chunks |   Avg Len |   HitRate@K |   MRR |   Context Precision |   Context Recall |   Faithfulness |   Answer Relevancy | Latency   |
|--------------------|----------|-----------|-------------|-------|---------------------|------------------|----------------|--------------------|-----------|
| fixed              |      118 |       510 |       1.000 | 0.955 |               0.849 |            0.726 |          0.807 |              0.808 | 327.1 ms  |
| recursive          |      117 |       472 |       1.000 | 0.977 |               0.925 |            0.750 |          0.853 |              0.890 | 351.4 ms  |
| sentence_paragraph |      190 |       445 |       1.000 | 0.955 |               0.763 |            0.677 |          0.812 |              0.899 | 403.9 ms  |
| semantic           |       84 |       646 |       1.000 | 0.932 |               0.780 |            0.833 |          0.892 |              0.917 | 373.0 ms  |

## 2. Recommendation

Based on the multi-dimensional evaluation (Ragas Context Precision/Recall, Faithfulness, Hit Rate@K, and MRR):
- **Optimal Strategy:** `recursive`
- **Recommendation:** Set `CHUNKING_STRATEGY=recursive` in `src/exocortex/config.py`.

Generated at: 2026-08-21 06:08:41
