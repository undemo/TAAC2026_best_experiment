# hyformer_pm_head_semantic_feature_v1 Architecture

```mermaid
flowchart TB
    subgraph Inputs["Raw Batch"]
        UI["user_int_feats"]
        UD["user_dense_feats"]
        II["item_int_feats"]
        SEQ["seq_a / seq_b / seq_c / seq_d"]
        TS["timestamp + seq_time_buckets"]
        MISS["missing masks"]
    end

    subgraph SemanticNS["Semantic NS Tokens"]
        UIG["3 user int semantic group tokens"]
        UDG["1 grouped user dense token"]
        IIG["4 item int semantic group tokens"]
        TT["1 TimeToken"]
    end

    UI --> UIG
    MISS --> UIG
    UD --> UDG
    MISS --> UDG
    II --> IIG
    MISS --> IIG
    TS --> TT

    UIG --> NS["NS token stream: 9 x d_model=68"]
    UDG --> NS
    IIG --> NS
    TT --> NS

    subgraph SeqEnc["Sequence Tokenization"]
        SEQEMB["per-domain side-info embeddings"]
        TIMEEMB["time-bucket embeddings"]
        SEQTOK["4 sequence token streams"]
    end

    SEQ --> SEQEMB
    TS --> TIMEEMB
    SEQEMB --> SEQTOK
    TIMEEMB --> SEQTOK

    NS --> QGEN["MultiSeqQueryGenerator"]
    SEQTOK --> QGEN
    QGEN --> QTOK["8 query tokens: 2 queries x 4 domains"]

    NS --> HY["MultiSeqHyFormerBlock x 2"]
    SEQTOK --> HY
    QTOK --> HY
    HY --> HREP["HyFormer representation: 68"]

    NS --> PM["PMHead feature extractor"]
    SEQTOK --> PM
    PM --> PMF["PM feature: 64"]

    HREP --> CAT["concat: 68 + 64 = 132"]
    PMF --> CAT
    CAT --> CLS["MLP classifier"]
    CLS --> LOGIT["binary logit"]

    LOGIT --> VAL["validation: val_auc only"]
```

## Current Run Configuration

- `ns_tokenizer_type=group`
- `d_model=68`
- `num_queries=2`
- `num_hyformer_blocks=2`
- `num_heads=4`
- `pm_head_enabled=True`
- `pm_feature_dim=64`
- `time_token_enabled=True`
- `rankmixer_input_tokens = 2 * 4 + 9 = 17`
- `classifier_input_dim = 68 + 64 = 132`
