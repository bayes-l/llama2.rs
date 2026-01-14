pub struct TransformerState {
    // current wave of activations
    pub x: Vec<f32>,      // activation at current time stamp (dim,)
    pub xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,      // query (dim,)
    pub k: Vec<f32>,      // key (dim,)
    pub v: Vec<f32>,      // value (dim,)
    pub att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<f32>, // output logits
    // kv cache
    pub key_cache: Vec<f32>,   // (layer, seq_len, dim)
    pub value_cache: Vec<f32>, // (layer, seq_len, dim)
}

pub struct TransformerWeights {
    // token embedding table
    pub token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls
    pub wq: Vec<f32>, // (layer, dim, dim)
    pub wk: Vec<f32>, // (layer, dim, dim)
    pub wv: Vec<f32>, // (layer, dim, dim)
    pub wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    pub w1: Vec<f32>, // (layer, hidden_dim, dim)
    pub w2: Vec<f32>, // (layer, dim, hidden_dim)
    pub w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    pub freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    pub freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Vec<f32>,
}

impl Default for TransformerWeights {
    fn default() -> Self {
        TransformerWeights {
            token_embedding_table: vec![],
            rms_att_weight: vec![],
            rms_ffn_weight: vec![],
            wq: vec![],
            wk: vec![],
            wv: vec![],
            wo: vec![],
            w1: vec![],
            w2: vec![],
            w3: vec![],
            rms_final_weight: vec![],
            freq_cis_real: vec![],
            freq_cis_imag: vec![],
            wcls: vec![],
        }
    }
}

pub struct TransformerOptions {
    pub dim: i32,        // transformer dimension
    pub hidden_dim: i32, // for ffn layers
    pub n_layers: i32,   // number of layers
    pub n_heads: i32,    // number of query heads
    pub n_kv_heads: i32, // number of key/value heads (can be < query heads for MQA)
    pub vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    pub seq_len: i32,    // max sequence length
}

impl Default for TransformerOptions {
    fn default() -> Self {
        TransformerOptions {
            dim: 0,
            hidden_dim: 0,
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            vocab_size: 0,
            seq_len: 0,
        }
    }
}

pub struct Transformer {
    options: TransformerOptions,
    weights: TransformerWeights,
    state: TransformerState,
}

impl Transformer {
    fn new(options: TransformerOptions) -> Self {
        let state = TransformerState {
            x: vec![0.0; options.dim as usize],
            xb: vec![0.0; options.dim as usize],
            xb2: vec![0.0; options.dim as usize],
            hb: vec![0.0; options.hidden_dim as usize],
            hb2: vec![0.0; options.hidden_dim as usize],
            q: vec![0.0; options.dim as usize],
            k: vec![0.0; (options.dim / options.n_heads * options.n_kv_heads) as usize],
            v: vec![0.0; (options.dim / options.n_heads * options.n_kv_heads) as usize],
            att: vec![0.0; (options.n_heads * options.seq_len) as usize],
            logits: vec![0.0; options.vocab_size as usize],
            key_cache: vec![
                0.0;
                (options.n_layers * options.seq_len * options.dim / options.n_heads
                    * options.n_kv_heads) as usize
            ],
            value_cache: vec![
                0.0;
                (options.n_layers * options.seq_len * options.dim / options.n_heads
                    * options.n_kv_heads) as usize
            ],
        };
        let weights = TransformerWeights::default();

        Self {
            options,
            weights,
            state,
        }
    }

    fn transform(&self, token: i32, pos: i32) {

    }
}
