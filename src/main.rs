use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use clap::Parser;

// ----------------------------------------------------------------------------
// Transformer model structures

#[derive(Debug)]
struct Config {
    dim: i32,        // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32,   // number of layers
    n_heads: i32,    // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads for MQA)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32,    // max sequence length
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls
    wq: Vec<f32>, // (layer, dim, dim)
    wk: Vec<f32>, // (layer, dim, dim)
    wv: Vec<f32>, // (layer, dim, dim)
    wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Vec<f32>,
}

struct RunState {
    // current wave of activations
    x: Vec<f32>,      // activation at current time stamp (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    k: Vec<f32>,      // key (dim,)
    v: Vec<f32>,      // value (dim,)
    att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

// ----------------------------------------------------------------------------
// Initialization: read from checkpoint

fn read_checkpoint(checkpoint: &str, config: &mut Config, weights: &mut TransformerWeights) -> io::Result<()> {
    let mut file = File::open(checkpoint)?;
    
    // read in the config header
    let mut buffer = [0u8; 4];
    file.read_exact(&mut buffer)?;
    config.dim = i32::from_le_bytes(buffer);
    file.read_exact(&mut buffer)?;
    config.hidden_dim = i32::from_le_bytes(buffer);
    file.read_exact(&mut buffer)?;
    config.n_layers = i32::from_le_bytes(buffer);
    file.read_exact(&mut buffer)?;
    config.n_heads = i32::from_le_bytes(buffer);
    file.read_exact(&mut buffer)?;
    config.n_kv_heads = i32::from_le_bytes(buffer);
    file.read_exact(&mut buffer)?;
    config.vocab_size = i32::from_le_bytes(buffer);
    file.read_exact(&mut buffer)?;
    config.seq_len = i32::from_le_bytes(buffer);

    // read in the weights
    let head_size = config.dim / config.n_heads;
    
    weights.token_embedding_table = read_floats(&mut file, (config.vocab_size * config.dim) as usize)?;
    weights.rms_att_weight = read_floats(&mut file, (config.n_layers * config.dim) as usize)?;
    weights.wq = read_floats(&mut file, (config.n_layers * config.dim * config.dim) as usize)?;
    weights.wk = read_floats(&mut file, (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize)?;
    weights.wv = read_floats(&mut file, (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize)?;
    weights.wo = read_floats(&mut file, (config.n_layers * config.dim * config.dim) as usize)?;
    weights.rms_ffn_weight = read_floats(&mut file, (config.n_layers * config.dim) as usize)?;
    weights.w1 = read_floats(&mut file, (config.n_layers * config.hidden_dim * config.dim) as usize)?;
    weights.w2 = read_floats(&mut file, (config.n_layers * config.dim * config.hidden_dim) as usize)?;
    weights.w3 = read_floats(&mut file, (config.n_layers * config.hidden_dim * config.dim) as usize)?;
    weights.rms_final_weight = read_floats(&mut file, config.dim as usize)?;
    
    let head_size_usize = head_size as usize;
    weights.freq_cis_real = read_floats(&mut file, config.seq_len as usize * head_size_usize / 2)?;
    weights.freq_cis_imag = read_floats(&mut file, config.seq_len as usize * head_size_usize / 2)?;
    
    // skip what used to be freq_cis_real and freq_cis_imag (for legacy reasons)
    weights.wcls = if let Ok(wcls) = read_floats(&mut file, (config.vocab_size * config.dim) as usize) {
        wcls
    } else {
        weights.token_embedding_table.clone()
    };

    Ok(())
}

fn read_floats(file: &mut File, count: usize) -> io::Result<Vec<f32>> {
    let mut buffer = vec![0u8; count * 4];
    file.read_exact(&mut buffer)?;
    let floats: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(floats)
}

// ----------------------------------------------------------------------------
// Neural net blocks

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    let size = x.len();
    let mut ss: f32 = 0.0;
    for i in 0..size {
        ss += x[i] * x[i];
    }
    ss /= size as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    for i in 0..size {
        o[i] = weight[i] * (ss * x[i]);
    }
}

fn softmax(x: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for i in 0..x.len() {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }
    for i in 0..x.len() {
        x[i] /= sum;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    for i in 0..d {
        let mut val = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// The Transformer

fn transformer(token: i32, pos: i32, config: &Config, s: &mut RunState, w: &TransformerWeights) {
    let dim = config.dim as usize;
    let hidden_dim = config.hidden_dim as usize;
    let head_size = (config.dim / config.n_heads) as usize;
    let n_heads = config.n_heads as usize;
    let kv_dim = (config.n_kv_heads * config.dim / config.n_heads) as usize;
    let kv_mul = config.n_heads / config.n_kv_heads;

    // copy the token embedding into x
    let content_row_start = (token * config.dim) as usize;
    s.x[..dim].copy_from_slice(&w.token_embedding_table[content_row_start..content_row_start + dim]);

    // forward all the layers
    for l in 0..config.n_layers as usize {
        // attention rmsnorm
        rmsnorm(&mut s.xb, &s.x, &w.rms_att_weight[l * dim..(l + 1) * dim]);

        // qkv matmuls for this position
        matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..(l + 1) * dim * dim], dim, dim);
        matmul(&mut s.k, &s.xb, &w.wk[l * dim * kv_dim..(l + 1) * dim * kv_dim], dim, kv_dim);
        matmul(&mut s.v, &s.xb, &w.wv[l * dim * kv_dim..(l + 1) * dim * kv_dim], dim, kv_dim);

        // RoPE relative positional encoding
        for h in 0..n_heads {
            let q_offset = h * head_size;
            for i in (0..head_size).step_by(2) {
                let head_dim = i % head_size;
                let freq_cis_real_idx = pos as usize * head_size / 2 + head_dim / 2;
                let fcr = w.freq_cis_real[freq_cis_real_idx];
                let fci = w.freq_cis_imag[freq_cis_real_idx];
                let q0 = s.q[q_offset + i];
                let q1 = s.q[q_offset + i + 1];
                s.q[q_offset + i] = q0 * fcr - q1 * fci;
                s.q[q_offset + i + 1] = q0 * fci + q1 * fcr;
            }
        }

        for h in 0..(config.n_kv_heads as usize) {
            let k_offset = h * head_size;
            for i in (0..head_size).step_by(2) {
                let head_dim = i % head_size;
                let freq_cis_real_idx = pos as usize * head_size / 2 + head_dim / 2;
                let fcr = w.freq_cis_real[freq_cis_real_idx];
                let fci = w.freq_cis_imag[freq_cis_real_idx];
                let k0 = s.k[k_offset + i];
                let k1 = s.k[k_offset + i + 1];
                s.k[k_offset + i] = k0 * fcr - k1 * fci;
                s.k[k_offset + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        let loff = l * config.seq_len as usize * kv_dim;
        let key_cache_row = loff + pos as usize * kv_dim;
        let value_cache_row = loff + pos as usize * kv_dim;
        s.key_cache[key_cache_row..key_cache_row + kv_dim].copy_from_slice(&s.k[..kv_dim]);
        s.value_cache[value_cache_row..value_cache_row + kv_dim].copy_from_slice(&s.v[..kv_dim]);

        // multihead attention
        for h in 0..n_heads {
            let q_offset = h * head_size;
            let att_offset = h * config.seq_len as usize;

            // get the query vector for this head
            let q = &s.q[q_offset..q_offset + head_size];

            // attention scores for this head
            for t in 0..=pos as usize {
                let k_offset = loff + t * kv_dim + (h / kv_mul as usize) * head_size;
                let mut score = 0.0;
                for i in 0..head_size {
                    score += q[i] * s.key_cache[k_offset + i];
                }
                score /= (head_size as f32).sqrt();
                s.att[att_offset + t] = score;
            }

            // softmax the scores to get attention weights
            softmax(&mut s.att[att_offset..att_offset + pos as usize + 1]);

            // weighted sum of the values
            let xb_offset = h * head_size;
            for i in 0..head_size {
                s.xb[xb_offset + i] = 0.0;
            }
            for t in 0..=pos as usize {
                let v_offset = loff + t * kv_dim + (h / kv_mul as usize) * head_size;
                let a = s.att[att_offset + t];
                for i in 0..head_size {
                    s.xb[xb_offset + i] += a * s.value_cache[v_offset + i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..(l + 1) * dim * dim], dim, dim);

        // residual connection back into x
        for i in 0..dim {
            s.x[i] += s.xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(&mut s.xb, &s.x, &w.rms_ffn_weight[l * dim..(l + 1) * dim]);

        // ffn
        matmul(&mut s.hb, &s.xb, &w.w1[l * hidden_dim * dim..(l + 1) * hidden_dim * dim], dim, hidden_dim);
        matmul(&mut s.hb2, &s.xb, &w.w3[l * hidden_dim * dim..(l + 1) * hidden_dim * dim], dim, hidden_dim);

        // SwiGLU non-linearity
        for i in 0..hidden_dim {
            let mut val = s.hb[i];
            // silu(x) = x * sigmoid(x)
            val = val / (1.0 + (-val).exp());
            // elementwise multiply with w3(x)
            s.hb[i] = val * s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(&mut s.xb, &s.hb, &w.w2[l * dim * hidden_dim..(l + 1) * dim * hidden_dim], hidden_dim, dim);

        // residual connection
        for i in 0..dim {
            s.x[i] += s.xb[i];
        }
    }

    // final rmsnorm
    let x_copy = s.x.clone();
    rmsnorm(&mut s.x, &x_copy, &w.rms_final_weight);

    // classifier into logits
    matmul(&mut s.logits, &s.x, &w.wcls, dim, config.vocab_size as usize);
}

// ----------------------------------------------------------------------------
// Byte Pair Encoding (BPE) Tokenizer

struct TokenIndex {
    str: String,
    id: i32,
}

fn compare_tokens(a: &TokenIndex, b: &TokenIndex) -> std::cmp::Ordering {
    a.str.cmp(&b.str)
}

fn build_tokenizer(tokenizer_path: &str, vocab_size: i32) -> io::Result<Vec<String>> {
    let mut vocab = vec![String::new(); vocab_size as usize];
    
    let file = File::open(tokenizer_path)?;
    let mut reader = io::BufReader::new(file);
    
    for i in 0..vocab_size as usize {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        let len = f32::from_le_bytes(buffer) as usize;
        
        let mut str_buffer = vec![0u8; len];
        reader.read_exact(&mut str_buffer)?;
        vocab[i] = String::from_utf8_lossy(&str_buffer).to_string();
    }
    
    Ok(vocab)
}

fn str_lookup(str: &str, sorted_vocab: &[TokenIndex]) -> i32 {
    // binary search
    let mut left = 0;
    let mut right = sorted_vocab.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        let cmp = str.cmp(&sorted_vocab[mid].str);
        
        match cmp {
            std::cmp::Ordering::Less => right = mid,
            std::cmp::Ordering::Greater => left = mid + 1,
            std::cmp::Ordering::Equal => return sorted_vocab[mid].id,
        }
    }
    -1
}

fn encode(text: &str, vocab: &[String], sorted_vocab: &[TokenIndex], tokens: &mut Vec<i32>) {
    tokens.clear();
    
    // encode every individual byte in the input string
    for byte in text.bytes() {
        let token_str = format!("{}", byte as char);
        let id = str_lookup(&token_str, sorted_vocab);
        if id != -1 {
            tokens.push(id);
        }
    }
    
    // merge the best consecutive pair each iteration
    loop {
        let mut best_score = -1e10;
        let mut best_id = -1;
        let mut best_idx = -1;
        
        for i in 0..tokens.len().saturating_sub(1) {
            let str_buffer = format!("{}{}", vocab[tokens[i] as usize], vocab[tokens[i + 1] as usize]);
            let id = str_lookup(&str_buffer, sorted_vocab);
            if id != -1 && (id as f32) > best_score {
                best_score = id as f32;
                best_id = id;
                best_idx = i as i32;
            }
        }
        
        if best_idx == -1 {
            break; // no more pairs to merge
        }
        
        // merge the consecutive pair into new token
        tokens[best_idx as usize] = best_id;
        tokens.remove(best_idx as usize + 1);
    }
}

// ----------------------------------------------------------------------------
// Sampling

fn sample(probabilities: &[f32]) -> usize {
    let r: f32 = rand::random();
    let mut cdf = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probabilities.len() - 1
}

fn argmax(v: &[f32]) -> usize {
    let mut max_i = 0;
    let mut max_p = v[0];
    for (i, &p) in v.iter().enumerate().skip(1) {
        if p > max_p {
            max_i = i;
            max_p = p;
        }
    }
    max_i
}

// ----------------------------------------------------------------------------
// CLI

#[derive(Parser)]
#[command(name = "llama2")]
#[command(about = "Run inference with LLaMA 2 transformer model", long_about = None)]
struct Cli {
    /// Path to the model checkpoint file
    checkpoint: String,
    
    /// Path to the tokenizer file
    #[arg(short = 'z', long, default_value = "tokenizer.bin")]
    tokenizer: String,
    
    /// Temperature for sampling (0.0 = greedy argmax sampling, 1.0 = sampling from distribution)
    #[arg(short = 't', long, default_value = "1.0")]
    temperature: f32,
    
    /// Number of steps to run for
    #[arg(short = 'n', long, default_value = "256")]
    steps: usize,
    
    /// Optional prompt to condition on
    #[arg(short = 'i', long)]
    prompt: Option<String>,
}

fn main() -> io::Result<()> {
    let args = Cli::parse();
    
    // parameter validation
    if args.temperature < 0.0 {
        eprintln!("Temperature must be non-negative");
        std::process::exit(1);
    }
    
    // build the Transformer
    let mut config = Config {
        dim: 0,
        hidden_dim: 0,
        n_layers: 0,
        n_heads: 0,
        n_kv_heads: 0,
        vocab_size: 0,
        seq_len: 0,
    };
    
    let mut weights = TransformerWeights {
        token_embedding_table: Vec::new(),
        rms_att_weight: Vec::new(),
        rms_ffn_weight: Vec::new(),
        wq: Vec::new(),
        wk: Vec::new(),
        wv: Vec::new(),
        wo: Vec::new(),
        w1: Vec::new(),
        w2: Vec::new(),
        w3: Vec::new(),
        rms_final_weight: Vec::new(),
        freq_cis_real: Vec::new(),
        freq_cis_imag: Vec::new(),
        wcls: Vec::new(),
    };
    
    read_checkpoint(&args.checkpoint, &mut config, &mut weights)?;
    
    println!("Config loaded:");
    println!("  dim: {}", config.dim);
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  n_layers: {}", config.n_layers);
    println!("  n_heads: {}", config.n_heads);
    println!("  n_kv_heads: {}", config.n_kv_heads);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  seq_len: {}", config.seq_len);
    
    // allocate the RunState buffers
    let mut state = RunState {
        x: vec![0.0; config.dim as usize],
        xb: vec![0.0; config.dim as usize],
        xb2: vec![0.0; config.dim as usize],
        hb: vec![0.0; config.hidden_dim as usize],
        hb2: vec![0.0; config.hidden_dim as usize],
        q: vec![0.0; config.dim as usize],
        k: vec![0.0; (config.dim / config.n_heads * config.n_kv_heads) as usize],
        v: vec![0.0; (config.dim / config.n_heads * config.n_kv_heads) as usize],
        att: vec![0.0; (config.n_heads * config.seq_len) as usize],
        logits: vec![0.0; config.vocab_size as usize],
        key_cache: vec![0.0; (config.n_layers * config.seq_len * config.dim / config.n_heads * config.n_kv_heads) as usize],
        value_cache: vec![0.0; (config.n_layers * config.seq_len * config.dim / config.n_heads * config.n_kv_heads) as usize],
    };
    
    // read in the tokenizer.bin file
    let vocab = if Path::new(&args.tokenizer).exists() {
        match build_tokenizer(&args.tokenizer, config.vocab_size) {
            Ok(v) => v,
            Err(_) => {
                eprintln!("Failed to load tokenizer from {}", args.tokenizer);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Tokenizer file not found: {}", args.tokenizer);
        std::process::exit(1);
    };
    
    // create and sort vocabulary for encoding
    let mut sorted_vocab: Vec<TokenIndex> = vocab
        .iter()
        .enumerate()
        .map(|(i, s)| TokenIndex {
            str: s.clone(),
            id: i as i32,
        })
        .collect();
    sorted_vocab.sort_by(compare_tokens);
    
    // encode the prompt, if provided
    let mut prompt_tokens = Vec::new();
    if let Some(ref prompt) = args.prompt {
        encode(prompt, &vocab, &sorted_vocab, &mut prompt_tokens);
    }
    
    // start the main loop
    let mut token = if prompt_tokens.is_empty() {
        1 // BOS token
    } else {
        prompt_tokens[0]
    };
    let mut pos = 0;
    
    print!("<s>\n");
    io::stdout().flush()?;
    
    let mut next;
    while pos < args.steps as i32 {
        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &mut state, &weights);
        
        // sample the next token
        if args.temperature == 0.0 {
            // greedy argmax sampling
            next = argmax(&state.logits);
        } else {
            // apply the temperature to the logits
            for i in 0..config.vocab_size as usize {
                state.logits[i] /= args.temperature;
            }
            // apply softmax to get probabilities
            softmax(&mut state.logits);
            // sample from the distribution
            next = sample(&state.logits);
        }
        
        // print the token as string
        if pos >= prompt_tokens.len() as i32 {
            let piece = &vocab[next];
            print!("{}", piece);
            io::stdout().flush()?;
        }
        
        // advance forward
        token = if (pos + 1) < prompt_tokens.len() as i32 {
            prompt_tokens[pos as usize + 1]
        } else {
            next as i32
        };
        pos += 1;
    }
    
    println!();
    Ok(())
}
