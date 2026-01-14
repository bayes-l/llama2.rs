use anyhow::anyhow;
use std::fmt::format;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const INVALID_SCORE: f32 = -1e10;
struct Tokenizer {
    vocab: Vec<String>,
    sorted_vocab: Vec<(i32, String)>,
}

impl Tokenizer {
    fn open(path: String, vocab_size: i32) -> Result<Self, anyhow::Error> {
        let file_path = Path::new(&path);
        if !file_path.exists() {
            return Err(anyhow!(
                "tokenizer does not exist. the path is: {:?}",
                file_path
            ));
        }
        let mut vocab = vec![String::new(); vocab_size as usize];
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        for item in vocab.iter_mut().take(vocab_size as usize) {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let len = u32::from_le_bytes(buf) as usize;

            let mut str_buffer = vec![0u8; len];
            reader.read_exact(&mut str_buffer)?;

            *item = String::from_utf8_lossy(&str_buffer).to_string();
        }

        let mut indexable_vocab: Vec<(i32, String)> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (i as i32, s.clone()))
            .collect();
        indexable_vocab.sort_by(|a, b| a.1.cmp(&b.1));

        Ok(Self {
            vocab,
            sorted_vocab: indexable_vocab,
        })
    }

    fn binary_search_token_index(&self, str_token: &str) -> i32 {
        let mut left = 0;
        let mut right = self.sorted_vocab.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let cmp = str_token.cmp(&self.sorted_vocab[mid].1);

            match cmp {
                std::cmp::Ordering::Less => right = mid,
                std::cmp::Ordering::Greater => left = mid + 1,
                std::cmp::Ordering::Equal => return self.sorted_vocab[mid].0,
            }
        }
        -1
    }

    fn encode_prompt(&self, prompt: &str) -> Vec<i32> {
        let mut tokens = Vec::new();

        for byte in prompt.bytes() {
            let token_str = format!("{}", byte);
            let index = self.binary_search_token_index(&token_str);
            if index != -1 {
                tokens.push(index)
            }
        }

        loop {
            //comparing
        }
    }
}
