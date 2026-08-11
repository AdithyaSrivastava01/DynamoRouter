use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

pub struct PromptTokenizer {
    inner: Tokenizer,
}

impl PromptTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_is_deterministic_and_prefix_stable() {
        let tok = PromptTokenizer::from_file("../../data/tokenizer.json").unwrap();
        let a = tok.tokenize("The quick brown fox jumps over the lazy dog.");
        let b = tok.tokenize("The quick brown fox jumps over the lazy dog. And more text.");
        assert!(!a.is_empty());
        assert_eq!(a, tok.tokenize("The quick brown fox jumps over the lazy dog."));
        assert_eq!(b[..a.len()], a[..]); // BPE prefix stability at word boundary
    }
}
