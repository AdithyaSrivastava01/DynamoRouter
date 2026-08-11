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

    /// Tokenize `text`. Returns `Err` only on an actual encode failure —
    /// an empty prompt still succeeds with an empty token list (callers
    /// treat that as a cold/empty-hash request, not an error).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_is_deterministic_and_prefix_stable() {
        let tok = PromptTokenizer::from_file("../../data/tokenizer.json").unwrap();
        let a = tok
            .tokenize("The quick brown fox jumps over the lazy dog.")
            .unwrap();
        let b = tok
            .tokenize("The quick brown fox jumps over the lazy dog. And more text.")
            .unwrap();
        assert!(!a.is_empty());
        assert_eq!(
            a,
            tok.tokenize("The quick brown fox jumps over the lazy dog.")
                .unwrap()
        );
        assert_eq!(b[..a.len()], a[..]); // BPE prefix stability at word boundary
    }

    #[test]
    fn empty_prompt_tokenizes_to_empty_not_error() {
        let tok = PromptTokenizer::from_file("../../data/tokenizer.json").unwrap();
        assert_eq!(tok.tokenize("").unwrap(), Vec::<u32>::new());
    }
}
