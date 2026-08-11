pub const BLOCK_SIZE: usize = 16;

/// Chained FNV-1a over BLOCK_SIZE-token blocks. Hash i depends on blocks 0..=i,
/// so equal hash prefixes == equal token prefixes (modulo collisions).
pub fn block_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut hashes = Vec::with_capacity(tokens.len() / BLOCK_SIZE);
    let mut prev: u64 = 0xcbf29ce484222325;
    for block in tokens.chunks_exact(BLOCK_SIZE) {
        let mut h = prev;
        for &t in block {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        hashes.push(h);
        prev = h;
    }
    hashes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize, offset: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i + offset).collect()
    }

    #[test]
    fn shared_prefix_shares_hashes() {
        let a = block_hashes(&toks(64, 0));
        let mut b_toks = toks(48, 0);
        b_toks.extend(toks(16, 999));
        let b = block_hashes(&b_toks);
        assert_eq!(a.len(), 4);
        assert_eq!(a[..3], b[..3]);
        assert_ne!(a[3], b[3]);
    }

    #[test]
    fn position_matters() {
        // same 16 tokens, but preceded by different block → different hash
        let x: Vec<u32> = toks(16, 0).into_iter().chain(toks(16, 100)).collect();
        let y: Vec<u32> = toks(16, 50).into_iter().chain(toks(16, 100)).collect();
        let hx = block_hashes(&x);
        let hy = block_hashes(&y);
        assert_ne!(hx[1], hy[1]);
    }

    #[test]
    fn partial_block_dropped() {
        assert_eq!(block_hashes(&toks(15, 0)).len(), 0);
        assert_eq!(block_hashes(&toks(17, 0)).len(), 1);
    }
}
