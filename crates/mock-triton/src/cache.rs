//! Ground-truth KV cache simulation for a mock replica.
//!
//! Block hashes are chained (hash i commits to blocks 0..=i), so a flat
//! `hash -> last_access` map is equivalent to a radix tree over token
//! blocks: presence of hash i implies the whole prefix chain up to i was
//! inserted, and counting leading present hashes is the same as walking a
//! tree from the root. Eviction mirrors `router::prefix_cache::PrefixCache`
//! (see that module for the full derivation): a lazy min-heap keyed by
//! `(last_access, seq)` gives amortized O(log n) eviction per block instead
//! of an O(capacity) `min_by_key` scan, and per-touch `seq` ordering evicts
//! deeper (later) blocks in a chain before their shallower ancestors that
//! share the same access tick — the flat equivalent of leaf-first eviction.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

/// Heap compaction fires once `heap.len() > 2 * map.len().max(COMPACT_FLOOR)`,
/// keeping amortized eviction cost bounded under hit-heavy traffic that
/// would otherwise grow the heap without bound (every touch pushes,
/// regardless of whether anything is evicted).
const COMPACT_FLOOR: usize = 1024;

pub struct KvCacheSim {
    capacity: usize,
    /// hash -> (last_access, seq): the current ground truth for a block.
    map: FxHashMap<u64, (u64, u64)>,
    /// Lazily-cleaned eviction candidates: `(Reverse(access), seq, hash)`.
    /// An entry is stale (discarded on pop without evicting anything) once
    /// `map`'s value for that hash no longer equals what was pushed.
    heap: BinaryHeap<(Reverse<u64>, u64, u64)>,
    clock: u64,
    seq: u64,
}

impl KvCacheSim {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: FxHashMap::default(),
            heap: BinaryHeap::new(),
            clock: 0,
            seq: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns count of leading blocks already cached, then caches all
    /// blocks (one call = one clock tick) and evicts LRU blocks over
    /// capacity.
    pub fn lookup_insert(&mut self, hashes: &[u64]) -> usize {
        self.clock += 1;
        let access = self.clock;
        let cached = hashes
            .iter()
            .take_while(|h| self.map.contains_key(h))
            .count();
        for &h in hashes {
            self.seq += 1;
            let seq = self.seq;
            self.map.insert(h, (access, seq));
            self.heap.push((Reverse(access), seq, h));
        }
        self.compact_if_bloated();
        self.evict_over_capacity();
        cached
    }

    /// Rebuild the heap from the map (dropping every stale entry) once the
    /// heap has grown past a constant factor of the map size. Amortized
    /// over the >= n pushes required to bloat the heap that far again.
    fn compact_if_bloated(&mut self) {
        let threshold = 2 * self.map.len().max(COMPACT_FLOOR);
        if self.heap.len() > threshold {
            self.heap = self
                .map
                .iter()
                .map(|(&h, &(a, s))| (Reverse(a), s, h))
                .collect();
        }
    }

    fn evict_over_capacity(&mut self) {
        while self.map.len() > self.capacity {
            let Some((Reverse(access), seq, hash)) = self.heap.pop() else {
                break;
            };
            if self.map.get(&hash) == Some(&(access, seq)) {
                self.map.remove(&hash);
            }
            // else: stale heap entry, superseded by a later touch — discard.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_then_warm() {
        let mut c = KvCacheSim::new(100);
        assert_eq!(c.lookup_insert(&[1, 2, 3]), 0);
        assert_eq!(c.lookup_insert(&[1, 2, 3]), 3);
        assert_eq!(c.lookup_insert(&[1, 2, 9]), 2);
    }

    #[test]
    fn leading_only() {
        let mut c = KvCacheSim::new(100);
        c.lookup_insert(&[1, 2, 3]);
        // 2 and 3 are cached but 9 isn't the right first block
        assert_eq!(c.lookup_insert(&[9, 2, 3]), 0);
    }

    #[test]
    fn lru_eviction_over_capacity() {
        let mut c = KvCacheSim::new(4);
        c.lookup_insert(&[1, 2]);
        c.lookup_insert(&[3, 4]);
        c.lookup_insert(&[1, 2]); // refresh 1,2
        c.lookup_insert(&[5, 6]); // over capacity → evict 3,4
        assert_eq!(c.len(), 4);
        assert_eq!(c.lookup_insert(&[1, 2]), 2);
        assert_eq!(c.lookup_insert(&[5, 6]), 2);
    }

    #[test]
    fn eviction_prefers_chain_tail_over_shallower_blocks() {
        // One insert = one tick, so all four blocks share `access`; only
        // `seq` (position in the chain) breaks the tie. The tail of the
        // chain (deepest block, highest seq) must be evicted first — not a
        // middle or leading block, which would strand the chain with a gap.
        let mut c = KvCacheSim::new(3);
        assert_eq!(c.lookup_insert(&[1, 2, 3, 4]), 0);
        assert_eq!(c.len(), 3);
        // The leading chain 1,2,3 must still report a full match — proof
        // that the evicted block was the tail (4), not something earlier.
        assert_eq!(c.lookup_insert(&[1, 2, 3]), 3);
    }
}
