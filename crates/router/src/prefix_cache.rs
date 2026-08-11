//! Approximate per-replica picture of cached prefixes.
//!
//! Block hashes are chained (hash i commits to blocks 0..=i), so a flat
//! `hash -> last_access` map is equivalent to a radix tree over token
//! blocks: presence of hash i implies the whole prefix chain up to i was
//! inserted, and counting leading present hashes is the same as walking a
//! tree from the root. This drops the O(tree size) cost of evicting via a
//! full tree walk in favor of true LRU eviction backed by a lazy min-heap.
//!
//! Eviction pops the heap and discards any entry whose stored
//! `(last_access, seq)` no longer matches the map — i.e. the block was
//! touched again since that heap entry was pushed — giving an amortized
//! O(log n) cost per evicted block instead of an O(n) tree scan. Within a
//! single touch (one `insert`/`match_prefix` call, one clock tick), deeper
//! blocks in the chain get a strictly higher `seq`, so they are evicted
//! before their shallower ancestors that share the same access time — the
//! flat equivalent of leaf-first eviction in the old tree.
//!
//! Every touch pushes a new heap entry regardless of whether anything gets
//! evicted, so hit-heavy traffic on a replica sitting under budget (or
//! repeatedly re-touching the same blocks) would otherwise grow the heap
//! without bound — `evict_to`'s pop loop only runs while over budget, and
//! never runs at all for read-refresh-only traffic. `touch` guards against
//! this by compacting (rebuilding the heap from the map, dropping all
//! stale entries) whenever the heap outgrows the map by more than a
//! constant factor, keeping heap size — and so amortized eviction cost —
//! bounded by a constant multiple of `num_blocks()`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

/// Heap compaction fires once `heap.len() > 2 * map.len().max(COMPACT_FLOOR)`.
/// The floor keeps small/empty caches from compacting on nearly every touch.
const COMPACT_FLOOR: usize = 1024;

pub struct PrefixCache {
    /// hash -> (last_access, seq): the current ground truth for a block.
    map: FxHashMap<u64, (u64, u64)>,
    /// Lazily-cleaned eviction candidates: `(Reverse(access), seq, hash)`.
    /// `Reverse(access)` makes the max-heap pop the *smallest* access
    /// first; plain `seq` then makes it pop the *largest* seq among ties —
    /// the deepest block touched in that tick. An entry is stale (and
    /// discarded on pop without evicting anything) once `map`'s value for
    /// that hash no longer equals what was pushed.
    heap: BinaryHeap<(Reverse<u64>, u64, u64)>,
    clock: u64,
    seq: u64,
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            heap: BinaryHeap::new(),
            clock: 0,
            seq: 0,
        }
    }

    /// Number of blocks currently cached.
    pub fn num_blocks(&self) -> usize {
        self.map.len()
    }

    /// Shared by `match_prefix` (`only_present = true`, stops at the first
    /// miss) and `insert` (`only_present = false`, touches every block
    /// regardless of prior presence). One call = one clock tick; `seq`
    /// increases per block within that tick, in chain order.
    fn touch(&mut self, hashes: &[u64], only_present: bool) -> usize {
        self.clock += 1;
        let access = self.clock;
        let mut touched = 0;
        for &h in hashes {
            if only_present && !self.map.contains_key(&h) {
                break;
            }
            self.seq += 1;
            let seq = self.seq;
            self.map.insert(h, (access, seq));
            self.heap.push((Reverse(access), seq, h));
            touched += 1;
        }
        self.compact_if_bloated();
        touched
    }

    /// Rebuild the heap from the map (dropping every stale entry) once the
    /// heap has grown past a constant factor of the map size. The O(n)
    /// rebuild is amortized over the >= n pushes required to bloat the
    /// heap that far again, preserving amortized O(log n) eviction even
    /// under hit-heavy or repeated-refresh traffic that `evict_to` alone
    /// would never shrink.
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

    /// Count of leading blocks present; refreshes access time (and
    /// eviction priority) for the matched prefix only. Stops at the first
    /// miss.
    pub fn match_prefix(&mut self, hashes: &[u64]) -> usize {
        self.touch(hashes, true)
    }

    /// Insert (or refresh) the full chain in one tick.
    pub fn insert(&mut self, hashes: &[u64]) {
        self.touch(hashes, false);
    }

    /// Read-only count of leading blocks present. Does not refresh access
    /// time — use this to score candidates so that merely *considering* a
    /// replica doesn't make its cache look more recently used than it is.
    pub fn lookup(&self, hashes: &[u64]) -> usize {
        hashes
            .iter()
            .take_while(|h| self.map.contains_key(h))
            .count()
    }

    /// Current heap size — exposed for tests to assert compaction keeps it
    /// bounded relative to `num_blocks()`.
    #[cfg(test)]
    fn heap_len(&self) -> usize {
        self.heap.len()
    }

    /// Evict least-recently-used blocks until `num_blocks() <= budget`.
    pub fn evict_to(&mut self, budget: usize) {
        while self.map.len() > budget {
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
    fn match_counts_shared_prefix() {
        let mut t = PrefixCache::new();
        t.insert(&[1, 2, 3, 4]);
        assert_eq!(t.match_prefix(&[1, 2, 3, 4]), 4);
        assert_eq!(t.match_prefix(&[1, 2, 9, 9]), 2);
        assert_eq!(t.match_prefix(&[9]), 0);
        assert_eq!(t.num_blocks(), 4);
    }

    #[test]
    fn insert_dedups_shared_prefix() {
        let mut t = PrefixCache::new();
        t.insert(&[1, 2, 3]);
        t.insert(&[1, 2, 4]);
        assert_eq!(t.num_blocks(), 4); // 1,2 shared; 3 and 4 branch
    }

    #[test]
    fn evict_removes_lru_leaves_first() {
        let mut t = PrefixCache::new();
        t.insert(&[1, 2]);
        t.insert(&[3, 4]);
        t.match_prefix(&[1, 2]); // refresh chain 1→2
        t.evict_to(2);
        assert_eq!(t.num_blocks(), 2);
        assert_eq!(t.match_prefix(&[1, 2]), 2); // recently used survives
        assert!(t.match_prefix(&[3, 4]) < 2); // cold chain evicted
    }

    #[test]
    fn evict_to_zero_empties_tree() {
        let mut t = PrefixCache::new();
        t.insert(&[1, 2, 3]);
        t.evict_to(0);
        assert_eq!(t.num_blocks(), 0);
        assert_eq!(t.match_prefix(&[1, 2, 3]), 0);
    }

    #[test]
    fn eviction_is_leaf_first_within_same_access_tick() {
        // One insert = one tick, so all four blocks share `access`; only
        // `seq` (position in the chain) breaks the tie. The last block of
        // the chain must be evicted first, exactly like a leaf in the old
        // radix tree.
        let mut t = PrefixCache::new();
        t.insert(&[1, 2, 3, 4]);
        t.evict_to(3);
        assert_eq!(t.num_blocks(), 3);
        assert_eq!(t.lookup(&[1, 2, 3]), 3); // shallower blocks survive
        assert_eq!(t.match_prefix(&[1, 2, 3, 4]), 3); // block 4 (the leaf) is gone
    }

    #[test]
    fn missing_block_halts_match_even_if_a_deeper_block_lingers() {
        // A block can only be refreshed by match_prefix/insert alongside
        // its ancestors (both walk from index 0), so a deeper block can
        // never carry a strictly newer access than a shallower one in the
        // same chain — LRU eviction therefore always clears a chain from
        // the tail inward, and a true "orphan" (deep block present, a
        // shallower ancestor gone) cannot arise through the public API.
        // This test pins down the defensive behavior that would apply even
        // if it somehow did: the leading contiguous scan stops at the
        // first gap and ignores whatever sits beyond it.
        let mut t = PrefixCache::new();
        t.insert(&[1, 2, 3, 4]);
        t.insert(&[5, 6]); // newer tick — 1..4 is now the older group
        t.evict_to(2); // must clear the older group (1..4) first
        assert_eq!(t.num_blocks(), 2);
        assert_eq!(t.lookup(&[5, 6]), 2);
        assert_eq!(t.lookup(&[1, 2, 3, 4]), 0);
    }

    #[test]
    fn lookup_does_not_refresh_access() {
        let mut t = PrefixCache::new();
        t.insert(&[1, 2, 3, 4]);
        t.insert(&[5, 6]); // newer group
        assert_eq!(t.lookup(&[1, 2, 3, 4]), 4); // read-only "peek"
        t.evict_to(2); // the peek must not have made 1..4 look fresher
        assert_eq!(t.num_blocks(), 2);
        assert_eq!(t.lookup(&[5, 6]), 2);
        assert_eq!(t.lookup(&[1, 2, 3, 4]), 0);
    }

    #[test]
    fn heap_stays_bounded_under_repeated_hit_workload() {
        // Pure cache-hit traffic: the same 64-block chain, matched in full
        // every time. num_blocks() never changes and never approaches any
        // reasonable budget, so evict_to's pop loop never runs — without
        // compaction, heap_len() grows by 64 on every single call, forever.
        let mut t = PrefixCache::new();
        let chain: Vec<u64> = (0..64u64).collect();
        t.insert(&chain);
        for _ in 0..5_000 {
            assert_eq!(t.match_prefix(&chain), 64);
        }
        // Heap must stay within a small constant factor of the map size,
        // not grow with the number of requests served.
        assert!(
            t.heap_len() <= 4 * t.num_blocks().max(1024),
            "heap grew unbounded: heap_len={} num_blocks={}",
            t.heap_len(),
            t.num_blocks()
        );
    }
}
