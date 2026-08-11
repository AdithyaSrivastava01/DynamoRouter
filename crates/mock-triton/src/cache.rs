use std::collections::HashMap;

pub struct KvCacheSim {
    capacity: usize,
    map: HashMap<u64, u64>, // block hash -> last access
    clock: u64,
}

impl KvCacheSim {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, map: HashMap::new(), clock: 0 }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns count of leading blocks already cached, then caches all blocks.
    pub fn lookup_insert(&mut self, hashes: &[u64]) -> usize {
        self.clock += 1;
        let cached = hashes.iter().take_while(|h| self.map.contains_key(h)).count();
        for &h in hashes {
            self.map.insert(h, self.clock);
        }
        while self.map.len() > self.capacity {
            if let Some((&victim, _)) = self.map.iter().min_by_key(|(_, &t)| t) {
                self.map.remove(&victim);
            } else {
                break;
            }
        }
        cached
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
}
