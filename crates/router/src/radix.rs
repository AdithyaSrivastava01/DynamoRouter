use std::collections::HashMap;

#[derive(Default)]
struct Node {
    children: HashMap<u64, Node>,
    last_access: u64,
}

pub struct RadixTree {
    root: Node,
    clock: u64,
    num_blocks: usize,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self { root: Node::default(), clock: 0, num_blocks: 0 }
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Count of leading blocks present; refreshes access time along the path.
    pub fn match_prefix(&mut self, hashes: &[u64]) -> usize {
        self.clock += 1;
        let clock = self.clock;
        let mut node = &mut self.root;
        let mut matched = 0;
        for h in hashes {
            match node.children.get_mut(h) {
                Some(child) => {
                    child.last_access = clock;
                    matched += 1;
                    node = child;
                }
                None => break,
            }
        }
        matched
    }

    pub fn insert(&mut self, hashes: &[u64]) {
        self.clock += 1;
        let clock = self.clock;
        let mut created = 0;
        let mut node = &mut self.root;
        for &h in hashes {
            if let std::collections::hash_map::Entry::Vacant(e) = node.children.entry(h) {
                e.insert(Node::default());
                created += 1;
            }
            let child = node.children.get_mut(&h).unwrap();
            child.last_access = clock;
            node = child;
        }
        self.num_blocks += created;
    }

    /// Evict least-recently-used leaves until num_blocks <= budget.
    pub fn evict_to(&mut self, budget: usize) {
        while self.num_blocks > budget {
            let mut path = Vec::new();
            let mut best: Option<(u64, Vec<u64>)> = None;
            Self::min_leaf(&self.root, &mut path, &mut best);
            let Some((_, leaf_path)) = best else { break };
            self.remove_leaf(&leaf_path);
        }
    }

    fn min_leaf(node: &Node, path: &mut Vec<u64>, best: &mut Option<(u64, Vec<u64>)>) {
        for (k, child) in &node.children {
            path.push(*k);
            if child.children.is_empty() {
                if best.as_ref().is_none_or(|(a, _)| child.last_access < *a) {
                    *best = Some((child.last_access, path.clone()));
                }
            } else {
                Self::min_leaf(child, path, best);
            }
            path.pop();
        }
    }

    fn remove_leaf(&mut self, path: &[u64]) {
        let Some((&last, parents)) = path.split_last() else { return };
        let mut node = &mut self.root;
        for k in parents {
            match node.children.get_mut(k) {
                Some(child) => node = child,
                None => return,
            }
        }
        if node.children.remove(&last).is_some() {
            self.num_blocks -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_counts_shared_prefix() {
        let mut t = RadixTree::new();
        t.insert(&[1, 2, 3, 4]);
        assert_eq!(t.match_prefix(&[1, 2, 3, 4]), 4);
        assert_eq!(t.match_prefix(&[1, 2, 9, 9]), 2);
        assert_eq!(t.match_prefix(&[9]), 0);
        assert_eq!(t.num_blocks(), 4);
    }

    #[test]
    fn insert_dedups_shared_prefix() {
        let mut t = RadixTree::new();
        t.insert(&[1, 2, 3]);
        t.insert(&[1, 2, 4]);
        assert_eq!(t.num_blocks(), 4); // 1,2 shared; 3 and 4 branch
    }

    #[test]
    fn evict_removes_lru_leaves_first() {
        let mut t = RadixTree::new();
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
        let mut t = RadixTree::new();
        t.insert(&[1, 2, 3]);
        t.evict_to(0);
        assert_eq!(t.num_blocks(), 0);
        assert_eq!(t.match_prefix(&[1, 2, 3]), 0);
    }
}
