use crate::radix::RadixTree;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    CacheAware,
    RoundRobin,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    pub alpha: f64,         // weight of matched blocks
    pub beta: f64,          // weight of inflight load
    pub min_match_blocks: usize, // below this → least-loaded fallback
    pub block_budget: usize,     // per-replica cache capacity (blocks)
    pub max_inflight: usize,     // per-replica backpressure cap
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 2.0,
            min_match_blocks: 2,
            block_budget: 65_536,
            max_inflight: 512,
        }
    }
}

struct ReplicaState {
    tree: RadixTree,
    inflight: usize,
    healthy: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RouteDecision {
    pub replica: usize,
    pub matched_blocks: usize,
}

pub struct Scheduler {
    replicas: Vec<ReplicaState>,
    policy: Policy,
    cfg: SchedulerConfig,
    rr_next: usize,
}

impl Scheduler {
    pub fn new(n: usize, policy: Policy, cfg: SchedulerConfig) -> Self {
        let replicas = (0..n)
            .map(|_| ReplicaState { tree: RadixTree::new(), inflight: 0, healthy: true })
            .collect();
        Self { replicas, policy, cfg, rr_next: 0 }
    }

    pub fn pick(&mut self, hashes: &[u64]) -> Option<RouteDecision> {
        let available: Vec<usize> = self
            .replicas
            .iter()
            .enumerate()
            .filter(|(_, r)| r.healthy && r.inflight < self.cfg.max_inflight)
            .map(|(i, _)| i)
            .collect();
        if available.is_empty() {
            return None;
        }

        let (idx, matched) = match self.policy {
            Policy::RoundRobin => {
                let n = self.replicas.len();
                let mut idx = None;
                for off in 0..n {
                    let cand = (self.rr_next + off) % n;
                    if available.contains(&cand) {
                        idx = Some(cand);
                        self.rr_next = (cand + 1) % n;
                        break;
                    }
                }
                (idx?, 0)
            }
            Policy::CacheAware => {
                let mut best = available[0];
                let mut best_score = f64::MIN;
                let mut best_matched = 0;
                for &i in &available {
                    let matched = self.replicas[i].tree.match_prefix(hashes);
                    let score = self.cfg.alpha * matched as f64
                        - self.cfg.beta * self.replicas[i].inflight as f64;
                    if score > best_score {
                        best_score = score;
                        best = i;
                        best_matched = matched;
                    }
                }
                if best_matched < self.cfg.min_match_blocks {
                    // cold traffic: pure least-loaded to avoid pinning
                    let &least = available
                        .iter()
                        .min_by_key(|&&i| self.replicas[i].inflight)
                        .unwrap();
                    (least, 0)
                } else {
                    (best, best_matched)
                }
            }
        };

        let r = &mut self.replicas[idx];
        r.tree.insert(hashes);
        r.tree.evict_to(self.cfg.block_budget);
        r.inflight += 1;
        Some(RouteDecision { replica: idx, matched_blocks: matched })
    }

    pub fn complete(&mut self, replica: usize) {
        let r = &mut self.replicas[replica];
        r.inflight = r.inflight.saturating_sub(1);
    }

    pub fn mark_unhealthy(&mut self, replica: usize) {
        let r = &mut self.replicas[replica];
        r.healthy = false;
        r.tree = RadixTree::new(); // cache assumed lost on restart
    }

    pub fn mark_healthy(&mut self, replica: usize) {
        self.replicas[replica].healthy = true;
    }

    pub fn is_healthy(&self, replica: usize) -> bool {
        self.replicas[replica].healthy
    }

    pub fn inflight(&self, replica: usize) -> usize {
        self.replicas[replica].inflight
    }

    pub fn replica_blocks(&self, replica: usize) -> usize {
        self.replicas[replica].tree.num_blocks()
    }

    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(n: usize, policy: Policy) -> Scheduler {
        Scheduler::new(n, policy, SchedulerConfig::default())
    }

    #[test]
    fn same_prefix_pins_to_same_replica() {
        let mut s = sched(4, Policy::CacheAware);
        let hashes: Vec<u64> = (1..=8).collect();
        let d1 = s.pick(&hashes).unwrap();
        s.complete(d1.replica);
        let d2 = s.pick(&hashes).unwrap();
        assert_eq!(d1.replica, d2.replica);
        assert_eq!(d2.matched_blocks, 8);
    }

    #[test]
    fn cold_traffic_goes_least_loaded() {
        let mut s = sched(3, Policy::CacheAware);
        // load replica 0 and 1 with one inflight each, replica 2 free
        let a = s.pick(&[100, 101]).unwrap().replica;
        let b = s.pick(&[200, 201]).unwrap().replica;
        let c = s.pick(&[300, 301]).unwrap().replica;
        // three cold requests with distinct prefixes spread across all three replicas
        let mut set = [a, b, c];
        set.sort();
        assert_eq!(set, [0, 1, 2]);
    }

    #[test]
    fn round_robin_cycles_and_ignores_cache() {
        let mut s = sched(3, Policy::RoundRobin);
        let hashes: Vec<u64> = (1..=4).collect();
        let picks: Vec<usize> = (0..6)
            .map(|_| {
                let d = s.pick(&hashes).unwrap();
                s.complete(d.replica);
                d.replica
            })
            .collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn unhealthy_replica_skipped_and_tree_dropped() {
        let mut s = sched(2, Policy::CacheAware);
        let hashes: Vec<u64> = (1..=8).collect();
        let d = s.pick(&hashes).unwrap();
        s.complete(d.replica);
        s.mark_unhealthy(d.replica);
        let d2 = s.pick(&hashes).unwrap();
        assert_ne!(d2.replica, d.replica);
        s.complete(d2.replica);
        s.mark_healthy(d.replica);
        // tree was dropped: no match credit on re-admitted replica
        let d3 = s.pick(&hashes).unwrap();
        assert_eq!(d3.replica, d2.replica); // pinned to the one that has it now
    }

    #[test]
    fn all_replicas_at_cap_returns_none() {
        let cfg = SchedulerConfig { max_inflight: 1, ..Default::default() };
        let mut s = Scheduler::new(2, Policy::CacheAware, cfg);
        s.pick(&[1, 2]).unwrap();
        s.pick(&[3, 4]).unwrap();
        assert!(s.pick(&[5, 6]).is_none());
    }

    #[test]
    fn eviction_respects_block_budget() {
        let cfg = SchedulerConfig { block_budget: 4, ..Default::default() };
        let mut s = Scheduler::new(1, Policy::CacheAware, cfg);
        for i in 0..10u64 {
            let d = s.pick(&[i * 10, i * 10 + 1]).unwrap();
            s.complete(d.replica);
        }
        assert!(s.replica_blocks(0) <= 4);
    }
}
