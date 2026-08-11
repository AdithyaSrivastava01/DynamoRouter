//! Owns all per-replica routing state: health, in-flight load, and each
//! replica's [`PrefixCache`] (the router's best guess at what that replica
//! actually has cached). [`Scheduler::pick`] is the entire routing
//! decision for one request; callers wrap the scheduler in a `Mutex` since
//! a decision costs microseconds.

use crate::prefix_cache::PrefixCache;

/// Routing strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Score replicas by (matched cached blocks, inflight load); falls
    /// back to least-loaded routing for cold or barely-matching traffic.
    CacheAware,
    /// Ignore the cache entirely and cycle through replicas in order.
    RoundRobin,
}

/// Tunables for [`Policy::CacheAware`] scoring and per-replica limits.
#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// Weight of matched blocks in the score.
    pub alpha: f64,
    /// Weight of inflight load in the score (subtracted).
    pub beta: f64,
    /// Matches strictly below this bypass scoring and fall back to
    /// least-loaded routing instead, to avoid pinning cold traffic onto a
    /// replica that merely scored best among weak options.
    pub min_match_blocks: usize,
    /// Per-replica cache capacity, in blocks.
    pub block_budget: usize,
    /// Per-replica in-flight request cap (backpressure).
    pub max_inflight: usize,
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
    cache: PrefixCache,
    inflight: usize,
    healthy: bool,
}

/// Why [`Scheduler::pick`] could not route a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickError {
    /// Every replica is currently marked unhealthy.
    NoHealthyReplicas,
    /// At least one replica is healthy, but every healthy replica is
    /// already at `max_inflight` — routing exists, capacity doesn't.
    AllSaturated,
}

/// The outcome of a single [`Scheduler::pick`] call.
#[derive(Clone, Copy, Debug)]
pub struct RouteDecision {
    pub replica: usize,
    pub matched_blocks: usize,
    /// The chosen replica's in-flight count immediately after this pick
    /// (i.e. it includes the request just routed). Lets callers update a
    /// per-replica inflight gauge without taking the scheduler mutex a
    /// second time.
    pub inflight_after: usize,
}

pub struct Scheduler {
    replicas: Vec<ReplicaState>,
    policy: Policy,
    cfg: SchedulerConfig,
    rr_next: usize,
}

impl Scheduler {
    /// Create a scheduler over `n` replicas, all initially healthy and idle.
    pub fn new(n: usize, policy: Policy, cfg: SchedulerConfig) -> Self {
        let replicas = (0..n)
            .map(|_| ReplicaState {
                cache: PrefixCache::new(),
                inflight: 0,
                healthy: true,
            })
            .collect();
        Self {
            replicas,
            policy,
            cfg,
            rr_next: 0,
        }
    }

    /// Route one request. Returns `Err(PickError::NoHealthyReplicas)` if
    /// every replica is unhealthy, or `Err(PickError::AllSaturated)` if at
    /// least one is healthy but all healthy replicas are at their
    /// `max_inflight` cap. On success, bumps the chosen replica's inflight
    /// count — callers must pair this with exactly one
    /// [`Scheduler::complete`] call for that replica.
    pub fn pick(&mut self, hashes: &[u64]) -> Result<RouteDecision, PickError> {
        let n = self.replicas.len();
        let mut avail_mask = vec![false; n];
        let mut any_available = false;
        let mut any_healthy = false;
        for (i, r) in self.replicas.iter().enumerate() {
            if r.healthy {
                any_healthy = true;
                if r.inflight < self.cfg.max_inflight {
                    avail_mask[i] = true;
                    any_available = true;
                }
            }
        }
        if !any_available {
            return Err(if any_healthy {
                PickError::AllSaturated
            } else {
                PickError::NoHealthyReplicas
            });
        }

        let (idx, matched) = match self.policy {
            Policy::RoundRobin => {
                let mut idx = None;
                for off in 0..n {
                    let cand = (self.rr_next + off) % n;
                    if avail_mask[cand] {
                        idx = Some(cand);
                        self.rr_next = (cand + 1) % n;
                        break;
                    }
                }
                // any_available is true, so the wrap-around scan above is
                // guaranteed to land on some healthy-and-available replica.
                (
                    idx.expect("any_available guarantees a round-robin candidate"),
                    0,
                )
            }
            Policy::CacheAware => {
                let available: Vec<usize> = (0..n).filter(|&i| avail_mask[i]).collect();
                let mut best = available[0];
                let mut best_score = f64::MIN;
                let mut best_matched = 0;
                for &i in &available {
                    // Read-only: scoring a replica must not make its cache
                    // look more recently used than it really is, or every
                    // pick would refresh every losing candidate's LRU too.
                    let matched = self.replicas[i].cache.lookup(hashes);
                    let score = self.cfg.alpha * matched as f64
                        - self.cfg.beta * self.replicas[i].inflight as f64;
                    if score > best_score {
                        best_score = score;
                        best = i;
                        best_matched = matched;
                    }
                }
                // Intentional: this checks the argmax-by-score replica's
                // match count, not the best match count across all
                // replicas. A replica could hold a longer match yet still
                // lose on score to heavier inflight load; falling back
                // based on ITS match count instead of the winner's would
                // re-pin traffic onto a busy replica. Comparing the
                // winner's own match count is what keeps the anti-pinning
                // behavior intact.
                if best_matched < self.cfg.min_match_blocks {
                    // Cold (or barely-matching) traffic: pure least-loaded,
                    // to avoid pinning based on a weak match.
                    let &least = available
                        .iter()
                        .min_by_key(|&&i| self.replicas[i].inflight)
                        .unwrap();
                    let least_matched = self.replicas[least].cache.lookup(hashes);
                    (least, least_matched)
                } else {
                    (best, best_matched)
                }
            }
        };

        let r = &mut self.replicas[idx];
        if self.policy == Policy::CacheAware {
            // RoundRobin never reads the cache, so maintaining it here
            // would be pure overhead.
            r.cache.insert(hashes);
            r.cache.evict_to(self.cfg.block_budget);
        }
        r.inflight += 1;
        let inflight_after = r.inflight;
        Ok(RouteDecision {
            replica: idx,
            matched_blocks: matched,
            inflight_after,
        })
    }

    /// Mark one in-flight request against `replica` as finished. Must be
    /// called exactly once per successful `pick()` of that replica, on
    /// every completion path (success and error) — otherwise inflight
    /// counts drift and the replica looks permanently busier than it is.
    /// If a phantom completion lands after `mark_unhealthy` has already
    /// reset inflight to 0, `saturating_sub` absorbs the underflow; that
    /// small accepted skew is harmless.
    pub fn complete(&mut self, replica: usize) {
        let r = &mut self.replicas[replica];
        r.inflight = r.inflight.saturating_sub(1);
    }

    /// Take a replica out of routing. Its in-flight count is reset to 0
    /// (those requests will never complete against a dead replica) and its
    /// cache is dropped (assumed lost, e.g. on restart).
    pub fn mark_unhealthy(&mut self, replica: usize) {
        let r = &mut self.replicas[replica];
        r.healthy = false;
        r.inflight = 0;
        r.cache = PrefixCache::new();
    }

    /// Re-admit a replica to routing. Its cache starts cold.
    pub fn mark_healthy(&mut self, replica: usize) {
        self.replicas[replica].healthy = true;
    }

    /// Whether `replica` currently participates in routing.
    pub fn is_healthy(&self, replica: usize) -> bool {
        self.replicas[replica].healthy
    }

    /// Current in-flight request count for `replica`.
    pub fn inflight(&self, replica: usize) -> usize {
        self.replicas[replica].inflight
    }

    /// Number of blocks currently in `replica`'s cache picture.
    pub fn replica_blocks(&self, replica: usize) -> usize {
        self.replicas[replica].cache.num_blocks()
    }

    /// Total number of replicas known to the scheduler.
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
        // Each pick has a distinct, never-seen prefix, so every request is
        // cold (matched_blocks 0 everywhere) and the least-loaded fallback
        // fires each time, spreading the three requests one per replica.
        let a = s.pick(&[100, 101]).unwrap().replica;
        let b = s.pick(&[200, 201]).unwrap().replica;
        let c = s.pick(&[300, 301]).unwrap().replica;
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
    fn unhealthy_replica_skipped_and_cache_dropped() {
        let mut s = sched(2, Policy::CacheAware);
        let hashes: Vec<u64> = (1..=8).collect();
        let d = s.pick(&hashes).unwrap();
        s.complete(d.replica);
        s.mark_unhealthy(d.replica);
        let d2 = s.pick(&hashes).unwrap();
        assert_ne!(d2.replica, d.replica);
        s.complete(d2.replica);
        s.mark_healthy(d.replica);
        // cache was dropped: no match credit on re-admitted replica
        let d3 = s.pick(&hashes).unwrap();
        assert_eq!(d3.replica, d2.replica); // pinned to the one that has it now
    }

    #[test]
    fn all_replicas_at_cap_returns_all_saturated() {
        let cfg = SchedulerConfig {
            max_inflight: 1,
            ..Default::default()
        };
        let mut s = Scheduler::new(2, Policy::CacheAware, cfg);
        s.pick(&[1, 2]).unwrap();
        s.pick(&[3, 4]).unwrap();
        assert_eq!(s.pick(&[5, 6]).unwrap_err(), PickError::AllSaturated);
    }

    #[test]
    fn no_healthy_replicas_returns_no_healthy_replicas_error() {
        let mut s = sched(2, Policy::CacheAware);
        s.mark_unhealthy(0);
        s.mark_unhealthy(1);
        assert_eq!(s.pick(&[1, 2]).unwrap_err(), PickError::NoHealthyReplicas);
    }

    #[test]
    fn eviction_respects_block_budget() {
        let cfg = SchedulerConfig {
            block_budget: 4,
            ..Default::default()
        };
        let mut s = Scheduler::new(1, Policy::CacheAware, cfg);
        for i in 0..10u64 {
            let d = s.pick(&[i * 10, i * 10 + 1]).unwrap();
            s.complete(d.replica);
        }
        assert!(s.replica_blocks(0) <= 4);
    }

    #[test]
    fn fallback_reports_actual_matched_blocks_not_hardcoded_zero() {
        let cfg = SchedulerConfig {
            min_match_blocks: 5,
            ..Default::default()
        };
        let mut s = Scheduler::new(2, Policy::CacheAware, cfg);
        let seed = s.pick(&[1, 2, 3]).unwrap();
        assert_eq!(seed.replica, 0);
        s.complete(seed.replica);

        // matched (3) is below min_match_blocks (5), so this still routes
        // via the least-loaded fallback — both replicas are tied at 0
        // inflight, so the tie-break lands back on replica 0, the one
        // holding the 3-block chain. The fallback's reported
        // matched_blocks must be that replica's real match count, not a
        // hardcoded 0.
        let d = s.pick(&[1, 2, 3]).unwrap();
        assert_eq!(d.replica, 0);
        assert_eq!(d.matched_blocks, 3);
    }

    #[test]
    fn matched_equal_to_min_match_blocks_routes_cache_aware_not_fallback() {
        let mut s = sched(2, Policy::CacheAware);
        let hashes: Vec<u64> = vec![1, 2]; // length == default min_match_blocks

        // Bump replica 0's inflight so the seeding pick's least-loaded
        // fallback is forced onto replica 1 (a genuine tie-break, not a
        // coincidence of iteration order).
        let warm = s.pick(&[999, 998]).unwrap();
        assert_eq!(warm.replica, 0);

        let seed = s.pick(&hashes).unwrap(); // cold: fallback picks replica 1
        assert_eq!(seed.replica, 1);

        s.complete(warm.replica);
        s.complete(seed.replica);

        // Both replicas are idle now; replica 1 has the chain cached with
        // an exact match count of min_match_blocks. `<` (not `<=`) must
        // treat this as a real cache hit and route by score, not fall back
        // to the least-loaded tie-break (which would land on replica 0).
        let d = s.pick(&hashes).unwrap();
        assert_eq!(d.replica, 1);
        assert_eq!(d.matched_blocks, 2);
    }

    #[test]
    fn empty_hashes_routes_least_loaded_without_panicking() {
        let mut s = sched(3, Policy::CacheAware);
        let d = s.pick(&[]).unwrap();
        assert_eq!(d.matched_blocks, 0);
        assert!(d.replica < 3);
    }

    #[test]
    fn round_robin_skips_unhealthy_replica() {
        let mut s = sched(3, Policy::RoundRobin);
        s.mark_unhealthy(1);
        let hashes: Vec<u64> = vec![1, 2];
        let picks: Vec<usize> = (0..4)
            .map(|_| {
                let d = s.pick(&hashes).unwrap();
                s.complete(d.replica);
                d.replica
            })
            .collect();
        assert_eq!(picks, vec![0, 2, 0, 2]);
    }
}
