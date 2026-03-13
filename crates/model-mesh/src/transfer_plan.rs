//! Multi-peer transfer scheduling via greedy bin-packing (LPT).
//!
//! When multiple NIXL peers hold the same model weights, we balance the
//! tensor-to-peer assignment so each peer serves roughly the same number
//! of bytes. This cuts transfer time roughly linearly with peer count.
//!
//! Algorithm: Longest Processing Time (LPT) first.
//!   1. Find the tensor intersection across all peers (should be identical
//!      for the same model/tp_rank).
//!   2. Sort tensors descending by size.
//!   3. Assign each tensor to the peer with the smallest running byte total.
//!
//! Single peer: returns an empty vec (no plan needed).

use std::collections::{HashMap, HashSet};

use crate::proto;

/// Compute a balanced tensor-to-peer assignment.
///
/// Returns one `PeerTransferAssignment` per peer, each containing the full
/// peer metadata plus the subset of tensor names that peer should serve.
///
/// Returns an empty vec when there are fewer than 2 peers (no scheduling
/// needed, the caller uses the single-peer fast path).
pub fn compute_transfer_plan(peers: &[proto::PeerNixlMd]) -> Vec<proto::PeerTransferAssignment> {
    if peers.len() < 2 {
        return Vec::new();
    }

    // Build the intersection of tensor names across all peers.
    // For the same model/tp_rank these should be identical, but we
    // intersect defensively.
    let mut sets: Vec<HashSet<&str>> = peers
        .iter()
        .map(|p| p.tensors.iter().map(|t| t.name.as_str()).collect())
        .collect();

    let mut common: HashSet<&str> = sets.remove(0);
    for s in &sets {
        common = common.intersection(s).copied().collect();
    }

    if common.is_empty() {
        return Vec::new();
    }

    // Build a size lookup from the first peer (sizes are identical across peers).
    let size_map: HashMap<&str, u64> = peers[0]
        .tensors
        .iter()
        .filter(|t| common.contains(t.name.as_str()))
        .map(|t| (t.name.as_str(), t.size))
        .collect();

    // Sort tensors descending by size (LPT).
    let mut sorted_tensors: Vec<&str> = common.into_iter().collect();
    sorted_tensors.sort_by(|a, b| {
        let sa = size_map.get(a).copied().unwrap_or(0);
        let sb = size_map.get(b).copied().unwrap_or(0);
        sb.cmp(&sa).then_with(|| a.cmp(b)) // tie-break by name for determinism
    });

    // Greedy bin-packing: assign each tensor to the peer with the
    // smallest running byte total.
    let n = peers.len();
    let mut totals = vec![0u64; n];
    let mut assignments: Vec<Vec<String>> = vec![Vec::new(); n];

    for name in sorted_tensors {
        let size = size_map.get(name).copied().unwrap_or(0);
        // find the peer with the smallest total
        let min_idx = totals
            .iter()
            .enumerate()
            .min_by_key(|(_, t)| **t)
            .map(|(i, _)| i)
            .unwrap_or(0);
        assignments[min_idx].push(name.to_string());
        totals[min_idx] += size;
    }

    peers
        .iter()
        .zip(assignments)
        .map(|(peer, assigned)| proto::PeerTransferAssignment {
            agent_name: peer.agent_name.clone(),
            nixl_md: peer.nixl_md.clone(),
            tensors: peer.tensors.clone(),
            assigned_tensors: assigned,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::proto;
    use crate::transfer_plan::compute_transfer_plan;

    fn make_peer(name: &str, tensors: &[(&str, u64)]) -> proto::PeerNixlMd {
        proto::PeerNixlMd {
            agent_name: name.to_string(),
            nixl_md: vec![0xCA, 0xFE],
            tensors: tensors
                .iter()
                .map(|(n, s)| proto::TensorInfo {
                    name: n.to_string(),
                    size: *s,
                    ..Default::default()
                })
                .collect(),
        }
    }

    #[test]
    fn single_peer_returns_empty() {
        let peers = vec![make_peer("peer-0", &[("w", 100)])];
        assert!(compute_transfer_plan(&peers).is_empty());
    }

    #[test]
    fn empty_peers_returns_empty() {
        assert!(compute_transfer_plan(&[]).is_empty());
    }

    #[test]
    fn two_peers_balanced() {
        // 4 tensors: 100, 80, 60, 40 = 280 total
        // LPT: peer-0 gets 100+40=140, peer-1 gets 80+60=140
        let tensors = &[
            ("layer.0", 100),
            ("layer.1", 80),
            ("layer.2", 60),
            ("layer.3", 40),
        ];
        let peers = vec![make_peer("peer-0", tensors), make_peer("peer-1", tensors)];

        let plan = compute_transfer_plan(&peers);
        assert_eq!(plan.len(), 2);

        let p0: u64 = plan[0]
            .assigned_tensors
            .iter()
            .map(|n| tensors.iter().find(|(t, _)| t == n).unwrap().1)
            .sum();
        let p1: u64 = plan[1]
            .assigned_tensors
            .iter()
            .map(|n| tensors.iter().find(|(t, _)| t == n).unwrap().1)
            .sum();

        // Should be perfectly balanced: 140 each
        assert_eq!(p0, 140);
        assert_eq!(p1, 140);
    }

    #[test]
    fn three_peers_reasonably_balanced() {
        // 6 tensors: 100, 90, 80, 70, 60, 50 = 450 total
        // Ideal: 150 each
        let tensors = &[
            ("a", 100),
            ("b", 90),
            ("c", 80),
            ("d", 70),
            ("e", 60),
            ("f", 50),
        ];
        let peers = vec![
            make_peer("p0", tensors),
            make_peer("p1", tensors),
            make_peer("p2", tensors),
        ];

        let plan = compute_transfer_plan(&peers);
        assert_eq!(plan.len(), 3);

        let totals: Vec<u64> = plan
            .iter()
            .map(|p| {
                p.assigned_tensors
                    .iter()
                    .map(|n| tensors.iter().find(|(t, _)| t == n).unwrap().1)
                    .sum()
            })
            .collect();

        // LPT should give 100+50=150, 90+60=150, 80+70=150
        assert_eq!(totals[0], 150);
        assert_eq!(totals[1], 150);
        assert_eq!(totals[2], 150);
    }

    #[test]
    fn unequal_sizes_assigns_all() {
        // One huge tensor, one tiny: 1000 + 1 = 1001
        let tensors = &[("big", 1000), ("small", 1)];
        let peers = vec![make_peer("p0", tensors), make_peer("p1", tensors)];

        let plan = compute_transfer_plan(&peers);
        let total_assigned: usize = plan.iter().map(|p| p.assigned_tensors.len()).sum();
        assert_eq!(total_assigned, 2);
    }

    #[test]
    fn preserves_peer_metadata() {
        let tensors = &[("w", 100)];
        let peers = vec![make_peer("p0", tensors), make_peer("p1", tensors)];

        let plan = compute_transfer_plan(&peers);
        assert_eq!(plan[0].agent_name, "p0");
        assert_eq!(plan[1].agent_name, "p1");
        assert_eq!(plan[0].nixl_md, vec![0xCA, 0xFE]);
        assert!(!plan[0].tensors.is_empty());
    }

    #[test]
    fn disjoint_tensors_returns_empty() {
        // Peers have no tensors in common
        let p0 = make_peer("p0", &[("only_a", 100)]);
        let p1 = make_peer("p1", &[("only_b", 100)]);
        assert!(compute_transfer_plan(&[p0, p1]).is_empty());
    }

    #[test]
    fn partial_overlap_uses_intersection() {
        // peer-0 has {a, b, c}, peer-1 has {b, c, d}
        // intersection is {b, c}
        let p0 = make_peer("p0", &[("a", 100), ("b", 80), ("c", 60)]);
        let p1 = make_peer("p1", &[("b", 80), ("c", 60), ("d", 40)]);

        let plan = compute_transfer_plan(&[p0, p1]);
        assert_eq!(plan.len(), 2);

        let all_assigned: Vec<&str> = plan
            .iter()
            .flat_map(|p| p.assigned_tensors.iter().map(|s| s.as_str()))
            .collect();
        assert!(all_assigned.contains(&"b"));
        assert!(all_assigned.contains(&"c"));
        assert!(!all_assigned.contains(&"a"));
        assert!(!all_assigned.contains(&"d"));
    }
}
