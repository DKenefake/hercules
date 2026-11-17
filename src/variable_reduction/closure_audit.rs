//! Offline diagnostic, deliberately not part of the solve path.
use super::*;
use crate::constraint::ImplicationGraph;
use crate::persistence::cold_gradient_probe;
use std::time::{Duration, Instant};

fn close(
    prepared: &PreparedPreprocess,
    graph: &ImplicationGraph,
    mut fixed: FixedVarMap,
) -> Option<FixedVarMap> {
    loop {
        let before = fixed.len();
        if !graph.propagate(&mut fixed) {
            return None;
        }
        fixed = cold_gradient_probe(prepared.qubo(), fixed, prepared.adjacency());
        if fixed.len() == before {
            return Some(fixed);
        }
    }
}

#[derive(Default)]
struct Audit {
    fixed: FixedVarMap,
    relations: Vec<Constraint>,
    pairs: usize,
    rounds: usize,
    failed: usize,
    complete: bool,
}

fn audit(q: &Qubo, initial: &FixedVarMap, use_relations: bool, budget: Duration) -> Audit {
    let start = Instant::now();
    let prepared = prepare_preprocess(q, false);
    let mut result = Audit {
        fixed: initial.clone(),
        ..Default::default()
    };
    loop {
        let before = (result.fixed.len(), result.relations.len());
        let candidates = select_probe_candidates(prepared.qubo(), &result.fixed, usize::MAX);
        result.rounds += 1;
        for i in candidates {
            if start.elapsed() >= budget {
                return result;
            }
            if result.fixed.contains_key(&i) {
                continue;
            }
            let graph = ImplicationGraph::new(
                q.num_x(),
                if use_relations {
                    &result.relations
                } else {
                    &[]
                },
            );
            let sides: [Option<FixedVarMap>; 2] = std::array::from_fn(|v| {
                let mut fixed = result.fixed.clone();
                fixed.insert(i, v);
                close(&prepared, &graph, fixed)
            });
            result.pairs += 1;
            for (v, side) in sides.iter().enumerate() {
                if side.is_none() {
                    result.failed += 1;
                    result.fixed.insert(i, 1 - v);
                }
            }
            assert!(sides.iter().any(Option::is_some), "lost every optimum");
            match (&sides[0], &sides[1]) {
                (Some(a), Some(b)) => {
                    for (&j, &v) in a {
                        if b.get(&j) == Some(&v) {
                            result.fixed.insert(j, v);
                        }
                    }
                }
                (Some(a), None) | (None, Some(a)) => result.fixed.extend(a),
                _ => unreachable!(),
            }
            // Each conditional strict fixing excludes precisely one binary pair.
            for (v, side) in sides.iter().enumerate() {
                if let Some(side) = side {
                    for (&j, &w) in side {
                        if i == j || result.fixed.contains_key(&j) {
                            continue;
                        }
                        let kind = match (v, w) {
                            (0, 0) => ConstraintType::GreaterThan,
                            (0, 1) => ConstraintType::AtLeastOne,
                            (1, 0) => ConstraintType::NoMoreThanOne,
                            (1, 1) => ConstraintType::LessThan,
                            _ => unreachable!(),
                        };
                        let (a, b, kind) = if i < j {
                            (i, j, kind)
                        } else {
                            (
                                j,
                                i,
                                match kind {
                                    ConstraintType::LessThan => ConstraintType::GreaterThan,
                                    ConstraintType::GreaterThan => ConstraintType::LessThan,
                                    other => other,
                                },
                            )
                        };
                        result.relations.push(Constraint::new(a, b, kind));
                    }
                }
            }
            result.relations.sort_unstable();
            result.relations.dedup();
        }
        let changed =
            result.fixed.len() != before.0 || (use_relations && result.relations.len() != before.1);
        if !changed {
            result.complete = true;
            return result;
        }
    }
}

// Sequential weak choices preserve one optimum, not every optimum. Never export
// these choices as implications. This diagnostic uses exact integer test data.
fn weak_close(q: &Qubo, initial: &FixedVarMap) -> FixedVarMap {
    let prepared = prepare_preprocess(q, false);
    let mut fixed = initial.clone();
    loop {
        let before = fixed.len();
        for i in 0..q.num_x() {
            if fixed.contains_key(&i) {
                continue;
            }
            let mut lo = prepared.qubo().c[i];
            let mut hi = lo;
            for &(j, w) in &prepared.adjacency()[i] {
                if let Some(&v) = fixed.get(&j) {
                    lo += w * v as f64;
                    hi += w * v as f64;
                } else {
                    lo += w.min(0.0);
                    hi += w.max(0.0);
                }
            }
            if lo >= 0.0 {
                fixed.insert(i, 0);
            } else if hi <= 0.0 {
                fixed.insert(i, 1);
            }
        }
        if fixed.len() == before {
            return fixed;
        }
    }
}

#[test]
fn closure_preserves_every_optimum() {
    use smolprng::{JsfLarge, PRNG};
    let mut rng = PRNG {
        generator: JsfLarge::from(789021u64),
    };
    for _ in 0..100 {
        let mut q = Qubo::make_random_qubo(8, &mut rng, 0.4);
        q.q.data_mut()
            .iter_mut()
            .for_each(|v| *v = (*v * 16.0).round());
        q.c.mapv_inplace(|v| (v * 8.0).round());
        for initial in [
            FixedVarMap::default(),
            [(0, 1), (3, 0)].into_iter().collect(),
        ] {
            let result = audit(&q, &initial, true, Duration::from_secs(1));
            assert!(result.complete);
            let assignments: Vec<_> = (0..256)
                .filter(|mask| initial.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
                .map(|mask| Array1::from_iter((0..8).map(|i| (mask >> i) & 1)))
                .collect();
            let best = assignments
                .iter()
                .map(|x| q.eval_usize(x))
                .fold(f64::INFINITY, f64::min);
            let weak = weak_close(&q, &initial);
            let weak_best = assignments
                .iter()
                .filter(|x| weak.iter().all(|(&i, &v)| x[i] == v))
                .map(|x| q.eval_usize(x))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(weak_best, best);
            let mut production = initial.clone();
            prepare_preprocess(&q, false).apply_weak_fixings(&mut production);
            let retained_best = assignments.iter()
                .filter(|x| production.iter().all(|(&i,&v)| x[i] == v))
                .map(|x| q.eval_usize(x)).fold(f64::INFINITY, f64::min);
            assert_eq!(retained_best, best);
            for x in assignments.iter().filter(|x| q.eval_usize(x) == best) {
                let full = (0..8).map(|i| (i, x[i])).collect();
                assert!(result.fixed.iter().all(|(&i, &v)| x[i] == v));
                assert!(result.relations.iter().all(|c| c.check(&full)));
            }
        }
    }
}

#[test]
fn weak_choices_are_not_strict_probe_implications() {
    let q = Qubo::new_with_c(sprs::CsMat::zero((3, 3)), ndarray::array![0.0, 0.5, (1u64 << 51) as f64]);
    let mut fixed = FixedVarMap::default();
    prepare_preprocess(&q, false).apply_weak_fixings(&mut fixed);
    // Fractional and oversized rows are deliberately left to strict presolve.
    assert_eq!(fixed, [(0, 0)].into_iter().collect());
    let (relations, strict) = probe(&q, &FixedVarMap::default(), false);
    assert!(!strict.contains_key(&0));
    assert!(relations.constraints.iter().all(|c| c.x_i != 0 && c.x_j != 0));
}

#[test]
#[ignore = "offline saved-node audit"]
fn saved_node_closure_audit() {
    println!(
        "AUDIT instance,call,mode,initial,fixed,relations,pairs,rounds,failed,complete,micros"
    );
    for id in [3506, 5725, 3642, 3877] {
        let dir = std::path::PathBuf::from(format!("target/fractional-cycle-screen/{id}"));
        let csv = std::fs::read_to_string(dir.join("comparison.csv")).unwrap();
        for line in csv.lines().skip(1) {
            let cols: Vec<_> = line.split(',').collect();
            if cols[2] != "Greedy" {
                continue;
            }
            let call = cols[0];
            let q = Qubo::read_qubo(dir.join(format!("node-{call}.qubo")).to_str().unwrap());
            let initial: FixedVarMap =
                std::fs::read_to_string(dir.join(format!("node-{call}.fix")))
                    .unwrap()
                    .lines()
                    .map(|line| {
                        let s: Vec<_> = line.split_whitespace().collect();
                        (s[0].parse().unwrap(), s[1].parse().unwrap())
                    })
                    .collect();
            let weak = weak_close(&q, &initial);
            println!("WEAK {id},{call},{},{}", initial.len(), weak.len());
            for mode in [false, true] {
                let start = Instant::now();
                let r = audit(&q, &initial, mode, Duration::from_millis(100));
                println!(
                    "AUDIT {id},{call},{mode},{},{},{},{},{},{},{},{}",
                    initial.len(),
                    r.fixed.len(),
                    r.relations.len(),
                    r.pairs,
                    r.rounds,
                    r.failed,
                    r.complete,
                    start.elapsed().as_micros()
                );
            }
        }
    }
}
