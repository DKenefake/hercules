use crate::qubo::Qubo;
use std::collections::HashMap;

/// Given a QUBO and a set of fixed variables, find all disconnected subgraphs in the QUBO graph
/// that do not include any fixed variables. Each subgraph is represented as a vector of variable
/// indices.
pub fn get_all_disconnected_graphs(
    qubo: &Qubo,
    fixed_vars: &HashMap<usize, usize>,
) -> Vec<Vec<usize>> {
    let mut visited = (0..qubo.num_x())
        .map(|x| fixed_vars.contains_key(&x))
        .collect::<Vec<bool>>();
    let v_check = (0..qubo.num_x())
        .filter(|x| !fixed_vars.contains_key(x))
        .collect::<Vec<usize>>();

    let mut output = Vec::new();

    fn dfs(
        v: usize,
        component: &mut Vec<usize>,
        visited: &mut Vec<bool>,
        qubo: &Qubo,
        fixed_vars: &HashMap<usize, usize>,
    ) {
        visited[v] = true;

        component.push(v);

        let q_i = qubo.q.outer_view(v).unwrap();
        for (j, _) in q_i.iter() {
            if !visited[j] && !fixed_vars.contains_key(&j) {
                dfs(j, component, visited, qubo, fixed_vars);
            }
        }
    }

    for v in v_check {
        if !visited[v] {
            let mut component = Vec::new();

            dfs(v, &mut component, &mut visited, qubo, fixed_vars);

            if !component.is_empty() {
                output.push(component);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use crate::graph_utils::get_all_disconnected_graphs;
    use crate::qubo::Qubo;
    use sprs::TriMat;
    use std::collections::HashMap;

    #[test]
    fn test_disconnected_graphs_1() {
        let mut q_tri = TriMat::<f64>::new((5, 5));

        // edges 0 - 1 - 2 and 3 - 4
        q_tri.add_triplet(0, 1, 1.0);
        q_tri.add_triplet(1, 0, 1.0);
        q_tri.add_triplet(1, 2, 1.0);
        q_tri.add_triplet(2, 1, 1.0);
        q_tri.add_triplet(3, 4, 1.0);
        q_tri.add_triplet(4, 3, 1.0);

        let p = Qubo::new(q_tri.to_csc());

        let fixed_vars = HashMap::new();
        let components = get_all_disconnected_graphs(&p, &fixed_vars);

        // we expect two components: [0, 1, 2] and [3, 4]
        assert_eq!(components.len(), 2);
        assert!(components
            .iter()
            .any(|c| c.len() == 3 && c.contains(&0) && c.contains(&1) && c.contains(&2)));
        assert!(components
            .iter()
            .any(|c| c.len() == 2 && c.contains(&3) && c.contains(&4)));

        // now test with fixing variable 1
        let mut fixed_vars = HashMap::new();
        fixed_vars.insert(1, 0);
        let components = get_all_disconnected_graphs(&p, &fixed_vars);

        // we expect three components: [0], [2], and [3, 4]
        assert_eq!(components.len(), 3);
        assert!(components.iter().any(|c| c.len() == 1 && c.contains(&0)));
        assert!(components.iter().any(|c| c.len() == 1 && c.contains(&2)));
        assert!(components
            .iter()
            .any(|c| c.len() == 2 && c.contains(&3) && c.contains(&4)));
    }

    #[test]
    fn test_disconnected_graphs_2() {
        // test a star graph
        let mut q_tri = TriMat::<f64>::new((5, 5));

        // edges 0 - 1, 0 - 2, 0 - 3, 0 - 4
        for i in 1..5 {
            q_tri.add_triplet(0, i, 1.0);
            q_tri.add_triplet(i, 0, 1.0);
        }

        let p = Qubo::new(q_tri.to_csc());

        let fixed_vars = HashMap::new();
        let components = get_all_disconnected_graphs(&p, &fixed_vars);
        // we expect one component: [0, 1, 2, 3, 4]
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 5);

        // now test with fixing variable 0
        let mut fixed_vars = HashMap::new();
        fixed_vars.insert(0, 0);

        let components = get_all_disconnected_graphs(&p, &fixed_vars);
        // we expect four components: [1], [2], [3], and [4]
        assert_eq!(components.len(), 4);
        for i in 1..5 {
            assert!(components.iter().any(|c| c.len() == 1 && c.contains(&i)));
        }
    }

    #[test]
    fn test_disconnected_graphs_3() {
        // we are testing a circle graph
        let mut q_tri = TriMat::<f64>::new((4, 4));

        // edges 0 - 1 - 2 - 3 - 0
        q_tri.add_triplet(0, 1, 1.0);
        q_tri.add_triplet(1, 0, 1.0);
        q_tri.add_triplet(1, 2, 1.0);
        q_tri.add_triplet(2, 1, 1.0);
        q_tri.add_triplet(2, 3, 1.0);
        q_tri.add_triplet(3, 2, 1.0);
        q_tri.add_triplet(3, 0, 1.0);
        q_tri.add_triplet(0, 3, 1.0);
        let p = Qubo::new(q_tri.to_csc());
        let fixed_vars = HashMap::new();
        let components = get_all_disconnected_graphs(&p, &fixed_vars);

        // we expect one component: [0, 1, 2, 3]
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 4);

        // now test with fixing variable 1
        let mut fixed_vars = HashMap::new();
        fixed_vars.insert(1, 0);
        let components = get_all_disconnected_graphs(&p, &fixed_vars);

        // we expect one component: [0, 2, 3]
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }
}
