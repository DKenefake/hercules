#![crate_name = "hercules"]
#![doc = include_str!("../README.md")] // imports the readme as the front page of the documentation
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
#![allow(non_snake_case)] // some of the variable names are taken from papers, and are not snake case
#![allow(clippy::let_and_return)] // in many instances we are recreating equations as written in papers, and helps with readability
#![allow(dead_code)] // this is a library module, so until all the tests are implemented this will needlessly warn
#![allow(clippy::must_use_candidate)] // somewhat of a nuisance introduced by clippy::pedantic
#![allow(clippy::doc_markdown)] // breaks some of the documentation written in latex
#![allow(clippy::match_bool)]
// I just think this is fine, I think having all possible actions shown in one place simplifies view
#![allow(clippy::module_name_repetitions)] // some names are just repeated, and that is fine

use pyo3::prelude::*;

pub mod initial_points;
pub mod local_search;
pub mod local_search_utils;
pub mod python_interopt;
pub mod qubo;
pub mod utils;

// imports to generate the python interface

use python_interopt::*;

/// Gives python access to the rust interface
#[pymodule]
fn hercules(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(pso_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(gls_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(mls_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(msls_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(pso, m)?)?;
    m.add_function(wrap_pyfunction!(gls, m)?)?;
    m.add_function(wrap_pyfunction!(mls, m)?)?;
    m.add_function(wrap_pyfunction!(msls, m)?)?;
    m.add_function(wrap_pyfunction!(read_qubo, m)?)?;
    m.add_function(wrap_pyfunction!(write_qubo, m)?)?;
    Ok(())
}

/// This is the test module. Very few of the tests are actually assert style tests, as we are likely to not hit the same
/// local minima when running the tests (and not break them) every time we change the seed of the prng and order of operations.
#[cfg(test)]
mod tests {
    use crate::qubo::Qubo;
    use crate::{initial_points, local_search, local_search_utils, utils};
    use ndarray::Array1;
    use rayon::prelude::*;
    use smolprng::*;
    use sprs::CsMat;

    fn make_solver_qubo() -> Qubo {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        Qubo::make_random_qubo(50, &mut prng, 0.1)
    }

    fn get_min_obj(p: &Qubo, xs: &Vec<Array1<f64>>) -> f64 {
        xs.par_iter()
            .map(|x| p.eval(&x))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn make_test_prng() -> PRNG<JsfLarge> {
        PRNG {
            generator: JsfLarge::default(),
        }
    }

    #[test]
    fn qubo_init() {
        let eye = CsMat::eye(3);
        let p = Qubo::new(eye);
        print!("{:?}", p.eval(&Array1::from_vec(vec![1.0, 2.0, 3.0])));
    }

    #[test]
    fn qubo_heuristics() {
        let eye = CsMat::eye(3);
        let p = Qubo::new(eye);
        let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let x_1 = local_search::simple_local_search(&p, &x_0, 10);
        print!("{:?}", x_1);
    }

    #[test]
    fn qubo_multi_heuristics() {
        let eye = CsMat::eye(3);
        let p = Qubo::new(eye);
        let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let x_1 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let x_2 = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let xs = vec![x_0, x_1, x_2];

        let x_3 = local_search::multi_simple_local_search(&p, &xs);
        print!("{:?}", x_3);
    }

    #[test]
    fn test_mutate() {
        let mut prng = make_test_prng();
        let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let target = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        let x_1 = utils::mutate_solution(&x_0, 1, &mut prng);

        assert_eq!(x_1, target);
    }

    #[test]
    fn test_flip() {
        let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let target = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let x_1 = utils::invert(&x_0);

        assert_eq!(x_1, target);
    }

    #[test]
    fn test_gen_random() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let starting_points = initial_points::generate_random_starting_points(&p, 5, &mut prng);

        let local_sols = starting_points
            .par_iter()
            .map(|x| local_search::simple_local_search(&p, &x, 500usize))
            .collect::<Vec<Array1<f64>>>();
        let mut local_objs = local_sols.iter().map(|x| p.eval(x)).collect::<Vec<f64>>();
        local_objs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{:?}", local_objs);
    }

    #[test]
    fn test_alpha() {
        let p = make_solver_qubo();
        let alpha = p.alpha();
        println!("{alpha}");
    }
    #[test]
    fn test_rho() {
        let p = make_solver_qubo();
        let rho = p.rho();
        println!("{rho}");
    }

    #[test]
    fn test_opt_criteria() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let mut x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
        for _ in 0..100 {
            x_0 = local_search_utils::get_gain_criteria(&p, &x_0);
            println!("{:?}", p.eval(&x_0));
        }
    }

    #[test]
    fn test_opt_heuristics() {
        let p = make_solver_qubo();
        let mut x_0 = Array1::ones(p.num_x()) * p.alpha();

        x_0 = local_search::simple_gain_criteria_search(&p, &x_0, 100);

        println!("{:?}", p.eval(&x_0));
    }

    #[test]
    fn test_multi_opt_heuristics() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let mut xs = initial_points::generate_random_starting_points(&p, 10, &mut prng);

        xs = local_search::multi_simple_gain_criteria_search(&p, &xs);

        let min_obj = get_min_obj(&p, &xs);
        println!("{:?}", min_obj);
    }

    #[test]
    fn test_multi_opt_heuristics_2() {
        let p = make_solver_qubo();
        let mut xs = vec![
            Array1::ones(p.num_x()) * p.alpha(),
            Array1::ones(p.num_x()) * p.rho(),
        ];

        xs = local_search::multi_simple_gain_criteria_search(&p, &xs);
        let min_obj = get_min_obj(&p, &xs);
        println!("{min_obj:?}");
    }

    #[test]
    fn test_multi_opt_heuristics_3() {
        let p = make_solver_qubo();
        let mut xs = Vec::new();
        for i in 0..1000 {
            xs.push(Array1::ones(p.num_x()) * (i as f64) / 1000.0);
        }

        xs = local_search::multi_simple_gain_criteria_search(&p, &xs);
        let min_obj = get_min_obj(&p, &xs);
        println!("{min_obj:?}");
    }

    #[test]
    fn test_mixed_search() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let mut xs = initial_points::generate_random_starting_points(&p, 10, &mut prng);

        xs = xs
            .par_iter()
            .map(|x| local_search::simple_mixed_search(&p, &x, 1000))
            .collect();
        let min_obj = get_min_obj(&p, &xs);
        println!("{min_obj:?}");
    }

    #[test]
    fn test_particle_swarm() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let x = local_search::particle_swarm_search(&p, 50, 1000, &mut prng);

        println!("{:?}", p.eval(&x));
    }

    #[test]
    fn write_qubo() {
        let mut prng = make_test_prng();

        let p = Qubo::make_random_qubo(10, &mut prng, 0.1);
        Qubo::write_qubo(&p, "test.qubo");
    }

    #[test]
    fn read_qubo() {
        let mut prng = make_test_prng();

        let p = Qubo::make_random_qubo(10, &mut prng, 0.1);
        Qubo::write_qubo(&p, "test_read.qubo");

        let q = Qubo::read_qubo("test_read.qubo");

        assert_eq!(p.q, q.q);
        assert_eq!(p.c, q.c);
    }

    #[test]
    fn large_scale_write_qubo() {
        let mut prng = make_test_prng();

        let p = Qubo::make_random_qubo(1000, &mut prng, 0.01);
        Qubo::write_qubo(&p, "test_large.qubo");
    }

    #[test]
    fn compare_methods() {
        let mut prng = make_test_prng();
        let p = Qubo::make_random_qubo(200, &mut prng, 0.01);

        let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
        let max_iter = p.num_x();

        let x_pso = local_search::particle_swarm_search(&p, 100, max_iter, &mut prng);
        let x_mixed = local_search::simple_mixed_search(&p, &x_0, max_iter);
        let x_gain = local_search::simple_gain_criteria_search(&p, &x_0, max_iter);
        let x_opt = local_search::simple_local_search(&p, &x_0, max_iter);
        let x_rand = local_search::random_search(&p, 100, &mut prng);

        println!(
            "PSO: {:?}, MIXED: {:?}, GAIN: {:?}, 1OPT: {:?}, Rand: {:?} ",
            p.eval(&x_pso),
            p.eval(&x_mixed),
            p.eval(&x_gain),
            p.eval(&x_opt),
            p.eval(&x_rand)
        );
    }
}
