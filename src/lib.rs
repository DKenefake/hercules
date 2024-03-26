#![crate_name = "hercules"]
#![doc = include_str!("../README.md")] // imports the readme as the front page of the documentation
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
#![allow(non_snake_case)] // some of the variable names are taken from papers, and are not snake case
#![allow(clippy::let_and_return)] // in many instances we are recreating equations as written in papers, and helps with readability
#![allow(dead_code)] // this is a library module, so until all the tests are implemented this will needlessly warn
#![allow(clippy::must_use_candidate)] // somewhat of a nuisance introduced by clippy::pedantic
#![allow(clippy::doc_markdown)] // breaks some of the documentation written in latex
#![allow(clippy::match_bool)]
#![allow(clippy::suboptimal_flops)] // far to many false positives, e.g. vector and matrix multiplication suggestions
#![allow(clippy::similar_names)]
// I think this is fine, as the names are similar for a reason, mostly in the Constraints function
// I just think this is fine, I think having all possible actions shown in one place simplifies view
#![allow(clippy::module_name_repetitions)] // some names are just repeated, and that is fine

const VERSION: &str = env!("CARGO_PKG_VERSION");

use pyo3::prelude::*;

mod branchbound;
pub mod branchbound_utils;
mod branchboundlogger;
mod constraint;
pub mod constraint_reduction;
pub mod initial_points;
mod kopt;
pub mod local_search;
pub mod local_search_utils;
pub mod persistence;
pub mod python_interopt;
pub mod qubo;
pub mod utils;
pub mod variable_reduction;

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
    m.add_function(wrap_pyfunction!(get_persistence, m)?)?;
    m.add_function(wrap_pyfunction!(solve_branch_bound, m)?)?;
    m.add_function(wrap_pyfunction!(convex_symmetric_form, m)?)?;
    m.add_function(wrap_pyfunction!(generate_rule_1_1, m)?)?;
    m.add_function(wrap_pyfunction!(generate_rule_2_1, m)?)?;
    Ok(())
}

/// This is the test module. It is used to test the code in the library
#[cfg(test)]
mod tests {
    use crate::qubo::Qubo;
    use ndarray::Array1;
    use rayon::prelude::*;
    use smolprng::*;

    pub(crate) fn make_solver_qubo() -> Qubo {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        Qubo::make_random_qubo(50, &mut prng, 0.1)
    }

    pub(crate) fn get_min_obj(p: &Qubo, xs: &Vec<Array1<f64>>) -> f64 {
        xs.par_iter()
            .map(|x| p.eval(&x))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    pub(crate) fn make_test_prng() -> PRNG<JsfLarge> {
        PRNG {
            generator: JsfLarge::default(),
        }
    }
}
