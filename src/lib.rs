#![crate_name = "hercules"]
#![doc = include_str!("../README.md")] // imports the readme as the front page of the documentation
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
#![allow(non_snake_case)] // some of the variable names are taken from papers, and are not snake case
#![allow(clippy::let_and_return)] // in many instances we are recreating equations as written in papers, and helps with readability
// #![allow(dead_code)] // this is a library module, so until all the tests are implemented this will needlessly warn
#![allow(clippy::must_use_candidate)] // somewhat of a nuisance introduced by clippy::pedantic
#![allow(clippy::doc_markdown)] // breaks some of the documentation written in latex
#![allow(clippy::match_bool)]
#![allow(clippy::suboptimal_flops)] // far to many false positives, e.g. vector and matrix multiplication suggestions
#![allow(clippy::similar_names)] // we are using similar names to the papers we are implementing
#![allow(clippy::implicit_hasher)] // we are using the default hasher
#![allow(clippy::needless_pass_by_value)]
// This is fine as it is lighting up functions that will eventually be interfaces that will consume the values
#![allow(clippy::cast_precision_loss)] // we are casting floats to ints, and this is fine as the max int is 1
#![allow(clippy::cast_sign_loss)] // we are casting floats to uints, and this is fine as the lowerbound is 0
#![allow(clippy::cast_possible_truncation)] // we are casting floats to uints, and this is fine as the max int is 1
#![allow(clippy::module_name_repetitions)] // some names are just repeated, and that is fine

const VERSION: &str = env!("CARGO_PKG_VERSION");

use pyo3::prelude::*;

mod branch_node;
mod branch_stratagy;
mod branch_subproblem;
mod branchbound;
pub mod branchbound_utils;
mod branchboundlogger;
mod constraint;
pub mod constraint_reduction;
pub mod initial_points;
mod kopt;
pub mod local_search;
pub mod local_search_utils;
mod lower_bound;
pub mod persistence;
mod preprocess;
pub mod python_interopt;
pub mod qubo;
mod solver_options;
pub mod utils;
pub mod variable_reduction;
mod early_termination;

// imports to generate the python interface

#[allow(clippy::wildcard_imports)]
// wildcard importing makes sense are we are importing everything anyway
use python_interopt::*;

/// Gives python access to the rust interface
#[pymodule]
fn hercules(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    use smolprng::*;

    pub(crate) fn make_solver_qubo() -> Qubo {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        Qubo::make_random_qubo(50, &mut prng, 0.1)
    }

    pub(crate) fn get_min_obj(p: &Qubo, xs: &Vec<Array1<usize>>) -> f64 {
        // get minimum objective value
        xs.into_iter()
            .fold(f64::INFINITY, |acc, x| p.eval_usize(&x).min(acc))
    }

    pub(crate) fn make_test_prng() -> PRNG<JsfLarge> {
        PRNG {
            generator: JsfLarge::default(),
        }
    }

    // pub(crate) fn pardalos_hard_problem(n:usize) -> Qubo{
    //
    // }
}
