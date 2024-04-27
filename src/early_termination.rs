use ndarray::Array1;
use sprs::TriMat;
/// QUBO Criteria for Early Termination
/// In that if we have a sufficiency condition for optimality, we can terminate early



use crate::qubo::Qubo;

/// Sufficiency condition for optimality as given by beck2000 as stated by chen2012. This allows for
/// early termination of the optimization algorithm if the condition is met. The condition is as follows:
///
/// $$2(2X - I)(Qx + c) <= min(Eig(Q))e$$
///
/// Example:
/// ```rust
/// use hercules::qubo::Qubo;
/// use ndarray::Array1;
///
/// let q = sprs::TriMat::new((2, 2));
/// q.add_triplet(0, 0, 1.0);
/// q.add_triplet(1, 1, 1.0);
/// let c = Array1::from_vec(vec![1.0, 1.0]);
/// let p = Qubo::new_with_c(q.to_csc(), c);
/// let x = Array1::from_vec(vec![1, 1]);
/// let suff = hercules::early_termination::beck_proof(&p, &x);
/// ```
pub fn beck_proof(qubo: &Qubo, x: &Array1<usize>) -> bool{
    // 2(2X - I)(Qx + c) <= min(Eig(Q))e
    // Where X is diag(x), lets just get a naive implementation done first

    let x_float = x.mapv(|x| x as f64);
    let qx_b = (&qubo.q * &x_float) + &qubo.c; // Qx + c

    // make the diagonal matrix 2X-I
    let mut X = TriMat::new((qubo.num_x(), qubo.num_x()));
    for i in 0..qubo.num_x(){
        X.add_triplet(i, i, 2.0*x_float[i]-1.0);
    }

    let X = X.to_csc::<usize>();

    // compute the full lhs e.g. 2(2X-I)(Qx + c)
    let lhs = 2.0*(&X * &qx_b);

    // find the minimum eigenvalue of the hessian
    let eigs = qubo.hess_eigenvalues();
    let min_eig = eigs.iter().fold(f64::INFINITY, |acc, &x| x.min(acc));

    // check if the Sufficiency condition is met for each index
    for i in 0..qubo.num_x(){

        // if we do not meet the sufficiency condition, return false
        if lhs[i] > min_eig {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests{
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_beck_proof(){
        // basically the idea here is to make a QUBO where the sufficency condition is met for a point
        // a really easy way to do this is to make a QUBO with a single variable, where the magnitute
        // of Q is dominated by c
        let mut Q = TriMat::new((1, 1));
        Q.add_triplet(0, 0, 0.1);
        let c = -Array1::ones(1);
        let p = Qubo::new_with_c(Q.to_csc(), c);

        // the solution is x = 1
        let cand_sol = Array1::ones(1);

        // the sufficiency condition should be met
        assert!(beck_proof(&p, &cand_sol));
    }
}