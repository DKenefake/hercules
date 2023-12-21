use ndarray::Array1;
use rayon::prelude::*;
use smolprng::*;
use sprs::CsMat;

mod initial_points;
mod local_search;
mod local_search_utils;
mod qubo;
mod qubo_heuristics;
mod utils;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::Qubo;
    use crate::{initial_points, local_search, local_search_utils, utils};

    #[test]
    fn qubo_init() {
        let eye = CsMat::eye(3);
        let p = Qubo::new(eye);
        // print!("{:?}", p.q.to_dense());
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
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
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
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);

        let starting_points = initial_points::generate_random_starting_points(&p, 5, &mut prng);
        // let local_sols = qubo_heuristics::multi_simple_local_search(&p, &starting_points);
        let local_sols = starting_points
            .par_iter()
            .map(|x| local_search::simple_local_search(&p, &x, 500 as usize))
            .collect::<Vec<Array1<f64>>>();
        let mut local_objs = local_sols.iter().map(|x| p.eval(x)).collect::<Vec<f64>>();
        local_objs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{:?}", local_objs);
    }

    #[test]
    fn test_alpha() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let alpha = p.alpha();
        println!("{:?}", alpha);
    }
    #[test]
    fn test_rho() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let rho = p.rho();
        println!("{:?}", rho);
    }

    #[test]
    fn test_opt_criteria() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let mut x_0 = Array1::zeros(p.num_x());
        for _ in 0..100 {
            x_0 = local_search_utils::get_opt_criteria(&p, &x_0);
            println!("{:?}", p.eval(&x_0));
        }
    }

    #[test]
    fn test_opt_heuristics() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let mut x_0 = Array1::ones(p.num_x()) * p.alpha();

        x_0 = local_search::simple_opt_criteria_search(&p, &x_0, 100);

        println!("{:?}", p.eval(&x_0));
    }

    #[test]
    fn test_multi_opt_heuristics() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let mut xs = initial_points::generate_random_starting_points(&p, 1000, &mut prng);

        xs = local_search::multi_simple_opt_criteria_search(&p, &xs);

        let min_obj = xs
            .iter()
            .map(|x| p.eval(&x))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        println!("{:?}", min_obj);
    }

    #[test]
    fn test_multi_opt_heuristics_2() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let mut xs = vec![
            Array1::ones(p.num_x()) * p.alpha(),
            Array1::ones(p.num_x()) * p.rho(),
        ];

        xs = local_search::multi_simple_opt_criteria_search(&p, &xs);
        let min_obj = xs
            .iter()
            .map(|x| p.eval(&x))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        println!("{:?}", min_obj);
    }

    #[test]
    fn test_multi_opt_heuristics_3() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let mut xs = Vec::new();
        for i in 0..1000 {
            xs.push(Array1::ones(p.num_x()) * (i as f64) / 1000.0);
        }

        xs = local_search::multi_simple_opt_criteria_search(&p, &xs);
        let min_obj = xs
            .iter()
            .map(|x| p.eval(&x))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        println!("{:?}", min_obj);
    }

    #[test]
    fn test_mixed_search() {
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };
        let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
        let mut xs = initial_points::generate_random_starting_points(&p, 100, &mut prng);

        xs = xs
            .par_iter()
            .map(|x| local_search::simple_mixed_search(&p, &x, 1000))
            .collect();
        let min_obj = xs
            .iter()
            .map(|x| p.eval(&x))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        println!("{:?}", min_obj);
    }

    // #[test]
    // fn test_boros2007() {
    //     let mut prng = PRNG {
    //         generator: JsfLarge::default(),
    //     };
    //
    //     let p = Qubo::make_random_qubo(500, &mut prng, 0.1);
    //     let x = local_search::boros_2007(&p);
    //
    //     println!("{:?}", x);
    //     println!("{:?}", p.eval(&x));
    // }
}
