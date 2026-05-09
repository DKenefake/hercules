window.BENCHMARK_DATA = {
  "lastUpdate": 1778349208804,
  "repoUrl": "https://github.com/DKenefake/hercules",
  "entries": {
    "hercules-criterion": [
      {
        "commit": {
          "author": {
            "email": "Dustin.Kenefake@gmail.com",
            "name": "Dustin Kenefake",
            "username": "DKenefake"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d8cbf5b9611a3434b9835a58ff4656f4e582daed",
          "message": "Merge pull request #10 from DKenefake/benchreorg\n\nRework the benchmarks",
          "timestamp": "2026-05-09T08:20:45-04:00",
          "tree_id": "2d5c2fda477eb31ea10b6b7c92f756c3712d21c5",
          "url": "https://github.com/DKenefake/hercules/commit/d8cbf5b9611a3434b9835a58ff4656f4e582daed"
        },
        "date": 1778329461503,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 1202,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1710,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 830,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 898,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 118636,
            "range": "± 166",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 1024,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 104408,
            "range": "± 317",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 27568,
            "range": "± 149",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 57473,
            "range": "± 257",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 793594,
            "range": "± 2488",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 398380,
            "range": "± 3084",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 84080,
            "range": "± 1053",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 49566358,
            "range": "± 105546",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 300617343,
            "range": "± 2862686",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "Dustin.Kenefake@gmail.com",
            "name": "Dustin Kenefake",
            "username": "DKenefake"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e03baa21d5986895a4909be8c357cb4ed7808fa3",
          "message": "Merge pull request #11 from DKenefake/preprocessupgrade\n\nperf upgrades to preprocessing, probing, add benchs",
          "timestamp": "2026-05-09T13:47:12-04:00",
          "tree_id": "572e2e40a942116f4fdcb638d6e4204bcfa3992f",
          "url": "https://github.com/DKenefake/hercules/commit/e03baa21d5986895a4909be8c357cb4ed7808fa3"
        },
        "date": 1778349208525,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 1187,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1619,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 852,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 924,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 80793,
            "range": "± 1147",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 961,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 104440,
            "range": "± 161",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 23224,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 14796,
            "range": "± 120",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/25",
            "value": 1751757002,
            "range": "± 7627353",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/50",
            "value": 3508937394,
            "range": "± 5203381",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/100",
            "value": 6971506044,
            "range": "± 19196916",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo_heavy/test_large",
            "value": 285506714,
            "range": "± 347647",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 372284,
            "range": "± 2244",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 1130420,
            "range": "± 10420",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 499406,
            "range": "± 4013",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 31364856,
            "range": "± 127189",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 67486659,
            "range": "± 181437",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}