window.BENCHMARK_DATA = {
  "lastUpdate": 1778436436903,
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
          "id": "54c0bbdb6f3498618abdd3d9958cabc9d191bb68",
          "message": "Merge pull request #12 from DKenefake/roofdual\n\nAdding roofdual solver",
          "timestamp": "2026-05-09T21:10:16-04:00",
          "tree_id": "56fcfc55aed82481717a3f4707aacb11bff0b37a",
          "url": "https://github.com/DKenefake/hercules/commit/54c0bbdb6f3498618abdd3d9958cabc9d191bb68"
        },
        "date": 1778375842421,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 1176,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1662,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 858,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 942,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 79528,
            "range": "± 937",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 894,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 103084,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 71140,
            "range": "± 271",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 12869,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/25",
            "value": 1780888621,
            "range": "± 2818554",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/50",
            "value": 3572347926,
            "range": "± 21034448",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/100",
            "value": 7081928064,
            "range": "± 3154384",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo_heavy/test_large",
            "value": 288144960,
            "range": "± 200294",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 410664,
            "range": "± 1366",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 1272595,
            "range": "± 14795",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 562505,
            "range": "± 9039",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 31776440,
            "range": "± 200633",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka1b",
            "value": 11092838,
            "range": "± 86656",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka2b",
            "value": 53712433,
            "range": "± 140305",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 65409567,
            "range": "± 241249",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka7a",
            "value": 42395803,
            "range": "± 227886",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/bqp50",
            "value": 428745054,
            "range": "± 2896561",
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
          "id": "8305615b528fe513df30ba7263dad3566ddb7219",
          "message": "Merge pull request #13 from DKenefake/roofdualbug\n\nfix the linear error problem",
          "timestamp": "2026-05-10T00:44:11-04:00",
          "tree_id": "c7a832220b34546e06e70cd87bbd704a74800f17",
          "url": "https://github.com/DKenefake/hercules/commit/8305615b528fe513df30ba7263dad3566ddb7219"
        },
        "date": 1778388625893,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 982,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1290,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 684,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 738,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 63803,
            "range": "± 1025",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 783,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 81734,
            "range": "± 258",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 23274,
            "range": "± 551",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 11296,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/25",
            "value": 1362647590,
            "range": "± 421873",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/50",
            "value": 2731476958,
            "range": "± 7058394",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/100",
            "value": 5423492489,
            "range": "± 15392925",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo_heavy/test_large",
            "value": 221293454,
            "range": "± 273499",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 358872,
            "range": "± 3438",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 870081,
            "range": "± 9482",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 391307,
            "range": "± 8540",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 10172765,
            "range": "± 325576",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka1b",
            "value": 9924262,
            "range": "± 20332",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka2b",
            "value": 48189884,
            "range": "± 706916",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 27430414,
            "range": "± 238555",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka7a",
            "value": 19779048,
            "range": "± 75559",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/bqp50",
            "value": 399664677,
            "range": "± 2942383",
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
          "id": "d08ffd8f96a7369cd904fc126468d80612df5757",
          "message": "Merge pull request #14 from DKenefake/usefree\n\nMake the side bounding problem to be optionally the roof dual",
          "timestamp": "2026-05-10T13:59:58-04:00",
          "tree_id": "0415a7df605c06594fbae3cba0e12b332a1d6e6e",
          "url": "https://github.com/DKenefake/hercules/commit/d08ffd8f96a7369cd904fc126468d80612df5757"
        },
        "date": 1778436436178,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 1258,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1642,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 878,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 945,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 81295,
            "range": "± 542",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 1001,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 105766,
            "range": "± 295",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 29962,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 14239,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/25",
            "value": 1768873927,
            "range": "± 1150094",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/50",
            "value": 3547825235,
            "range": "± 18724815",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/100",
            "value": 7037533504,
            "range": "± 3456111",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo_heavy/test_large",
            "value": 287062069,
            "range": "± 360142",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 92043,
            "range": "± 277",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 1128875,
            "range": "± 3891",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 498107,
            "range": "± 412",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 2085298,
            "range": "± 20125",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka1b",
            "value": 12750542,
            "range": "± 26692",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka2b",
            "value": 61663904,
            "range": "± 150937",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 30745269,
            "range": "± 66874",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka7a",
            "value": 25067026,
            "range": "± 537226",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/bqp50",
            "value": 465088215,
            "range": "± 4246884",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}