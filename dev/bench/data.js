window.BENCHMARK_DATA = {
  "lastUpdate": 1780288290903,
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
          "id": "d5534e1adee086737cd112949692cff6866dfcd5",
          "message": "Merge pull request #15 from DKenefake/hapgd-integration\n\nadd the new subsolver",
          "timestamp": "2026-05-30T16:48:46-04:00",
          "tree_id": "337bd11611412287392ec2195d4183ab64687f08",
          "url": "https://github.com/DKenefake/hercules/commit/d5534e1adee086737cd112949692cff6866dfcd5"
        },
        "date": 1780174550201,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 1184,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1693,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 833,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 898,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 79148,
            "range": "± 955",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 972,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 103056,
            "range": "± 216",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 22511,
            "range": "± 81",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 14354,
            "range": "± 169",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/25",
            "value": 1724771839,
            "range": "± 922446",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/50",
            "value": 3457930380,
            "range": "± 6030503",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/100",
            "value": 6858270196,
            "range": "± 44854464",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo_heavy/test_large",
            "value": 280331215,
            "range": "± 543525",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 89840,
            "range": "± 287",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 1148266,
            "range": "± 4462",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 496150,
            "range": "± 1742",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 1698391,
            "range": "± 16976",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka1b",
            "value": 10754289,
            "range": "± 41506",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka2b",
            "value": 56830075,
            "range": "± 124628",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 25613812,
            "range": "± 98594",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka7a",
            "value": 20862626,
            "range": "± 74944",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/bqp50",
            "value": 468515806,
            "range": "± 2780884",
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
          "id": "b543179f40ea1ce0f5e853a4429fcd67413c0823",
          "message": "Merge pull request #16 from DKenefake/hashimprovement\n\nMove to fx hashing",
          "timestamp": "2026-05-31T00:23:02-04:00",
          "tree_id": "e92babb0353eec9665620c63f80ccccc7153e35e",
          "url": "https://github.com/DKenefake/hercules/commit/b543179f40ea1ce0f5e853a4429fcd67413c0823"
        },
        "date": 1780201724176,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 1258,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1698,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 874,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 944,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 24210,
            "range": "± 696",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 1045,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 24760,
            "range": "± 247",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 24571,
            "range": "± 181",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 12014,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/25",
            "value": 398221571,
            "range": "± 3527009",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/50",
            "value": 799243107,
            "range": "± 3965577",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/100",
            "value": 1591905505,
            "range": "± 11167768",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo_heavy/test_large",
            "value": 66247775,
            "range": "± 123230",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 73548,
            "range": "± 142",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 1160876,
            "range": "± 6122",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 485831,
            "range": "± 3060",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 398630,
            "range": "± 7470",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka1b",
            "value": 6207154,
            "range": "± 45269",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka2b",
            "value": 29748300,
            "range": "± 239325",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 16155799,
            "range": "± 77822",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka7a",
            "value": 12841254,
            "range": "± 56486",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/bqp50",
            "value": 250258404,
            "range": "± 463301",
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
          "id": "03c87ce66a0389f46bc42134f113686ed995c1ac",
          "message": "Merge pull request #17 from DKenefake/perfbranch\n\nthe union of a log of different improvements",
          "timestamp": "2026-06-01T00:25:51-04:00",
          "tree_id": "875b123506c3b9b9573e71adcddb6c800b3937bd",
          "url": "https://github.com/DKenefake/hercules/commit/03c87ce66a0389f46bc42134f113686ed995c1ac"
        },
        "date": 1780288290613,
        "tool": "cargo",
        "benches": [
          {
            "name": "helpers/eval_usize/128",
            "value": 1458,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/eval_grad_usize/128",
            "value": 1866,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_flip_objective/64",
            "value": 875,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "helpers/one_step_local_search/64",
            "value": 946,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/iterative_persistence/64",
            "value": 23426,
            "range": "± 270",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/disconnected_graphs/64",
            "value": 973,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/solve_small_components/64",
            "value": 24798,
            "range": "± 87",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/enumerate_solve/10",
            "value": 22843,
            "range": "± 164",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo/64",
            "value": 11562,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/25",
            "value": 414330818,
            "range": "± 313283",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/50",
            "value": 832790262,
            "range": "± 6334385",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/probe_limited/test_large/100",
            "value": 1649451883,
            "range": "± 8176743",
            "unit": "ns/iter"
          },
          {
            "name": "preprocess/preprocess_qubo_heavy/test_large",
            "value": 68391786,
            "range": "± 81239",
            "unit": "ns/iter"
          },
          {
            "name": "solver/process_node/64",
            "value": 71181,
            "range": "± 235",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/random64",
            "value": 1143345,
            "range": "± 5560",
            "unit": "ns/iter"
          },
          {
            "name": "solver/convex_symmetric_form/gka6a",
            "value": 482201,
            "range": "± 916",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/random96",
            "value": 403431,
            "range": "± 8117",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka1b",
            "value": 6107509,
            "range": "± 9693",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka2b",
            "value": 29547777,
            "range": "± 85128",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka6a",
            "value": 15431763,
            "range": "± 54136",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/gka7a",
            "value": 12186721,
            "range": "± 67482",
            "unit": "ns/iter"
          },
          {
            "name": "solver/branch_bound_solve/bqp50",
            "value": 247151942,
            "range": "± 227747",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}