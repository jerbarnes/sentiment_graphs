#!/usr/bin/env python
# coding: utf-8

import numpy as np
from bootstrap_single import read_data, fill_scores, color

setups = ["point_to_root", "head_first", "head_first-inside_label",
          "head_final", "head_final-inside_label",
          "head_final-inside_label-dep_edges",
          "head_final-inside_label-dep_edges-dep_labels"]


def get_best(all_scores):
    overall = 0
    ov_setup_r_i = ("", -1)
    for setup in all_scores:
        best_setup = 0
        se_setup_r_i = ("", -1)
        for r_i in range(1, 6):
            print(f"run for {setup} was {r_i}")
            print(all_scores[setup][r_i])
            if all_scores[setup][r_i][0][-1] > best_setup:
                se_setup_r_i = (setup, r_i)
                best_setup = all_scores[setup][r_i][0][-1]
            if all_scores[setup][r_i][0][-1] > overall:
                ov_setup_r_i = (setup, r_i)
                overall = all_scores[setup][r_i][0][-1]
        print(f"{color.BLUE}Best run for {setup} was {se_setup_r_i[1]}{color.END}")
        print(all_scores[setup][se_setup_r_i[1]])
    print()
    print(f"{color.RED}Best overall run was {ov_setup_r_i[0]} with {ov_setup_r_i[0]}{color.END}")
    print(all_scores[ov_setup_r_i[0]][ov_setup_r_i[1]])
    return


def best_run(golddir, preddir):
    all_scores = {}
    for setup in setups:
        all_scores[setup] = {}
        for r_i in [1, 2, 3, 4, 5]:
            L, n = read_data(golddir, preddir, setup, r_i)

            n_features = int(len(L) / n)
            M = np.array(L).reshape(int(len(L) / n_features), n_features)

            scores = fill_scores(np.zeros((1, 8)), np.sum(M, axis=0))
            all_scores[setup][r_i] = scores
    get_best(all_scores)


if __name__ == "__main__":
    import sys
    golddir = sys.argv[1]
    preddir = sys.argv[2]
    best_run(golddir, preddir)
