#!/bin/bash

for dataset in ds_unis mpqa; do
    for setup in head_final head_first; do

        python3 convert_json.py --json_dir $dataset --out_dir sent_graphs/$dataset --setup $setup
        python3 convert.py --json_dir $dataset --out_dir sent_graphs/$dataset --setup $setup --inside_label
        python3 convert.py --json_dir $dataset --out_dir sent_graphs/$dataset --setup $setup --inside_label --use_dep_edges
        python3 convert.py --json_dir $dataset --out_dir sent_graphs/$dataset --setup $setup --inside_label --use_dep_edges --use_dep_labels
    done;
        python3 convert.py --json_dir $dataset --out_dir sent_graphs/$dataset --setup point_to_root
done;
