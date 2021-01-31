#! /usr/bin/bash

lang=norec;
a=head_final-inside_label-dep_edges
b=head_final-inside_label
python bootstrap.py ../data/sent_graphs/$lang/ ../experiments/$lang/ $a $b;

lang=eu;
a=head_final-inside_label-dep_edges
b=head_final-inside_label-dep_edges-dep_labels
python bootstrap.py ../data/sent_graphs/$lang/ ../experiments/$lang/ $a $b;

lang=ca;
a=head_first
b=head_first-inside_label
python bootstrap.py ../data/sent_graphs/$lang/ ../experiments/$lang/ $a $b;

lang=mpqa;
a=head_final
b=head_final-inside_label
python bootstrap.py ../data/sent_graphs/$lang/ ../experiments/$lang/ $a $b;

lang=ds_unis;
a=head_final-inside_label-dep_edges
b=head_final
python bootstrap.py ../data/sent_graphs/$lang/ ../experiments/$lang/ $a $b;
