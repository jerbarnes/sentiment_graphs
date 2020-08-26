BEGIN {
  FS = " \\| ";
  file = ARGV[1];
}

/^Cues:/ {
  cp = $6; cr = $7; cf = $8;
}

/^Scopes\(cue match\):/ {
  sp = $6; sr = $7; sf = $8;
}

/^Scopes\(no cue match\):/ {
  snp = $6; snr = $7; snf = $8;
}

/^Scope tokens\(no cue match\):/ {
  tp = $6; tr = $7; tf = $8;
}


/^Negated\(no cue match\):/ {
  ep = $6; er = $7; ef = $8;
}


/^Full negation:/ {
  np = $6; nr = $7; nf = $8;
}

END {
  #printf("%s\n%s %s %s\t%s %s %s\t%s %s %s\t%s %s %s\t%s %s %s\n", file, cp, cr, cf, sp, sr, sf, tp, tr, tf, ep, er, ef, np, nr, nf);
  printf("%s\t\t%s\t%s\t%s\t%s\t%s\t%s\n", file, cf, sf, snf, tf, ef, nf);
}
