import subprocess
from collections import defaultdict
import numpy as np


def shell_command(command):
    peval = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT)
    stdout,stderr = peval.communicate()
    return stdout.decode("utf-8")


def make_oes(gold_star, pred_star):
    with open(gold_star) as gh, open(pred_star) as ph:
        gsen = []
        psen = []
        counter = 0
        shell_command("rm -f madness/*starsem".split())
        oes = defaultdict(lambda : {x: 
            {y: 0 for y in "gold system tp fp fn".split()}
            for x in "Cues#Scopes(cue match)#Scopes(no cue match)#Scope tokens#Negated#Full negation".split("#")})
        for gl, pl in zip(gh, ph):
            gl = gl.strip()
            pl = pl.strip()
            if gl == "":
                with open("./madness/g.{}.starsem".format(counter), "w") as go, open("./madness/p.{}.starsem".format(counter), "w") as po:
                    print("\n".join(gsen), end="\n\n", file=go)
                    print("\n".join(psen), end="\n\n", file=po)
                    gsen = []
                    psen = []
                
                peval = shell_command(f"perl eval.cd-sco.pl -g madness/g.{counter}.starsem -s madness/p.{counter}.starsem".split())
                for line in peval.split("\n"):
                    for category in oes[counter].keys():
                        if line.startswith(category):
                            line = line.split("|")
                            a,b,c,d,e = line[:5]
                            a = a.split(":")[1]
                            oes[counter][category]["gold"] = int(a)
                            oes[counter][category]["system"] = int(b)
                            oes[counter][category]["tp"] = int(c)
                            oes[counter][category]["fp"] = int(d)
                            oes[counter][category]["fn"] = int(e)
                            break


                counter += 1
                print(counter, end="\r")
            else:
                gsen.append(gl)
                psen.append(pl)
        print(f"Done {counter} sentences")
    return oes

def compute_scores_vectorized(counts):
    if len(counts.shape) == 3:
        scores = np.zeros((counts.shape[0], 6, 3))
    elif len(counts.shape) == 2:
        scores = np.zeros((6, 3))
    else:
        raise Exception
    prec = lambda x: x[...,2] / (x[...,2] + x[...,3])
    reca = lambda x: x[...,2] / (x[...,2] + x[...,4])
    fsco = lambda p,r: 2*p*r / (p + r)
    for i in range(counts.shape[0]):
        scores[i,...,0,] = prec(counts[i])
        scores[i,...,1] = reca(counts[i])
        scores[i,...,2] = fsco(scores[i,...,0], scores[i,...,1])
    return scores

def bootstrap_vectorized(oes1, oes2, b=10):
    import time
    c2i = {k: i for i,k in enumerate(oes1[0])}
    m2i = {k: i for i,k in enumerate(oes1[0]["Cues"])}
    m1 = np.zeros((len(oes1), len(c2i), len(m2i)))
    m2 = np.zeros((len(oes2), len(c2i), len(m2i)))
    
    for i in oes1:
        for k1 in oes1[i]:
            for k2 in oes1[i][k1]:
                m1[i,c2i[k1],m2i[k2]] = oes1[i][k1][k2]
                m2[i,c2i[k1],m2i[k2]] = oes2[i][k1][k2]

    s = time.time()
    samples_ids = np.random.choice(m1.shape[0], m1.shape[0]*b).reshape(b, m1.shape[0])
    print(time.time() - s)
    s = time.time()
    samples = np.zeros((m1.shape[0], b))
    for j in range(b):
        for i in range(m1.shape[0]):
            samples[samples_ids[j,i], j] += 1
    
    print(time.time() - s)
    s = time.time()
    evals1 = (np.einsum('ijk,il->ljk', m1, samples))
    
    evals2 = (np.einsum('ijk,il->ljk', m2, samples))
    print(time.time() - s)
    s = time.time()
    
    sample_scores1 = compute_scores_vectorized(evals1)
    sample_scores2 = compute_scores_vectorized(evals2)

    print(time.time() - s)
    s = time.time()
    oscores1 = compute_scores_vectorized(np.sum(m1, axis=0))
    oscores2 = compute_scores_vectorized(np.sum(m2, axis=0))

    deltas = oscores1 - oscores2
    deltas *= 2

    diffs = sample_scores1 - sample_scores2
    diffs_plus = np.where(diffs >= 0, diffs, 0)
    diffs_minus = np.where(diffs < 0, diffs, 0)

    deltas_plus = np.where(deltas > 0, deltas, np.float("inf"))

    deltas_minus = np.where(deltas < 0, deltas, -np.float("inf"))
    s1 = np.sum(diffs_plus > deltas_plus, axis=0)
    s2 = np.sum(diffs_minus < deltas_minus, axis=0)
    print(time.time() - s)

    print(oscores1)
    print(oscores2)
    
    print(s1 / b)
    print(s2 / b)


if __name__ == "__main__":
    import sys
    gold_star_1 = sys.argv[1]
    pred_star_1 = sys.argv[2]

    gold_star_2 = sys.argv[3]
    pred_star_2 = sys.argv[4]

    oes1 = make_oes(gold_star_1, pred_star_1)
    oes2 = make_oes(gold_star_2, pred_star_2)
    
    bootstrap_vectorized(oes1, oes2, int(sys.argv[5]))
