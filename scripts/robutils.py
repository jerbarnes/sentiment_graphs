from collections import Counter, defaultdict
import re
import numpy as np
import json

#from pprint import pprint


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def flatten(l):
    return [item for sublist in l for item in sublist]


class ConllU_Entry:
    def __init__(self, id, form, lemma, tag, pos, morph, head, deprel, deps, other):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.lemma = lemma
        self.tag = tag 
        self.pos = pos
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.morph = morph
        self.other = other

    def __repr__(self):
        if self.deps != []:
            deps = "|".join(["{}:{}".format(i,l) for i,l in self.deps])
        else:
            deps = "_"
        return "\t".join([str(self.id), self.form, self.lemma, self.tag, self.pos, self.morph, str(self.head), self.deprel, deps, self.other])

def read_conllu(fh, UD=False, cpos=False):
    ROOT = ConllU_Entry(0, "ROOT", "ROOT", "ROOT", "ROOT", "_", 0, "root", [], "_")
    tokens = [ROOT]
    for line in fh:
        tok = line.strip().split("\t")
        if line.startswith("#"):
            if UD:
                line = line.strip().split()
                if len(line) == 1:
                    graphID = line[0]
                elif line[1] == "sent_id":
                    graphID = "#" + line[3]
                else:
                    continue
            else:
                graphID = line.strip()
        elif tok == [""]:
            if len(tokens) > 1:
                yield (graphID, tokens)
            tokens = [ROOT]
        else:
            try:
                id = int(tok[0])
            except ValueError:
                if UD:
                    continue
                else:
                    raise ValueError
            form = tok[1]
            lemma = tok[2]
            tag = tok[3]
            pos = tok[4]
            if cpos:
                pos = tag
            feat = tok[5]
            try:
                head = int(tok[6])
            except:
                head = "_"
            deprel = tok[7].lower()
            if tok[8] == "_":
                deps = []
            elif UD:
                deps = []
            else:
                deps = []
                for x in tok[8].split("|"):
                    x = x.split(":")
                    i = x[0]
                    l = ":".join(x[1:])
                    deps.append((int(i), l))
            other = tok[9]
            entry = ConllU_Entry(id, form, lemma, tag, pos, feat, head, deprel, deps, other)
            tokens.append(entry)

def conllu_to_matrix(sentence, tree=False):
    N = len(sentence)
    matrix = np.zeros((N, N), dtype=np.int) 
    labels = np.ndarray((N, N), dtype=object)
    if tree:
        for token in sentence[1:]:
            m = token.id
            h = token.head
            matrix[h,m] = 1
            labels[h,m] = token.deprel
    else:
        for token in sentence[1:]:
            m = token.id
            for h,l in token.deps:
                matrix[h,m] = 1
                labels[h,m] = l
    return matrix, labels
    
def edges_to_conllu(sentence, edges):
    for token in sentence:
        token.deps = []
    for h,m,l in edges:
        sentence[m].deps.append((h,l))

def matlab_to_edges(labels):
    return [(h,m, labels[h,m]) for h,m in zip(*np.where(labels))]

def write_conllu(fn, conllu_gen):
    with open(fn, 'w', encoding='UTF8') as fh:
        for graphID, sentence in conllu_gen:
            fh.write(graphID)
            fh.write("\n")
            for entry in sentence[1:]:
                fh.write(str(entry))
                fh.write("\n")
            fh.write("\n")


def vocab_conllu(fn, tree=False, UD=False):
    wordsCount = Counter()
    lemmasCount = Counter()
    posCount = Counter()
    relCount = Counter()
    numSen = 0

    with open(fn, 'r', encoding='UTF8') as fh:
        for graphID, sentence in read_conllu(fh, UD, UD):
            numSen += 1
            wordsCount.update([node.norm for node in sentence])
            lemmasCount.update([node.lemma for node in sentence])
            posCount.update([node.pos for node in sentence])
#            relCount.update(flatten([[l for i,l in node.deps]
#                                     for node in sentence]))
            if tree:
                relCount.update([node.deprel for node in sentence])
            else:
                relCount.update([l for node in sentence for i,l in node.deps])


    #return wordsCount, posCount, relCount, numSen, lemmasCount
    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())},
            list(posCount.keys()), list(relCount.keys()))



def edges_to_matrix(edges, N):
    matrix = np.zeros((N, N), dtype=np.int)
    labels = np.ndarray((N, N), dtype=object)
    for h, m, l in edges:
        matrix[h, m] = 1
        labels[h, m] = l
    return matrix, labels


def stupid_tree_(scores):
    out = np.zeros(scores.shape, dtype=np.int)
    for m in range(len(scores)):
        h = np.argmax(scores[:,m])
        out[h][m] = 1
    return out

def stupid_tree(scores):
    out = np.zeros(scores.shape, dtype=np.int)
    r = 0
    r_ = float("-inf")
    for m in range(1, len(scores)):
        h_ = float("-inf")
        h = 0
        if scores[0,m] >= r_:
            r = m
            r_ = scores[0,m]
        for j in range(1, len(scores)):
            if j == m:
                continue
            if scores[j,m] >= h_:
                h_ = scores[j,m]
                h = j
        out[h,m] = 1
    out[:,r] = 0
    out[0,r] = 1
    return out


def greater(input1, input2, dtype=None):
    ones = np.ones_like(input1, dtype=dtype)
    zeros = np.zeros_like(input1, dtype=dtype)
    return np.where(np.greater(input1, input2), ones, zeros)

def condoy(fname, count=False, semedges=False):
    from src import col_data as cd
    xs = []
    with open(fname) as f:
        for line in f:
            xs.append(json.loads(line))
    # extra ids for distributed negations
    neg_dist_id = 0
    cnt = 0
    for x in xs:
        cnt += 1
        ns = x["nodes"]
        cues = {}
        arcs = defaultdict(list)
        # extra ids for distributed negations
        neg_dist_id += 1
        for n in ns:
            if "negation" not in n: continue
            for neg in n["negation"]:
                if "cue" in neg:
                    cues[neg["id"]] = cues.get(neg["id"], []) + [int(n["id"])]

        #print([ns[i]["form"] for i in cues.values()])
        scopes = defaultdict(lambda : [])
        edges = {i: (0, "root") for i in range(1,len(ns)+1)}
        deps = {i: [] for i in range(1, len(ns)+1)}
        for n in ns:
            if "edges" in n:
                for edge in n["edges"]:
                    edges[edge["target"]] = (n["id"], edge["label"])
                    deps[edge["target"]] += [(n["id"], edge["label"])]
            if "negation" not in n: continue
            for neg in n["negation"]:
                if "scope" in neg:
                    if "event" in neg:
                        scopes[cues[neg["id"]][0]].append((int(n["id"]),"event"))
                    else:
                        scopes[cues[neg["id"]][0]].append((int(n["id"]), "scope"))
        for k,v in scopes.items():
            for e,se in v:
                arcs[e].append((k,se))
        for c in cues:
            if len(cues[c]) == 1:
                arcs[cues[c][0]].append((0,"cue"))
            else:
                for i in range(len(cues[c])):
                    #arcs[cues[c][i]].append((0,"cue"))
                    #arcs[cues[c][i+1]].append((cues[c][i], "mwc"))
                    #arcs[cues[c][i]].append((cues[c][-1], "mwc"))
                    arcs[cues[c][i]].append((cues[c][0], "mwc"))
                arcs[cues[c][0]].append((0,"cue"))


        sid = str(x["id"]) if not count else cnt
        if semedges:
            for n in ns:
                edges[n["id"]] = ("_","_")
                if deps[n["id"]] == []:
                    deps[n["id"]] = "_"
                else:
                    deps[n["id"]] = "|".join([f"{a}:{b}" for a,b in deps[n["id"]]])
        print(cd.Sentence(sid, [cd.Token(int(n["id"]), n["form"], 
            n["properties"]["lemma"], n["properties"]["upos"], n["properties"]["xpos"], 
            "_", edges[n["id"]][0], edges[n["id"]][1], deps[n["id"]], "_", 
            "|".join([(str(sc)+":"+sce) for sc,sce in arcs[int(n["id"])]]) if arcs[int(n["id"])] else "_")
            for n in ns])
            )


def condoy_old(fname):
    from src import col_data as cd
    xs = []
    with open(fname) as f:
        for line in f:
            xs.append(json.loads(line))
    # extra ids for distributed negations
    neg_dist_id = 0
    for x in xs:
        ns = x["nodes"]
        cues = {}
        arcs = defaultdict(list)
        # extra ids for distributed negations
        neg_dist_id += 1
        if x["negations"] > 0:
            for n in ns:
                for neg in n["negation"]:
                    if "cue" in neg:
                        cues[neg["id"]] = cues.get(neg["id"], []) + [int(n["id"])]

            #print([ns[i]["form"] for i in cues.values()])
            scopes = defaultdict(lambda : [])
            for n in ns:
                for neg in n["negation"]:
                    if "scope" in neg:
                        if "event" in neg:
                            #scopes[cues[neg["id"]][-1]].append((int(n["id"]),"event"))
                            scopes[cues[neg["id"]][0]].append((int(n["id"]),"event"))
                        else:
                            #scopes[cues[neg["id"]][-1]].append((int(n["id"]), "scope"))
                            scopes[cues[neg["id"]][0]].append((int(n["id"]), "scope"))
            for k,v in scopes.items():
                for e,se in v:
                    arcs[e].append((k,se))
            for c in cues:
                if len(cues[c]) == 1:
                    arcs[cues[c][0]].append((-1,"cue"))
                else:
                    for i in range(len(cues[c])):
                        #arcs[cues[c][i]].append((-1,"cue"))
                        #arcs[cues[c][i+1]].append((cues[c][i], "mwc"))
                        #arcs[cues[c][i]].append((cues[c][-1], "mwc"))
                        arcs[cues[c][i]].append((cues[c][0], "mwc"))
                    #arcs[cues[c][-1]].append((-1,"cue"))
                    arcs[cues[c][0]].append((-1,"cue"))



        yield cd.Sentence(x["id"] + " " + x["source"] + " _ " + str(neg_dist_id), [cd.Token(int(n["id"])+1, n["form"], 
            n["properties"]["lemma"], "_", n["properties"]["xpos"], 
            "_", "_", "_", "_", "_", 
            "|".join([(str(sc+1)+":"+sce) for sc,sce in arcs[int(n["id"])]]) if arcs[int(n["id"])] else "_")
            for n in ns])
    # id, form, lemma, upos, xpos,
    # feats, head, deprel, deps, misc,
    # scope

def check_cue(form):
    pre = "in un im ir dis".split()
    suf = ["less"]
    form = form.lower()
    for p in pre:
        if re.search("^"+p, form):#form.startswith(p):
            return True, p, re.sub("^"+p, "", form)#form.lstrip(p)
    for s in suf:
        if re.search(s+r"(ly|ness)?"+r"$", form):#form.endswith(s):
            return True, s, re.sub(s+r"(ly|ness)?"+r"$", r"", form)#form.rstrip(s)
    return False, "_", form

def conllup_to_starsem(fname, sherlock_train="data/sherlock/cdt.conllup", semcue=False):
    from src import col_data as cd
    sentences = cd.read_col_data(fname)
    from src import vocab as vcb
    train = cd.read_col_data(sherlock_train)
    _vocabs = vcb.make_vocabs(train, 0)
    vocabs = vcb.Vocabs(*_vocabs)
    w2i = vocabs.scoperels.w2i
    for sentence in sentences:
        #sid, story = sentence.id.split(maxsplit=1)
        story = "_"
        sid = sentence.id.split()[0]
        matrix = sentence.make_matrix("scope", label=True, w2i=w2i)
        if semcue:
            cmatrix = sentence.make_matrix("sem", label=True, w2i=w2i)
            cues = [i for i in range(len(cmatrix)) if cmatrix[0,i] == w2i["cue"] and not w2i["mwc"] in cmatrix[:,i]]
        else:
            #cues = [i for i in range(len(matrix)) if matrix[0,i] == w2i["cue"] and not w2i["mwc"] in matrix[:,i]]
            cues = [i for i in range(len(matrix)) if matrix[0,i] == w2i["cue"]]
        #for h in range(len(matrix)):
        #    if sum(matrix[h,:]) > 0:
        #        cues.append(h)

        if len(cues) > 0:
            # cue scope event
            for word in sentence:
                negs = ["_", "_", "_"] * len(cues)
                for i,c in enumerate(cues):
                    p = len(cues) - 1 - i
                    p = i
                    myev = word.form
                    if c == word.id:
                        is_incue, mycue, myev = check_cue(word.form)
                        if not is_incue:
                            mycue = word.form
                        negs[3*p] = mycue
                    if matrix[c,word.id] == w2i["event"]:
                        negs[3*p+1] = myev
                        negs[3*p+2] = myev
                    elif matrix[c,word.id] == w2i["scope"]:
                        negs[3*p+1] = myev#word.form
                    elif matrix[c,word.id] == w2i["mwc"]:
                        negs[3*p+0] = myev#word.form
                    elif semcue:
                        if cmatrix[c,word.id] == w2i["mwc"]:
                            negs[3*p+0] = myev#word.form
                print("\t".join([story, sid, str(word.id-1), word.form, word.lemma, word.xpos, "_", *negs]))
        else:
            for word in sentence:
                print("\t".join([story, sid, str(word.id-1), word.form, word.lemma, word.xpos, "_", "***"]))

        print()

def conllup_to_epe(fname):
    import col_data as cd
    s = 0
    for sentence in cd.read_col_data(fname):
        sid, story = sentence.id.split(maxsplit=1)
        epe = {"id": sid, "nodes": []}
        cues = {}
        nodes = epe["nodes"]
        c = 0
        for token in sentence:
            node = {"id": token.id,
                    "form": token.form,
                    "start": s,
                    "end": s + len(token.form),
                    "properties": {"xpos": token.xpos,
                                   "upos": token.upos,
                                   "lemma": token.lemma},
                    "edges": [],
                    "negation": []}
            s += len(token.form) + 1
            nodes.append(node)

            if "cue" in [l for h,l in token.scope] and not "mwc" in [l for h,l in token.scope]:
                if token.id not in cues:
                    cues[token.id] = c
                    c += 1
            elif "cue" in [l for h,l in token.scope] and "mwc" in [l for h,l in token.scope]:
                try:
                    cues[token.id] = cues[[h for h,l in token.scope if l == "mwc"][0]]
                except KeyError:
                    cues[[h for h,l in token.scope if l == "mwc"][0]] = c
                    c += 1
                    cues[token.id] = cues[[h for h,l in token.scope if l == "mwc"][0]]

        for token in sentence:
            if token.head == 0:
                nodes[token.id-1]["top"] = True
            elif token.head > 0:
                nodes[token.head-1]["edges"].append({"label": token.deprel, "target": token.id})
            for h,l in token.deps:
                nodes[h-1]["edges"].append({"label": l, "target": token.id})
            for h,l in token.scope:
                if h == token.id:
                    is_incue, mycue, myev = check_cue(token.form)
                    if not is_incue:
                        mycue = token.form
                        myev = token.form
                    if l == "scope":
                        nodes[token.id-1]["negation"].append({"id": cues[token.id], "cue": mycue, "scope": myev})
                    elif l == "event":
                        nodes[token.id-1]["negation"].append({"id": cues[token.id], "cue": mycue, "scope": myev, "event": myev})
                elif l == "cue":
                    #print((cues[token.id], [x["id"] for x in nodes[token.id-1]["negation"]]))
                    if not (cues[token.id] in [x["id"] for x in nodes[token.id-1]["negation"]]):
                        nodes[token.id-1]["negation"].append({"id": cues[token.id], "cue": token.form})
                elif l == "scope":
                    nodes[token.id-1]["negation"].append({"id": cues[h], "scope": token.form})
                elif l == "event":
                    nodes[token.id-1]["negation"].append({"id": cues[h], "scope": token.form, "event": token.form})
        print(json.dumps(epe))
        #yield epe






if __name__ == "__main__":
    import sys
    #conllup_to_epe(sys.argv[1])
    conllup_to_starsem(sys.argv[1], semcue="semcue" in sys.argv[1])
    #for s in condoy(sys.argv[1]):
    #    print(s)
    #with open(sys.argv[1]) as f:
    #    import col_data as cd
    #    for gid, sentence in read_conllu(f):
    #        print(cd.Sentence(gid.lstrip("#"), [cd.Token(*str(t).split(), "_") for t in sentence[1:]]))
