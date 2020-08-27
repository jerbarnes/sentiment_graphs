def get_text(sent):
    for line in sent.split("\n"):
        if line.startswith("# text ="):
            return line.strip().split("text = ")[-1].split()

def get_tokens(sent):
    tokens = []
    for line in sent.split("\n"):
        if line.startswith("#"):
            pass
        else:
            ll = line.strip().split("\t")
            try:
                tok = ll[1]
                tokens.append(tok)
            except:
                print(sent)
                print(line)
    return tokens

if __name__ == "__main__":
    """
    Currently, there are mismatches between norec fine sent_ids and norec 2.0 sent_ids. This leads to problems during preprocessing, so we have to remove the sentences that don't match up.
    """

    for split in ["train.conllu", "dev.conllu", "test.conllu"]:
        file = open(split).read()
        sents = file.split("\n\n")

        filtered = []

        for sent in sents:
            # check if text is the same as the tokens
            if get_text(sent) == get_tokens(sent):
                filtered.append(sent)

        with open(split, "w") as outfile:
            for sent in filtered:
                outfile.write(sent + "\n\n")
