from sklearn.metrics.pairwise import cosine_similarity

pos2pos = {"ADJ":"a", "ADV":"r", "NOUN":"n", "VERB":"v"}

def get_lemma2syn(f_dictionary):
    dictionary = open(f_dictionary, "r").readlines()
    lemma2syn = {}
    for line in dictionary:
        fields = line.strip().split(" ")
        lemma, synsets = fields[0], [syn[:10] for syn in fields[1:]]
        lemma2syn[lemma] = synsets
    return lemma2syn

def calculate_accuracy(outs, gold_synsets, lemmas, pos_filters, embeddings, dictionary):
    lemma2syn = get_lemma2syn(dictionary)
    count_correct = 0
    count_all = 0
    unavailable_syn_emb = set()
    unavailable_syn_cases = 0
    # out.write("Lemma\tSelected synset\tGold synset\tDistance\n")
    for count, gold in enumerate(gold_synsets):
        # if using the test data, use "test_lemmas", otherwise use "train_lemmas"
        lemma = lemmas[count]
        pos = pos_filters[count]
        if lemma in lemma2syn:
            possible_syns = lemma2syn[lemma]
        else:
            count_all += 1
            continue
        output = outs[count]
        max_sim = -10000.0
        selected_syn = ""
        for syn in possible_syns:
            if pos2pos[pos] != syn.split("-")[1]:
                continue
            if syn in embeddings:
                cos_sim = cosine_similarity(output.reshape(1,-1), embeddings[syn].reshape(1,-1))[0][0]
            else:
                unavailable_syn_cases += 1
                unavailable_syn_emb.add(syn)
                cos_sim = 0.0
            if cos_sim > max_sim:
                max_sim = cos_sim
                selected_syn = syn
        # gold_cos_sim = cosine_similarity(output.reshape(1,-1), embeddings[gold].reshape(1,-1))[0][0]
        # line_to_write = lemma + "\t" + selected_syn + "\t" + gold + "\t" + str(max_sim - gold_cos_sim) + "\n"
        # out.write(line_to_write)
        if selected_syn in gold:
            count_correct += 1
        count_all += 1
    return count_correct, count_all