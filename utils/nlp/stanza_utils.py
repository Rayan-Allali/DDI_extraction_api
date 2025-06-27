
import numpy as np
def extract_dependency_tree_stanford(spacy_doc,nlp_Stanza):
    tokens = [token.text for token in spacy_doc]
    stanza_doc = nlp_Stanza([tokens])

    tokens = []
    edges = []
    node_to_index = {}

    for sent in stanza_doc.sentences:
        for i, word in enumerate(sent.words):
            tokens.append(word.text)
            node_to_index[word.id] = i
        for word in sent.words:
            if word.head != 0:
                child_idx = node_to_index[word.id]
                parent_idx = node_to_index[word.head]
                edges.append((child_idx, parent_idx, word.deprel))

    return tokens, edges

ud_deprel = [
        "acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "aux:pass",
        "case", "cc", "cc:preconj", "ccomp", "clf", "compound", "compound:prt",
        "conj", "cop", "csubj", "csubj:pass", "dep", "det", "det:predet",
        "discourse", "dislocated", "expl", "fixed", "flat", "flat:foreign", "flat:name",
        "goeswith", "iobj", "list", "mark", "nmod", "nmod:npmod", "nmod:poss",
        "nmod:tmod", "nsubj", "nsubj:pass", "nummod", "obj", "obl", "obl:agent",
        "obl:npmod", "obl:tmod", "orphan", "parataxis", "punct", "reparandum", "root",
        "vocative", "xcomp"
    ]
def create_deprel_scalar_vocab():
    
    deprel_to_index = {deprel: i for i, deprel in enumerate(ud_deprel)}
    
    indices = np.array(list(deprel_to_index.values()), dtype=np.float32)
    min_val = indices.min()
    max_val = indices.max()
    if max_val > min_val: 
        normalized = 2 * (indices - min_val) / (max_val - min_val) - 1
    else:
        normalized = np.zeros_like(indices) 
    deprel_to_scalar = dict(zip(ud_deprel, normalized))
    
    deprel_to_scalar['unk'] = 0.0
    return deprel_to_scalar
deprel_to_scalar = create_deprel_scalar_vocab()