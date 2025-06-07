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