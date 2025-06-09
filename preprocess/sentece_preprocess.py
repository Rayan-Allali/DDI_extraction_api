def blind_drug_mentions(sentence,ner_bio_tags):
    words = sentence.split()
    drugs = []
    word = ""
    for idx, label in ner_bio_tags.items():
        if label == "B-DRUG":
             if word != "":
                drugs.append(word.strip())
                word = ""
             word = words[idx]
        elif label == "I-DRUG":
            if word != "":
              word += f' {words[idx]}'
        else:
            if word != "":
                drugs.append(word.strip())
                word = ""

    if word:
        drugs.append(word.strip())
    print(drugs)
    
    
    drug_map = {}
    def drug_blinding(sentence, drugs):
        copy_sentence = sentence
        for drug_index, drug in enumerate(drugs):
            blinded_token = f"DRUG{drug_index}"
            copy_sentence = copy_sentence.replace(drug, blinded_token, 1)
            normalise_drug = f"{drug}________{drug_index}"
            drug_map[normalise_drug] = blinded_token
        return copy_sentence

    sentence = drug_blinding(sentence, drugs)
    print(f'drug_map: {drug_map}')
    if len(drug_map) <= 1:
        print('No DDI found')
        return None
    return sentence,drug_map


def extracting_drugs(sentence,ner_bio_tags):
    words = sentence.split()
    drugs = []
    word = ""
    for idx, label in ner_bio_tags.items():
        if label == "B-DRUG":
             if word != "":
                drugs.append(word.strip())
                word = ""
             word = words[idx]
        elif label == "I-DRUG":
            if word != "":
              word += f' {words[idx]}'
        else:
            if word != "":
                drugs.append(word.strip())
                word = ""

    if word:
        drugs.append(word.strip())
    
    return drugs