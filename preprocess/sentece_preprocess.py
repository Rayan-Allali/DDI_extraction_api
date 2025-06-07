def blind_drug_mentions(sentence,ner_bio_tags):
    words = sentence.split()
    print(f"words {words}")
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
    
    # drugs_indexes=[]
    # # drugs=filter_drugs_with_synonyms(drugs)
    # # def rename_drug_mentions(sentence, drugs):
    # #     copy_sentence = sentence
    # #     for drug in drugs:
    # #         drug_synonyms = get_drug_synonyms(drug)
    # #         for synonym in drug_synonyms:
    # #             copy_sentence = sentence.replace(synonym, drug, 1)
    # #     return copy_sentence
    # def drug_blinding(sentence, drugs):
    #     copy_sentence = sentence
    #     for drugIndex, drug in enumerate(drugs):
    #       copy_sentence = copy_sentence.replace(drug, f"DRUG{drugIndex}", 1)
    #       drugs_indexes.append(f"DRUG{drugIndex}")
    #     return copy_sentence
    # sentence = rename_drug_mentions(sentence, drugs)
    sentence = drug_blinding(sentence, drugs)
    print(f'drug_map: {drug_map}')
    if len(drug_map) <= 1:
        print('No DDI found')
        return None
    return sentence,drug_map