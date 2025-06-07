import itertools
import re
from utils.negative_filtering import is_in_par
from utils.drug_pair_sentence import DrugPairSentence
def generate_drugPairSentences_for_pairs(sentence, drug_map):
    results=[]
    reversed_map = {v: k for k, v in drug_map.items()}
    drug_indexes = list(reversed_map.keys()) 
    for drug1, drug2 in itertools.combinations(drug_indexes, 2):
        pattern1 = re.escape(drug1) + r'(?!\d)'
        pattern2 = re.escape(drug2) + r'(?!\d)'
        if is_in_par(sentence,drug1,drug2):
          continue
        else:
          newSentence = re.sub(pattern1, f"[e1] {drug1} [/e1]", sentence)
          newSentence = re.sub(pattern2, f"[e2] {drug2} [/e2]", newSentence)
          
          
          real_drug1 = reversed_map[drug1]
          real_drug2 = reversed_map[drug2]

          drug_pair_sentence = DrugPairSentence(real_drug1, real_drug2, newSentence)
          # drug_pair_sentence = DrugPairSentence(drug1, drug2, newSentence)
          results.append(drug_pair_sentence)
    return results
