class DrugPairSentence:
    def __init__(self, drug1, drug2, sentence):
        self.drug1 = drug1
        self.drug2 = drug2
        self.sentence = sentence

    def __repr__(self):
        return f"DrugPairSentence(drug1={self.drug1}, drug2={self.drug2}, sentence={self.sentence})"