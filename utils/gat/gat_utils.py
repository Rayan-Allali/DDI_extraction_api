from utils.ner_utils import label_mapping_reverse
def re_prediction_with_gat(data_loader_graph,gat_model,device):
    pair_ddi_results = []
    for batch in data_loader_graph:
            batch=batch.to(device)
            sentences = batch["sentence"]
            drugs1 =batch["drug1"]
            drugs2 =batch["drug2"]
            print(sentences)
            logits = gat_model(batch)
            preds = logits.argmax(dim=1)
            results = preds.cpu().numpy()

            for index, result in enumerate(results):
              drug1 = drugs1[index].split("________")[0]
              drug2 = drugs2[index].split("________")[0]
              label = label_mapping_reverse.get(result, "unknown")
              pair_ddi_result = {
                "drug1": drug1,
                "drug2": drug2,
                "label": label,
                "index":index
                }

              pair_ddi_results.append(pair_ddi_result)
              print(f'Sentence:"{index}" has interaction: {label}')
              print(label)
    return pair_ddi_results