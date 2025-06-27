from utils.ner_utils import label_mapping_reverse
import numpy as np
def re_prediction_with_gat(data_loader_graph,gat_model,device):
    pair_ddi_results = []
    for batch in data_loader_graph:
            batch=batch.to(device)
            sentences = batch["sentence"]
            drugs1 =batch["drug1"]
            drugs2 =batch["drug2"]
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
    return pair_ddi_results


def re_prediction_with_ensemble_gat(data_loader_graph, re_gat_models, device):
    pair_ddi_results = []

    for batch in data_loader_graph:
        batch = batch.to(device)
        sentences = batch["sentence"]
        drugs1 = batch["drug1"]
        drugs2 = batch["drug2"]
        all_logits = []
        for model in re_gat_models:
            logits = model(batch)
            all_logits.append(logits)

        stacked_preds = [logits.argmax(dim=1).cpu().numpy() for logits in all_logits]
        stacked_preds = np.stack(stacked_preds, axis=0)  # shape: (n_models, batch_size)
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_preds)

        for index, result in enumerate(final_preds):
            drug1 = drugs1[index].split("________")[0]
            drug2 = drugs2[index].split("________")[0]
            label = label_mapping_reverse.get(result, "unknown")
            pair_ddi_result = {
                "drug1": drug1,
                "drug2": drug2,
                "label": label,
                "index": index
            }
            pair_ddi_results.append(pair_ddi_result)

    return pair_ddi_results

     