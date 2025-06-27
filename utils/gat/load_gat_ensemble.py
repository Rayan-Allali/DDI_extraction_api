from utils.gat.gat_model import GATModelWithAttention,GATModelWithAttentionWithEdges
import torch
def load_model_and_predict(model_config,device):
    if model_config["with_edges"]:
         model = GATModelWithAttentionWithEdges(
            node_in_dim=model_config["node_in_dim"],
            gat_hidden_channels=256,
            cls_dim=768,
            num_classes=5,
            dropout_rate=model_config["dropout_rate"]
        ).to(device)
    else:
         model = GATModelWithAttention(
            node_in_dim=model_config["node_in_dim"],
            gat_hidden_channels=256,
            cls_dim=768,
            num_classes=5,
            dropout_rate=model_config["dropout_rate"]
        ).to(device)
       
    print("path:",model_config["path"])
    state_dict = torch.load(model_config["path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model