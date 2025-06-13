import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
from torch.nn.functional import cosine_similarity


def extract_embedding(img_path, preprocessing, feature_extractor):
    img = Image.open(img_path).convert("RGB")
    tensor_img = preprocessing(img).unsqueeze(0)
    with torch.no_grad():
        vector = feature_extractor(tensor_img).squeeze()
    return vector


def compute_similarity(path_a, path_b, preprocessing, feature_extractor):
    vec_a = extract_embedding(path_a, preprocessing, feature_extractor)
    vec_b = extract_embedding(path_b, preprocessing, feature_extractor)
    return cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()


def resnet(object_pairs):
    valid_matches = []

    backbone = resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval()

    preprocessing = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    for idx, (obj_l, obj_r) in enumerate(object_pairs):
        path_l = obj_l["image"]
        path_r = obj_r["image"]

        print(f"\nПара №{idx + 1}:")

        sim_score = compute_similarity(path_l, path_r, preprocessing, backbone)
        print(f"Косинусна подібність: {sim_score:.4f}")

        if sim_score > 0.8:
            valid_matches.append((obj_l, obj_r))
            print("Збіг: ймовірно, це один і той самий об'єкт.")
        else:
            print(" Різні об'єкти.")

    return valid_matches