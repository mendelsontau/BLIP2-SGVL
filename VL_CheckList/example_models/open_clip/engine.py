import os
from VL_CheckList.vl_checklist.vlp_model import VLPModel
from VL_CheckList.example_models.utils.helpers import LRUCache, chunks
import torch.cuda
from PIL import Image
from open_clip import tokenize

class OPEN_CLIP(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    MAX_CACHE = 20

    def __init__(self, model_id, model, preprocess):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = "resources"
        self.model_id = model_id
        self.model = model
        self.preprocess = preprocess


    def model_name(self):
        return self.model_id

    def _load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")
        if not self._models.has(model_id):
            self._models.put(model_id, [self.model, self.preprocess])
        return self._models.get(model_id)

    def _load_data(self, src_type, data):
        pass

    def predict(self,
                images: list,
                texts: list,
                src_type: str = 'local'
                ):

        model_list = self._load_model(self.model_id)
        model = model_list[0]
        preprocess = model_list[1]
        # process images by batch
        probs = []
        for i, chunk_i in enumerate(chunks(images, self.batch_size)):
            for j in range(len(chunk_i)):
                image = preprocess(Image.open(chunk_i[j])).unsqueeze(0).to(self.device)
                # text format is [["there is a cat","there is a dog"],[...,...]...]
                text = tokenize(texts[j]).to(self.device)

                with torch.no_grad():
                    image_features, _, text_features, logit_scale = model(image, text)
                    #image_features, text_features, logit_scale = model(image, text)
                    logits_per_image = logit_scale*image_features @ text_features.T
                    probs.extend(logits_per_image.cpu().numpy())

        return {"probs":probs}
        


