import os
from VL_CheckList.vl_checklist.vlp_model import VLPModel
from VL_CheckList.example_models.utils.helpers import LRUCache, chunks
import torch.cuda
from PIL import Image
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast


class BLIP2(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    MAX_CACHE = 20

    def __init__(self, model_id, model, preprocess, device, use_amp, args):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 32
        self.device = device
        self.model_dir = "resources"
        self.model_id = model_id
        self.model = model
        self.preprocess = preprocess
        self.use_amp = use_amp
        self.args = args


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
        images_collect = []
        for chunk_i, chunk_t in zip(chunks(images, self.batch_size),chunks(texts, self.batch_size)):
            for j in range(len(chunk_i)):
                image = preprocess(Image.open(chunk_i[j]).convert("RGB")).to(self.device)
                images_collect.append(image)
            images_collect = torch.stack(images_collect)
            with torch.no_grad():
                with autocast():
                    image_embeds = model.ln_vision(model.visual_encoder(images_collect))
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                        image.device
                    )

                    text = model.tokenizer(
                        chunk_t,
                        truncation=True,
                        padding=True,
                        max_length=model.max_txt_len,
                        return_tensors="pt",
                    ).to(image.device)

                    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

                    if self.args.through_query:
                        sg_tokens = model.object_queries.expand(image_embeds.shape[0], -1, -1)
                        if self.args.relations:
                            sg_tokens = torch.cat([sg_tokens,model.relation_queries.expand(image_embeds.shape[0], -1, -1)],dim=1)
                        query_tokens = torch.cat([query_tokens, sg_tokens],dim=1)
                        query_output = model.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            use_cache=True,
                            return_dict=True,
                        )
                        sg_tokens = query_output.last_hidden_state[:,model.num_query_token:,:]
                        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                        query_tokens = torch.cat([query_tokens, sg_tokens],dim=1)

                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                        image.device
                    )
                    attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
                    output_itm = model.Qformer.bert(
                        text.input_ids,
                        query_embeds=query_tokens,
                        attention_mask=attention_mask,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    itm_embeddings = output_itm.last_hidden_state[:, : model.num_query_token, :]
                    itm_logit = model.itm_head(itm_embeddings)
                    itm_logit = itm_logit.mean(dim=1)
                probabilities = torch.nn.functional.softmax(itm_logit, dim=1)[:, 1].cpu().tolist()
                probs.extend(probabilities)
        return {"probs":probs}
        


