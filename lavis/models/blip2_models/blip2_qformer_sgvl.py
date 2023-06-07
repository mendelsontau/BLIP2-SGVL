"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import math
import functools
from operator import mul

from torch.nn.modules.utils import _pair

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion


class MLP_Head(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class HNLoss(nn.Module):
    def __init__(
            self,
            alpha = 1.0
    ):
        super().__init__()
        self.alpha = alpha


    def forward(self, image_features, text_features, ignore_mask, vg_batch_size):
        vg_image_features = image_features[-vg_batch_size:,:]
        positive_text_features = text_features[-2*vg_batch_size:-vg_batch_size,:]
        negative_text_features = text_features[-vg_batch_size:,:]
        positive_similarity,_ = torch.bmm(vg_image_features, positive_text_features.unsqueeze(-1)).squeeze().max(-1)
        negative_similarity,_ = torch.bmm(vg_image_features, negative_text_features.unsqueeze(-1)).squeeze().max(-1)
        positive_similarity = torch.exp(positive_similarity)
        negative_similarity = torch.exp(negative_similarity)
        denominator = positive_similarity + negative_similarity
        loss_per_sample = -torch.log(torch.div(positive_similarity,denominator))
        loss = self.alpha * torch.dot(loss_per_sample, ignore_mask)/torch.sum(ignore_mask)
        return loss 


@registry.register_model("blip2_sgvl")
@registry.register_model("blip2_sgvl_feature_extractor")
class Blip2QformerSGVL(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        args = None
    ):
        super().__init__()

        device = torch.device(args.device)

        self.objects = args.objects
        self.object_tokens = args.object_tokens
        self.relations = args.relations
        self.relation_tokens = args.relation_tokens
        self.num_query_token = num_query_token

        args.vit_precision = vit_precision
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, args
        )
        vision_width = self.visual_encoder.embed_dim if vit_model == "eva_clip_g" else self.visual_encoder.num_features
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq, args
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        self.negatives_loss = args.negatives

        self.vg_loss_lambda = args.vg_loss_lambda

        if self.negatives_loss:
            self.hn_loss = HNLoss()

        self.vg_loss_lambda = args.vg_loss_lambda

        self.through_query = args.through_query

        if self.vg_loss_lambda > 0.0:
            weight_dict = {'loss_ce': args.loss_ce, 'loss_bbox': 5}
            weight_dict['loss_giou'] = 2
            losses = ['labels','boxes','cardinality']
            matcher = HungarianMatcher(cost_class=weight_dict["loss_ce"],cost_bbox=weight_dict["loss_bbox"],cost_giou=weight_dict["loss_giou"]) 


            self.num_matcher_classes = args.vg_batch_size * args.objects

            self.vgcriterion = SetCriterion(self.num_matcher_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=(5.5/self.object_tokens), losses=losses)
            self.vgcriterion.to(args.device)


            if not args.through_query:
                self.class_head = MLP_Head(vision_width, vision_width, embed_dim,args.head_layers).to(device)
                self.bb_head = MLP_Head(vision_width, vision_width, 4, args.head_layers).to(device)
                self.random_row = nn.Parameter(torch.zeros(1,embed_dim))
                self.no_object_row = nn.Parameter(torch.zeros(1,embed_dim))
                if self.relations > 0:
                    self.num_relation_classes = args.vg_batch_size * args.relations
                    self.vgrelcriterion = SetCriterion(self.num_relation_classes, matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=(1.8/self.relation_tokens), losses=losses)
                    self.vgrelcriterion.to(args.device)
                    self.no_relation_row = nn.Parameter(torch.zeros(1,embed_dim))
                    self.rel_class_head = MLP_Head(vision_width, vision_width, embed_dim,args.head_layers).to(device)
                    self.rel_bb_head = MLP_Head(vision_width, vision_width, 4, args.head_layers).to(device)
            else:
                if self.objects > 0:
                    val = math.sqrt(6. / float(3 * functools.reduce(mul, _pair(16), 1) + self.Qformer.config.hidden_size))  #prompt init per visual prompt tuning

                    self.object_queries = nn.Parameter(torch.zeros(1, self.object_tokens, self.Qformer.config.hidden_size))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.object_queries, -val, val)

                if self.relations > 0:
                    val = math.sqrt(6. / float(3 * functools.reduce(mul, _pair(16), 1) + self.Qformer.config.hidden_size))  #prompt init per visual prompt tuning

                    self.relation_queries = nn.Parameter(torch.zeros(1, self.relation_tokens, self.Qformer.config.hidden_size))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.relation_queries, -val, val)



                self.class_head = MLP_Head(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size, self.Qformer.config.hidden_size,args.head_layers).to(device)
                self.bb_head = MLP_Head(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size, 4, args.head_layers).to(device)
                self.random_row = nn.Parameter(torch.zeros(1,self.Qformer.config.hidden_size))
                self.no_object_row = nn.Parameter(torch.zeros(1,self.Qformer.config.hidden_size))
                if self.relations > 0:
                    self.num_relation_classes = args.vg_batch_size * args.relations
                    self.vgrelcriterion = SetCriterion(self.num_relation_classes, matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=(1.8/self.relation_tokens), losses=losses)
                    self.vgrelcriterion.to(args.device)
                    self.no_relation_row = nn.Parameter(torch.zeros(1,self.Qformer.config.hidden_size))
                    self.rel_class_head = MLP_Head(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size, self.Qformer.config.hidden_size,args.head_layers).to(device)
                    self.rel_bb_head = MLP_Head(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size, 4, args.head_layers).to(device)
                



    def forward(self, image, text, vg_batch_size = 0, ignore_mask=None, objects_descs = None, objects_targets = None, relations_descs = None, relations_targets = None
    , laion_negs = None, laion_neg_mask = None):


        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        if self.vg_loss_lambda > 0:
            graph_descriptions = []
            if not self.through_query:
                object_tokens = image_embeds[-vg_batch_size:,1 : 1 + self.object_tokens ,:]    
            graph_descriptions += objects_descs
            num_object_descs = len(objects_descs)
            if self.relations > 0:
                if not self.through_query:
                    relation_tokens = image_embeds[-vg_batch_size:,1 + self.object_tokens : 1 + self.object_tokens + self.relation_tokens,:]
                graph_descriptions += relations_descs
                num_relation_descs = len(relations_descs)
            text_descriptions = self.tokenizer(
                graph_descriptions,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if self.vg_loss_lambda > 0 and self.through_query:
            sg_tokens = self.object_queries.expand(image_embeds.shape[0], -1, -1)
            if self.relations:
                sg_tokens = torch.cat([sg_tokens,self.relation_queries.expand(image_embeds.shape[0], -1, -1)],dim=1)
            query_tokens = torch.cat([query_tokens, sg_tokens],dim=1)
            


        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        if self.vg_loss_lambda > 0 and self.through_query:
            object_tokens = query_output.last_hidden_state[-vg_batch_size:,self.num_query_token:self.num_query_token + self.object_tokens,:]
            if self.relations:
               relation_tokens = query_output.last_hidden_state[-vg_batch_size:,self.num_query_token + self.object_tokens:,:] 


        image_feat = F.normalize(
            self.vision_proj(query_output.last_hidden_state[:,:self.num_query_token,:]), dim=-1
        )

        text_no_adds = self.tokenizer(
            text[:image.shape[0]],
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        if self.negatives_loss:
                        text_negs = self.tokenizer(text[image.shape[0]:image.shape[0] + vg_batch_size],padding='max_length', truncation=True, max_length=self.max_txt_len, 
                            return_tensors="pt").to(image.device)
        

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        if self.vg_loss_lambda > 0:
            with torch.no_grad():
                text_output_descriptions = self.Qformer.bert(
                    text_descriptions.input_ids,
                    attention_mask=text_descriptions.attention_mask,
                    return_dict=True,
                )
                if self.through_query:
                    text_feat_descriptions = F.normalize(text_output_descriptions.last_hidden_state[:, 0, :],dim = -1)
                else:
                    text_feat_descriptions = F.normalize(self.text_proj(text_output_descriptions.last_hidden_state[:, 0, :]),dim = -1)
                if self.relations > 0:
                    relations_descs_feat_m = text_feat_descriptions[-num_relation_descs:]
                objects_descs_feat_m = text_feat_descriptions[:num_object_descs]







        #calculate negatives loss
        neg_loss = 0.0
        if self.negatives_loss:
            neg_loss = self.hn_loss(image_feat, text_feat, ignore_mask, vg_batch_size)
            #remove neagtives
            text_feat = text_feat[:-vg_batch_size]

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feat
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feat.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2



        ###============== Hungarian Matching ===================###
        if self.vg_loss_lambda > 0.0:
            no_object_rows_to_add = self.num_matcher_classes - num_object_descs
            random_rows = self.random_row
            no_object_row = self.no_object_row.to(image.device)
            random_rows = random_rows.expand(no_object_rows_to_add,-1).to(image.device)
            objects_descs_feat_m = torch.cat([objects_descs_feat_m,random_rows,no_object_row])
            label_embeddings = self.class_head(object_tokens)
            label_predictions = label_embeddings @ objects_descs_feat_m.t() / self.temp 
            bb_predictions = self.bb_head(object_tokens).sigmoid()
            predictions_dict = {"pred_logits" : label_predictions, "pred_boxes": bb_predictions}
            loss_dict = self.vgcriterion(predictions_dict, objects_targets)
            weight_dict = self.vgcriterion.weight_dict
            if self.relations > 0:
                no_relation_rows_to_add = self.num_relation_classes - num_relation_descs
                random_rows = self.random_row
                no_relation_row = self.no_relation_row.to(image.device)
                random_rows = random_rows.expand(no_relation_rows_to_add,-1).to(image.device)
                relations_descs_feat_m = torch.cat([relations_descs_feat_m,random_rows,no_relation_row])
                label_embeddings = self.rel_class_head(relation_tokens)
                label_predictions = label_embeddings @ relations_descs_feat_m.t() / self.temp 
                bb_predictions = self.rel_bb_head(relation_tokens).sigmoid()
                predictions_dict = {"pred_logits" : label_predictions, "pred_boxes": bb_predictions}
                relation_loss_dict = self.vgrelcriterion(predictions_dict, relations_targets)
                loss_dict = {k: loss_dict[k] + relation_loss_dict[k] for k in loss_dict}    
        else:
            loss_dict = None
            weight_dict = None
        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_no_adds.input_ids)
        text_attention_mask_world = concat_all_gather(text_no_adds.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_no_adds.input_ids, text_no_adds.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_no_adds.attention_mask, text_no_adds.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        if self.through_query:
            sg_tokens = self.object_queries.expand(text_ids_all.shape[0], -1, -1)
            if self.relations:
                sg_tokens = torch.cat([sg_tokens,self.relation_queries.expand(text_ids_all.shape[0], -1, -1)],dim=1)
            query_tokens_itm = torch.cat([query_tokens_itm, sg_tokens],dim=1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : self.num_query_token, :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        if self.negatives_loss:
            text_neg_input_ids = text_negs.input_ids
            query_tokens_itm_neg = self.query_tokens.expand(text_neg_input_ids.shape[0], -1, -1)
            if self.through_query:
                sg_tokens = self.object_queries.expand(text_neg_input_ids.shape[0], -1, -1)
                if self.relations:
                    sg_tokens = torch.cat([sg_tokens,self.relation_queries.expand(text_neg_input_ids.shape[0], -1, -1)],dim=1)
                query_tokens_itm_neg = torch.cat([query_tokens_itm_neg, sg_tokens],dim=1)
            query_atts_itm_neg = torch.ones(query_tokens_itm_neg.size()[:-1], dtype=torch.long).to(
                image.device
            )
            text_neg_attention_mask = text_negs.attention_mask
            attention_mask_neg = torch.cat([query_atts_itm_neg, text_neg_attention_mask], dim=1)
            output_neg_vg = self.Qformer.bert(text_neg_input_ids,
                                        query_embeds = query_tokens_itm_neg,
                                        attention_mask = attention_mask_neg,
                                        encoder_hidden_states = image_embeds[-vg_batch_size:],
                                        encoder_attention_mask = image_atts[-vg_batch_size:],      
                                        return_dict = True,
                                        )  

            vl_vg_embeddings = torch.cat([vl_embeddings[image.shape[0] - vg_batch_size:image.shape[0],:,:], output_neg_vg.last_hidden_state[:,:self.num_query_token,:]],dim=0)
            vl_vg_output = self.itm_head(vl_vg_embeddings)
            vl_vg_output = vl_vg_output.mean(dim=1)
            itm_vg_labels = torch.cat([torch.ones(vg_batch_size,dtype=torch.long),torch.zeros(vg_batch_size,dtype=torch.long)],
                            dim=0).to(image.device)
            loss_vg_itm = F.cross_entropy(vl_vg_output, itm_vg_labels)
            neg_loss += loss_vg_itm
            neg_loss /= 2

        return loss_itc, loss_itm, neg_loss, loss_dict, weight_dict

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg, args):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            args=args
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
