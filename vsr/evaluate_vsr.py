
import sys
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
sys.path.insert(0, "/home/gamir/DER-Roei/alon/LAVIS")
from tqdm import tqdm
import pandas as pd
import json
import logging
from .vsr_dataset import VsrDataset, get_vsr_loader
from torch.cuda.amp import autocast as autocast




def evaluate_vsr(blip_model, blip_processor, device, use_amp, args):
    blip_model.eval()
    vsr_dataset = VsrDataset(blip_processor=blip_processor)
    vsr_loader = get_vsr_loader(vsr_dataset,32)
    all_results = []
    all_relations = []
    for batch in tqdm(vsr_loader):
        images, captions, labels, relations = batch
        images = images.to(device)
        labels = labels.to(device)
        with autocast(enabled=use_amp):
            image_embeds = blip_model.ln_vision(blip_model.visual_encoder(images))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                images.device
            )

            text = blip_model.tokenizer(
                captions,
                truncation=True,
                padding=True,
                max_length=blip_model.max_txt_len,
                return_tensors="pt",
            ).to(images.device)

            query_tokens = blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1)

            if args.through_query:
                sg_tokens = blip_model.object_queries.expand(image_embeds.shape[0], -1, -1)
                if args.relations:
                    sg_tokens = torch.cat([sg_tokens,blip_model.relation_queries.expand(image_embeds.shape[0], -1, -1)],dim=1)
                query_tokens = torch.cat([query_tokens, sg_tokens],dim=1)
                query_output = blip_model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    use_cache=True,
                    return_dict=True,
                )
                sg_tokens = query_output.last_hidden_state[:,blip_model.num_query_token:,:]
                query_tokens = blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_tokens = torch.cat([query_tokens, sg_tokens],dim=1)

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                images.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            output_itm = blip_model.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, : blip_model.num_query_token, :]
            itm_logit = blip_model.itm_head(itm_embeddings)
        itm_logit = itm_logit.mean(dim=1)
        predicted_label =  torch.argmax(itm_logit,1)
        results = torch.eq(labels,predicted_label).detach().cpu().tolist()
        all_results += results
        all_relations += relations

    rel2cat = {}
    cats = []
    rel_count = 0
    with open("vsr/rel_meta_category_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            cat, rels = line.strip().split(": ")
            cat = cat.strip()
            rel_list = rels.split(",")
            rel_count+=len(rel_list)
            rel_list = [rel.strip() for rel in rel_list]
            for rel in rel_list:
                rel2cat[rel] = cat
            cats.append(cat)
    print (f"# rel: {rel_count}")
    cats = list(set(cats))

    results_by_cat = {}
    results_by_meta_cat = {"Adjacency":{"corrects":0,"samples":0}, "Directional":{"corrects":0,"samples":0}, "Orientation":{"corrects":0,"samples":0},
    "Projective":{"corrects":0,"samples":0},"Proximity":{"corrects":0,"samples":0},"Topological":{"corrects":0,"samples":0},"Unallocated":{"corrects":0,"samples":0}, "total":{"corrects":0,"samples":0}}
    for i in range(len(all_results)):
        result = all_results[i]
        relation = all_relations[i]
        if relation not in results_by_cat:
            if result == True:
                results_by_cat[relation] = {"corrects":1,"samples":1}
            else:
                results_by_cat[relation] = {"corrects":0,"samples":1}  
        else:
            results_by_cat[relation]["samples"] +=1
            if result == True:
                results_by_cat[relation]["corrects"] +=1
        results_by_cat[relation]["accuracy"] = results_by_cat[relation]["corrects"]/results_by_cat[relation]["samples"]
        if relation not in rel2cat:
            continue
        metacat = rel2cat[relation]
        results_by_meta_cat[metacat]["samples"] +=1
        results_by_meta_cat["total"]["samples"] +=1
        if result == True:
            results_by_meta_cat[metacat]["corrects"] += 1
            results_by_meta_cat["total"]["corrects"] += 1
        results_by_meta_cat[metacat]["accuracy"] = results_by_meta_cat[metacat]["corrects"]/results_by_meta_cat[metacat]["samples"]
        results_by_meta_cat["total"]["accuracy"] = results_by_meta_cat["total"]["corrects"]/results_by_meta_cat["total"]["samples"]
        
    return results_by_cat, results_by_meta_cat