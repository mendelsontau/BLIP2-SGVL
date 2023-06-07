from datasets import load_dataset
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
from torch.cuda.amp import autocast as autocast

def blip_processor(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    return transform


def load_model(model_url, image_size, device, vit='base'):
    model = blip_itm(pretrained=model_url, image_size=image_size, vit=vit)
    model.eval()
    model = model.to(device)
    return model

def compute_itm(blip_model, caption, image, use_amp, args):
    with autocast(enabled=use_amp):
        image_embeds = blip_model.ln_vision(blip_model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = blip_model.tokenizer(
            caption,
            truncation=True,
            max_length=blip_model.max_txt_len,
            return_tensors="pt",
        ).to(image.device)


        query_tokens = blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if args.through_query:
            sg_tokens = blip_model.object_queries.expand(image_embeds.shape[0], -1, -1)
            if args.relations:
                sg_tokens = torch.cat([sg_tokens,blip_model.relation_queries.expand(image_embeds.shape[0], -1, -1)],dim=1)
            query_tokens = torch.cat([query_tokens, sg_tokens],dim=1)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
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

    return itm_logit

def evaluate_winoground(blip_model, blip_processor, device, use_amp, args):
    blip_model.eval()
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    #check if winoground folder exists
    result_dict_itm = {}
    auth_token = "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"
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    categories_blip_scores_itm = {}
    categories_blip_scores_itm["All Dataset"] = []
    categories_blip_scores_itm["Ambiguously Correct"] = []
    categories_blip_scores_itm["Visually Difficult"] = []
    categories_blip_scores_itm["Unusual Text"] = []
    categories_blip_scores_itm["Complex Reasoning"] = []
    categories_blip_scores_itm["Unusual Image"] = []
    categories_blip_scores_itm["Non Minimal"] = []
    categories_blip_scores_itm["No Tag"] = []
    categories_blip_scores_itc = {}
    categories_blip_scores_itc["All Dataset"] = []
    categories_blip_scores_itc["Ambiguously Correct"] = []
    categories_blip_scores_itc["Visually Difficult"] = []
    categories_blip_scores_itc["Unusual Text"] = []
    categories_blip_scores_itc["Complex Reasoning"] = []
    categories_blip_scores_itc["Unusual Image"] = []
    categories_blip_scores_itc["Non Minimal"] = []
    categories_blip_scores_itc["No Tag"] = []

    #load tag assignments
    f = open("Winoground/tag_assignments.json")
    tag_assignments = json.load(f)

    for example in tqdm(winoground):
    # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
    # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
        image_0 = blip_processor(example["image_0"].convert("RGB")).unsqueeze(0)
        image_1 = blip_processor(example["image_1"].convert("RGB")).unsqueeze(0)
        caption_0 = example["caption_0"]
        caption_1 = example["caption_1"]
        image_0 = image_0.to(device=device)
        image_1 = image_1.to(device=device)
        with torch.no_grad():
            output_c0_i0 = compute_itm(blip_model=blip_model,caption=caption_0, image=image_0, use_amp = use_amp, args=args)
            output_c1_i0 = compute_itm(blip_model=blip_model,caption=caption_1, image=image_0, use_amp = use_amp, args=args)
            output_c0_i1 = compute_itm(blip_model=blip_model,caption=caption_0, image=image_1, use_amp = use_amp, args=args)
            output_c1_i1 = compute_itm(blip_model=blip_model,caption=caption_1, image=image_1, use_amp = use_amp, args=args)

            blip_itm_scores_c0_i0 = torch.nn.functional.softmax(output_c0_i0, dim=1)[:, 1].item()
            blip_itm_scores_c1_i0 = torch.nn.functional.softmax(output_c1_i0, dim=1)[:, 1].item()
            blip_itm_scores_c0_i1 = torch.nn.functional.softmax(output_c0_i1, dim=1)[:, 1].item()
            blip_itm_scores_c1_i1 = torch.nn.functional.softmax(output_c1_i1, dim=1)[:, 1].item()

        example_id = str(example["id"])
        all_tags = tag_assignments[example_id]
        if len(all_tags) == 0:
            all_tags = ["No Tag"]
        all_tags.append("All Dataset")
        sample_dict_itm = {"id" : example["id"], "c0_i0": blip_itm_scores_c0_i0, "c0_i1": blip_itm_scores_c0_i1, "c1_i0": blip_itm_scores_c1_i0, "c1_i1": blip_itm_scores_c1_i1}
        for tag in all_tags:
            categories_blip_scores_itm[tag].append(sample_dict_itm)
        
        sample_result_dict_itm = {"text": True if text_correct(sample_dict_itm) else False, "image": True if image_correct(sample_dict_itm) else False, "group": True if group_correct(sample_dict_itm) else False}

        result_dict_itm[example_id] = sample_result_dict_itm

    winoground_dict = {}
    for category in categories_blip_scores_itm:
        category_blip_scores_itm = categories_blip_scores_itm[category]
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in category_blip_scores_itm:
            text_correct_count += 1 if text_correct(result) else 0
            image_correct_count += 1 if image_correct(result) else 0
            group_correct_count += 1 if group_correct(result) else 0

        denominator = len(category_blip_scores_itm)
        winoground_text_score = text_correct_count/denominator
        winoground_image_score = image_correct_count/denominator
        winoground_group_score = group_correct_count/denominator

        metrics = {category + " text score": text_correct_count/denominator, 
        category + " image score": image_correct_count/denominator,
        category + " group score": group_correct_count/denominator,}

        winoground_dict[category] = metrics



        print(
        f"winoground " + category
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

    return winoground_dict, result_dict_itm


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#image_size = 384
#model_path = "pretrained_checkpoints/model_base.pth"
#blip_model = load_model(model_path, image_size, device, vit='base')
#blip_processor  = blip_processor(image_size)
#evaluate_winoground(blip_model=blip_model,blip_processor=blip_processor,device=device)