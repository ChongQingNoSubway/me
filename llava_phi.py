from typing import List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import numpy as np
import pickle
import torch
import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom


from ..mipha.model.language_model.mipha_phi import (MiphaPhiForCausalLM, MiphaPhiModel)
from ..mipha.model.mipha_arch import MiphaMetaModel, MiphaMetaForCausalLM

from instructseg.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, SEG_TOKEN_INDEX, REFER_TOKEN_INDEX, TEMPORAL_TOKEN_INDEX

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import \
    MultiScaleMaskedTransformerDecoderForOPTPreTrain
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..mask_encoder.swin_trans import build_swin_b, build_swin_l

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from .ovp import OVPsampler
from .vmtf import VMTF

from ..datasets_mapper.IVS_mapper import IVSDatasetMapper
from instructseg.model.mask_decoder.mask_criterion.Mask_Criterion import InstructSeg_criterion, hungarian_matcher_InstructSeg
from transformers import PhiModel, PhiForCausalLM, PhiConfig
from .sampler import StokenAttentionLayerWithSampler,MultiStokenWithSampler


@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_SEG_class: Optional[torch.FloatTensor] = None
    loss_rvos_class: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None
    

class InstructSegModel(MiphaPhiModel):
    # config_class = LlavaConfig

    def __init__(self, config: PhiConfig, mask_decoder_cfg=None):
        super(InstructSegModel, self).__init__(config)
        self.cfg = mask_decoder_cfg
        self.projector_outdim = config.hidden_size 
        if hasattr(config, "mm_vision_tower"):
            swin_type = getattr(config,'swin_type','base')
            if swin_type == 'base':
                self.vision_tower_mask = build_swin_b(None)
            else:
                self.vision_tower_mask = build_swin_l(None)

            self.vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vision_tower_mask(self):
        vision_tower = getattr(self, 'vision_tower_mask', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower if hasattr(model_args, 'vision_tower') else model_args.mm_vision_tower
        vision_tower_mask = model_args.vision_tower_mask if hasattr(model_args, 'vision_tower_mask') else model_args.mm_vision_tower_mask

        self.config.mm_vision_tower = vision_tower
        swin_type = getattr(model_args,'swin_type','base')
        self.config.swin_type = swin_type
        if swin_type == 'base':
            vision_tower_mask = build_swin_b(vision_tower_mask)
        else:
            print('current visual encoder is swin large')
            vision_tower_mask = build_swin_l(vision_tower_mask)
            # self.config.mm_input_embeds = 1536

        if fsdp is not None and len(fsdp) > 0:
            # self.vision_tower = [vision_tower]
            self.vision_tower_mask = [vision_tower_mask]
        else:
            # self.vision_tower = vision_tower
            self.vision_tower_mask = vision_tower_mask

        self.config.use_mm_proj = True
        vision_tower_mask.hidden_size = 256
        vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

        

class InstructSeg(MiphaPhiForCausalLM):
    # config_class = LlavaConfig

    def __init__(self, config, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None):
        super(InstructSeg, self).__init__(config)

        self.model = InstructSegModel(config, mask_decoder_cfg)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index

        # TODO: the tokenizer length of phi-2 is 50295, but the output class of lm_head is 51200
        self.lm_head = nn.Linear(config.hidden_size, 51200, bias=False)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        self.is_train_mask_decode = is_train_mask_decode
        self.refer_pooling = nn.AdaptiveAvgPool1d(output_size=1)

        self.use_temporal_query = True
        self.use_vmtf = True

        if is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()
        self.post_init()

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print('Initialize mask modules...')
            self.config.mask_decode_train = True
        self.seg_query = nn.Parameter(
            torch.zeros([self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, self.config.hidden_size]))

        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.seg_query_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
    
        # self.token_sampler = StokenAttentionLayerWithSampler(embed_dim=2560,dropout=0.1,num_samples=160,slot_iters=6,slot_eps=1e-8)
        
        self.token_sampler = MultiStokenWithSampler(dim=2560, n_iter=1, num_heads=4, sampler_k=480, slot_iters=8, slot_eps=1e-8)
        if self.use_temporal_query:
            self.ovp_on_img = True
            self.num_temporal_queries = 128
            self.ovp_layers_num = 3
            self.ovp_layers = OVPsampler(dim=self.config.hidden_size, depth=self.ovp_layers_num, dim_head=128, heads=8, ff_mult=2)
            self.temporal_query = nn.Parameter(
                torch.randn(self.num_temporal_queries, self.config.hidden_size))

        if self.use_vmtf:
            additional_dim = 1024
            self.vmtf_layers_num = 3
            self.vmtf_layers = VMTF(dim=additional_dim, depth=self.vmtf_layers_num, dim_head=128, heads=8, ff_mult=2)
            local_fea_dim = [256, 512, 1024]
            self.local_project = nn.Conv2d(local_fea_dim[-1], additional_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.local_project)

            
            self.text_vmtf_projector = nn.Linear(self.config.hidden_size, additional_dim)

            self.origin_SEG_token_projector = nn.Linear(self.config.hidden_size, additional_dim)


            self.SEG_token_projector = nn.Linear(additional_dim*2, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        
        else:
            self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
            
    

        self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith('.pkl'):
                with open(pretrained_path, 'rb') as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt['model'],'sem_seg_head.pixel_decoder')
            predictor_weights = get_w(ckpt['model'],'sem_seg_head.predictor')
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            #deal some diff keys
            change_w(pixel_decoder_weights,'adapter_1.weight','adapter_1.0.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.weight','adapter_1.1.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.bias','adapter_1.1.bias')
            change_w(pixel_decoder_weights,'layer_1.weight','layer_1.0.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.weight','layer_1.1.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.bias','layer_1.1.bias')
            if 'static_query.weight' in predictor_weights:
                change_w(predictor_weights,'static_query.weight','query_feat.weight')
            if predictor_weights['query_embed.weight'].shape[0] == 200:
                predictor_weights['query_embed.weight'] = predictor_weights['query_embed.weight'][:100,:]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights,strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights,strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)


    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower_mask()(images)
        features_dict = {
            'res2': features[0], # bs, 128, 256, 256
            'res3': features[1], # bs, 256, 128, 128
            'res4': features[2], # bs, 512, 64, 64
            'res5': features[3], # bs, 1024, 32, 32
        }
        return features_dict
    def mask_decoder_training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT
        
        matcher = hungarian_matcher_InstructSeg(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        
        weight_dict = {"loss_SEG_class": class_weight,  "loss_mask": mask_weight,
                       "loss_dice": dice_weight,
                       }
        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["SEG_labels", "masks",]
        self.criterion = InstructSeg_criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )
        self.size_divisibility = 32
        self.referring_on = True

        self.sem_seg_postprocess_before_inference = self.referring_on
    
    
    def SEG_instance_inference(self, SEG_cls, mask_pred, use_soft=False):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        scores = F.sigmoid(SEG_cls)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        if use_soft:
            clean_mask = (mask_pred > 0).float()
            clean_mask_counts = clean_mask.sum(dim=(1,2))
            result.pred_masks = mask_pred.sigmoid()
            
        else:
            result.pred_masks = (mask_pred > 0).float()
            clean_mask = result.pred_masks
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * clean_mask.flatten(1)).sum(1) / (
                clean_mask.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        return result


    
    # def encode_images(self, images):
    #     image_features = self.get_model().get_vision_tower()(images)
    #     image_features = self.get_model().mm_projector(image_features)

    #     sample_ratio=0.01
    #     diversity_strength=0.9
    #     if image_features.ndim == 3:
    #         B, L, D = image_features.shape
    #         K = max(1, int(L * sample_ratio))

    #         selected_feats = []

    #         for b in range(B):
    #             feats = image_features[b]  # (L, D)
    #             normed = F.normalize(feats, dim=1)

    #         # Inline cosine diversity sampling
    #             selected = [torch.randint(L, (1,)).item()]
    #             candidates = set(range(L)) - set(selected)

    #             for _ in range(K - 1):
    #                 selected_feats_sub = normed[selected]  # (k, D)
    #                 remaining_indices = list(candidates)
    #                 remaining_feats = normed[remaining_indices]  # (~L, D)

    #                 sim = F.cosine_similarity(
    #                     remaining_feats.unsqueeze(1),       # (R, 1, D)
    #                     selected_feats_sub.unsqueeze(0),    # (1, k, D)
    #                     dim=-1
    #                 )  # (R, k)

    #                 avg_sim = sim.mean(dim=1)              # (R,)
    #                 diversity_score = 1 - avg_sim          # Higher = more diverse

    #                 if diversity_strength < 1.0:
    #                     noise = torch.rand_like(diversity_score)
    #                     diversity_score = (
    #                         diversity_strength * diversity_score +
    #                         (1 - diversity_strength) * noise
    #                     )

    #                 best_idx = torch.argmax(diversity_score).item()
    #                 next_token = remaining_indices[best_idx]
    #                 selected.append(next_token)
    #                 candidates.remove(next_token)

    #             selected_feats.append(feats[torch.tensor(selected, dtype=torch.long)])

    #         image_features = torch.stack(selected_feats, dim=0)  # (B, K, D)

    #     return image_features

    
    # def get_text_image_tokens(self, images):
    #     return self.encode_images(images)


    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = self.token_sampler(image_features)
        return image_features
    

    def get_text_image_tokens(self, images):
        return self.encode_images(images)
    
    
    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(in_channels,
                                                                     hidden_dim,
                                                                     num_queries,
                                                                     nheads,
                                                                     dim_feedforward,
                                                                     dec_layers,
                                                                     pre_norm,
                                                                     mask_dim,
                                                                     enforce_input_project,
                                                                     seg_norm,
                                                                     seg_proj,
                                                                     seg_fuse_score)
        return predictor


    def get_model(self):
        return self.model
    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i) for i in
                        range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {'channel': out_feature_channels[name], 'stride': out_feature_strides[name]})
        return backbone_feature_shape

    def get_encoder_image(self, images):
        encode_image_features = self.get_model().get_vision_tower()(images)
        return encode_image_features

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder
    
    def prepare_targets(self, targets, images):
        
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        has_gt_ids = False
        if hasattr(targets[0], 'gt_ids'):
            has_gt_ids = True
        for targets_per_image in targets:
            if has_gt_ids:
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids!=-1
            else:
                inst_ids = None
                valid_id = None
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "valid": valid_id,
                    "inst_id": inst_ids,
                    # "positive_map": positive_map,
                }
            )
        return new_targets

    def filter_valid_targets(self, det_targets, ref_targets):
        bz = len(det_targets)
        for i in range(bz):  # fliter empety object in key frame. Note that det_loss is only computed on the key frame !
            det_target = det_targets[i]
            ref_target = ref_targets[i]
            if False in det_target['valid']:
                valid_i = det_target['valid'].clone()
                for k,v in det_target.items():
                    det_target[k] = v[valid_i]
                for k,v in ref_target.items():                    
                    ref_target[k] = v[valid_i]


        return det_targets,ref_targets

    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS


    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer


    def concat_image_seg_cls_embeds(self, input_id, img_feature, label, seg_query, seg_query_mask,
                                    refer_embedding_indices=None, refer_embedding=None, temporal_query=None, temporal_query_mask=None):
        image_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
        seg_query_indices = torch.where(input_id == SEG_TOKEN_INDEX)[0]
        assert len(image_token_indices) == 1, 'not supporting multi image index'
        # assert len(seg_query_indices) == 1, 'not supporting multi seg index'
        
        cur_new_input_embeds = []
        cur_new_seg_query_mask = []
        if label is not None:
            cur_new_label = []
            assert label.shape == input_id.shape
        else:
            cur_new_label = None
        
        cur_refer_embedding_indices = [] if refer_embedding_indices is not None else None


        if temporal_query_mask is not None:
            enable_temporal_mask = True
            cur_new_temporal_query_mask = []
        else:
            enable_temporal_mask = False
            cur_new_temporal_query_mask = None
        chunks = []
        current_chunk = []

        for id in input_id:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=input_id.device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=input_id.device))

       
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX:
                cur_new_input_embeds.append(img_feature)
                cur_new_seg_query_mask.append(torch.zeros(img_feature.shape[0]))
                
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((img_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )

                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(img_feature.shape[0]))
            elif chunk_len == 1 and chunk[0] == SEG_TOKEN_INDEX:
                cur_new_input_embeds.append(seg_query)
                cur_new_seg_query_mask.append(torch.ones(seg_query.shape[0]))
                
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((seg_query.shape[0],), 0, device=input_id.device,
                                                                       dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((seg_query.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype))
        
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(seg_query.shape[0]))

            elif chunk_len == 1 and chunk[0] == TEMPORAL_TOKEN_INDEX:
                cur_new_input_embeds.append(temporal_query)
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.ones(temporal_query.shape[0]))
                cur_new_seg_query_mask.append(torch.zeros(temporal_query.shape[0]))
                
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((temporal_query.shape[0],), 0, device=input_id.device,
                                                                       dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((temporal_query.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype))

                  
            elif chunk_len == 1 and chunk[0] == REFER_TOKEN_INDEX:
                refer_embed = refer_embedding
                if len(refer_embed.shape) == 1:
                    refer_embed = refer_embed.unsqueeze(0)
                cur_new_input_embeds.append(refer_embed)
                cur_new_seg_query_mask.append(torch.zeros(refer_embed.shape[0]))
                
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 1, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((refer_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(refer_embed.shape[0]))
            
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(input_id[:chunk_len]))
                cur_new_seg_query_mask.append(seg_query_mask[:chunk_len])
                
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(refer_embedding_indices[:chunk_len])
                if label is not None:
                    cur_new_label.append(label[:chunk_len])
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(temporal_query_mask[:chunk_len])

                

            input_id = input_id[chunk_len:]
            seg_query_mask = seg_query_mask[chunk_len:]
            
            if refer_embedding_indices is not None:
                refer_embedding_indices = refer_embedding_indices[chunk_len:]
            if label is not None:
                label = label[chunk_len:]
            
            if enable_temporal_mask:
                temporal_query_mask = temporal_query_mask[chunk_len:]

        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        if label is not None:
            cur_new_label = [x.to(device=self.device) for x in cur_new_label]
            cur_new_label = torch.cat(cur_new_label, dim=0)
        cur_new_seg_query_mask = [x.to(device=self.device) for x in cur_new_seg_query_mask]
        cur_new_seg_query_mask = torch.cat(cur_new_seg_query_mask, dim=0)
        
        if refer_embedding_indices is not None:
            cur_refer_embedding_indices = [x.to(device=self.device) for x in cur_refer_embedding_indices]
            cur_refer_embedding_indices = torch.cat(cur_refer_embedding_indices, dim=0)


        if enable_temporal_mask:
            cur_new_temporal_query_mask = [x.to(device=self.device) for x in cur_new_temporal_query_mask]
            cur_new_temporal_query_mask = torch.cat(cur_new_temporal_query_mask, dim=0)

        return cur_new_input_embeds, cur_new_label, cur_new_seg_query_mask, cur_new_temporal_query_mask, cur_refer_embedding_indices
    

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images,
            token_refer_id=None, refer_embedding_indices=None, expanded_seg_query=None, temporal_query=None):
        vision_tower = self.get_vision_tower()
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, None, None, None

        seg_query_mask = torch.zeros_like(input_ids)
        # if type(images) is list or images.ndim == 5:
        #     concat_images = torch.cat([image for image in images], dim=0)
        #     image_features = self.encode_images(concat_images)
        #     split_sizes = [image.shape[0] for image in images]
        #     image_features = torch.split(image_features, split_sizes, dim=0)
        #     image_features = [x.flatten(0, 1) for x in image_features]
        # else:
        image_features = self.encode_images(images[:, 0])
        
        
        temporal_query_mask = None
        # for video temporal_query
        if temporal_query is not None:
            temporal_query_mask = None # torch.zeros_like(input_ids)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_seg_query_masks = []
        
        new_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        new_temporal_query_masks = [] if temporal_query_mask is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_seg_query_mask = seg_query_mask[batch_idx]
            cur_temporal_query_mask = temporal_query_mask[batch_idx] if temporal_query_mask is not None else None
            cur_seg_query = expanded_seg_query[batch_idx]
            cur_temporal_query = temporal_query[batch_idx] if temporal_query is not None else None
            cur_image_feature = image_features[batch_idx]
            
            cur_refer_embedding_indices = refer_embedding_indices[batch_idx] if refer_embedding_indices is not None else None
            
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_seg_query_masks.append(cur_seg_query_mask)
                # cur_image_idx += 1
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None

            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            cur_input_embeds, cur_label, cur_seg_query_mask, cur_temporal_query_mask, cur_refer_embedding_indices = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                label=cur_label,
                temporal_query=cur_temporal_query,
                temporal_query_mask=cur_temporal_query_mask,
                seg_query=cur_seg_query,
                seg_query_mask=cur_seg_query_mask,
                refer_embedding_indices=cur_refer_embedding_indices,
                refer_embedding=cur_refer_embedding
            )
            assert cur_input_embeds.shape[0] == cur_seg_query_mask.shape[0]

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)
            new_seg_query_masks.append(cur_seg_query_mask)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices.append(cur_refer_embedding_indices)
            if temporal_query_mask is not None:
                new_temporal_query_masks.append(cur_temporal_query_mask)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            new_seg_query_masks_align = []
            for new_seg_query_mask in new_seg_query_masks:
                new_seg_query_mask = torch.cat(
                    (new_seg_query_mask, torch.zeros((max_len - new_seg_query_mask.shape[0]),dtype=new_seg_query_mask.dtype, device=new_seg_query_mask.device)),
                    dim=0)
                new_seg_query_masks_align.append(new_seg_query_mask)
            new_seg_query_masks = torch.stack(new_seg_query_masks_align, dim=0)

            if refer_embedding_indices is not None:
                new_refer_embedding_indices_align = []
                for new_refer_embedding_indice in new_refer_embedding_indices:
                    new_refer_embedding_indice = torch.cat(
                        (new_refer_embedding_indice,
                         torch.zeros((max_len - new_refer_embedding_indice.shape[0]),dtype=new_refer_embedding_indice.dtype, device=new_refer_embedding_indice.device)),
                        dim=0)
                    new_refer_embedding_indices_align.append(new_refer_embedding_indice)
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices_align, dim=0)


            if temporal_query_mask is not None:
                new_temporal_query_masks_align = []
                for new_temporal_query_mask in new_temporal_query_masks:
                    new_temporal_query_mask = torch.cat(
                        (new_temporal_query_mask, torch.zeros((max_len - new_temporal_query_mask.shape[0]),dtype=new_temporal_query_mask.dtype, device=new_temporal_query_mask.device)),
                        dim=0)
                    new_temporal_query_masks_align.append(new_temporal_query_mask)
                new_temporal_query_masks = torch.stack(new_temporal_query_masks_align, dim=0)


            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            new_seg_query_masks = torch.stack(new_seg_query_masks, dim=0)
            
            if refer_embedding_indices is not None:
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices, dim=0)

            
            if new_temporal_query_masks is not None:
                new_temporal_query_masks = torch.stack(new_temporal_query_masks, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_seg_query_masks, new_refer_embedding_indices, new_temporal_query_masks
    

    def get_SEG_embedding(self,hidden_states, refer_embedding_indices, return_all=False):
        refer_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, refer_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            current_pool_refer_state = self.refer_pooling(current_refer_state.transpose(-2, -1)).transpose(-2, -1)
            if return_all:
                current_pool_refer_state = torch.cat([current_pool_refer_state, current_refer_state], dim=0)

            refer_embedding_list.append(current_pool_refer_state)
        
        return torch.stack(refer_embedding_list, dim=0) if not return_all else refer_embedding_list




    def get_ovp_feat(self, cur_temporal_query, image_features, token_refer_id):

        out_temporal_query = []

        for bs_idx in len(token_refer_id):   
            cur_refer_embedding = self.embed_refer_ids(token_refer_id[bs_idx])
            cur_image_text_features = torch.cat((image_features[bs_idx], cur_refer_embedding), dim=0)
            cur_out_query = self.ovp_layers(
                latents=cur_temporal_query[bs_idx:bs_idx+1].unsqueeze(1), x=cur_image_text_features.unsqueeze(0).unsqueeze(1))

            out_temporal_query.append(cur_out_query)
        out_temporal_query = torch.cat(out_temporal_query, dim=0)

        return out_temporal_query

    def get_ovp_feat_time(self, cur_temporal_query, image_features, token_refer_id):

        T = image_features.shape[1]

        out_temporal_query = []

        for bs_idx in range(len(token_refer_id)):   
            cur_refer_embedding = self.embed_refer_ids(token_refer_id[bs_idx]).unsqueeze(0).expand(T, -1, -1) 
 
 
            cur_image_text_features = torch.cat((image_features[bs_idx], cur_refer_embedding), dim=1)
            cur_out_query = self.ovp_layers(
                latents=cur_temporal_query[bs_idx:bs_idx+1].unsqueeze(1).expand(-1, T, -1, -1), x=cur_image_text_features.unsqueeze(0))

            out_temporal_query.append(cur_out_query)
        out_temporal_query = torch.cat(out_temporal_query, dim=0)

        return out_temporal_query
           



    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            global_step=None,
            dataset_type=None,) -> Union[Tuple, CausalLMOutputWithPast]:
        
        
        # multiple frame training for rvos or revos
        multi_ref_vos_data = ['rvos', 'revos']
        if dataset_type is not None and dataset_type[0] in multi_ref_vos_data:
            output = self.train_rvos(input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                images,
                images_clip,
                return_dict,
                seg_info,
                token_refer_id,
                refer_embedding_indices,
                dataset_type,)
            return output

        
        

        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
            # print(batch_dataset_type)
        else:
            batch_dataset_type = []
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        expanded_seg_query = None

        cur_temporal_query = None

        if images is not None and len(images.shape) < 5:
            # append T dim
            images = images.unsqueeze(1)
        if images_clip is not None and len(images_clip.shape) < 5:
            # append T dim
            images_clip = images_clip.unsqueeze(1)

        if (input_ids == SEG_TOKEN_INDEX).sum() != 0 or (batch_dataset_type != [] and 'mm_conv' not in batch_dataset_type):

            # for generative mode only the 1th stage need
            if input_ids.shape[1] != 1:
                image_features = self.get_vision_tower_feature(images[:, 0])
                bs = input_ids.shape[0]
                # 100 * 2560 -->> bs * 100 * 2560
                expanded_seg_query = self.seg_query.unsqueeze(0).expand(bs, -1, -1)
                    
                if self.ovp_on_img:
                    cur_temporal_query = self.temporal_query.unsqueeze(0).expand(bs, -1, -1) # sequential temporal_query between each frame

                    cur_temporal_query = self.get_ovp_feat_time(cur_temporal_query, self.encode_images(images_clip[:, 0]).unsqueeze(1), token_refer_id)
                    cur_temporal_query = cur_temporal_query.flatten(1,2)
            
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, refer_embedding_indices, cur_temporal_query_masks = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images_clip,
                token_refer_id, refer_embedding_indices, expanded_seg_query, temporal_query=cur_temporal_query)
            
            
        else:
            seg_query_mask = None
            SEG_token_indices = None

            cur_images_clip = None
            if images_clip is not None:
                cur_images_clip = images_clip[:, 0]

            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.mm_conv_prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, cur_images_clip)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)


        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices, return_all=self.use_vmtf)
            if self.use_vmtf: 

                origin_SEG_embedding = torch.cat([self.origin_SEG_token_projector(kk.unsqueeze(0)[:, 0:1]) for kk in SEG_embedding])

                local_vision = self.local_project(image_features['res5']).flatten(2)
                local_vision = local_vision.permute(0, 2, 1)
                new_SEG_embedding = []
                for bs_idx, cur_SEG_embedding in enumerate(SEG_embedding):
                    cur_SEG_embedding = self.text_vmtf_projector(cur_SEG_embedding.unsqueeze(0))
                    # bs(1), 1, dim
                    cur_SEG_embedding = self.vmtf_layers(latents=cur_SEG_embedding.unsqueeze(1), 
                        x=local_vision[bs_idx:bs_idx+1].unsqueeze(1))
                    new_SEG_embedding.append(cur_SEG_embedding)
                new_SEG_embedding = torch.cat(new_SEG_embedding, dim=0)

                SEG_embedding = torch.cat((origin_SEG_embedding, new_SEG_embedding), dim=-1)


        else:
            SEG_embedding = None


        loss = None
        llm_loss = None
        seg_llm_loss_dataset = ['reason_seg']
        if labels is not None:
            if seg_query_mask is None or batch_dataset_type in seg_llm_loss_dataset:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                vocab_size = shift_logits.shape[-1]
                shift_logits = shift_logits.view(-1, vocab_size)  # self.config.vocab_size
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                llm_loss = loss_fct(shift_logits, shift_labels)

        
        if seg_query_mask is not None:
            
            seg_query = self.get_seg_query(hidden_states, seg_query_mask)  # bs N 2560
            
            seg_query = self.seg_query_projector(seg_query) # bs 100 256

            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)

            if refer_embedding_indices is not None:
                SEG_embedding = self.SEG_token_projector(SEG_embedding)


            mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,)
            if seg_info is not None:
                if "instances" in seg_info[0]:
                    if isinstance(seg_info[0]["instances"], list):
                        gt_instances = [x["instances"][0].to(self.device) for x in seg_info]
                    else:
                        gt_instances = [x["instances"].to(self.device) for x in seg_info]

                    targets = self.prepare_targets(gt_instances, images[:, 0])
                else:
                    targets = None

                mask_losses = self.criterion(mask_outputs, targets)
                weight_dict = self.weight_dict

                loss_mask = 0.0
                loss_dice = 0.0
                loss_SEG_class = 0.0
                loss_rvos_class = 0.0
            
                for k in list(mask_losses.keys()):
                    if k in weight_dict:
                        if mask_losses[k] is not None:
                            mask_losses[k] *= weight_dict[k]
                        if '_SEG' in k and mask_losses[k] is not None and batch_dataset_type != 'rvos':
                            loss_SEG_class += mask_losses[k]
                        elif '_SEG' in k and mask_losses[k] is not None and batch_dataset_type == 'rvos':
                            loss_rvos_class += mask_losses[k]
                        
                        elif '_mask' in k:
                            loss_mask += mask_losses[k]
                        
                        elif '_dice' in k:
                            loss_dice += mask_losses[k]
                    else:
                        mask_losses.pop(k)
                mask_loss = loss_mask + loss_dice + loss_SEG_class + loss_rvos_class
            
                if isinstance(loss_SEG_class, float):
                    loss_SEG_class = torch.tensor(loss_SEG_class, device=mask_loss.device)
                
                if isinstance(loss_rvos_class, float):
                    loss_rvos_class = torch.tensor(loss_rvos_class, device=mask_loss.device)

        
            
            if labels is not None:
                if llm_loss is not None:
                    loss = llm_loss + mask_loss
                    llm = llm_loss
                else:
                    loss = mask_loss
                    llm = torch.tensor(0.0, device=mask_loss.device)

            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                loss_mask=loss_mask.detach(),
                loss_dice=loss_dice.detach(),
                loss_SEG_class=loss_SEG_class.detach(),
                loss_llm=llm.detach(),
                loss_rvos_class=loss_rvos_class.detach(),
            )

        if labels is not None and seg_query_mask is None:
            loss_mask = torch.tensor(0.0, device=llm_loss.device)
            loss_dice = torch.tensor(0.0, device=llm_loss.device)
            loss_SEG_class = torch.tensor(0.0, device=llm_loss.device)
            loss_rvos_class = torch.tensor(0.0, device=llm_loss.device)
            loss = llm_loss
        else:
            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_SEG_class=loss_SEG_class.detach(),
            loss_llm=llm_loss.detach(),
            loss_rvos_class=loss_rvos_class.detach(),
            
        )



    def train_rvos(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            dataset_type=None,) -> Union[Tuple, CausalLMOutputWithPast]:
            
        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
            # print(batch_dataset_type)
        else:
            batch_dataset_type = []
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        expanded_seg_query = None

        T = images.shape[1]
        T_all = images_clip.shape[1]


        bs = input_ids.shape[0]
        # 100 * 2560 -->> bs * 100 * 2560
        expanded_seg_query = self.seg_query.unsqueeze(0).expand(bs, -1, -1) # share same seg_query for each frame
        if self.use_temporal_query:
            origin_temporal_query = self.temporal_query.unsqueeze(0).expand(bs, -1, -1) # sequential temporal_query between each frame
            reference_frame_features = [self.encode_images(images_clip[:, k]) for k in range(T, T_all)]

            cur_temporal_query = self.get_ovp_feat_time(origin_temporal_query, torch.stack(reference_frame_features, 1), token_refer_id)
            cur_temporal_query = cur_temporal_query.flatten(1,2)

            # cur_temporal_query = []
            # for k in range(len(reference_frame_features)):
            #     cur_temporal_query.append(self.get_ovp_feat(origin_temporal_query, reference_frame_features[k], token_refer_id))

            # cur_temporal_query = torch.cat(cur_temporal_query, dim=1)

            
        else:
            cur_temporal_query = None

        


        
        # for every frame t
        loss = 0.
        for t in range(T):

            image_features = self.get_vision_tower_feature(images[:, t])
            
            cur_input_ids, cur_attention_mask, cur_past_key_values, cur_inputs_embeds, cur_labels, cur_seg_query_mask, cur_refer_embedding_indices, cur_temporal_query_masks = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images_clip[:, t:t+1],
                token_refer_id, refer_embedding_indices, expanded_seg_query, temporal_query=cur_temporal_query)
            
        
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=cur_past_key_values,
                inputs_embeds=cur_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)



            if cur_refer_embedding_indices is not None:
                SEG_embedding = self.get_SEG_embedding(hidden_states, cur_refer_embedding_indices, return_all=self.use_vmtf)
                if self.use_vmtf:
                    
                    origin_SEG_embedding = torch.cat([self.origin_SEG_token_projector(kk.unsqueeze(0)[:, 0:1]) for kk in SEG_embedding])

                    local_vision = self.local_project(image_features['res5']).flatten(2)
                    local_vision = local_vision.permute(0, 2, 1)
                    new_SEG_embedding = []
                    for bs_idx, cur_SEG_embedding in enumerate(SEG_embedding):
                        cur_SEG_embedding = self.text_vmtf_projector(cur_SEG_embedding.unsqueeze(0))
                        # bs(1), 1, dim
                        cur_SEG_embedding = self.vmtf_layers(latents=cur_SEG_embedding.unsqueeze(1), 
                            x=local_vision[bs_idx:bs_idx+1].unsqueeze(1))
                        new_SEG_embedding.append(cur_SEG_embedding)
                    new_SEG_embedding = torch.cat(new_SEG_embedding, dim=0)

                    SEG_embedding = torch.cat((origin_SEG_embedding, new_SEG_embedding), dim=-1)
                
                SEG_embedding = self.SEG_token_projector(SEG_embedding)
            else:
                SEG_embedding = None

            if cur_seg_query_mask is not None:
                
                seg_query = self.get_seg_query(hidden_states, cur_seg_query_mask)  # bs N 2560

                seg_query = self.seg_query_projector(seg_query) # bs 100 256

                
                mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                    image_features)

                mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding)
                if seg_info is not None:
                    if "instances" in seg_info[0]:
                        if isinstance(seg_info[0]["instances"], list):
                            gt_instances = [x["instances"][t].to(self.device) for x in seg_info]
                        else:
                            gt_instances = [x["instances"].to(self.device) for x in seg_info]

                        targets = self.prepare_targets(gt_instances, images[:, t])
                    else:
                        targets = None

                    mask_losses = self.criterion(mask_outputs, targets)
                    weight_dict = self.weight_dict

                    loss_mask = 0.0
                    loss_dice = 0.0
                    loss_SEG_class = 0.0
                    
                    loss_rvos_class = 0.0
                    for k in list(mask_losses.keys()):
                        if k in weight_dict:
                            if mask_losses[k] is not None:
                                mask_losses[k] *= weight_dict[k]

                            if '_SEG' in k and mask_losses[k] is not None:
                                loss_rvos_class += mask_losses[k]
                            
                            elif '_mask' in k:
                                loss_mask += mask_losses[k]
                            elif '_dice' in k:
                                loss_dice += mask_losses[k]
                            
                        else:
                            mask_losses.pop(k)
                    
                    mask_loss = loss_mask + loss_dice + loss_rvos_class
                    
                    if isinstance(loss_SEG_class, float):
                        loss_SEG_class = torch.tensor(loss_SEG_class, device=mask_loss.device)
                    
                    
                    if isinstance(loss_rvos_class, float):
                        loss_rvos_class = torch.tensor(loss_rvos_class, device=mask_loss.device)

                    loss += mask_loss
                    llm = torch.tensor(0.0, device=mask_loss.device)

        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_SEG_class=loss_SEG_class.detach(),
            loss_rvos_class=loss_rvos_class.detach(),
            loss_llm=llm.detach(),
        )


    def get_seg_query(self, hidden_states, seg_query_masks):
        seg_query_list = []
        for sample_hidden_state, sample_query_mask in zip(hidden_states, seg_query_masks):
            if torch.sum(sample_query_mask) == 0:
                continue

            unique_query_value = torch.unique(sample_query_mask)
            unique_query_value = unique_query_value[unique_query_value != 0]

            for value in unique_query_value:
                current_query_mask = (sample_query_mask == value)
                current_query = sample_hidden_state[current_query_mask]

                seg_query_list.append(current_query)

        seg_query = torch.stack(seg_query_list, dim=0)

        return seg_query
    
    
    def eval_seg(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            padding_mask=None,
            use_soft=False,
            is_thing_list=None,
            video_on = False,):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if len(images.shape) < 5:
            # append T dim
            images = images.unsqueeze(1)
            images_clip = images_clip.unsqueeze(1)
        
        image_features = self.get_vision_tower_feature(images[:, 0])
        
        bs = input_ids.shape[0]
        # 100 * 2560 -->> bs * 100 * 2560
        expanded_seg_query = self.seg_query.unsqueeze(0).expand(bs, -1, -1)

        if self.use_temporal_query:
        
            if video_on:
                T_all = images_clip.shape[1]
                if T_all == 1:
                    cur_temporal_query = self.temporal_query.unsqueeze(0).expand(bs, -1, -1) # sequential temporal_query between each frame
                    cur_temporal_query = self.get_ovp_feat_time(cur_temporal_query, self.encode_images(images_clip[:, 0]).unsqueeze(1), token_refer_id)
                    cur_temporal_query = cur_temporal_query.flatten(1,2)
                else:
                    origin_temporal_query = self.temporal_query.unsqueeze(0).expand(bs, -1, -1) # sequential temporal_query between each frame
                    try:
                        reference_frame_features = [self.encode_images(images_clip[:, k]) for k in range(1, T_all)]
                    except:
                        print(T_all, images_clip.shape)

                    cur_temporal_query = self.get_ovp_feat_time(origin_temporal_query, torch.stack(reference_frame_features, 1), token_refer_id)
                    cur_temporal_query = cur_temporal_query.flatten(1,2)

            elif self.ovp_on_img:
                cur_temporal_query = self.temporal_query.unsqueeze(0).expand(bs, -1, -1) # sequential temporal_query between each frame
                cur_temporal_query = self.get_ovp_feat_time(cur_temporal_query, self.encode_images(images_clip[:, 0]).unsqueeze(1), token_refer_id)
                cur_temporal_query = cur_temporal_query.flatten(1,2)

        
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, refer_embedding_indices, cur_temporal_query_masks = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images_clip,
            token_refer_id, refer_embedding_indices, expanded_seg_query, temporal_query=cur_temporal_query)
        

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state


        seg_query = self.get_seg_query(hidden_states, seg_query_mask)

        seg_query = self.seg_query_projector(seg_query)

        bs = seg_query.shape[0]

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices, return_all=self.use_vmtf)
            if self.use_vmtf:

                origin_SEG_embedding = torch.cat([self.origin_SEG_token_projector(kk.unsqueeze(0)[:, 0:1]) for kk in SEG_embedding])

                local_vision = self.local_project(image_features['res5']).flatten(2)
                local_vision = local_vision.permute(0, 2, 1)
                new_SEG_embedding = []
                for bs_idx, cur_SEG_embedding in enumerate(SEG_embedding):
                    cur_SEG_embedding = self.text_vmtf_projector(cur_SEG_embedding.unsqueeze(0))
                    # bs(1), 1, dim
                    cur_SEG_embedding = self.vmtf_layers(latents=cur_SEG_embedding.unsqueeze(1), 
                        x=local_vision[bs_idx:bs_idx+1].unsqueeze(1))
                    new_SEG_embedding.append(cur_SEG_embedding)
                new_SEG_embedding = torch.cat(new_SEG_embedding, dim=0)

                SEG_embedding = torch.cat((origin_SEG_embedding, new_SEG_embedding), dim=-1)

        else:
            SEG_embedding = None

        
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.SEG_token_projector(SEG_embedding)
        

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding)

        
        mask_pred_results = mask_outputs["pred_masks"]
        images = [x[0] for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        
        SEG_cls_results = mask_outputs['pred_SEG_logits']

        # mask_pred_results_0 = (mask_pred_results>0).float()
        # mask_pred_results_sum = mask_pred_results_0.sum(dim=(2,3))
        del mask_outputs
        processed_results = []
        if SEG_cls_results is None:
            SEG_cls_results = [None]
        
        for _seg_info, SEG_cls_result, mask_pred_result, input_per_image, image_size in zip(
                seg_info, SEG_cls_results, mask_pred_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            if padding_mask is None:
                padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~ np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            # gt = _seg_info['instances'].gt_masks
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)


            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(SEG_cls_result.float(),
                                                                            mask_pred_result.float(), use_soft=use_soft)
                processed_results[-1]["instances"] = instance_r
           

            return processed_results




    def mm_conv_prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # concat text and image embedding. prepare labels, IGNORE_INDEX for image tokens
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Align embedddings, labels, attn_mask from different sample into a batch
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels




    def eval_vqa(
            self,
            do_sample=True,
            temperature=0.2,
            num_beams=1,
            max_new_tokens=128,
            eos_token_id = None,
            use_cache=True,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,):
        
        # input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.mm_conv_prepare_inputs_labels_for_multimodal(
        #         input_ids, attention_mask, past_key_values, labels, images_clip)
        

        output_ids = self.generate(
            input_ids=input_ids,
            images_clip=images_clip,
            do_sample=do_sample,
            eos_token_id = eos_token_id,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache)

        
        return output_ids
