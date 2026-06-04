import logging

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import base64
import torch.nn.functional as F
import io
from PIL import Image
import numpy as np
import copy
from fastapi.staticfiles import StaticFiles
from openpyxl import Workbook
from io import BytesIO
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote

import os
import sys
import cv2
import torch
import numpy as np

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.clicker import Click, Clicker
from isegm.utils.vis import draw_with_blend_and_clicks
from isegm.inference.predictors import get_predictor
from isegm.utils.gland_utils import GlandUtils
from models.UNet import UNet
from Utils.uncertainty_utils import *
from Utils.TTA_utils import make_reverse_aug_prob_list, make_aug_data_list, make_reverse_aug_mask
from Utils.diversity_utils import furthest_first
from Utils.adaptive_thresholding_utils import RedundancyThresholdEWMA


class ALHelper:
    def __init__(self, device):
        os.makedirs(save_path, exist_ok=True)
        self.unlabeled_names = []  # 保存文件名
        self.unlabeled_images = []  # 保存 base64 字符串
        self.labeled_names = []
        self.labeled_images = []
        self.device = device
        self.TARGET_H = 352
        self.TARGET_W = 736
        self.model = None
        self.strategy = None
        self.num_inferences = 8
        self.alpha = 0.5
        self.transform_config = {
            # 随机水平翻转
            "hflip": True,
            # 旋转（随机 -15° ~ +15°）
            "rotation": 10,
            # 随机 0~3 次 90° 旋转（90/180/270）
            "rot90": True,
            # 加性高斯噪声
            "gauss_noise": {"mean": 0, "std": 0.01}
        }
        self.budget = 0
        self.al_cycle = 0
        self.threshold = 0.9
        self.stage = 1
        self.threshold_calculator = RedundancyThresholdEWMA(alpha=1.0, gamma=0.8)

    def clear_unlabeled(self):
        self.unlabeled_names = []
        self.unlabeled_images = []

    def clear_labeled(self):
        self.labeled_images = []
        self.labeled_names = []

    def load_image(self, content: bytes):
        # === bytes → PIL
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        # === PIL → numpy
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        # === resize
        img_resized = cv2.resize(
            img_np,
            (al_helper.TARGET_W, al_helper.TARGET_H),
            interpolation=cv2.INTER_LINEAR
        )
        # === numpy → torch (C×H×W)
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(self.device)

        return img_t

    def load_model(self, ckpt_bytes: bytes):
        buffer = io.BytesIO(ckpt_bytes)
        checkpoint = torch.load(buffer, map_location=self.device)

        self.model = UNet(**checkpoint['hyper_parameters'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, images, pool_kernel=4, pool_stride=4, tag="Unlabeled"):
        """
        通用特征提取方法（用于 Coreset / SDS / 其他采样策略）

        Args:
            images (List[torch.Tensor]): 一组图像（每个形状为 [C, H, W]）
            pool_kernel (int): 池化核大小
            pool_stride (int): 池化步长
            tag (str): 打印日志时的标签（如 'Unlabeled' / 'Labeled'）
        Returns:
            np.ndarray: 特征矩阵 [N, D]
        """
        print(f"[Embedding] 提取 {tag} 样本特征...")
        embeddings = []

        self.model.eval()
        with torch.no_grad():
            for img_t in images:
                x = img_t.unsqueeze(0).to(self.device)
                _, [enc1, enc2, enc3, center, dec1, dec2, dec3] = self.model(x)
                pooled_dec = F.max_pool2d(dec3, kernel_size=pool_kernel, stride=pool_stride, padding=0)
                feat = pooled_dec.view(pooled_dec.size(0), -1).cpu().numpy()
                embeddings.append(feat[0])

        embeddings = np.array(embeddings)
        print(f"[Embedding] {tag} 特征维度: {embeddings.shape}")
        return embeddings

    def entropy_sample(self):
        """
        对所有未标注图像计算熵（基于 softmax 输出），
        返回熵值最高的前 `budget` 张图像名。
        """

        if self.model is None:
            raise ValueError("模型未加载，请先上传 .ckpt 文件")
        if len(self.unlabeled_images) == 0:
            raise ValueError("未加载未标注图像数据")

        self.model.eval()
        mean_entropies = []
        with torch.no_grad():
            for img_t in self.unlabeled_images:
                # img_t: C×H×W
                x = img_t.unsqueeze(0).to(self.device)  # 扩维 → [1, C, H, W]
                logits, _ = self.model(x)  # [1, num_classes, H, W]

                # softmax 概率
                prob = F.softmax(logits, dim=1)  # [1, num_classes, H, W]

                # 熵计算：H = -∑p·log(p)
                entropy_map = entropy(prob, 1)  # [1, H, W]

                # 求每张图的平均熵
                mean_entropy = entropy_map.mean().item()
                mean_entropies.append(mean_entropy)

        mean_entropies = np.array(mean_entropies)

        # 取熵最大的前 budget 个
        topk_idx = np.argsort(-mean_entropies)[:self.budget]
        selected_names = [self.unlabeled_names[i] for i in topk_idx]

        # 打印调试信息
        print(f"[Entropy] 平均熵值排序前{self.budget}个索引: {topk_idx}")
        print(f"[Entropy] 选中图像: {selected_names}")

        return selected_names

    def dropout_sample(self):
        """
        基于 Monte Carlo Dropout 的不确定性采样。
        对所有未标注图像进行多次前向传播，计算每张图的加权 JSD，
        返回不确定性最高的前 `budget` 张图像名。
        """

        if self.model is None:
            raise ValueError("模型未加载，请先上传 .ckpt 文件")
        if len(self.unlabeled_images) == 0:
            raise ValueError("未加载未标注图像数据")

        self.model.eval()
        mean_jsds = []


        with torch.no_grad():
            for img_t in self.unlabeled_images:
                # img_t: C×H×W
                x = img_t.unsqueeze(0).to(self.device)  # [1, C, H, W]

                # 启用 Dropout
                self.model.apply(lambda m: m.train() if isinstance(m, torch.nn.Dropout) else None)

                # 多次前向传播，收集 softmax 概率
                prob_list = []
                for _ in range(self.num_inferences):
                    logits, _ = self.model(x)
                    prob = F.softmax(logits, dim=1)
                    prob_list.append(prob)

                # 形状整理 → (1, num_inf, C, H, W)
                prob_tensor = torch.stack(prob_list, dim=1)

                # 计算 JSD 不确定性
                jsd_map = weighted_jsd(prob_tensor, alpha=self.alpha)  # [1, H, W]

                # 求每张图的不确定性均值
                mean_jsd = jsd_map.mean().item()
                mean_jsds.append(mean_jsd)

        mean_jsds = np.array(mean_jsds)

        # 取 JSD 最大的前 budget 个
        topk_idx = np.argsort(-mean_jsds)[:self.budget]
        selected_names = [self.unlabeled_names[i] for i in topk_idx]

        # 打印调试信息
        print(f"[Dropout] 平均JSD排序前{self.budget}个索引: {topk_idx}")
        print(f"[Dropout] 选中图像: {selected_names}")

        return selected_names

    def tta_sample(self):
        """
        基于 Test-Time Augmentation (TTA) 的不确定性采样。
        对每张未标注图像生成多种增强版本，计算加权 JSD，
        返回不确定性最高的前 `budget` 张图像名。
        """

        if self.model is None:
            raise ValueError("模型未加载，请先上传 .ckpt 文件")
        if len(self.unlabeled_images) == 0:
            raise ValueError("未加载未标注图像数据")

        self.model.eval()
        mean_jsds = []


        with torch.no_grad():
            for img_t in self.unlabeled_images:
                # img_t: [C, H, W]
                x = img_t.unsqueeze(0).to(self.device)  # [1, C, H, W]
                all_data_list = [x]

                # 生成增强后的样本列表与变换参数
                x_aug_list, param_dic_list = make_aug_data_list(
                    x, self.num_inferences - 1, self.transform_config
                )

                # 拼接原图与增强图像
                all_data_list.extend(x_aug_list)
                all_data = torch.cat(all_data_list, dim=0)  # [num_inferences, C, H, W]

                # 模型前向
                all_logits, _ = self.model(all_data)
                logits, aug_logits = torch.split(
                    all_logits, [1, self.num_inferences - 1], dim=0
                )

                # softmax 概率
                prob = F.softmax(logits, dim=1)  # 原图预测
                prob_TTA_list = make_reverse_aug_prob_list(
                    aug_logits,
                    param_dic_list,
                    model_norm_fct=self.model.out_channels,
                    data_shape=logits.shape,
                )
                prob_list = [prob] + prob_TTA_list

                # 整理形状 → (1, num_inf, C, H, W)
                prob_tensor = torch.stack(prob_list, dim=1)

                # 计算 mask 并加权
                aug_mask = make_reverse_aug_mask(
                    logits.shape, param_dic_list, device=self.device
                )

                # 计算 JSD
                jsd_map = weighted_jsd(prob_tensor * aug_mask, alpha=self.alpha)  # [1, H, W]

                # 求平均不确定性
                mean_jsd = jsd_map.mean().item()
                mean_jsds.append(mean_jsd)

        mean_jsds = np.array(mean_jsds)

        # 取 JSD 最大的前 budget 个
        topk_idx = np.argsort(-mean_jsds)[:self.budget]
        selected_names = [self.unlabeled_names[i] for i in topk_idx]

        # 调试输出
        print(f"[TTA] 平均JSD排序前{self.budget}个索引: {topk_idx}")
        print(f"[TTA] 选中图像: {selected_names}")

        return selected_names

    def coreset_sample(self):
        """
        基于 Coreset (Furthest-First Traversal) 的主动学习采样方法。
        对所有未标注图像提取模型特征嵌入，并选取与已标注样本最远的未标注样本。
        返回多样性最高的前 `budget` 张图像名。
        """

        if self.model is None:
            raise ValueError("模型未加载，请先上传 .ckpt 文件")
        if len(self.unlabeled_images) == 0:
            raise ValueError("未加载未标注图像数据")
        if not hasattr(self, "labeled_images") or len(self.labeled_images) == 0:
            raise ValueError("未加载已标注图像数据，用于计算距离")

        self.model.eval()

        # === Step 1: 提取未标注样本 embedding ===
        embedding_unlabeled = self.get_embeddings(self.unlabeled_images, pool_kernel=4, pool_stride=4, tag="Unlabeled")

        # === Step 2: 提取已标注样本 embedding ===
        embedding_labeled = self.get_embeddings(self.labeled_images, pool_kernel=4, pool_stride=4, tag="Labeled")

        # === Step 3: Furthest-First 采样 ===
        print("[Coreset] 执行 Furthest-First 采样...")
        chosen_indices = furthest_first(
            embedding_unlabeled, embedding_labeled, self.budget
        )

        # === Step 4: 根据索引匹配文件名 ===
        selected_names = [self.unlabeled_names[i] for i in chosen_indices]

        print(f"[Coreset] 选中索引: {chosen_indices}")
        print(f"[Coreset] 选中图像: {selected_names}")

        return selected_names

    def sds_sample(self):
        """
        基于 SDS (Similarity-based Diversity Sampling) 的主动学习采样方法。
        使用余弦相似度计算未标注与已标注样本之间的距离，
        选取与已标注样本最不相似（最具信息性）的前 `budget` 张图像。
        """

        if self.model is None:
            raise ValueError("模型未加载，请先上传 .ckpt 文件")
        if len(self.unlabeled_images) == 0:
            raise ValueError("未加载未标注图像数据")
        if not hasattr(self, "labeled_images") or len(self.labeled_images) == 0:
            raise ValueError("未加载已标注图像数据，用于计算相似度")

        self.model.eval()

        embedding_unlabeled = self.get_embeddings(self.unlabeled_images, pool_kernel=64, pool_stride=64, tag="Unlabeled")
        embedding_labeled = self.get_embeddings(self.labeled_images, pool_kernel=64, pool_stride=64, tag="Labeled")

        # === Step 3: 计算相似度并选出最“不相似”的样本 ===
        print("[SDS] 计算余弦相似度...")
        cos_sim_unlabeled_unlabeled = cosine_similarity(embedding_unlabeled, embedding_unlabeled)
        np.fill_diagonal(cos_sim_unlabeled_unlabeled, -np.inf)
        max_intra_similarity = np.amax(cos_sim_unlabeled_unlabeled, axis=1)

        cos_sim_unlabeled_labeled = cosine_similarity(embedding_unlabeled, embedding_labeled)
        max_exter_similarity = np.amax(cos_sim_unlabeled_labeled, axis=1)

        total_max_similarity = max_intra_similarity + max_exter_similarity

        # 选取总相似度最低的样本
        chosen_indices = np.argsort(total_max_similarity)[:self.budget]

        selected_names = [self.unlabeled_names[i] for i in chosen_indices]

        print(f"[SDS] 选中索引: {chosen_indices}")
        print(f"[SDS] 选中图像: {selected_names}")

        return selected_names

    def phs_sample(self):
        """
        Progressive Hybrid Sampling (PHS)
        阶段 1：先基于熵采样（Entropy），计算候选与标注集相似度，
                 若 mean_similarity > threshold → 进入阶段 2。
        阶段 2：基于 SDS + Entropy 混合采样。
        阈值自适应更新 (EWMA) 由 threshold_calculator 提供。
        """

        if self.model is None:
            raise ValueError("模型未加载，请先上传 .ckpt 文件")
        if len(self.unlabeled_images) == 0:
            raise ValueError("未加载未标注图像数据")
        if not hasattr(self, "labeled_images") or len(self.labeled_images) == 0:
            raise ValueError("未加载已标注图像数据，用于计算相似度")

        self.model.eval()

        print(f"\n[PHS] ===== 第 {self.al_cycle + 1} 轮主动学习 =====")
        print(f"[PHS] 当前阶段: {self.stage}, 阈值 = {self.threshold:.4f}")

        # === 提取特征 (embedding) ===
        embedding_unlabeled = self.get_embeddings(self.unlabeled_images, pool_kernel=64, pool_stride=64,
                                                  tag="Unlabeled")
        embedding_labeled = self.get_embeddings(self.labeled_images, pool_kernel=64, pool_stride=64, tag="Labeled")

        # === Stage 1: 熵采样 + 阈值更新判断 ===
        if self.stage == 1:
            print("[PHS] 阶段 1 → 仅执行熵采样 (Entropy Sampling)")

            mean_entropies = []
            with torch.no_grad():
                for img_t in self.unlabeled_images:
                    x = img_t.unsqueeze(0).to(self.device)
                    logits, _ = self.model(x)
                    prob = F.softmax(logits, dim=1)
                    entropy_map = entropy(prob, 1)
                    mean_entropy = entropy_map.mean().item()
                    mean_entropies.append(mean_entropy)

            mean_entropies = np.array(mean_entropies)
            # 取熵值最高的前 budget 个索引
            top_uncertainty_indices = np.argsort(-mean_entropies)[:self.budget]
            print(f"[PHS][Stage1] 熵最高的前 {self.budget} 个索引: {top_uncertainty_indices}")

            # === 计算这些候选样本与已标注集的外部相似度 ===
            mean_similarity, std_similarity, adaptive_threshold = self.threshold_calculator.calculate_exter_similarity(
                top_uncertainty_indices,
                embedding_unlabeled,
                embedding_labeled,
                self.al_cycle
            )

            print(
                f"[PHS][Stage1] mean_similarity={mean_similarity:.4f}, std={std_similarity:.4f}, threshold={self.threshold:.4f}")
            print(f"[PHS][Stage1] 自适应阈值(adaptive_threshold)={adaptive_threshold:.4f}")

            # 判断是否进入阶段 2
            if mean_similarity > self.threshold:
                print(f"[PHS][Stage1] mean_similarity 超过阈值 → 进入 Stage 2")
                self.stage = 2
            else:
                print(f"[PHS][Stage1] mean_similarity 未超过阈值 → 保持 Stage 1")
                self.threshold = adaptive_threshold # 更新动态自适应阈值

            chosen_indices = top_uncertainty_indices

        # === Stage 2: SDS + Entropy 组合采样 ===
        else:
            print("[PHS] 阶段 2 → 执行 SDS + Entropy 组合采样")

            # Step 1: SDS 多样性采样（前 3×budget）
            cos_sim_unlabeled_unlabeled = cosine_similarity(embedding_unlabeled, embedding_unlabeled)
            np.fill_diagonal(cos_sim_unlabeled_unlabeled, -np.inf)
            max_intra_similarity = np.amax(cos_sim_unlabeled_unlabeled, axis=1)

            cos_sim_unlabeled_labeled = cosine_similarity(embedding_unlabeled, embedding_labeled)
            max_exter_similarity = np.amax(cos_sim_unlabeled_labeled, axis=1)

            total_max_similarity = max_intra_similarity + max_exter_similarity
            top_diverse_num = min(3 * self.budget, len(self.unlabeled_images))
            select_top_diverse_indices = np.argsort(total_max_similarity)[:top_diverse_num]
            print(f"[PHS][Stage2] SDS 选出 {top_diverse_num} 个候选样本")

            # Step 2: 熵计算（在 SDS 候选集中再选前 budget）
            mean_entropies = []
            with torch.no_grad():
                for img_t in self.unlabeled_images:
                    x = img_t.unsqueeze(0).to(self.device)
                    logits, _ = self.model(x)
                    prob = F.softmax(logits, dim=1)
                    entropy_map = entropy(prob, 1)
                    mean_entropies.append(entropy_map.mean().item())

            mean_entropies = np.array(mean_entropies)
            sub_entropies = mean_entropies[select_top_diverse_indices]
            top_uncertainty_order = np.argsort(-sub_entropies)[:self.budget]
            chosen_indices = [select_top_diverse_indices[i] for i in top_uncertainty_order]

            print(f"[PHS][Stage2] 最终选出 {len(chosen_indices)} 张图像（SDS+Entropy）")

        # === 输出选中样本 ===
        selected_names = [self.unlabeled_names[i] for i in chosen_indices]
        print(f"[PHS] 选中图像: {selected_names}")

        # === 更新状态 ===
        self.al_cycle += 1

        return selected_names


class Segmenter:
    def __init__(self, resume_path, save_path, device):
        os.makedirs(save_path, exist_ok=True)
        self.resume_path = resume_path
        self.device = device
        self.pred_thr = 0.49
        self.load_model()
        self.image = None
        self.previous_mask = None
        self.previous_click_num = 0
        self.TARGET_H = 350
        self.TARGET_W = 740
        self.pseudo_clicker = Clicker()
        self.previous_result = None
        self.previous_pseudo_clicks = []

    def load_model(self):
        model = utils.load_is_model(self.resume_path, self.device)
        zoom_in_params = None
        predictor_params = {}
        predictor_params['net_clicks_limit'] = 740 * 350
        self.predictor = get_predictor(model, 'NoBRS', self.device,
                              prob_thresh=0.49,
                              predictor_params=predictor_params,
                              zoom_in_params=zoom_in_params)
    
    def reset(self):
        self.previous_mask = None
        self.image = None
        self.previous_click_num = 0
        self.pseudo_clicker = Clicker()
        self.previous_result = None
        self.previous_pseudo_clicks = []

    def revert(self):
        """Revert to previous segmentation result."""
        if self.previous_result is None:
            return None  # 没有可回退版本
        # 将 previous_result 作为当前 mask
        self.previous_mask = self.previous_result.copy()
        # 恢复 pseudo_clicker 的 clicks_list
        self.pseudo_clicker.clicks_list = copy.deepcopy(self.previous_pseudo_clicks)
        return self.previous_mask

    def segment(self, input_image, foreground_points, background_points, designated_click_radius=-1, scribble_enabled = True, scribble_count = 1) -> Image:
        input_image = np.array(input_image).astype(np.uint8)
        if self.image is None:
            self.image = copy.deepcopy(input_image)
        h, w, _ = input_image.shape
        # 固定输入图像尺寸为350x740,降低推理时间开销
        input_image = cv2.resize(input_image, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_LINEAR)
        h_rs = self.TARGET_H
        w_rs = self.TARGET_W
        designated_click_radius = min(h, w, designated_click_radius)
        # 保存上一轮结果
        if self.previous_mask is not None:
            self.previous_result = self.previous_mask.copy()
        
        with torch.no_grad():
            self.predictor.set_input_image(input_image)
            clicker = Clicker()
            for foreground_point in foreground_points:
                y = foreground_point['y'] / h * h_rs
                x = foreground_point['x'] / w * w_rs
                clicker.add_click(Click(True, (y, x)))
            for background_point in background_points:
                y = background_point['y'] / h * h_rs
                x = background_point['x'] / w * w_rs
                clicker.add_click(Click(False, (y, x)))
            # if self.previous_mask is None:
            #     self.previous_mask = np.zeros(input_image.shape[:2])
            #     pred_probs, perform_time, updated_feedback1, updated_feedback2 = self.predictor.get_prediction(clicker, self.previous_mask, 0.0, designated_click_radius, new_click_num=len(clicker) - self.previous_click_num)
            # else:
            #     previous_mask_rs = cv2.resize(self.previous_mask, (w_rs, h_rs))
            #     pred_probs, perform_time, updated_feedback1, updated_feedback2 = self.predictor.get_prediction(clicker, previous_mask_rs, 1.0, designated_click_radius, new_click_num=len(clicker) - self.previous_click_num)
            self.previous_pseudo_clicks = copy.deepcopy(self.pseudo_clicker.clicks_list)
            # 阶段一：用户点击
            pred_probs, fp_map_probs, fn_map_probs, execution_time = self.predictor.get_prediction(clicker,
                                                                                              pseudo_clicker=self.pseudo_clicker,
                                                                                              generate_pseu=scribble_enabled,
                                                                                              designated_click_radius=designated_click_radius)
            # 伪涂鸦生成策略
            if scribble_enabled:
                fn_map = (fn_map_probs > 0.49).astype(np.uint8) * 255
                fp_map = (fp_map_probs > 0.49).astype(np.uint8) * 255
                output = (pred_probs > 0.49).astype(np.uint8) * 255
                self.pseudo_clicker.make_next_pseudo_scribbles(fn_map, fp_map, output, scribble_count)
                # 阶段二：模型涂鸦
                pred_probs, fp_map_probs, fn_map_probs, execution_time = self.predictor.get_prediction(clicker,
                                                                                                  pseudo_clicker=self.pseudo_clicker,
                                                                                                  generate_pseu=False,
                                                                                                  designated_click_radius=designated_click_radius)

            pred_probs = cv2.resize(pred_probs, (w, h))
            self.previous_mask = pred_probs
            self.previous_click_num = len(clicker)

app = FastAPI()
app.mount("/static", StaticFiles(directory="demo/static"), name="static")
templates = Jinja2Templates(directory="demo/templates")

save_path = './results'
resume_path = './weights/mg_pse.pth'
device = torch.device(f"cuda:{0}")
segmenter_instance = Segmenter(resume_path, save_path, device)
al_helper = ALHelper(device)

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@app.post("/submit_post_message")
async def post_message(request: Request):
    request_data = await request.json()
    original_image = request_data['image']
    base64_decoded = base64.b64decode(original_image.split(',')[1])
    image = Image.open(io.BytesIO(base64_decoded))

    background_points = request_data['backgroundPoints']
    foreground_points = request_data['foregroundPoints']
    designated_click_radius = request_data['sliderValue']
    scribble_enabled = request_data['pseudoScribbleEnabled']
    scribble_count = request_data['scribbleCount']

    segmenter_instance.segment(image, foreground_points, background_points, designated_click_radius, scribble_enabled, scribble_count)
    
    output_mask = (segmenter_instance.previous_mask > segmenter_instance.pred_thr).astype(np.uint8)
    # 计算形态学参数
    # 计算连通区域
    num_labels, labels = cv2.connectedComponents(output_mask, connectivity=8)
    # 根据分辨率动态阈值（可选）
    H, W = output_mask.shape
    baseline_pixels = 350 * 740
    scale = (H * W) / baseline_pixels
    MIN_AREA = int(200 * scale)

    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area < MIN_AREA:
            labels[labels == label] = 0
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # 去掉背景
    # 收集每个腺体的水平位置（x 坐标最小值，即最左边像素）
    glands = []
    for label in unique_labels:
        ys, xs = np.where(labels == label)
        leftmost_x = xs.min()
        glands.append((label, leftmost_x))

    # 按 x 坐标排序，只考虑水平位置
    glands_sorted = sorted(glands, key=lambda x: x[1])  # 最左边的腺体排在前面

    # 创建新的 labels 数组，按顺序重新编号
    new_labels = np.zeros_like(labels)
    for new_label, (old_label, _) in enumerate(glands_sorted, start=1):
        new_labels[labels == old_label] = new_label

    # 替换原来的 labels
    labels = new_labels
    num_labels = len(glands_sorted) + 1  # 更新 num_labels（包含背景）
    # 腺体总数量
    total_gland_num = int(num_labels - 1)
    # 计算各腺体形态学参数
    gland_lengths = []
    gland_areas = []
    gland_widths = []
    gland_curvatures = []
    for label in range(1, num_labels):  # 1 ~ num_labels-1, 0为背景
        # 获取单个腺体
        gland = (labels == label).astype(np.uint8)
        # 获取中心线
        center_points = GlandUtils.get_center_points(gland)
        # 计算腺体长度
        if len(center_points) > 1:
            length = GlandUtils.calculate_length(center_points)
        else:
            length = 0
        gland_lengths.append(length)
        area = GlandUtils.calculate_area(gland)
        gland_areas.append(area)
        width_list, width, left_points, right_points = GlandUtils.calculate_width(gland, center_points)
        gland_widths.append(width)
        curvature = GlandUtils.calculate_deformation(left_points, right_points, width_list, width)
        gland_curvatures.append(curvature)

    output_mask_PIL = 255 * np.repeat(output_mask[:,:,None], 3, axis=-1)
    output_mask_PIL = Image.fromarray(output_mask_PIL)
    buf = io.BytesIO()
    output_mask_PIL.save(buf, format='JPEG')
    buf.seek(0)
    mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    output_image = draw_with_blend_and_clicks(
        img=segmenter_instance.image, mask=output_mask, clicks_list=None,
        mask_color=(0, 180, 255),
    )
    pseudo_clicks_list = segmenter_instance.pseudo_clicker.clicks_list
    # resized image size
    H_rs = segmenter_instance.TARGET_H  # 350
    W_rs = segmenter_instance.TARGET_W  # 740
    # original image size
    H_org, W_org, _ = segmenter_instance.image.shape
    # convert
    new_clicks_list = resize_click_list_to_original(
        pseudo_clicks_list, H_rs, W_rs, H_org, W_org
    )
    output_image = draw_with_blend_and_clicks(output_image, clicks_list=new_clicks_list,
                                                 pos_color=(255, 150, 220), neg_color=(255,255,0), radius=1)

    output_image = Image.fromarray(output_image)
    buf = io.BytesIO()
    output_image.save(buf, format='JPEG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    response_data = {
        "output_mask": mask_base64,
        "output_image": image_base64,
        "total_gland_num": total_gland_num,
        "gland_lengths": gland_lengths,
        "gland_areas": gland_areas,
        "gland_widths": gland_widths,
        "gland_curvatures": gland_curvatures,
        "labels": labels.tolist()
    }
    return JSONResponse(response_data)


@app.post("/clear_post_message")
async def post_message(request: Request):
    segmenter_instance.reset()

@app.post("/revert")
async def revert():
    # 从 segmenter 拿到 previous_result
    mask = segmenter_instance.revert()

    if mask is None:
        return JSONResponse({
            "output_mask": None,
            "output_image": None,
        })

    # -----------------------------
    # 1) mask → base64
    # -----------------------------
    output_mask = (mask > segmenter_instance.pred_thr).astype(np.uint8)
    output_mask_PIL = 255 * np.repeat(output_mask[:, :, None], 3, axis=-1)
    output_mask_PIL = Image.fromarray(output_mask_PIL.astype(np.uint8))

    buf = io.BytesIO()
    output_mask_PIL.save(buf, format='JPEG')  # JPEG OK 因为是RGB
    buf.seek(0)
    mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # -----------------------------
    # 2) 生成 overlay 图像 (带颜色)
    # -----------------------------
    output_image = draw_with_blend_and_clicks(
        img=segmenter_instance.image,
        mask=output_mask,
        clicks_list=None,
        mask_color=(0, 180, 255),
    )

    # -----------------------------
    # ✅ 3) 可视化 pseudo scribbles
    # -----------------------------
    pseudo_clicks_list = segmenter_instance.pseudo_clicker.clicks_list

    # resized image size
    H_rs = segmenter_instance.TARGET_H
    W_rs = segmenter_instance.TARGET_W

    # original size
    H_org, W_org, _ = segmenter_instance.image.shape

    # click 坐标恢复
    new_clicks_list = resize_click_list_to_original(
        pseudo_clicks_list, H_rs, W_rs, H_org, W_org
    )

    # 叠加 pseudo scribbles
    output_image = draw_with_blend_and_clicks(
        output_image,
        clicks_list=new_clicks_list,
        pos_color=(255, 150, 220),
        neg_color=(255, 255, 0),
        radius=1,
    )

    # -----------------------------
    # 转 base64
    # -----------------------------
    output_image = Image.fromarray(output_image)
    buf = io.BytesIO()
    output_image.save(buf, format='JPEG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "output_mask": mask_base64,
        "output_image": image_base64,
    })


@app.post("/change_color_post_message")
async def post_message(request: Request):
    if segmenter_instance.previous_mask is None:
        return JSONResponse({'output_mask': None, 'output_image': None})
  
    request_data = await request.json()
    color_r = request_data['color_r']
    color_g = request_data['color_g']
    color_b = request_data['color_b']
    color_a = request_data['color_a']
    print(color_r, color_g, color_b, color_a)

    output_mask = (segmenter_instance.previous_mask > segmenter_instance.pred_thr).astype(np.uint8)
    output_mask_PIL = Image.fromarray(output_mask, mode='L')
    buf = io.BytesIO()
    output_mask_PIL.save(buf, format='JPEG')
    buf.seek(0)
    mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    output_image = draw_with_blend_and_clicks(
        img=segmenter_instance.image, mask=output_mask,
        alpha=color_a,
        clicks_list=None,
        mask_color=(color_r, color_g, color_b),
    )
    pseudo_clicks_list = segmenter_instance.pseudo_clicker.clicks_list
    # resized image size
    H_rs = segmenter_instance.TARGET_H  # 350
    W_rs = segmenter_instance.TARGET_W  # 740
    # original image size
    H_org, W_org, _ = segmenter_instance.image.shape
    # convert
    new_clicks_list = resize_click_list_to_original(
        pseudo_clicks_list, H_rs, W_rs, H_org, W_org
    )
    output_image = draw_with_blend_and_clicks(output_image, clicks_list=new_clicks_list,
                                              pos_color=(255, 150, 220), neg_color=(255, 255, 0), radius=1)

    output_image = Image.fromarray(output_image)
    buf = io.BytesIO()
    output_image.save(buf, format='JPEG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = JSONResponse({'output_mask': mask_base64, 'output_image': image_base64})
    return response

@app.get("/")
async def main(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/interactive-segmentation")
async def test_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("interface-cn.html", {"request": request})

@app.get("/active-learning")
async def test_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("al.html", {"request": request})

@app.post("/export_excel")
async def export_excel(request: Request):
    """
    导出中文版本的睑板腺形态分析报告 Excel 文件
    """
    data = await request.json()

    patient = data.get("patient", {})
    glands = data.get("glands", {})
    overall = data.get("overall", {})

    gland_lengths = glands.get("lengths", [])
    gland_widths  = glands.get("widths", [])
    gland_areas   = glands.get("areas", [])
    gland_curv    = glands.get("curvatures", [])

    px_to_mm = float(data.get("px_to_mm", 0))

    wb = Workbook()
    ws = wb.active
    ws.title = "睑板腺分析报告"

    row = 1

    # === 患者信息 ===
    ws["A{}".format(row)] = "患者检查信息"
    row += 1
    patient_field_map = {
        "name": "姓名",
        "sex": "性别",
        "age": "年龄",
        "eye": "眼睑部位",
        "exam_date": "就诊日期",
        "meiboscore": "Meiboscore 评分",
    }

    for key, label in patient_field_map.items():
        value = patient.get(key, "--")
        if key == "sex":
            if value == "male":
                value = "男"
            elif value == "female":
                value = "女"
        elif key == "eye":
            value_map = {
                "left upper eyelid": "左上眼睑",
                "right upper eyelid": "右上眼睑",
                "left lower eyelid": "左下眼睑",
                "right lower eyelid": "右下眼睑"
            }
            value = value_map.get(value, value)
        ws["A{}".format(row)] = label
        ws["B{}".format(row)] = value
        row += 1

    row += 1

    # === 单腺体指标 ===
    ws["A{}".format(row)] = "单个睑板腺指标"
    row += 1
    ws.append(["序号", "长度 (mm)", "宽度 (mm)", "面积 (mm²)", "弯曲度"])

    n = len(gland_lengths)
    for i in range(n):
        length_px = gland_lengths[i] if i < len(gland_lengths) else None
        width_px = gland_widths[i] if i < len(gland_widths) else None
        area_px = gland_areas[i] if i < len(gland_areas) else None
        curv = gland_curv[i] if i < len(gland_curv) else None

        length_mm = (length_px * px_to_mm) if length_px is not None else None
        width_mm = (width_px * px_to_mm) if width_px is not None else None
        area_mm2 = (area_px * (px_to_mm ** 2)) if area_px is not None else None

        ws.append([
            i + 1,
            round(length_mm, 3) if length_mm is not None else None,
            round(width_mm, 3) if width_mm is not None else None,
            round(area_mm2, 3) if area_mm2 is not None else None,
            round(curv, 3) if curv is not None else None,
        ])

    row = ws.max_row + 2

    # === 总体指标 ===
    ws["A{}".format(row)] = "总体睑板腺指标"
    row += 1

    overall_field_map = {
        "number": "睑板腺数量",
        "avg_length": "睑板腺平均长度 (mm)",
        "avg_width": "睑板腺平均宽度 (mm)",
        "avg_area": "睑板腺平均面积 (mm²)",
        "avg_curvature": "睑板腺平均弯曲度",
    }

    for key, label in overall_field_map.items():
        value = overall.get(key, "--")
        clean_value = None if value == "--" else value
        ws["A{}".format(row)] = label
        ws["B{}".format(row)] = clean_value
        row += 1

    # === 导出 Excel ===
    output = BytesIO()
    wb.save(output)
    output.seek(0)

    file_name = "患者睑板腺形态学参数分析报告.xlsx"
    encoded_filename = quote(file_name)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            # 注意这里！filename* 使用 UTF-8 编码，可以正确显示中文
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


@app.post("/upload_unlabeled_data")
async def upload_unlabeled(files: list[UploadFile] = File(...)):
    al_helper.clear_unlabeled()

    for f in files:
        name = f.filename
        content = await f.read()  # bytes
        img = al_helper.load_image(content)
        al_helper.unlabeled_names.append(name)
        al_helper.unlabeled_images.append(img)

    return {"msg": "Unlabeled data uploaded", "count": len(al_helper.unlabeled_names)}


@app.post("/upload_labeled_data")
async def upload_labeled(files: list[UploadFile] = File(...)):
    al_helper.clear_labeled()

    for f in files:
        name = f.filename
        content = await f.read()  # bytes
        img = al_helper.load_image(content)
        al_helper.labeled_names.append(name)
        al_helper.labeled_images.append(img)

    return {"msg": "Labeled data uploaded", "count": len(al_helper.labeled_names)}


@app.post("/upload_model")
async def upload_model(model_file: UploadFile = File(...)):
    try:
        # 读取上传文件的字节
        ckpt_bytes = await model_file.read()
        # 直接从字节加载模型
        al_helper.load_model(ckpt_bytes)
        return {"msg": f"加载模型成功"}
    except Exception as e:
        return {"msg": f"加载模型失败: {e}"}

@app.post("/run_active_learning")
async def run_active_learning(request: Request):
    data = await request.json()
    strategy = data.get("strategy")
    al_helper.budget = data.get("budget")
    selected_names = []
    if strategy == 'Entropy':
        selected_names = al_helper.entropy_sample()
    elif strategy == 'Dropout':
        selected_names = al_helper.dropout_sample()
    elif strategy == 'TTA':
        selected_names = al_helper.tta_sample()
    elif strategy == 'Core-set':
        selected_names = al_helper.coreset_sample()
    elif strategy == 'SDS':
        selected_names = al_helper.sds_sample()
    elif strategy == 'PHS':
        selected_names = al_helper.phs_sample()

    return {"selected_names": selected_names}

def resize_click_list_to_original(pseudo_clicks_list, H_rs, W_rs, H_org, W_org):
    """
    将 Click 坐标从resize空间映射回原图尺寸
    """
    new_clicks = []

    scale_y = H_org / H_rs
    scale_x = W_org / W_rs

    for ck in pseudo_clicks_list:
        y_rs, x_rs = ck.coords

        # Scale back
        y_org = y_rs * scale_y
        x_org = x_rs * scale_x

        # 生成新的 Click
        new_ck = ck.copy(coords=(y_org, x_org))
        new_clicks.append(new_ck)

    return new_clicks

