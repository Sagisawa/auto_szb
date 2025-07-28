"""
费用识别模块
实现卡牌费用数字的识别功能
"""

import cv2
import numpy as np
import glob
import os
import logging
from typing import Tuple, Optional
from src.utils.gpu_utils import get_easyocr_reader

logger = logging.getLogger(__name__)


class CostRecognition:
    """费用识别类"""
    
    def __init__(self, constants_manager=None):
        self.reader = get_easyocr_reader()
        # 费用数字模板缓存
        self.cost_templates = {}
        
        # 从常量管理器获取参数
        if constants_manager:
            self.constants = constants_manager
            self.edge_thresholds = constants_manager.get_edge_thresholds()
            self.pyramid_levels = constants_manager.pyramid_levels
            self.cost_digit_height, self.cost_digit_width = constants_manager.get_cost_digit_size()
            self.cost_min, self.cost_max = constants_manager.get_cost_range()
            self.cost_confidence_threshold = constants_manager.cost_confidence_threshold
            self.default_hp_value = constants_manager.default_hp_value
            self.angle_range = constants_manager.angle_range
            self.angle_steps = constants_manager.get_angle_steps()
            self.template_path = constants_manager.get_template_path("cost_numbers")
        else:
            # 使用默认值
            from src.config.game_constants import (
                EDGE_THRESHOLDS, PYRAMID_LEVELS, COST_DIGIT_HEIGHT, COST_DIGIT_WIDTH,
                COST_MIN, COST_MAX, COST_CONFIDENCE_THRESHOLD,
                DEFAULT_HP_VALUE, ANGLE_RANGE, ANGLE_STEPS, TEMPLATE_PATHS
            )
            self.edge_thresholds = EDGE_THRESHOLDS
            self.pyramid_levels = PYRAMID_LEVELS
            self.cost_digit_height, self.cost_digit_width = COST_DIGIT_HEIGHT, COST_DIGIT_WIDTH
            self.cost_min, self.cost_max = COST_MIN, COST_MAX
            self.cost_confidence_threshold = COST_CONFIDENCE_THRESHOLD
            self.default_hp_value = DEFAULT_HP_VALUE
            self.angle_range = ANGLE_RANGE
            self.angle_steps = ANGLE_STEPS
            self.template_path = TEMPLATE_PATHS["cost_numbers"]
    
    def recognize_cost_number(self, card_roi, device_state=None):
        """使用多级旋转匹配识别手牌卡费数字（多模板支持）"""
        # 数字模板路径
        template_dir = "templates/cost_numbers"

        # 截取数字区域（左上角）
        digit_roi = card_roi[0:27, 0:20]  # 高27，宽20
        
        # 预处理目标区域
        gray_digit = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
        
        # 创建图像金字塔
        pyramid = [gray_digit]
        for _ in range(self.pyramid_levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        
        # 在金字塔底层提取边缘特征
        base_level = pyramid[-1]
        edged_target = cv2.Canny(base_level, self.edge_thresholds[0], self.edge_thresholds[1])
        
        best_cost = 0
        best_confidence = 0
        
        # 遍历所有数字模板（多模板支持）
        for cost in range(0, 11):
            # 加载所有该数字的模板
            template_paths = glob.glob(os.path.join(template_dir, f"{cost}_*.png"))
            if not template_paths:
                continue
            for template_path in template_paths:
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is None:
                    continue
                # 创建模板金字塔
                template_pyramid = [template]
                for _ in range(self.pyramid_levels - 1):
                    template_pyramid.append(cv2.pyrDown(template_pyramid[-1]))
                # 在底层提取边缘
                edged_template = cv2.Canny(template_pyramid[-1], self.edge_thresholds[0], self.edge_thresholds[1])
                # 执行多角度匹配
                confidence, angle = self._multi_angle_digit_match(edged_target, edged_template)
                # 更新最佳匹配
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_cost = cost
        
        # 如果模板匹配置信度低于阈值，尝试使用OCR
        if best_confidence < self.cost_confidence_threshold:
            ocr_cost, ocr_confidence = self.recognize_cost_with_easyocr(card_roi, device_state)
            # 比较两种方法的置信度，选择置信度更高的结果
            if ocr_cost > 0 and ocr_confidence > 0:
                if ocr_confidence > best_confidence:
                    return ocr_cost, ocr_confidence
                else:
                    return best_cost, best_confidence
            else:
                return best_cost, best_confidence
        else:
            # 模板匹配成功
            if device_state and device_state.logger:
                device_state.logger.debug(f"模板匹配成功: {best_cost}费, 置信度: {best_confidence:.2f}")
        return best_cost, best_confidence

    def recognize_cost_with_easyocr(self, card_roi, device_state=None):
        """使用EasyOCR识别费用数字"""
        try:
            # 截取数字区域（左上角）
            digit_roi = card_roi[0:27, 0:20]  # 高27，宽20
            
            # 使用EasyOCR识别数字，获取详细信息包括置信度
            if self.reader is not None:
                results = self.reader.readtext(digit_roi, allowlist='0123456789', detail=1)
            else:
                results = []
            
            # 处理OCR结果
            if results and isinstance(results, list):
                # 找到置信度最高的结果
                best_result = None
                best_confidence = 0.0
                
                for result in results:
                    # result格式: (bbox, text, confidence)
                    if len(result) >= 3:
                        text = result[1]
                        confidence = result[2]
                        
                        # 检查文本是否为有效数字
                        if text.isdigit():
                            # 更新最佳结果
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_result = text
                
                if best_result:
                    # 尝试转换为整数
                    try:
                        cost = int(best_result)
                        # 检查费用是否在合理范围内（0-10）
                        if 0 <= cost <= 10:
                            if device_state and device_state.logger:
                            ##    device_state.logger.info(f"EasyOCR识别成功: {cost}费, 置信度: {best_confidence:.2f}, 原始文本: {best_result}")
                                return cost, best_confidence
                        else:
                            if device_state and device_state.logger:
                                device_state.logger.info(f"EasyOCR识别费用超出范围: {cost}费, 置信度: {best_confidence:.2f}")
                            return 0, 0.0
                    except ValueError:
                        if device_state and device_state.logger:
                            device_state.logger.info(f"EasyOCR识别结果无法转换为数字: {best_result}, 置信度: {best_confidence:.2f}")
                        return 0, 0.0
                else:
                    if device_state and device_state.logger:
                     return 0, 0.0
            else:
                if device_state and device_state.logger:
                 return 0, 0.0
                
        except Exception as e:
            if device_state and device_state.logger:
                device_state.logger.error(f"EasyOCR识别出错: {str(e)}")
            return 0, 0.0

    def _rotate_image_with_mask(self, image, angle, border_value=0):
        """
        旋转图像并生成有效区域掩码
        :param image: 单通道图像
        :param angle: 旋转角度(度)
        :return: (旋转后图像, 有效区域掩码)
        """
        h, w = image.shape
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新边界尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        
        # 调整旋转矩阵中心偏移
        M[0, 2] += (nW - w) / 2
        M[1, 2] += (nH - h) / 2
        
        # 执行旋转
        rotated = cv2.warpAffine(image, M, (nW, nH),
                                borderValue=border_value)
        
        # 生成有效区域掩码（排除黑边）
        mask = np.ones_like(image, dtype=np.uint8) * 255
        mask = cv2.warpAffine(mask, M, (nW, nH),
                             borderValue=0)
        
        return rotated, mask

    def _multi_angle_digit_match(self, target_img, template_img, angle_range=30):
        """
        在目标图像中匹配旋转的数字模板
        :param target_img: 目标图像（单通道，边缘图）
        :param template_img: 模板图像（单通道，边缘图）
        :param angle_range: 角度范围（±度）
        :return: (最佳匹配置信度, 最佳匹配角度)
        """
        best_angle = 0
        best_val = -1
        
        # 三级角度扫描：粗、中、细
        angle_steps = [5.0, 1.0, 0.2]
        current_angle_range = angle_range
        
        for step in angle_steps:
            angles = np.arange(-current_angle_range, current_angle_range + step, step)
            for angle in angles:
                # 旋转模板并生成掩码
                rotated_template, mask = self._rotate_image_with_mask(template_img, angle)
                
                # 如果旋转后的模板比目标图像大，则跳过
                if rotated_template.shape[0] > target_img.shape[0] or rotated_template.shape[1] > target_img.shape[1]:
                    continue
                    
                # 执行模板匹配
                res = cv2.matchTemplate(target_img, rotated_template, cv2.TM_CCOEFF_NORMED, mask=mask)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > best_val:
                    best_val = max_val
                    best_angle = angle
            
            # 缩小搜索范围到当前最佳角度附近
            current_angle_range = step * 2
        
        return best_val, best_angle 