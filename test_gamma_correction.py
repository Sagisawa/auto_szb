#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import adbutils
import matplotlib.pyplot as plt
from skimage import exposure
import os

def analyze_image_gamma(image, method_name: str):
    """分析图像的 gamma 特性"""
    print(f"\n=== {method_name} Gamma 分析 ===")
    
    # 转换为 numpy 数组
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # 确保是 RGB 格式
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # 计算每个通道的直方图
        hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256])
        
        # 计算平均亮度
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        # 计算对比度（标准差）
        contrast = np.std(gray)
        
        # 计算 gamma 值（通过拟合幂律分布）
        # 使用累积分布函数来估计 gamma
        hist_normalized = hist_r.flatten() / np.sum(hist_r)
        cumulative = np.cumsum(hist_normalized)
        
        # 简单的 gamma 估计（通过亮度分布）
        # 如果图像偏暗，可能 gamma > 1；如果偏亮，可能 gamma < 1
        if mean_brightness < 100:
            estimated_gamma = 1.5  # 偏暗，可能需要 gamma 校正
        elif mean_brightness > 150:
            estimated_gamma = 0.7  # 偏亮，可能需要 gamma 校正
        else:
            estimated_gamma = 1.0  # 正常亮度
        
        print(f"  平均亮度: {mean_brightness:.2f}")
        print(f"  对比度: {contrast:.2f}")
        print(f"  估计 Gamma: {estimated_gamma:.2f}")
        print(f"  图像尺寸: {img_array.shape}")
        
        return {
            'mean_brightness': mean_brightness,
            'contrast': contrast,
            'estimated_gamma': estimated_gamma,
            'histogram': hist_r.flatten(),
            'image_array': img_array
        }
    
    return None

def apply_gamma_correction(image, gamma=2.2):
    """应用 gamma 校正"""
    # 将图像转换为 0-1 范围
    if isinstance(image, Image.Image):
        img_array = np.array(image).astype(np.float32) / 255.0
    else:
        img_array = image.astype(np.float32) / 255.0
    
    # 应用 gamma 校正
    corrected = np.power(img_array, 1.0 / gamma)
    
    # 转换回 0-255 范围
    corrected = (corrected * 255).astype(np.uint8)
    
    return corrected

def save_comparison_images(fb_image, sc_image, fb_corrected, sc_corrected, output_dir="gamma_test"):
    """保存对比图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始图像
    if isinstance(fb_image, Image.Image):
        fb_image.save(f"{output_dir}/framebuffer_original.png")
    else:
        Image.fromarray(fb_image).save(f"{output_dir}/framebuffer_original.png")
    
    if isinstance(sc_image, Image.Image):
        sc_image.save(f"{output_dir}/screenshot_original.png")
    else:
        Image.fromarray(sc_image).save(f"{output_dir}/screenshot_original.png")
    
    # 保存校正后的图像
    Image.fromarray(fb_corrected).save(f"{output_dir}/framebuffer_gamma_corrected.png")
    Image.fromarray(sc_corrected).save(f"{output_dir}/screenshot_gamma_corrected.png")
    
    print(f"\n图像已保存到 {output_dir}/ 目录")

def plot_histograms(fb_data, sc_data):
    """绘制直方图对比"""
    plt.figure(figsize=(15, 10))
    
    # 原始图像直方图
    plt.subplot(2, 2, 1)
    plt.plot(fb_data['histogram'], 'r-', alpha=0.7, label='Framebuffer')
    plt.title('Framebuffer 原始直方图')
    plt.xlabel('像素值')
    plt.ylabel('频率')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(sc_data['histogram'], 'b-', alpha=0.7, label='Screenshot')
    plt.title('Screenshot 原始直方图')
    plt.xlabel('像素值')
    plt.ylabel('频率')
    plt.legend()
    
    # 亮度分布对比
    plt.subplot(2, 2, 3)
    plt.bar(['Framebuffer', 'Screenshot'], 
            [fb_data['mean_brightness'], sc_data['mean_brightness']],
            color=['red', 'blue'], alpha=0.7)
    plt.title('平均亮度对比')
    plt.ylabel('亮度值')
    
    plt.subplot(2, 2, 4)
    plt.bar(['Framebuffer', 'Screenshot'], 
            [fb_data['contrast'], sc_data['contrast']],
            color=['red', 'blue'], alpha=0.7)
    plt.title('对比度对比')
    plt.ylabel('标准差')
    
    plt.tight_layout()
    plt.savefig('gamma_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("直方图对比已保存为 gamma_analysis.png")

def main():
    """主测试函数"""
    print("=== Gamma 校正分析测试 ===\n")
    
    # 连接设备
    try:
        adb = adbutils.AdbClient()
        devices = adb.device_list()
        
        if not devices:
            print("未找到连接的设备")
            return
        
        device = devices[0]
        print(f"使用设备: {device.serial}")
        
        # 获取 framebuffer 截图
        print("\n获取 framebuffer 截图...")
        fb_image = device.framebuffer()
        if fb_image is None:
            print("framebuffer 截图失败")
            return
        
        # 获取 screenshot 截图
        print("获取 screenshot 截图...")
        sc_image = device.screenshot()
        if sc_image is None:
            print("screenshot 截图失败")
            return
        
        # 分析图像
        fb_data = analyze_image_gamma(fb_image, "Framebuffer")
        sc_data = analyze_image_gamma(sc_image, "Screenshot")
        
        if fb_data and sc_data:
            # 应用 gamma 校正
            print("\n=== 应用 Gamma 校正 ===")
            fb_corrected = apply_gamma_correction(fb_image, gamma=2.2)
            sc_corrected = apply_gamma_correction(sc_image, gamma=2.2)
            
            # 分析校正后的图像
            fb_corrected_data = analyze_image_gamma(fb_corrected, "Framebuffer (Gamma 校正后)")
            sc_corrected_data = analyze_image_gamma(sc_corrected, "Screenshot (Gamma 校正后)")
            
            # 保存对比图像
            save_comparison_images(fb_image, sc_image, fb_corrected, sc_corrected)
            
            # 绘制直方图对比
            plot_histograms(fb_data, sc_data)
            
            # 总结
            print("\n=== 分析总结 ===")
            print(f"Framebuffer 原始亮度: {fb_data['mean_brightness']:.2f}")
            print(f"Screenshot 原始亮度: {sc_data['mean_brightness']:.2f}")
            print(f"亮度差异: {abs(fb_data['mean_brightness'] - sc_data['mean_brightness']):.2f}")
            
            if fb_data['mean_brightness'] < sc_data['mean_brightness']:
                print("结论: Framebuffer 图像偏暗，可能需要 gamma 校正")
            elif fb_data['mean_brightness'] > sc_data['mean_brightness']:
                print("结论: Framebuffer 图像偏亮，可能需要 gamma 校正")
            else:
                print("结论: 两种方法亮度相近")
                
    except Exception as e:
        print(f"测试过程中出错: {e}")

if __name__ == "__main__":
    main() 