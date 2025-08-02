#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截图性能测试脚本
比较普通截图和framebuffer截图的性能差异
"""

import time
import statistics
from typing import List, Tuple
import adbutils
from PIL import Image

def test_framebuffer_performance(device, duration: int = 10) -> Tuple[List[float], float]:
    """测试 framebuffer() 方法的性能"""
    print(f"测试 framebuffer() 方法，持续 {duration} 秒...")
    
    start_time = time.time()
    frame_times = []
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            # 获取截图
            screenshot = device.framebuffer()
            if screenshot is not None:
                frame_count += 1
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
            
            # 短暂休息，避免过度占用资源
            time.sleep(0.01)
            
    except Exception as e:
        print(f"framebuffer() 测试出错: {e}")
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    return frame_times, avg_fps

def test_screenshot_performance(device, duration: int = 10) -> Tuple[List[float], float]:
    """测试 screenshot() 方法的性能"""
    print(f"测试 screenshot() 方法，持续 {duration} 秒...")
    
    start_time = time.time()
    frame_times = []
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            # 获取截图
            screenshot = device.screenshot()
            if screenshot is not None:
                frame_count += 1
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
            
            # 短暂休息，避免过度占用资源
            time.sleep(0.01)
            
    except Exception as e:
        print(f"screenshot() 测试出错: {e}")
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    return frame_times, avg_fps

def analyze_performance(frame_times: List[float], method_name: str):
    """分析性能数据"""
    if not frame_times:
        print(f"{method_name}: 无有效数据")
        return
    
    avg_time = statistics.mean(frame_times)
    min_time = min(frame_times)
    max_time = max(frame_times)
    std_dev = statistics.stdev(frame_times) if len(frame_times) > 1 else 0
    
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    max_fps = 1.0 / min_time if min_time > 0 else 0
    
    print(f"\n{method_name} 性能分析:")
    print(f"  平均帧时间: {avg_time:.4f} 秒")
    print(f"  最小帧时间: {min_time:.4f} 秒")
    print(f"  最大帧时间: {max_time:.4f} 秒")
    print(f"  标准差: {std_dev:.4f} 秒")
    print(f"  平均帧率: {avg_fps:.2f} FPS")
    print(f"  最大帧率: {max_fps:.2f} FPS")
    print(f"  总帧数: {len(frame_times)}")

def main():
    """主测试函数"""
    print("=== 截图方法性能对比测试 ===\n")
    
    # 连接设备
    try:
        adb = adbutils.AdbClient()
        devices = adb.device_list()
        
        if not devices:
            print("未找到连接的设备")
            return
        
        device = devices[0]
        print(f"使用设备: {device.serial}")
        
        # 测试参数
        test_duration = 10  # 每种方法测试10秒
        
        # 测试 framebuffer 方法
        print("\n" + "="*50)
        fb_times, fb_fps = test_framebuffer_performance(device, test_duration)
        analyze_performance(fb_times, "framebuffer()")
        
        # 测试 screenshot 方法
        print("\n" + "="*50)
        sc_times, sc_fps = test_screenshot_performance(device, test_duration)
        analyze_performance(sc_times, "screenshot()")
        
        # 对比结果
        print("\n" + "="*50)
        print("性能对比总结:")
        if fb_fps > 0 and sc_fps > 0:
            ratio = fb_fps / sc_fps
            print(f"framebuffer() 帧率是 screenshot() 的 {ratio:.2f} 倍")
            
            if ratio > 1.5:
                print("结论: framebuffer() 帧率显著高于 screenshot()")
            elif ratio > 1.1:
                print("结论: framebuffer() 帧率略高于 screenshot()")
            elif ratio < 0.9:
                print("结论: screenshot() 帧率高于 framebuffer()")
            else:
                print("结论: 两种方法帧率相近")
        else:
            print("无法进行有效对比")
            
    except Exception as e:
        print(f"测试过程中出错: {e}")

if __name__ == "__main__":
    main() 