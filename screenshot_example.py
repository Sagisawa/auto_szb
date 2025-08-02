#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截图功能使用示例
演示如何使用不同的截图方式
"""

import time
import json
import logging
from src.device.device_state import DeviceState
from adbutils import adb

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_screenshot_methods(serial: str):
    """演示不同的截图方法"""
    
    # 加载配置
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建设备状态
    device_state = DeviceState(serial, config)
    
    # 连接设备
    try:
        adb_device = adb.device(serial)
        if adb_device is None:
            logger.error(f"无法连接设备: {serial}")
            return
        
        device_state.adb_device = adb_device
        logger.info(f"已连接设备: {serial}")
        
    except Exception as e:
        logger.error(f"连接设备失败: {str(e)}")
        return
    
    logger.info("="*60)
    logger.info("截图功能演示")
    logger.info("="*60)
    
    # 1. framebuffer截图
    logger.info("\n1. framebuffer截图")
    start_time = time.time()
    screenshot1 = device_state.take_screenshot()
    end_time = time.time()
    if screenshot1:
        logger.info(f"✓ 成功，耗时: {(end_time-start_time)*1000:.2f}ms")
        logger.info(f"  图像尺寸: {screenshot1.size}")
    else:
        logger.error("✗ 失败")
    
    # 2. 重复测试
    logger.info("\n2. 重复测试")
    start_time = time.time()
    screenshot2 = device_state.take_screenshot()
    end_time = time.time()
    if screenshot2:
        logger.info(f"✓ 成功，耗时: {(end_time-start_time)*1000:.2f}ms")
    else:
        logger.error("✗ 失败")
    
    # 3. 再次测试
    logger.info("\n3. 再次测试")
    start_time = time.time()
    screenshot3 = device_state.take_screenshot()
    end_time = time.time()
    if screenshot3:
        logger.info(f"✓ 成功，耗时: {(end_time-start_time)*1000:.2f}ms")
    else:
        logger.error("✗ 失败")
    
    logger.info("\n" + "="*60)
    logger.info("演示完成")
    logger.info("="*60)

def performance_comparison(serial: str, test_count: int = 5):
    """性能对比测试"""
    
    # 加载配置
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建设备状态
    device_state = DeviceState(serial, config)
    
    # 连接设备
    try:
        adb_device = adb.device(serial)
        if adb_device is None:
            logger.error(f"无法连接设备: {serial}")
            return
        
        device_state.adb_device = adb_device
        
    except Exception as e:
        logger.error(f"连接设备失败: {str(e)}")
        return
    
    logger.info("\n" + "="*60)
    logger.info("性能对比测试")
    logger.info("="*60)
    
    # 测试普通截图
    logger.info(f"\n测试普通截图 ({test_count}次)...")
    normal_times = []
    for i in range(test_count):
        start_time = time.time()
        screenshot = device_state.take_screenshot(use_framebuffer=False)
        end_time = time.time()
        
        if screenshot is not None:
            normal_times.append(end_time - start_time)
            logger.info(f"  第{i+1}次: {(end_time - start_time)*1000:.2f}ms")
        else:
            logger.error(f"  第{i+1}次: 失败")
    
    # 测试framebuffer截图
    logger.info(f"\n测试framebuffer截图 ({test_count}次)...")
    framebuffer_times = []
    for i in range(test_count):
        start_time = time.time()
        screenshot = device_state.take_screenshot()
        end_time = time.time()
        
        if screenshot is not None:
            framebuffer_times.append(end_time - start_time)
            logger.info(f"  第{i+1}次: {(end_time - start_time)*1000:.2f}ms")
        else:
            logger.error(f"  第{i+1}次: 失败")
    
    # 输出统计结果
    logger.info("\n" + "-"*40)
    logger.info("性能统计结果:")
    logger.info("-"*40)
    
    if framebuffer_times:
        avg_framebuffer = sum(framebuffer_times) / len(framebuffer_times)
        logger.info(f"framebuffer截图平均时间: {avg_framebuffer*1000:.2f}ms")
    else:
        logger.error("所有截图测试都失败了")

if __name__ == "__main__":
    # 从配置文件读取设备信息
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        devices = config.get("devices", [])
        if not devices:
            logger.error("配置文件中未找到设备信息")
            exit(1)
        
        # 使用第一个设备
        device_config = devices[0]
        serial = device_config.get("serial")
        if not serial:
            logger.error("设备配置缺少serial字段")
            exit(1)
        
        logger.info(f"使用设备: {serial}")
        
        # 运行演示
        demo_screenshot_methods(serial)
        
        # 运行性能测试
        performance_comparison(serial, test_count=3)
        
    except Exception as e:
        logger.error(f"运行过程中出错: {str(e)}")
        exit(1) 