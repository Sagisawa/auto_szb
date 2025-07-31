import json
import os

"""
卡牌优先级配置
定义各种卡牌的使用优先级
"""

# 默认高优先级卡牌（在可用费用内优先使用）
DEFAULT_HIGH_PRIORITY_CARDS = {
    "蛇神之怒": {"priority": 3},
    "无极猎人阿拉加维": {"priority": 2},
    "命运黄昏奥丁": {"priority": 1},
    "鸣咽的圣骑士维尔伯特": {"priority": 2},
    "纯白圣女贞德": {"priority": 2},
    "夭之守护神埃忒耳": {"priority": 3},
    "怨灵": {"priority": 3},
    "水之守护神萨蕾法": {"priority": 3}
}

# 默认进化优先卡牌（进化/超进化时优先考虑）
DEFAULT_EVOLVE_PRIORITY_CARDS = {
    "鸣咽的圣骑士维尔伯特": {"priority": 2},
    "夭之守护神埃忒耳": {"priority": 1},
    "无极猎人阿拉加维": {"priority": 3},
    "水之守护神萨蕾法": {"priority": 3}
}

# 手牌出牌时特殊处理卡牌（需要特殊操作逻辑）
SPECIAL_CARDS = {
    "蛇神之怒": {
        "target_type": "enemy_player"  # 目标类型：敌方玩家
    },
    "命运黄昏奥丁": {
        "target_type": "shield_or_highest_hp"  # 目标类型：护盾或最高血量
    }
}

# 进化/超进化后需要特殊操作的卡牌
EVOLVE_SPECIAL_ACTIONS = {
    "铁拳神父": {
        "action": "attack_enemy_follower_hp_less_than_4",
        "super_evolution_action": "attack_two_enemy_followers_hp_less_than_4"
    }
}

def load_user_config():
    """加载用户配置文件，支持PyInstaller打包后的路径"""
    import sys
    
    # 尝试多种路径来找到config.json
    possible_paths = []
    
    # 1. 相对于当前文件的路径（开发环境）
    current_dir = os.path.dirname(__file__)
    possible_paths.append(os.path.join(current_dir, '../../config.json'))
    
    # 2. 相对于可执行文件的路径（PyInstaller打包后）
    if getattr(sys, 'frozen', False):
        # PyInstaller打包后的情况
        exe_dir = os.path.dirname(sys.executable)
        possible_paths.append(os.path.join(exe_dir, 'config.json'))
    
    # 3. 相对于工作目录的路径
    possible_paths.append('config.json')
    
    # 4. 相对于脚本运行目录的路径
    script_dir = os.getcwd()
    possible_paths.append(os.path.join(script_dir, 'config.json'))
    
    for config_path in possible_paths:
        config_path = os.path.abspath(config_path)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"成功加载配置文件: {config_path}")
                    return config
            except Exception as e:
                print(f"加载配置文件失败 {config_path}: {e}")
                continue
    
    print("警告: 未找到config.json文件，使用默认配置")
    return {}

# 全局变量，用于缓存配置
_user_config = None
_HIGH_PRIORITY_CARDS = None
_EVOLVE_PRIORITY_CARDS = None

def reload_config():
    """重新加载配置文件"""
    global _user_config, _HIGH_PRIORITY_CARDS, _EVOLVE_PRIORITY_CARDS
    _user_config = load_user_config()
    _HIGH_PRIORITY_CARDS = _user_config.get('high_priority_cards', DEFAULT_HIGH_PRIORITY_CARDS)
    _EVOLVE_PRIORITY_CARDS = _user_config.get('evolve_priority_cards', DEFAULT_EVOLVE_PRIORITY_CARDS)
    print(f"重新加载配置完成，高优先级卡牌: {list(_HIGH_PRIORITY_CARDS.keys())}")
    print(f"重新加载配置完成，进化优先级卡牌: {list(_EVOLVE_PRIORITY_CARDS.keys())}")

# 初始化配置
reload_config()

def get_high_priority_cards():
    """获取高优先级卡牌列表"""
    return _HIGH_PRIORITY_CARDS

def get_special_cards():
    """获取特殊处理卡牌列表"""
    return SPECIAL_CARDS

def is_high_priority_card(card_name):
    """检查是否为高优先级卡牌"""
    return card_name in _HIGH_PRIORITY_CARDS

def is_special_card(card_name):
    """检查是否为特殊处理卡牌"""
    return card_name in SPECIAL_CARDS

def get_card_priority(card_name):
    """获取卡牌优先级（数字越小优先级越高）"""
    if card_name in _HIGH_PRIORITY_CARDS:
        return _HIGH_PRIORITY_CARDS[card_name].get("priority", 999)
    return 999  # 默认低优先级

def get_card_info(card_name):
    """获取卡牌信息"""
    if card_name in _HIGH_PRIORITY_CARDS:
        return _HIGH_PRIORITY_CARDS[card_name]
    elif card_name in SPECIAL_CARDS:
        return SPECIAL_CARDS[card_name]
    return None 

def get_evolve_priority_cards():
    """获取进化优先卡牌列表"""
    return _EVOLVE_PRIORITY_CARDS

def is_evolve_priority_card(card_name):
    """检查是否为进化优先卡牌"""
    return card_name in _EVOLVE_PRIORITY_CARDS 

def get_evolve_special_actions():
    """获取进化/超进化特殊操作卡牌列表"""
    return EVOLVE_SPECIAL_ACTIONS

def is_evolve_special_action_card(card_name):
    """检查是否为进化/超进化特殊操作卡牌"""
    return card_name in EVOLVE_SPECIAL_ACTIONS 