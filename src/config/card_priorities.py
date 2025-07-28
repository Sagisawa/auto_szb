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

# 特殊处理卡牌（需要特殊操作逻辑）
SPECIAL_CARDS = {
    "蛇神之怒": {
        "target_type": "enemy_player"  # 目标类型：敌方玩家
    },
    "命运黄昏奥丁": {
        "target_type": "shield_or_highest_hp"  # 目标类型：护盾或最高血量
    }
}

def load_user_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
    config_path = os.path.abspath(config_path)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

user_config = load_user_config()

HIGH_PRIORITY_CARDS = user_config.get('high_priority_cards', DEFAULT_HIGH_PRIORITY_CARDS)
EVOLVE_PRIORITY_CARDS = user_config.get('evolve_priority_cards', DEFAULT_EVOLVE_PRIORITY_CARDS)

def get_high_priority_cards():
    """获取高优先级卡牌列表"""
    return HIGH_PRIORITY_CARDS

def get_special_cards():
    """获取特殊处理卡牌列表"""
    return SPECIAL_CARDS

def is_high_priority_card(card_name):
    """检查是否为高优先级卡牌"""
    return card_name in HIGH_PRIORITY_CARDS

def is_special_card(card_name):
    """检查是否为特殊处理卡牌"""
    return card_name in SPECIAL_CARDS

def get_card_priority(card_name):
    """获取卡牌优先级（数字越小优先级越高）"""
    if card_name in HIGH_PRIORITY_CARDS:
        return HIGH_PRIORITY_CARDS[card_name].get("priority", 999)
    return 999  # 默认低优先级

def get_card_info(card_name):
    """获取卡牌信息"""
    if card_name in HIGH_PRIORITY_CARDS:
        return HIGH_PRIORITY_CARDS[card_name]
    elif card_name in SPECIAL_CARDS:
        return SPECIAL_CARDS[card_name]
    return None 

def get_evolve_priority_cards():
    """获取进化优先卡牌列表"""
    return EVOLVE_PRIORITY_CARDS

def is_evolve_priority_card(card_name):
    """检查是否为进化优先卡牌"""
    return card_name in EVOLVE_PRIORITY_CARDS 