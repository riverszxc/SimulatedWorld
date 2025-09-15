import random
import json
import os
import logging
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
# 日志文件写入 log 子文件夹
log_dir = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(log_dir, exist_ok=True)
# 日志文件名带时间戳
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(log_dir, f'world_{now}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler()
    ]
)

# 加载Qwen3-1.7B模型和分词器（全局只加载一次）
LLM_MODEL_PATH = "/Users/zxc/project/data/model/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 物品类
class Item:
    def __init__(self, name, price=10, effect=None):
        self.name = name
        self.price = price
        self.effect = effect or {}

# 事件类
class Event:
    def __init__(self, name, effect_func, probability=0.1):
        self.name = name
        self.effect_func = effect_func
        self.probability = probability
    def trigger(self, character, world):
        self.effect_func(character, world)

# 角色类
class Character:
    def __init__(self, name, config=None):
        config = config or {}
        self.name = name
        self.age = config.get('age', random.randint(20, 40))
        self.health = config.get('health', random.randint(60, 100))
        self.wealth = config.get('wealth', random.randint(50, 150))
        self.survival_skill = config.get('survival_skill', random.randint(1, 10))
        self.social_skill = config.get('social_skill', random.randint(1, 10))
        self.work_skill = config.get('work_skill', random.randint(1, 10))
        self.experience = config.get('experience', 0)
        self.mood = config.get('mood', random.randint(40, 100))
        self.hunger = config.get('hunger', random.randint(0, 30))
        self.items = config.get('items', {})
        # 关系结构：{name: {'level': int, 'type': '亲密'|'合作'|'敌对'}}
        self.relationships = config.get('relationships', {})
        self.location = config.get('location', random.choice(World.LOCATIONS))
        self.alive = True
        self.personality = config.get('personality', random.choice(['外向', '内向', '冒险', '保守', '慷慨', '吝啬']))
        self.preference = config.get('preference', random.choice(['娱乐', '学习', '购物', '社交', '投资']))
        self.reputation = config.get('reputation', 0)
        self.family = config.get('family', None)
        self.social_circle = config.get('social_circle', set())

    def _update_relationship(self, other, delta):
        rel = self.relationships.get(other.name, {'level': 0, 'type': '合作'})
        if isinstance(rel, int):
            rel = {'level': rel, 'type': '合作'}
        rel['level'] += delta
        # 关系类型判定
        if rel['level'] >= 10:
            rel['type'] = '亲密'
        elif rel['level'] <= -5:
            rel['type'] = '敌对'
        else:
            rel['type'] = '合作'
        self.relationships[other.name] = rel

    def live(self, world):
        # 偏好影响心情
        if self.preference == '娱乐' and self.location == '公园':
            self.mood = min(100, self.mood + 20)
        elif self.preference == '学习' and self.location in ['家', '公园']:
            self.mood = min(100, self.mood + 10)
        if self.items.get("食物", 0) > 0:
            self.items["食物"] -= 1
            self.hunger = max(0, self.hunger - 20)
            self.health = min(100, self.health + 5)
        elif self.wealth >= 10:
            self.wealth -= 10
            self.items["食物"] = self.items.get("食物", 0) + 1
        else:
            self.hunger += 10
            self.health -= 5
        if self.location == "家":
            self.mood = min(100, self.mood + 10)
            self.health = min(100, self.health + 2)
        else:
            self.mood = max(0, self.mood - 5)

    def work(self, world):
        # 性格影响工作收益
        base = self.work_skill * random.randint(5, 10)
        if self.personality == '冒险':
            earn = int(base * 1.2)
        elif self.personality == '保守':
            earn = int(base * 0.8)
        else:
            earn = base
        if self.location == "公司":
            self.wealth += earn
            self.health -= 3
            self.experience += 2

    def learn(self, world):
        if self.location in ["家", "公园"]:
            self.survival_skill += 1
            self.work_skill += 1
            self.experience += 3
            self.mood -= 2

    def shop(self, world):
        if self.location == "商店":
            item = random.choice(list(world.items.values()))
            if self.wealth >= item.price:
                self.wealth -= item.price
                self.items[item.name] = self.items.get(item.name, 0) + 1

    def entertain(self, world):
        if self.location == "公园":
            self.mood = min(100, self.mood + 15)

    def heal(self, world):
        if self.location == "医院" and self.wealth >= 20:
            self.wealth -= 20
            self.health = min(100, self.health + 20)

    def invest(self, world):
        if self.items.get("股票", 0) > 0:
            gain = random.choice([-20, 10, 30])
            self.wealth += gain
            self.items["股票"] -= 1

    def move(self, world):
        self.location = random.choice(World.LOCATIONS)

    def interact(self, other, world):
        if self.name == other.name:
            return
        # 偏好社交的角色更容易提升关系
        delta = 1
        if self.preference == '社交' or other.preference == '社交':
            delta += 1
        # 性格影响关系变化
        if self.personality == '慷慨':
            delta += 1
        elif self.personality == '吝啬':
            delta -= 1
        # 社会规则：有概率帮助或攻击
        action = random.choices(['help', 'normal', 'attack'], [0.1, 0.8, 0.1])[0]
        if action == 'help':
            if self.wealth > 10:
                self.wealth -= 5
                other.wealth += 5
                self.reputation += 2
                delta += 1
        elif action == 'attack':
            other.health -= 5
            self.reputation -= 3
            delta -= 2
        # 随机正负
        if random.random() < 0.7:
            self._update_relationship(other, delta)
            other._update_relationship(self, delta)
        else:
            self._update_relationship(other, -delta)
            other._update_relationship(self, -delta)
        # 资助行为
        if self.wealth > 20 and other.wealth < 20 and self.personality == '慷慨':
            self.wealth -= 5
            other.wealth += 5
        # 社交圈扩展
        self.social_circle.add(other.name)
        other.social_circle.add(self.name)

    def use_item(self, world):
        if self.items.get("药品", 0) > 0 and self.health < 80:
            self.items["药品"] -= 1
            self.health = min(100, self.health + 15)

    def random_event(self, world):
        for event in world.events:
            # 声望高降低负面事件概率，提高正面事件概率
            prob = event.probability
            if event.name in ['生病', '失业'] and self.reputation > 5:
                prob *= 0.5
            if event.name in ['中奖', '结交新朋友'] and self.reputation > 5:
                prob *= 1.5
            if random.random() < prob:
                event.trigger(self, world)

    def grow(self, world):
        if self.experience > 10:
            self.survival_skill += 1
            self.work_skill += 1
            self.experience -= 10
            # 声望提升
            self.reputation += 1

    def check_alive(self, world):
        if self.health <= 0 or self.age > 80:
            self.alive = False

    def status(self):
        # 展示主要社交关系类型，兼容旧格式
        rel_summary = {k: (v['type'] if isinstance(v, dict) else '合作') for k, v in self.relationships.items()}
        return f"{self.name} | 年龄:{self.age} 健康:{self.health} 财富:{self.wealth} 心情:{self.mood} 饥饿:{self.hunger} 技能(生存:{self.survival_skill} 社交:{self.social_skill} 工作:{self.work_skill}) 经验:{self.experience} 位置:{self.location} 物品:{self.items} 关系:{rel_summary} 性格:{self.personality} 偏好:{self.preference} 声望:{self.reputation} 家族:{self.family} 社交圈:{list(self.social_circle)}"

    def llm_decide_action(self, world, enable_thinking=False):
        # 构造prompt，包含角色属性、世界状态等
        prompt = f"""
You are a character in a simulated world. Your state:, Name: {self.name}, Age: {self.age}, Health: {self.health}, Wealth: {self.wealth}, Mood: {self.mood}, Hunger: {self.hunger}, Personality: {self.personality}, Preference: {self.preference}, Reputation: {self.reputation}, Location: {self.location}, Relationships: {self.relationships}.
What is the best action to take next? Choose from: work, shop, entertain, learn, heal, invest, interact, rest, move. Only output the action word."""
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = tokenizer([chat_prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.6 if enable_thinking else 0.7,
            top_p=0.95 if enable_thinking else 0.8,
            top_k=20,
            do_sample=True
        )
        output_ids = outputs[0][inputs.input_ids.shape[-1]:].tolist()
        action = tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
        # 只保留第一个单词
        action_res = action.split()[0] if action else "rest"
        logging.info(f"{messages} -> {action} -> {action_res}")
        return action_res

# 世界类
class World:
    LOCATIONS = ["家", "公司", "商店", "公园", "医院"]
    def __init__(self, config=None):
        self.config = config or load_config()
        self.characters = []
        self.items = {}
        self.events = []
        self.day = 1
        self.load_items()
        self.load_events()
        self.load_characters()

    def load_items(self):
        items_conf = self.config.get('items', None)
        if items_conf:
            for item in items_conf:
                self.items[item['name']] = Item(item['name'], item.get('price', 10), item.get('effect', {}))
        else:
            self.items = {
                "食物": Item("食物", 10, {"health": 5, "hunger": -20}),
                "药品": Item("药品", 20, {"health": 15}),
                "书籍": Item("书籍", 15, {"experience": 5}),
                "玩具": Item("玩具", 12, {"mood": 10}),
                "股票": Item("股票", 30, {"wealth": "random"}),
            }

    def load_events(self):
        def make_event_func(effect):
            def func(c, w):
                for k, v in effect.items():
                    if k == 'health':
                        c.health += v
                    elif k == 'wealth':
                        c.wealth += v
                    elif k == 'work_skill':
                        c.work_skill += v
                    elif k == 'relationship':
                        # 随机加关系，兼容dict/int
                        others = [x for x in w.characters if x.name != c.name and x.alive]
                        if others:
                            friend = random.choice(others)
                            rel = c.relationships.get(friend.name, {'level': 0, 'type': '合作'})
                            if isinstance(rel, int):
                                rel = {'level': rel, 'type': '合作'}
                            rel['level'] += v
                            if rel['level'] >= 10:
                                rel['type'] = '亲密'
                            elif rel['level'] <= -5:
                                rel['type'] = '敌对'
                            else:
                                rel['type'] = '合作'
                            c.relationships[friend.name] = rel
            return func
        events_conf = self.config.get('events', None)
        if events_conf:
            for event in events_conf:
                self.events.append(Event(event['name'], make_event_func(event['effect']), event.get('probability', 0.1)))
        else:
            self.events = [
                Event("生病", lambda c, w: setattr(c, 'health', c.health - 15), 0.1),
                Event("中奖", lambda c, w: setattr(c, 'wealth', c.wealth + 50), 0.05),
                Event("失业", lambda c, w: setattr(c, 'work_skill', max(1, c.work_skill - 2)), 0.05),
                Event("结交新朋友", self.make_friend, 0.1),
            ]

    def load_characters(self):
        chars_conf = self.config.get('characters', None)
        if chars_conf:
            for char in chars_conf:
                self.characters.append(Character(char['name'], char))

    def make_friend(self, character, world):
        others = [c for c in self.characters if c.name != character.name and c.alive]
        if others:
            friend = random.choice(others)
            character.relationships[friend.name] = character.relationships.get(friend.name, 0) + 2

    def add_character(self, character):
        self.characters.append(character)

    def remove_dead(self):
        for i, c in enumerate(self.characters):
            if not c.alive:
                new_name = f"Newbie{self.day}{i}"
                self.characters[i] = Character(new_name)

    def run_day(self, actions=None):
        logging.info(f"\n=== 第{self.day}天 ===")
        day_actions = {}
        for c in self.characters:
            if not c.alive:
                day_actions[c.name] = 'dead'
                continue
            # 决策行为
            if actions and c.name in actions:
                action = actions[c.name]
            else:
                action = c.llm_decide_action(self)
            day_actions[c.name] = action
            if action == "work":
                c.work(self)
            elif action == "shop":
                c.shop(self)
            elif action == "entertain":
                c.entertain(self)
            elif action == "learn":
                c.learn(self)
            elif action == "heal":
                c.heal(self)
            elif action == "invest":
                c.invest(self)
            elif action == "interact":
                others = [x for x in self.characters if x.name != c.name and x.alive]
                if others:
                    c.interact(random.choice(others), self)
            elif action == "move":
                c.move(self)
            elif action == "rest":
                c.live(self)
            else:
                c.live(self)
            c.use_item(self)
            c.random_event(self)
            c.grow(self)
            c.check_alive(self)
        alive_chars = [c for c in self.characters if c.alive]
        if len(alive_chars) > 1:
            pairs = random.sample(alive_chars, 2)
            pairs[0].interact(pairs[1], self)
        self.remove_dead()
        # 打印行为信息
        logging.info("角色行为：")
        for name, act in day_actions.items():
            logging.info(f"{name}: {act}")
        logging.info("\n角色状态：")
        for c in self.characters:
            status = c.status()
            logging.info(status)
        self.day += 1
        return day_actions

# 可视化接口预留
class Visualizer:
    @staticmethod
    def plot_stats(stats, names, actions=None):
        days = list(range(1, len(stats[names[0]]['health']) + 1))
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for name in names:
            axs[0, 0].plot(days, stats[name]['health'], label=name)
            axs[0, 1].plot(days, stats[name]['wealth'], label=name)
            axs[1, 0].plot(days, stats[name]['mood'], label=name)
            axs[1, 1].plot(days, stats[name]['reputation'], label=name)
        axs[0, 0].set_title('Health')
        axs[0, 1].set_title('Wealth')
        axs[1, 0].set_title('Mood')
        axs[1, 1].set_title('Reputation')
        for ax in axs.flat:
            ax.set_xlabel('Day')
            ax.legend()
        plt.tight_layout()
        plt.show()
        # 行为轨迹可视化（离散标签）
        if actions:
            plt.figure(figsize=(12, 4))
            # 收集所有出现过的action
            all_actions = sorted({a for acts in actions.values() for a in acts})
            action2idx = {a: i for i, a in enumerate(all_actions)}
            idx2action = {i: a for a, i in action2idx.items()}
            for name in names:
                y = [action2idx[a] for a in actions[name]]
                plt.plot(days, y, label=name, marker='o')
            plt.title('Action Trajectory')
            plt.xlabel('Day')
            plt.ylabel('Action')
            plt.yticks(list(idx2action.keys()), [idx2action[i] for i in idx2action])
            plt.legend()
            plt.tight_layout()
            plt.show()

# 简单单元测试
def test_character_life():
    logging.info("\n[TEST] 角色生活行为测试")
    c = Character("Test", {"health": 50, "wealth": 20, "items": {"食物": 1}, "location": "家", "hunger": 10})
    w = World()
    hunger_before = c.hunger
    health_before = c.health
    c.live(w)
    assert c.health >= health_before, "吃饭或休息后健康应增加"
    assert c.hunger < hunger_before, "吃饭后饥饿应减少"
    logging.info("通过")

def test_event_system():
    logging.info("\n[TEST] 事件系统测试")
    c = Character("Test", {"health": 100, "wealth": 100, "work_skill": 5})
    w = World()
    # 强制触发生病事件
    event = Event("生病", lambda c, w: setattr(c, 'health', c.health - 15), 1.0)
    event.trigger(c, w)
    assert c.health == 85, "生病后健康应减少15"
    logging.info("通过")

if __name__ == "__main__":
    # 测试
    test_character_life()
    test_event_system()
    # 主模拟
    config = load_config()
    world = World(config)
    if not world.characters:
        names = ["Ava", "Ben", "Cara", "Dan", "Eve"]
        for name in names:
            world.add_character(Character(name))
    # 可视化数据收集
    stats = {c.name: {'health': [], 'wealth': [], 'mood': [], 'reputation': []} for c in world.characters}
    actions = {c.name: [] for c in world.characters}
    names = [c.name for c in world.characters]
    for _ in range(5):
        day_actions = world.run_day()
        for c in world.characters:
            stats[c.name]['health'].append(c.health)
            stats[c.name]['wealth'].append(c.wealth)
            stats[c.name]['mood'].append(c.mood)
            stats[c.name]['reputation'].append(c.reputation)
            actions[c.name].append(day_actions.get(c.name, 'dead'))
    # Visualizer.plot_stats(stats, names, actions)
