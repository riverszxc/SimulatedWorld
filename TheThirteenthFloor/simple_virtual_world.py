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
tokenizer = None
model = None

# 检查模型路径是否存在
if os.path.exists(LLM_MODEL_PATH):
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("LLM模型加载成功")
    except Exception as e:
        logging.warning(f"LLM模型加载失败: {e}")
        tokenizer = None
        model = None
else:
    logging.warning(f"LLM模型路径不存在: {LLM_MODEL_PATH}")

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
        
        # 新增记忆和认知系统
        self.memory = config.get('memory', {
            'action_history': [],  # 行为历史记录
            'interaction_history': [],  # 互动历史
            'event_history': [],  # 事件经历
            'learned_patterns': {},  # 学习到的行为模式
            'world_knowledge': {}  # 对世界的认知
        })
        self.goals = config.get('goals', {
            'short_term': [],  # 短期目标
            'long_term': []   # 长期目标
        })
        self.world_perception = {
            'other_characters': {},  # 对其他角色的了解
            'location_states': {},   # 对各位置状态的了解
            'market_trends': {},     # 对市场趋势的感知
            'social_dynamics': {}    # 对社会动态的理解
        }

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

    def record_action(self, action, result, day):
        """记录行为及其结果"""
        self.memory['action_history'].append({
            'day': day,
            'action': action,
            'result': result,
            'context': {
                'location': self.location,
                'health': self.health,
                'wealth': self.wealth,
                'mood': self.mood
            }
        })
        # 保持记忆长度合理
        if len(self.memory['action_history']) > 50:
            self.memory['action_history'].pop(0)
    
    def record_interaction(self, other_name, interaction_type, outcome, day):
        """记录互动经历"""
        self.memory['interaction_history'].append({
            'day': day,
            'other': other_name,
            'type': interaction_type,
            'outcome': outcome
        })
        if len(self.memory['interaction_history']) > 30:
            self.memory['interaction_history'].pop(0)
    
    def record_event(self, event_name, effect, day):
        """记录事件经历"""
        self.memory['event_history'].append({
            'day': day,
            'event': event_name,
            'effect': effect
        })
        if len(self.memory['event_history']) > 20:
            self.memory['event_history'].pop(0)
    
    def update_world_perception(self, world):
        """更新对世界状态的感知"""
        # 感知其他角色状态
        for char in world.characters:
            if char.name != self.name and char.alive:
                # 根据社交技能和关系决定能感知到多少信息
                perception_level = self.social_skill + self.relationships.get(char.name, {'level': 0})['level'] / 10
                
                self.world_perception['other_characters'][char.name] = {
                    'location': char.location,
                    'apparent_health': 'good' if char.health > 70 else 'poor' if char.health < 40 else 'average',
                    'apparent_wealth': 'rich' if char.wealth > 100 else 'poor' if char.wealth < 50 else 'average',
                    'relationship_level': self.relationships.get(char.name, {'level': 0})['level']
                }
                
                # 高感知能力能了解更多信息
                if perception_level > 8:
                    self.world_perception['other_characters'][char.name].update({
                        'mood': char.mood,
                        'recent_actions': char.memory['action_history'][-3:] if char.memory['action_history'] else []
                    })
        
        # 感知位置状态（基于经验和观察）
        location_crowds = {}
        for char in world.characters:
            if char.alive:
                location_crowds[char.location] = location_crowds.get(char.location, 0) + 1
        
        for location in world.LOCATIONS:
            self.world_perception['location_states'][location] = {
                'crowded': location_crowds.get(location, 0) > len(world.characters) / len(world.LOCATIONS),
                'visited_recently': any(h['context']['location'] == location for h in self.memory['action_history'][-5:])
            }
    
    def learn_from_experience(self):
        """从经验中学习行为模式"""
        if len(self.memory['action_history']) < 5:
            return
        
        # 分析成功的行为模式
        for action_record in self.memory['action_history'][-10:]:
            action = action_record['action']
            result = action_record['result']
            
            # 简单的学习逻辑：记录哪些行为在什么情况下效果好
            if action not in self.memory['learned_patterns']:
                self.memory['learned_patterns'][action] = {'success': 0, 'total': 0}
            
            self.memory['learned_patterns'][action]['total'] += 1
            if self._evaluate_action_success(result):
                self.memory['learned_patterns'][action]['success'] += 1
    
    def _evaluate_action_success(self, result):
        """评估行为是否成功（简化版）"""
        if isinstance(result, dict):
            # 如果结果是字典，检查是否有正面影响
            return any(v > 0 for k, v in result.items() if k in ['health', 'wealth', 'mood', 'experience'])
        return result  # 如果是布尔值直接返回
    
    def set_goals(self):
        """根据当前状态设定目标"""
        self.goals['short_term'].clear()
        self.goals['long_term'].clear()
        
        # 基于当前状态设定短期目标
        if self.health < 50:
            self.goals['short_term'].append('improve_health')
        if self.wealth < 30:
            self.goals['short_term'].append('earn_money')
        if self.mood < 40:
            self.goals['short_term'].append('improve_mood')
        if self.hunger > 70:
            self.goals['short_term'].append('find_food')
        
        # 基于性格和偏好设定长期目标
        if self.personality == '冒险':
            self.goals['long_term'].append('explore_opportunities')
        elif self.personality == '保守':
            self.goals['long_term'].append('maintain_stability')
        
        if self.preference == '社交':
            self.goals['long_term'].append('build_relationships')
        elif self.preference == '投资':
            self.goals['long_term'].append('accumulate_wealth')

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
        
        result = {'wealth': 0, 'health': 0, 'experience': 0}
        if self.location == "公司":
            self.wealth += earn
            self.health -= 3
            self.experience += 2
            result = {'wealth': earn, 'health': -3, 'experience': 2}
        
        self.record_action('work', result, world.day)
        return result

    def learn(self, world):
        result = {'survival_skill': 0, 'work_skill': 0, 'experience': 0, 'mood': 0}
        if self.location in ["家", "公园"]:
            self.survival_skill += 1
            self.work_skill += 1
            self.experience += 3
            self.mood -= 2
            result = {'survival_skill': 1, 'work_skill': 1, 'experience': 3, 'mood': -2}
        
        self.record_action('learn', result, world.day)
        return result

    def shop(self, world):
        result = {'wealth': 0, 'items': {}}
        if self.location == "商店":
            item = random.choice(list(world.items.values()))
            if self.wealth >= item.price:
                self.wealth -= item.price
                self.items[item.name] = self.items.get(item.name, 0) + 1
                result = {'wealth': -item.price, 'items': {item.name: 1}}
        
        self.record_action('shop', result, world.day)
        return result

    def entertain(self, world):
        result = {'mood': 0}
        if self.location == "公园":
            mood_gain = min(15, 100 - self.mood)
            self.mood = min(100, self.mood + 15)
            result = {'mood': mood_gain}
        
        self.record_action('entertain', result, world.day)
        return result

    def heal(self, world):
        result = {'wealth': 0, 'health': 0}
        if self.location == "医院" and self.wealth >= 20:
            self.wealth -= 20
            health_gain = min(20, 100 - self.health)
            self.health = min(100, self.health + 20)
            result = {'wealth': -20, 'health': health_gain}
        
        self.record_action('heal', result, world.day)
        return result

    def invest(self, world):
        result = {'wealth': 0, 'items': {}}
        if self.items.get("股票", 0) > 0:
            gain = random.choice([-20, 10, 30])
            self.wealth += gain
            self.items["股票"] -= 1
            result = {'wealth': gain, 'items': {'股票': -1}}
        
        self.record_action('invest', result, world.day)
        return result

    def move(self, world):
        old_location = self.location
        self.location = random.choice(World.LOCATIONS)
        result = {'location_change': f'{old_location} -> {self.location}'}
        
        self.record_action('move', result, world.day)
        return result

    def interact(self, other, world):
        if self.name == other.name:
            return {'interaction': 'failed', 'reason': 'self-interaction'}
        
        result = {'relationship_change': 0, 'wealth_change': 0, 'reputation_change': 0, 'interaction_type': 'normal'}
        
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
        result['interaction_type'] = action
        
        if action == 'help':
            if self.wealth > 10:
                self.wealth -= 5
                other.wealth += 5
                self.reputation += 2
                delta += 1
                result['wealth_change'] = -5
                result['reputation_change'] = 2
        elif action == 'attack':
            other.health -= 5
            self.reputation -= 3
            delta -= 2
            result['reputation_change'] = -3
        
        # 随机正负
        if random.random() < 0.7:
            self._update_relationship(other, delta)
            other._update_relationship(self, delta)
            result['relationship_change'] = delta
        else:
            self._update_relationship(other, -delta)
            other._update_relationship(self, -delta)
            result['relationship_change'] = -delta
        
        # 资助行为
        if self.wealth > 20 and other.wealth < 20 and self.personality == '慷慨':
            self.wealth -= 5
            other.wealth += 5
            result['wealth_change'] += -5
        
        # 社交圈扩展
        self.social_circle.add(other.name)
        other.social_circle.add(self.name)
        
        # 记录互动
        self.record_interaction(other.name, action, result, world.day)
        other.record_interaction(self.name, action + '_received', result, world.day)
        
        return result

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
                # 记录事件前的状态
                before_state = {
                    'health': self.health,
                    'wealth': self.wealth,
                    'work_skill': self.work_skill
                }
                event.trigger(self, world)
                # 计算事件效果
                effect = {
                    'health': self.health - before_state['health'],
                    'wealth': self.wealth - before_state['wealth'],
                    'work_skill': self.work_skill - before_state['work_skill']
                }
                self.record_event(event.name, effect, world.day)

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
        # 更新世界感知和目标
        self.update_world_perception(world)
        self.set_goals()
        self.learn_from_experience()
        
        # 如果LLM不可用，使用智能规则决策
        if tokenizer is None or model is None:
            return self.rule_based_decide_action(world)
        
        # 构造丰富的prompt
        recent_actions = [f"Day {h['day']}: {h['action']} -> {h['result']}" for h in self.memory['action_history'][-5:]]
        recent_interactions = [f"Day {h['day']}: {h['type']} with {h['other']} -> {h['outcome']}" for h in self.memory['interaction_history'][-3:]]
        recent_events = [f"Day {h['day']}: {h['event']} -> {h['effect']}" for h in self.memory['event_history'][-3:]]
        
        other_chars_info = []
        for name, info in self.world_perception['other_characters'].items():
            other_chars_info.append(f"{name}: at {info['location']}, {info['apparent_health']} health, {info['apparent_wealth']} wealth, relationship level {info['relationship_level']}")
        
        location_info = []
        for loc, info in self.world_perception['location_states'].items():
            status = "crowded" if info['crowded'] else "quiet"
            recent = "recently visited" if info['visited_recently'] else "not recently visited"
            location_info.append(f"{loc}: {status}, {recent}")
        
        learned_patterns_info = []
        for action, pattern in self.memory['learned_patterns'].items():
            if pattern['total'] > 0:
                success_rate = pattern['success'] / pattern['total']
                learned_patterns_info.append(f"{action}: {success_rate:.2f} success rate ({pattern['success']}/{pattern['total']})")
        
        prompt = f"""You are {self.name}, a character in a simulated world. Make a strategic decision for your next action.

CURRENT STATUS:
- Age: {self.age}, Health: {self.health}/100, Wealth: {self.wealth}, Mood: {self.mood}/100, Hunger: {self.hunger}/100
- Skills: Survival {self.survival_skill}, Social {self.social_skill}, Work {self.work_skill}
- Personality: {self.personality}, Preference: {self.preference}, Reputation: {self.reputation}
- Location: {self.location}, Items: {self.items}

RELATIONSHIPS: {self.relationships}

CURRENT GOALS:
- Short-term: {', '.join(self.goals['short_term']) if self.goals['short_term'] else 'None'}
- Long-term: {', '.join(self.goals['long_term']) if self.goals['long_term'] else 'None'}

RECENT HISTORY:
Actions: {'; '.join(recent_actions) if recent_actions else 'None'}
Interactions: {'; '.join(recent_interactions) if recent_interactions else 'None'}
Events: {'; '.join(recent_events) if recent_events else 'None'}

WORLD PERCEPTION:
Other Characters: {'; '.join(other_chars_info) if other_chars_info else 'None visible'}
Locations: {'; '.join(location_info)}

LEARNED PATTERNS: {'; '.join(learned_patterns_info) if learned_patterns_info else 'No patterns learned yet'}

Based on all this information, analyze your situation and choose the best action. Available actions: work, shop, entertain, learn, heal, invest, interact, rest, move.

CRITICAL: Respond with ONLY a valid JSON object in this exact format:
{{
    "reasoning": "Your decision reasoning in 50 words or less",
    "action": "chosen_action"
}}

The reasoning must be under 50 words. The action must be one of the valid options."""
        logging.info(f"{self.name} \nprompt: {prompt}")
        try:
            messages = [
                {"role": "system", "content": "You are a decision-making AI for a character simulation. Always respond with valid JSON containing 'reasoning' (under 50 words) and 'action' fields. Be concise and focused. Analyze only the most important factors."},
                {"role": "user", "content": prompt}
            ]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            inputs = tokenizer([chat_prompt], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256 if enable_thinking else 128,  # 增加thinking模式的token数量
                temperature=0.6 if enable_thinking else 0.7,
                top_p=0.95 if enable_thinking else 0.8,
                top_k=20,
                do_sample=True
            )
            output_ids = outputs[0][inputs.input_ids.shape[-1]:].tolist()
            
            # 解析JSON输出
            raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            print("raw_output: ", raw_output)
            
            # 解析JSON响应
            reasoning = ""
            action = "rest"  # 默认行为
            
            try:
                import json
                import re
                
                # 尝试提取JSON内容（可能包含在其他文本中）
                json_match = re.search(r'\{[^}]*"reasoning"[^}]*"action"[^}]*\}', raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    response_data = json.loads(json_str)
                    reasoning = response_data.get('reasoning', '').strip()
                    action = response_data.get('action', 'rest').strip().lower()
                else:
                    # 如果没有找到JSON，尝试解析原始文本
                    raw_output = raw_output.lower()
                    valid_actions = ["work", "shop", "entertain", "learn", "heal", "invest", "interact", "rest", "move"]
                    for valid_action in valid_actions:
                        if valid_action in raw_output:
                            action = valid_action
                            reasoning = f"Parsed from raw output: {raw_output[:50]}..."
                            break
                
                print("parsed reasoning: ", reasoning)
                print("parsed action: ", action)
                
            except Exception as e:
                logging.warning(f"JSON解析失败: {e}, 原始输出: {raw_output}")
                # 如果JSON解析失败，尝试从原始文本中提取行为
                raw_output = raw_output.lower()
                valid_actions = ["work", "shop", "entertain", "learn", "heal", "invest", "interact", "rest", "move"]
                for valid_action in valid_actions:
                    if valid_action in raw_output:
                        action = valid_action
                        reasoning = f"Fallback parsing: {raw_output[:50]}..."
                        break
            
            # 验证行为是否有效
            valid_actions = ["work", "shop", "entertain", "learn", "heal", "invest", "interact", "rest", "move"]
            if action not in valid_actions:
                # 如果不是有效行为，尝试在文本中寻找有效行为
                for valid_action in valid_actions:
                    if valid_action in action:
                        action = valid_action
                        break
                else:
                    action = "rest"  # 默认行为
            
            # 记录决策过程
            logging.info(f"\n{'='*50}")
            logging.info(f"{self.name} JSON决策过程:")
            logging.info(f"目标: 短期{self.goals['short_term']}, 长期{self.goals['long_term']}")
            logging.info(f"最近行为: {[h['action'] for h in self.memory['action_history'][-3:]]}")
            logging.info(f"决策理由: {reasoning}")
            logging.info(f"LLM原始输出: {raw_output}")
            logging.info(f"最终选择行为: {action}")
            logging.info(f"{'='*50}\n")
            
            return action
        except Exception as e:
            logging.warning(f"LLM决策失败，使用规则决策: {e}")
            return self.rule_based_decide_action(world)
    
    def rule_based_decide_action(self, world):
        """基于规则的智能决策系统，考虑目标、状态和经验"""
        # 优先处理紧急需求
        if self.health < 30:
            if self.location != '医院' and self.wealth >= 20:
                return 'move' if self.location != '医院' else 'heal'
            elif self.items.get('药品', 0) > 0:
                return 'rest'  # 会在use_item中使用药品
            else:
                return 'heal'
        
        if self.hunger > 80:
            if self.items.get('食物', 0) == 0:
                if self.wealth >= 10:
                    return 'shop' if self.location == '商店' else 'move'
                else:
                    return 'work' if self.location == '公司' else 'move'
            else:
                return 'rest'  # 会在live中吃食物
        
        # 根据目标决策
        for goal in self.goals['short_term']:
            if goal == 'improve_health' and self.location != '医院':
                return 'move'
            elif goal == 'earn_money' and self.location != '公司':
                return 'move'
            elif goal == 'improve_mood' and self.location != '公园':
                return 'move'
            elif goal == 'find_food' and self.location != '商店':
                return 'move'
        
        # 基于学习到的模式决策
        best_action = None
        best_success_rate = 0
        for action, pattern in self.memory['learned_patterns'].items():
            if pattern['total'] > 3:  # 有足够样本
                success_rate = pattern['success'] / pattern['total']
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_action = action
        
        if best_action and best_success_rate > 0.6:
            logging.info(f"{self.name} 基于经验选择 {best_action} (成功率: {best_success_rate:.2f})")
            return best_action
        
        # 基于性格和偏好的默认行为
        if self.preference == '娱乐':
            return 'entertain' if self.location == '公园' else 'move'
        elif self.preference == '学习':
            return 'learn' if self.location in ['家', '公园'] else 'move'
        elif self.preference == '购物':
            return 'shop' if self.location == '商店' else 'move'
        elif self.preference == '社交':
            others = [c for c in world.characters if c.name != self.name and c.alive]
            if others:
                return 'interact'
            else:
                return 'move'
        elif self.preference == '投资':
            if self.items.get('股票', 0) > 0:
                return 'invest'
            elif self.wealth >= 30:
                return 'shop' if self.location == '商店' else 'move'
        
        # 基于当前状态的决策
        if self.wealth < 50:
            return 'work' if self.location == '公司' else 'move'
        elif self.mood < 50:
            return 'entertain' if self.location == '公园' else 'move'
        elif self.experience < 20:
            return 'learn' if self.location in ['家', '公园'] else 'move'
        
        return 'rest'

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
        day_results = {}
        
        for c in self.characters:
            if not c.alive:
                day_actions[c.name] = 'dead'
                day_results[c.name] = 'dead'
                continue
            
            # 决策行为
            if actions and c.name in actions:
                action = actions[c.name]
            else:
                action = c.llm_decide_action(self)  # 使用JSON格式输出
            
            day_actions[c.name] = action
            result = None
            
            # 执行行为并记录结果
            if action == "work":
                result = c.work(self)
            elif action == "shop":
                result = c.shop(self)
            elif action == "entertain":
                result = c.entertain(self)
            elif action == "learn":
                result = c.learn(self)
            elif action == "heal":
                result = c.heal(self)
            elif action == "invest":
                result = c.invest(self)
            elif action == "interact":
                others = [x for x in self.characters if x.name != c.name and x.alive]
                if others:
                    result = c.interact(random.choice(others), self)
                else:
                    result = {'interaction': 'failed', 'reason': 'no_targets'}
            elif action == "move":
                result = c.move(self)
            elif action == "rest":
                c.live(self)
                result = {'action': 'rest', 'location_bonus': c.location == '家'}
                c.record_action('rest', result, self.day)
            else:
                c.live(self)
                result = {'action': 'default', 'location_bonus': c.location == '家'}
                c.record_action('default', result, self.day)
            
            day_results[c.name] = result
            
            # 其他日常活动
            c.use_item(self)
            c.random_event(self)
            c.grow(self)
            c.check_alive(self)
        
        # 额外随机互动
        alive_chars = [c for c in self.characters if c.alive]
        if len(alive_chars) > 1:
            pairs = random.sample(alive_chars, 2)
            pairs[0].interact(pairs[1], self)
        
        self.remove_dead()
        
        # 打印详细行为和结果信息
        logging.info("角色行为及结果：")
        for name, act in day_actions.items():
            result = day_results.get(name, {})
            logging.info(f"{name}: {act} -> {result}")
        
        logging.info("\n角色状态：")
        for c in self.characters:
            status = c.status()
            logging.info(status)
            # 显示最近的记忆和目标
            if c.memory['action_history']:
                recent_actions = [h['action'] for h in c.memory['action_history'][-3:]]
                logging.info(f"  最近行为: {recent_actions}")
            if c.goals['short_term'] or c.goals['long_term']:
                logging.info(f"  目标: 短期{c.goals['short_term']}, 长期{c.goals['long_term']}")
        
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
    for _ in range(3):
        day_actions = world.run_day()
        for c in world.characters:
            stats[c.name]['health'].append(c.health)
            stats[c.name]['wealth'].append(c.wealth)
            stats[c.name]['mood'].append(c.mood)
            stats[c.name]['reputation'].append(c.reputation)
            actions[c.name].append(day_actions.get(c.name, 'dead'))
    # Visualizer.plot_stats(stats, names, actions)
