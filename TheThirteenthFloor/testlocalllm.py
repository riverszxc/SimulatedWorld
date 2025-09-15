from transformers import AutoTokenizer, AutoModelForCausalLM

modelpath = "/Users/zxc/project/data/model/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModelForCausalLM.from_pretrained(modelpath)

messages = []
enable_thinking = True
print("欢迎使用Qwen3多轮对话，输入 exit 退出。每轮输入可加 /think 或 /no_think 切换思考模式。")

while True:
    user_input = input("用户: ")
    if user_input.strip().lower() == "exit":
        break
    # 检查思考模式切换
    if "/think" in user_input:
        enable_thinking = True
        user_input = user_input.replace("/think", "").strip()
    elif "/no_think" in user_input:
        enable_thinking = False
        user_input = user_input.replace("/no_think", "").strip()
    messages.append({"role": "user", "content": user_input})
    # 生成 prompt，tokenize=False
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=32768,
        temperature=0.6 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8,
        top_k=20,
        do_sample=True
    )
    output_ids = outputs[0][inputs.input_ids.shape[-1]:].tolist()
    # 解析 <think> 块和最终回复
    think_token_id = tokenizer.convert_tokens_to_ids("</think>")
    try:
        index = len(output_ids) - output_ids[::-1].index(think_token_id)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        thinking_content = None
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    if thinking_content:
        print("[思考]", thinking_content)
    print("助手:", content)
    messages.append({"role": "assistant", "content": content})