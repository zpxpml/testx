import os
import torch
from llm.model import LightForCausalLM, LightConfig
from transformers import AutoTokenizer, TextStreamer
import argparse

def get_prompt_datas(model_mode):
    if model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理是',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有',
            '为什么有些植物能够'
        ]
    else:
        prompt_datas = [
            '请介绍一下自己。',
            '你更擅长哪一个学科？',
            '鲁迅的《狂人日记》是如何批判封建礼教的？',
            '我咳嗽已经持续了两周，需要去医院检查吗？',
            '详细的介绍光速的物理概念。',
            '推荐一些杭州的特色美食吧。',
            '请为我讲解“大语言模型”这个概念。'
        ]

    return prompt_datas

def init_model(args):
    modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
    mode = args.model_mode
    # share_atten/model_output_no_o
    ckp_dir = '' # '/layers16/ffn_n'#'/sota_pre'
    ckp = f'./checkpoints{ckp_dir}/model_output_interp.pth' if mode == 0 else f'./checkpoints{ckp_dir}/model_output_post.pth'
    model = LightForCausalLM(LightConfig(hidden_size=args.hidden_size, use_moe=False))
    model.eval()
    print(args.device)
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

    tokenizer_path = "./tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f'Light模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Light")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--hidden_size', default=576, type=int)
    parser.add_argument('--max_seq_len', default=500, type=int)
    parser.add_argument('--model_mode', default=0, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型，4: RLAIF-Chat模型")
    args = parser.parse_args()
    model, tokenizer = init_model(args)

    
    max_new_tokens = args.max_seq_len
    temperature = args.temperature
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    prompts = get_prompt_datas(0) #args.model_mode
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        if test_mode == 0: print(f'👶: {prompt}')

        messages = []
        messages.append({"role": "user", "content": prompt})
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) if args.model_mode != 0 else (tokenizer.bos_token + prompt)
        # print(new_prompt, messages)
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to("cuda")
        print('🤖️: ', end='')
        generated_ids = model.generate(
            input_ids = inputs["input_ids"],
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.2 # 惩罚重复
        )

        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


