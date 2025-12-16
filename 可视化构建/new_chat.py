import gradio as gr
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置环境变量，避免 TRANSFORMERS_CACHE 警告
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

# 在这里指定包含多个模型文件夹的根路径
MODELS_ROOT_PATH = "E:/llama_factory/LLaMA-Factory/output_model"  # 请修改为您的实际模型根路径

# 全局变量存储模型和tokenizer
model = None
tokenizer = None
current_model_name = None
conversation_history = []


def get_available_models():
    """获取指定路径下的所有可用模型"""
    models = []

    if not os.path.exists(MODELS_ROOT_PATH):
        return ["错误：模型根路径不存在"]

    if not os.path.isdir(MODELS_ROOT_PATH):
        return ["错误：模型根路径不是目录"]

    try:
        # 遍历根目录下的所有子文件夹
        for item in os.listdir(MODELS_ROOT_PATH):
            item_path = os.path.join(MODELS_ROOT_PATH, item)

            if os.path.isdir(item_path):
                # 检查子文件夹是否包含模型文件
                model_files = []
                try:
                    model_files = os.listdir(item_path)
                except:
                    continue

                # 检查常见的模型文件扩展名
                model_extensions = ('.bin', '.safetensors', '.pt', '.pth', '.msgpack')
                has_model_files = any(fname.endswith(model_extensions) for fname in model_files)

                # 检查是否有配置文件
                has_config = any(
                    fname in ['config.json', 'pytorch_model.bin', 'model.safetensors'] for fname in model_files)

                if has_model_files or has_config:
                    models.append(item)

        # 按名称排序
        models.sort()

        if not models:
            models = ["未找到可用模型"]

        return models

    except Exception as e:
        return [f"扫描模型时出错: {str(e)}"]


def load_model(selected_model):
    """加载选定的模型"""
    global model, tokenizer, current_model_name, conversation_history

    if selected_model in ["未找到可用模型", "错误：模型根路径不存在",
                          "错误：模型根路径不是目录"] or selected_model.startswith("错误："):
        return f"错误：{selected_model}", ""

    model_path = os.path.join(MODELS_ROOT_PATH, selected_model)

    if not os.path.exists(model_path):
        return f"错误：模型路径 '{model_path}' 不存在", ""

    try:
        # 卸载之前的模型
        if model is not None:
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 清空对话历史
        conversation_history = []

        print(f"正在加载模型: {selected_model}")
        print(f"模型路径: {model_path}")

        # 检查设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 加载模型和tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 如果使用CPU，手动移动模型到设备
        if device == "cpu":
            model = model.to(device)

        current_model_name = selected_model

        # 添加系统消息
        conversation_history.append({
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        })

        return f"模型加载成功！\n模型: {selected_model}\n路径: {model_path}\n设备: {device}\n可以开始对话了！", ""

    except Exception as e:
        return f"加载模型时出错: {str(e)}", ""


def unload_model():
    """卸载当前模型"""
    global model, tokenizer, current_model_name, conversation_history

    if model is None:
        return "没有模型需要卸载", ""

    try:
        model_name = current_model_name
        # 释放模型内存
        del model
        del tokenizer
        model = None
        tokenizer = None
        current_model_name = None
        conversation_history = []

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return f"模型 '{model_name}' 已成功卸载，内存已释放", ""

    except Exception as e:
        return f"卸载模型时出错: {str(e)}", ""


def chat_with_model(message, chat_history):
    """与模型对话"""
    global model, tokenizer, conversation_history

    if model is None:
        return chat_history, "错误：请先加载模型！"

    if not message or not message.strip():
        return chat_history, "错误：请输入有效消息！"

    try:
        # 立即将用户消息添加到聊天历史
        chat_history.append({"role": "user", "content": message})
        yield chat_history, ""

        # 添加用户消息到对话历史
        conversation_history.append({"role": "user", "content": message.strip()})

        # 准备输入
        text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 显示思考中状态
        chat_history.append({"role": "assistant", "content": "🤔 思考中..."})
        yield chat_history, ""

        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # 提取新生成的token
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]

        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 添加助手回复到历史
        conversation_history.append({"role": "assistant", "content": response})

        # 移除思考中消息
        chat_history.pop()

        # 逐字显示回复
        displayed_response = ""
        for i in range(len(response)):
            displayed_response = response[:i + 1]
            chat_history.append({"role": "assistant", "content": displayed_response + "▌"})
            yield chat_history, ""
            time.sleep(0.02)  # 控制逐字显示速度
            chat_history.pop()  # 移除临时消息

        # 最终显示完整回复
        chat_history.append({"role": "assistant", "content": displayed_response})
        yield chat_history, ""

    except Exception as e:
        error_msg = f"生成回复时出错: {str(e)}"
        print(error_msg)
        # 移除思考中消息并显示错误
        if chat_history and chat_history[-1].get("content") == "🤔 思考中...":
            chat_history.pop()
        chat_history.append({"role": "assistant", "content": f"❌ {error_msg}"})
        yield chat_history, error_msg


def clear_history():
    """清空对话历史"""
    global conversation_history

    conversation_history = []
    # 重新添加系统消息
    if model is not None:
        conversation_history.append({
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        })

    return [], "对话历史已清空！"


def refresh_models():
    """刷新模型列表"""
    models = get_available_models()
    return gr.Dropdown(choices=models, value=models[0] if models else "")


def get_root_path_info():
    """获取根路径信息"""
    return f"模型根路径: {MODELS_ROOT_PATH}"


# 创建Gradio界面
with gr.Blocks(title="中医药知识问答系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#中医药知识问答系统")
    gr.Markdown("### 实时对话体验 - 问题即时显示，回复逐字输出")

    # 显示根路径信息
    path_info = gr.Textbox(
        value=get_root_path_info(),
        label="路径配置",
        interactive=False
    )

    with gr.Row():
        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                label="选择模型",
                info="从下拉框中选择要加载的模型"
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 刷新列表", size="sm")

    with gr.Row():
        load_btn = gr.Button("✅ 加载模型", variant="primary")
        unload_btn = gr.Button("❌ 卸载模型", variant="stop")
        clear_btn = gr.Button("🗑️ 清空历史", variant="secondary")

    status_display = gr.Textbox(
        label="状态信息",
        interactive=False,
        lines=4,
        placeholder="模型状态将显示在这里..."
    )

    # 使用新的 messages 格式，修复弃用警告
    chatbot = gr.Chatbot(
        label="对话内容",
        height=400,
        placeholder="加载模型后，在这里开始对话...",
        type="messages",  # 使用新的 messages 格式
        show_copy_button=True
    )

    with gr.Row():
        msg = gr.Textbox(
            label="输入消息",
            placeholder="请输入您的问题...",
            scale=4,
            max_lines=3
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    # 绑定事件
    refresh_btn.click(
        refresh_models,
        outputs=model_dropdown
    )

    load_btn.click(
        load_model,
        inputs=[model_dropdown],
        outputs=[status_display, msg]
    )

    unload_btn.click(
        unload_model,
        outputs=[status_display, msg]
    )

    submit_btn.click(
        chat_with_model,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    msg.submit(
        chat_with_model,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    clear_btn.click(
        clear_history,
        outputs=[chatbot, status_display]
    )

if __name__ == "__main__":
    # 检查根路径
    if not os.path.exists(MODELS_ROOT_PATH):
        print(f"警告：指定的模型根路径不存在: {MODELS_ROOT_PATH}")
        print("请在代码开头修改 MODELS_ROOT_PATH 变量为您的实际路径")

    print("启动Gradio界面...")
    print(f"模型根路径: {MODELS_ROOT_PATH}")

    # 显示可用的模型
    available_models = get_available_models()
    print(f"找到 {len(available_models)} 个模型: {available_models}")

    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False
    )