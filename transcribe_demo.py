import io
import os
import re
import sys
import ffmpeg
import locale
import importlib.util

import numpy as np
import gradio as gr
import soundfile as sf 
import torch

import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info
from argparse import ArgumentParser, BooleanOptionalAction

def _parse_csv_list(value):
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _resolve_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _ensure_bitsandbytes():
    if importlib.util.find_spec("bitsandbytes") is None:
        raise SystemExit(
            "bitsandbytes is not installed. Install it to use --bnb-4bit, then re-run."
        )


def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = args.device_map

    # Check if flash-attn2 flag is enabled and load model accordingly
    # Flash Attention is enabled by default in default parameters.
    # You can also apply 4 bit quantization using BitsAndBytes and have them both running
    if args.flash_attn2:
        try:
            common_kwargs = {
                "torch_dtype": "auto",
                "attn_implementation": "flash_attention_2",
                "device_map": device_map,
            }
            if args.bnb_4bit and not args.cpu_only:
                _ensure_bitsandbytes()
                skip_modules = _parse_csv_list(args.bnb_skip_modules)
                common_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=_resolve_dtype(args.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
                    llm_int8_skip_modules=skip_modules or None,
                )
            elif args.bnb_4bit and args.cpu_only:
                print("[bnb] --bnb-4bit ignored on CPU.")

            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                args.checkpoint_path, **common_kwargs
            )
        except Exception as exc:
            if sys.platform.startswith("win"):
                py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
                supported = {"cp310", "cp311", "cp312", "cp313"}
                suggested = ""
                if py_tag in supported:
                    suggested = (
                        f"Suggested wheel filename (if matching your CUDA/Torch):\n"
                        f"    flash_attn-2.8.3+cu128torch2.7.0cxx11abiFALSE-{py_tag}-{py_tag}-win_amd64.whl\n"
                    )
                hint = (
                    "flash_attention_2 was requested but is not available. "
                    "On Windows, install a prebuilt FlashAttention wheel that "
                    "matches your Python/CUDA/PyTorch. For example (Python 3.10, "
                    "CUDA 12.8, Torch 2.7):\n"
                    f"  wheel tag should include: {py_tag}-win_amd64\n"
                    "  example wheel name:\n"
                    "    flash_attn-2.8.3+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl\n"
                    f"{suggested}"
                    "  releases page:\n"
                    "    https://github.com/kingbri1/flash-attention/releases\n"
                    "If you don't want FlashAttention, re-run with --flash-attn2 disabled."
                )
            else:
                hint = (
                    "flash_attention_2 was requested but is not available. "
                    "Install FlashAttention2 for your platform (matching your CUDA/PyTorch), "
                    "or re-run with --flash-attn2 disabled."
                )
            raise RuntimeError(hint) from exc
    else:
        common_kwargs = {
            "torch_dtype": "auto",
            "device_map": device_map,
        }

        # Quantization Configuration is in default args at the bottom of this script if you're interested.  
        # default config is 197 thinker layers quantized as linear4bit. Audio tower, video thinker, etc are skipped.
        # (You can try experimental quantizations using command line arguments)
        if args.bnb_4bit and not args.cpu_only:
            _ensure_bitsandbytes()
            skip_modules = _parse_csv_list(args.bnb_skip_modules)
            common_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=_resolve_dtype(args.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
                llm_int8_skip_modules=skip_modules or None,
            )
        elif args.bnb_4bit and args.cpu_only:
            print("[bnb] --bnb-4bit ignored on CPU.")

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.checkpoint_path, **common_kwargs
        )

    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint_path)
    return model, processor

def _launch_demo(args, model, processor):
    # Disable the vestigial and cursed text output.
    # With bitsandbytes enabled, it can also cause a problem with this enabled.
    
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    # Voice settings
    VOICE_LIST = ['Chelsie', 'Ethan']
    DEFAULT_VOICE = 'Chelsie'

    default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
    default_task_prompt = "*Task* Transcribe this audio in detail\n<audio>"

    language = args.ui_language

    UI_TEXT = {
        "title": {"en": "Ace Step-Transcriber Demo", "zh": "Ace Step-Transcriber 演示"},
        "instructions_header": {"en": "🎯 Instructions for use:", "zh": "🎯 使用说明："},
        "instructions_1": {
            "en": "1️⃣ Click the Upload audio space or drop a song on the Audio space for transcribing.",
            "zh": "1️⃣ 点击上传音频区域或将歌曲拖放到音频区域进行转录。",
        },
        "instructions_2": {
            "en": "2️⃣ Click Submit and wait for the model's response.",
            "zh": "2️⃣ 点击提交并等待模型的回复。",
        },
        "system_prompt": {"en": "System Prompt", "zh": "系统提示词"},
        "task_prompt": {"en": "Task Prompt", "zh": "任务提示词"},
        "do_sample": {"en": "Do Sample", "zh": "采样生成"},
        "temperature": {"en": "Temperature", "zh": "温度"},
        "top_p": {"en": "Top-p", "zh": "Top-p"},
        "top_k": {"en": "Top-k", "zh": "Top-k"},
        "repetition_penalty": {"en": "Repetition Penalty", "zh": "重复惩罚"},
        "no_repeat_ngram": {"en": "No Repeat Ngram Size", "zh": "禁止重复 N-gram 长度"},
        "min_new_tokens": {"en": "Min New Tokens", "zh": "最小新 Token 数"},
        "num_beams": {"en": "Num Beams", "zh": "Beam 数"},
        "upload_audio": {"en": "Upload Audio", "zh": "上传音频"},
        "upload_image": {"en": "Upload Image", "zh": "上传图片"},
        "upload_video": {"en": "Upload Video", "zh": "上传视频"},
        "context_transcription": {"en": "Context transcription", "zh": "上下文转录"},
        "placeholder": {"en": "Enter text here...", "zh": "在此输入文本..."},
        "submit": {"en": "Submit", "zh": "提交"},
        "stop": {"en": "Stop", "zh": "停止"},
        "clear_history": {"en": "Clear History", "zh": "清除历史"},
        "voice_choice": {"en": "Voice Choice", "zh": "语音选择"},
        "offline_tab": {"en": "Offline", "zh": "离线"},
        "online_tab": {"en": "Online", "zh": "在线"},
    }

    def _normalize_language_tag(tag: str) -> str:
        if not tag:
            return ""
        lower = tag.strip().lower()
        if lower.startswith("zh"):
            return "zh"
        if lower.startswith("en"):
            return "en"
        return ""

    def _pick_language(browser_lang: str, request: gr.Request | None) -> str:
        # Prefer explicit browser/gradio language override
        lang = _normalize_language_tag(browser_lang)
        if not lang and request is not None:
            header = request.headers.get("accept-language", "")
            if header:
                lang = _normalize_language_tag(header.split(",")[0])
        if not lang:
            try:
                sys_lang = locale.getdefaultlocale()[0] or ""
            except Exception:
                sys_lang = ""
            lang = _normalize_language_tag(sys_lang)
        if not lang:
            lang = "en" if args.ui_language not in ("en", "zh") else args.ui_language
        return lang

    def _t(key: str, lang: str) -> str:
        entry = UI_TEXT.get(key, {})
        return entry.get(lang, entry.get("en", ""))

    def get_text(text: str, cn_text: str):
        if language == 'en':
            return text
        if language == 'zh':
            return cn_text
        return text
    
    def convert_webm_to_mp4(input_file, output_file):
        try:
            (
                ffmpeg
                .input(input_file)
                .output(output_file, acodec='aac', ar='16000', audio_bitrate='192k')
                .run(quiet=True, overwrite_output=True)
            )
            print(f"Conversion successful: {output_file}")
        except ffmpeg.Error as e:
            print("An error occurred during conversion.")
            print(e.stderr.decode('utf-8'))

    def format_history(history: list, system_prompt: str):
        messages = []
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item['role'], "content": item['content']})
            elif item["role"] == "user" and (isinstance(item["content"], list) or
                                            isinstance(item["content"], tuple)):
                file_path = item["content"][0]

                mime_type = client_utils.get_mimetype(file_path)
                if mime_type.startswith("image"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "image",
                            "image": file_path
                        }]
                    })
                elif mime_type.startswith("video"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "video",
                            "video": file_path
                        }]
                    })
                elif mime_type.startswith("audio"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "audio",
                            "audio": file_path,
                        }]
                    })
        return messages
    
    # Since we're getting all of the special tags from transformers now, we have to get the sections that we want.
    def _extract_assistant_text(decoded: str):
        if not decoded:
            return ""
        marker = "<|im_start|>assistant"
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
        end = "<|im_end|>"
        if end in decoded:
            decoded = decoded.split(end)[0]
        decoded = re.sub(r"<\|[^|]*\|>", "", decoded)
        return decoded.strip()

    # Odd output from the model.
    def _normalize_section_tags(text: str) -> str:
        if not text:
            return text
        replacements = {
            "ch-chorus": "chorus",
            "choruses": "chorus",
            "pre chorus": "pre-chorus",
            "post chorus": "post-chorus",
        }

        def repl(match):
            tag = match.group(1).strip().lower()
            tag = replacements.get(tag, tag)
            # Title-case known tags
            if tag.startswith("verse"):
                tag = "Verse" + tag[5:]
            elif tag == "chorus":
                tag = "Chorus"
            elif tag == "pre-chorus":
                tag = "Pre-Chorus"
            elif tag == "post-chorus":
                tag = "Post-Chorus"
            elif tag == "bridge":
                tag = "Bridge"
            elif tag == "intro":
                tag = "Intro"
            elif tag == "outro":
                tag = "Outro"
            elif tag == "acapella breakdown":
                tag = "Acapella Breakdown"
            return f"[{tag}]"
        return re.sub(r"\[([^\[\]]+)\]", repl, text)
    
    # deal with odd output from the model occasionally.
    def _clean_output_text(text: str) -> str:
        if not text:
            return text
        # Normalize common non-ASCII punctuation to ASCII
        text = text.replace("，", ",").replace("。", ".").replace("؛", ";").replace("،", ",")
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        # Remove replacement characters
        text = text.replace("�", "")
        # Fix camel-case glitches (e.g., GotSniped -> Got sniped)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Collapse extra spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Fix spacing around punctuation
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)
        # Normalize section tags
        text = _normalize_section_tags(text)
        # Normalize stray bracket spacing
        text = re.sub(r"\[\s+", "[", text)
        text = re.sub(r"\s+\]", "]", text)
        return text.strip()

    def predict(
        messages,
        voice=DEFAULT_VOICE,
        do_sample=False,
        temperature=0.3,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        min_new_tokens=0,
        num_beams=1,
    ):
        print('predict history: ', messages)    

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(model.device).to(model.dtype)

        gen_kwargs = {
            "speaker": voice,
            "use_audio_in_video": True,
            "thinker_do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "thinker_temperature": temperature,
                    "thinker_top_p": top_p,
                    "thinker_top_k": top_k,
                }
            )
        if repetition_penalty and repetition_penalty != 1.0:
            gen_kwargs["thinker_repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["thinker_no_repeat_ngram_size"] = no_repeat_ngram_size
        if min_new_tokens and min_new_tokens > 0:
            gen_kwargs["thinker_min_new_tokens"] = min_new_tokens
        if num_beams and num_beams > 1:
            gen_kwargs["thinker_num_beams"] = num_beams

        gen_out = model.generate(**inputs, **gen_kwargs)
        if isinstance(gen_out, (tuple, list)) and len(gen_out) == 2:
            text_ids, audio = gen_out
        else:
            text_ids = gen_out
            audio = None

        response = processor.batch_decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        response = _extract_assistant_text(response[0])
        response = _clean_output_text(response)
        yield {"type": "text", "data": response}

        if audio is not None:
            audio = np.array(audio * 32767).astype(np.int16)
            wav_io = io.BytesIO()
            sf.write(wav_io, audio, samplerate=24000, format="WAV")
            wav_io.seek(0)
            wav_bytes = wav_io.getvalue()
            audio_path = processing_utils.save_bytes_to_cache(
                wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
            yield {"type": "audio", "data": audio_path}

    def media_predict(
        audio,
        video,
        history,
        system_prompt,
        task_prompt,
        voice_choice,
        do_sample,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        no_repeat_ngram_size,
        min_new_tokens,
        num_beams,
    ):
        # Always start with a fresh context on submit
        history = []
        # First yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        if video is not None:
            convert_webm_to_mp4(video, video.replace('.webm', '.mp4'))
            video = video.replace(".webm", ".mp4")
        files = [audio, video]

        for f in files:
            if f:
                if task_prompt:
                    history.append({"role": "user", "content": task_prompt})
                history.append({"role": "user", "content": (f, )})

        formatted_history = format_history(history=history,
                                        system_prompt=system_prompt,)


        history.append({"role": "assistant", "content": ""})

        for chunk in predict(
            formatted_history,
            voice_choice,
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            min_new_tokens,
            num_beams,
        ):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    None,  # webcam
                    history,  # media_chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })

        # Final yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    def chat_predict(
        text,
        audio,
        image,
        video,
        history,
        system_prompt,
        task_prompt,
        voice_choice,
        do_sample,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        no_repeat_ngram_size,
        min_new_tokens,
        num_beams,
    ):
        # Always start with a fresh context on submit
        history = []
        # Process text input
        if text:
            history.append({"role": "user", "content": text})

        # Process audio input
        if audio:
            if task_prompt:
                history.append({"role": "user", "content": task_prompt})
            history.append({"role": "user", "content": (audio, )})

        # Process image input
        if image:
            history.append({"role": "user", "content": (image, )})

        # Process video input
        if video:
            history.append({"role": "user", "content": (video, )})

        formatted_history = format_history(history=history,
                                        system_prompt=system_prompt)

        yield None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        for chunk in predict(
            formatted_history,
            voice_choice,
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            min_new_tokens,
            num_beams,
        ):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(
                ), history
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history

    with gr.Blocks() as demo, ms.Application(), antd.ConfigProvider():
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label="System Prompt",
                                            value=default_system_prompt)
            task_prompt_textbox = gr.Textbox(label="Task Prompt",
                                            value=default_task_prompt,
                                            lines=3)
            do_sample_chk = gr.Checkbox(label="Do Sample", value=True)
            temperature_slider = gr.Slider(label="Temperature",
                                        minimum=0.0,
                                        maximum=1.5,
                                        step=0.05,
                                        value=0.7)
            top_p_slider = gr.Slider(label="Top-p",
                                    minimum=0.05,
                                    maximum=1.0,
                                    step=0.05,
                                    value=1.0)
            top_k_slider = gr.Slider(label="Top-k",
                                    minimum=0,
                                    maximum=200,
                                    step=1,
                                    value=30)
            repetition_penalty_slider = gr.Slider(label="Repetition Penalty",
                                                minimum=1.0,
                                                maximum=1.5,
                                                step=0.01,
                                                value=1.2)
            no_repeat_ngram_slider = gr.Slider(label="No Repeat Ngram Size",
                                            minimum=0,
                                            maximum=10,
                                            step=1,
                                            value=0)
            min_new_tokens_slider = gr.Slider(label="Min New Tokens",
                                            minimum=0,
                                            maximum=2048,
                                            step=1,
                                            value=0)
            num_beams_slider = gr.Slider(label="Num Beams",
                                        minimum=1,
                                        maximum=10,
                                        step=1,
                                        value=1)
        with antd.Flex(gap="small", justify="center", align="center"):
            with antd.Flex(vertical=True, gap="small", align="center"):
                title_text = antd.Typography.Title("Ace Step-Transcriber Demo",
                                    level=1,
                                    elem_style=dict(margin=0, fontSize=28))
                with antd.Flex(vertical=True, gap="small"):
                    instruction_header = antd.Typography.Text(
                        get_text("🎯 Instructions for use:",
                                 "🎯 使用说明："),
                        strong=True)
                    instruction_line_1 = antd.Typography.Text(
                        get_text(
                            "1️⃣ Click the Upload audio space or drop a song on the Audio space for transcribing.",
                            "1️⃣ 点击上传音频区域或将歌曲拖放到音频区域进行转录。"))
                    instruction_line_2 = antd.Typography.Text(
                        get_text(
                            "2️⃣ Click Submit and wait for the model's response.",
                            "2️⃣ 点击提交并等待模型的回答。"))
        voice_choice = gr.Dropdown(label="Voice Choice",
                                choices=VOICE_LIST,
                                value=DEFAULT_VOICE,
                                visible=False)
        with gr.Tabs():


            with gr.Tab("Offline"):
                # Media upload section in one row (top)
                with gr.Row(equal_height=True):
                    audio_input = gr.Audio(sources=["upload"],
                                        type="filepath",
                                        label="Upload Audio",
                                        elem_classes="media-upload",
                                        scale=1)
                    image_input = gr.Image(sources=["upload"],
                                        type="filepath",
                                        label="Upload Image",
                                        elem_classes="media-upload",
                                        scale=1,
                                        visible=False)
                    video_input = gr.Video(sources=["upload"],
                                        label="Upload Video",
                                        elem_classes="media-upload",
                                        scale=1,
                                        visible=False)

                chatbot = gr.Chatbot(type="messages", height=650, label="Context transcription")

                # Text input section
                text_input = gr.Textbox(show_label=False,
                                        placeholder="Enter text here...",
                                        elem_classes="hide-text-input")

                # Control buttons
                with gr.Row():
                    submit_btn = gr.Button(get_text("Submit", "提交"),
                                        variant="primary",
                                        size="lg")
                    stop_btn = gr.Button(get_text("Stop", "停止"),
                                        visible=False,
                                        size="lg")
                    clear_btn = gr.Button(get_text("Clear History", "清除历史"),
                                        size="lg")
                    submit_btn_offline = submit_btn
                    stop_btn_offline = stop_btn
                    clear_btn_offline = clear_btn

                def clear_chat_history():
                    return [], gr.update(value=None), gr.update(
                        value=None), gr.update(value=None), gr.update(value=None)

                submit_event = gr.on(
                    triggers=[submit_btn.click, text_input.submit],
                    fn=chat_predict,
                    inputs=[
                        text_input, audio_input, image_input, video_input, chatbot,
                        system_prompt_textbox,
                        task_prompt_textbox,
                        voice_choice,
                        do_sample_chk,
                        temperature_slider,
                        top_p_slider,
                        top_k_slider,
                        repetition_penalty_slider,
                        no_repeat_ngram_slider,
                        min_new_tokens_slider,
                        num_beams_slider,
                    ],
                    outputs=[
                        text_input, audio_input, image_input, video_input, chatbot
                    ])

                stop_btn.click(fn=lambda:
                            (gr.update(visible=True), gr.update(visible=False)),
                            inputs=None,
                            outputs=[submit_btn, stop_btn],
                            cancels=[submit_event],
                            queue=False)

                clear_btn.click(fn=clear_chat_history,
                                inputs=None,
                                outputs=[
                                    chatbot, text_input, audio_input, image_input,
                                    video_input
                                ])

                # Add some custom CSS to improve the layout
                gr.HTML("""
                    <style>
                        .media-upload {
                            margin: 10px;
                            min-height: 160px;
                        }
                        .media-upload > .wrap {
                            border: 2px dashed #ccc;
                            border-radius: 8px;
                            padding: 10px;
                            height: 100%;
                        }
                        .media-upload:hover > .wrap {
                            border-color: #666;
                        }
                        /* Make upload areas equal width */
                        .media-upload {
                            flex: 1;
                            min-width: 0;
                        }
                        .hide-text-input {
                            display: none !important;
                        }
                    </style>
                """)
            # Online is left over from the orgiginal demo.   I didn't see a reason to remove it completely.
            with gr.Tab("Online", visible=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        microphone = gr.Audio(sources=['microphone'],
                                            type="filepath")
                        webcam = gr.Video(sources=['webcam'],
                                        height=400,
                                        include_audio=True)
                        submit_btn = gr.Button(get_text("Submit", "提交"),
                                            variant="primary")
                        stop_btn = gr.Button(get_text("Stop", "停止"), visible=False)
                        clear_btn = gr.Button(get_text("Clear History", "清除历史"))
                        submit_btn_online = submit_btn
                        stop_btn_online = stop_btn
                        clear_btn_online = clear_btn
                    with gr.Column(scale=2):
                        media_chatbot = gr.Chatbot(height=650, type="messages", label="Context transcription")

                    def clear_history():
                        return [], gr.update(value=None), gr.update(value=None)

                    submit_event = submit_btn.click(fn=media_predict,
                                                    inputs=[
                                                        microphone, webcam,
                                                        media_chatbot,
                                                        system_prompt_textbox,
                                                        task_prompt_textbox,
                                                        voice_choice,
                                                        do_sample_chk,
                                                        temperature_slider,
                                                        top_p_slider,
                                                        top_k_slider,
                                                    repetition_penalty_slider,
                                                    no_repeat_ngram_slider,
                                                    min_new_tokens_slider,
                                                    num_beams_slider,
                                                    ],
                                                    outputs=[
                                                        microphone, webcam,
                                                        media_chatbot, submit_btn,
                                                        stop_btn
                                                    ])
                    stop_btn.click(
                        fn=lambda:
                        (gr.update(visible=True), gr.update(visible=False)),
                        inputs=None,
                        outputs=[submit_btn, stop_btn],
                        cancels=[submit_event],
                        queue=False)
                    clear_btn.click(fn=clear_history,
                                    inputs=None,
                                    outputs=[media_chatbot, microphone, webcam])

        # Language auto-detect + gradio language override (when available)
        ui_language_probe = gr.Textbox(visible=False)

        def _apply_language(browser_lang: str, request: gr.Request):
            lang = _pick_language(browser_lang, request)
            return (
                gr.update(value=_t("title", lang)),
                gr.update(value=_t("instructions_header", lang)),
                gr.update(value=_t("instructions_1", lang)),
                gr.update(value=_t("instructions_2", lang)),
                gr.update(label=_t("system_prompt", lang)),
                gr.update(label=_t("task_prompt", lang)),
                gr.update(label=_t("do_sample", lang)),
                gr.update(label=_t("temperature", lang)),
                gr.update(label=_t("top_p", lang)),
                gr.update(label=_t("top_k", lang)),
                gr.update(label=_t("repetition_penalty", lang)),
                gr.update(label=_t("no_repeat_ngram", lang)),
                gr.update(label=_t("min_new_tokens", lang)),
                gr.update(label=_t("num_beams", lang)),
                gr.update(label=_t("upload_audio", lang)),
                gr.update(label=_t("upload_image", lang)),
                gr.update(label=_t("upload_video", lang)),
                gr.update(label=_t("context_transcription", lang)),
                gr.update(label=_t("context_transcription", lang)),
                gr.update(placeholder=_t("placeholder", lang)),
                gr.update(value=_t("submit", lang)),
                gr.update(value=_t("stop", lang)),
                gr.update(value=_t("clear_history", lang)),
                gr.update(value=_t("submit", lang)),
                gr.update(value=_t("stop", lang)),
                gr.update(value=_t("clear_history", lang)),
                gr.update(label=_t("voice_choice", lang)),
            )
        # Localization labels
        demo.load(
            fn=_apply_language,
            inputs=[ui_language_probe],
            outputs=[
                title_text,
                instruction_header,
                instruction_line_1,
                instruction_line_2,
                system_prompt_textbox,
                task_prompt_textbox,
                do_sample_chk,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                repetition_penalty_slider,
                no_repeat_ngram_slider,
                min_new_tokens_slider,
                num_beams_slider,
                audio_input,
                image_input,
                video_input,
                chatbot,
                media_chatbot,
                text_input,
                submit_btn_offline,
                stop_btn_offline,
                clear_btn_offline,
                submit_btn_online,
                stop_btn_online,
                clear_btn_online,
                voice_choice,
            ],
            js=( #Attempt to capture localization settings
                "() => (window.gradio_config && ("
                "window.gradio_config.language || "
                "(window.gradio_config.i18n && window.gradio_config.i18n.lang)"
                ")) || navigator.language || ''"
            ),
        )

    demo.queue(default_concurrency_limit=100, max_size=100).launch(max_threads=100,
                                                                ssr_mode=False,
                                                                share=args.share,
                                                                inbrowser=args.inbrowser,
                                                                server_port=args.server_port,
                                                                server_name=args.server_name,)


DEFAULT_CKPT_PATH = "acestep-transcriber"
def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')
    parser.add_argument('--device-map',
                        type=str,
                        default='auto',
                        help="Device map for model loading (e.g., 'cuda:0').")

    parser.add_argument('--flash-attn2',
                        action=BooleanOptionalAction,
                        default=True,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument('--ui-language', type=str, choices=['en', 'zh'], default='en', help='Display language for the UI.')
    
    # 4-bit Quantize Config
    parser.add_argument('--bnb-4bit', action='store_true', help='Enable bitsandbytes 4-bit quantization.')
    parser.add_argument(
        '--bnb-4bit-quant-type',
        type=str,
        default='nf4',
        choices=['nf4', 'fp4'],
        help='4-bit quantization type for bitsandbytes.',
    )
    parser.add_argument(
        '--bnb-4bit-compute-dtype',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'],
        help='Compute dtype for 4-bit matmuls.',
    )
    parser.add_argument(
        '--bnb-4bit-double-quant',
        action=BooleanOptionalAction,
        default=True,
        help='Enable nested (double) quantization for 4-bit weights.',
    )
    # Quantize the thinker transformer stack only. leave the audio and video layers as is.
    parser.add_argument(
        '--bnb-skip-modules',
        type=str,
        default='thinker.visual,thinker.audio_tower,token2wav,talker',
        help='Comma-separated module names to keep in fp16/bf16.',
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)
