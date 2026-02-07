import io
import os
import re
import sys
import ffmpeg

import numpy as np
import gradio as gr
import soundfile as sf 

import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info
from argparse import ArgumentParser

def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = args.device_map

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
        try:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                args.checkpoint_path,
                torch_dtype='auto',
                attn_implementation='flash_attention_2',
                device_map=device_map,
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
                    "If you don't want FlashAttention, re-run with --flash-attn2 disabled (but it will probably crash with out of memory)."
                )
            else:
                hint = (
                    "flash_attention_2 was requested but is not available. "
                    "Install FlashAttention2 for your platform (matching your CUDA/PyTorch), "
                    "or re-run with --flash-attn2 disabled.(but it will probably crash with out of memory)"
                )
            raise RuntimeError(hint) from exc
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.checkpoint_path,
            device_map=device_map,
            torch_dtype='auto',
        )

    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint_path)
    return model, processor

def _launch_demo(args, model, processor):
    # Voice settings
    VOICE_LIST = ['Chelsie', 'Ethan']
    DEFAULT_VOICE = 'Chelsie'

    default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
    default_task_prompt = "*Task* Transcribe this audio in detail\n<audio>"

    language = args.ui_language

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

    # Because we are telling Transformers to send us all the things, we have to remove the audio tags and various others
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

    # Some anomalous output from the model
    def _normalize_section_tags(text: str) -> str:
        if not text:
            return text
        replacements = {
            "ch-chorus": "chorus",
            "choruses": "chorus",
            "pre chorus": "pre-chorus",
            "post chorus": "post-chorus",
        }

        # more anomalous output
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

    # Yes, even more anomalous output.
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
        do_sample=True,
        temperature=0.7,
        top_p=1,
        top_k=30,
        repetition_penalty=1.2,
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

        text_ids, audio = model.generate(**inputs, **gen_kwargs)

        # At first, it seemed that there was a beam search problem causing
        # early termination but it turned out that transformers was hiding most of the output
        response = processor.batch_decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        response = _extract_assistant_text(response[0])
        response = _clean_output_text(response)
        yield {"type": "text", "data": response}

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
        # Always start with a fresh context on submit otherwise the model will mix-in the output from the last inference
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

            # Added these so that users can experiment. The model is surprisingly stable in its outputs though.
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
                antd.Typography.Title("Ace Step-Transcriber Demo",
                                    level=1,
                                    elem_style=dict(margin=0, fontSize=28))
                with antd.Flex(vertical=True, gap="small"):
                    antd.Typography.Text(
                        get_text("🎯 Instructions for use:",
                                 "🎯 使用说明："),
                        strong=True)
                    antd.Typography.Text(
                        get_text(
                            "1️⃣ Click the Upload audio space or drop a song on the Audio space for transcribing.",
                            "1️⃣ 点击上传音频区域或将歌曲拖放到音频区域进行转录。"))
                    antd.Typography.Text(
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

                # Add some custom CSS to improve the layout since we do not need them for this demo
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
            # We don't need this so hide it.
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
                        action='store_true',
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)
