#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parakeet TDT 0.6B v2 语音识别 worker —— **Intel NPU (OpenVINO)** 版。

与 asr.py(sherpa-onnx / CUDA) 并列的第二个 Parakeet 后端, 供「创建初稿」和
「选区重新识别」调用。二者**命令行契约完全一致**, server.js 只需按引擎换 provider。

用法:
    python asr_npu.py --model <模型目录> --audio <16kHz单声道wav> --out <结果.json>
                      [--provider npu|gpu|cpu] [--threads N]

模型目录需含(取自 istupakov/parakeet-tdt-0.6b-v2-onnx):
    encoder-model.onnx (+ encoder-model.onnx.data)  FastConformer 编码器
    decoder_joint-model.onnx                        LSTM 预测 + Joint 网络
    vocab.txt                                       SentencePiece 词表(1025 行, blank=1024)

运行期往 stderr 输出 JSON 行(每行一个 JSON 对象), 供 Node 端增量解析:
    {"type":"log",     "msg":"..."}
    {"type":"progress","pct":42,"stage":"asr","msg":"识别中 3/10 块"}
    {"type":"error",   "msg":"..."}

成功时把结果写到 --out:
    {"duration":123.4, "language":"en", "segments":[{"start":..,"end":..,"text":..,
     "words":[{"word":..,"start":..,"end":..}]}]}

退出码 != 0 表示失败(错误信息在 stderr 的 error 行里)。

为什么需要这个后端: asr.py 走 sherpa-onnx 的 CUDA provider, 官方明确不支持 CPU,
没有 N 卡的机器(含只有 Intel 核显 / NPU 的笔记本)完全用不了 Parakeet。本文件用
OpenVINO 把同一份权重跑在 **Intel NPU** 上, 预测/联合网络退到核显(该图在 NPU 上
编译不过), 于是没有独显也能拿到逐词时间戳。

注意: parakeet-tdt-0.6b-v2 **仅支持英语**。
"""

from __future__ import annotations

import argparse
import array
import json
import os
import sys
import time
import wave

# 本文件所在目录即 asr/(与 asr.py 同级) —— 复用它已验证的音频读取, 不重复实现
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE_RATE = 16000
MEL_BINS = 128
BLANK_TOKEN_ID = 1024
DURATION_BINS = (0, 1, 2, 3, 4)
WORD_BOUNDARY = "\u2581"

# 编码器输入宽度固定为 1501 帧 mel(实测 100 帧/秒 => 15.01 秒); 更长音频按此窗口推进,
# 窗口之间重叠一部分, 并把 LSTM 状态与已解码 token 带过去, 保证跨窗连贯。
ENCODER_WINDOW_FRAMES = 1501
OVERLAP_FRAMES = 166
ENCODER_FRAME_RATE = 12.5          # 编码器帧率 = 100 / 下采样 8

# ── mel 前端参数(NeMo / parakeet 规范) ──
N_FFT = 512
WIN_LENGTH = 400                   # 25 ms
HOP_LENGTH = 160                   # 10 ms
LOG_GUARD = 2.0 ** -24
NORM_EPS = 1e-5
NORM_MAX = 4.0

# 基础断句阈值(与 asr.py 保持一致, 纯本地规则, 不涉及 LLM)
SENT_END = (".", "?", "!", "。", "？", "！", "…")
SOFT_END = (",", ";", ":")
PAUSE_SPLIT = 0.8
MAX_LINE_SEC = 10.0
MAX_LINE_WORDS = 30


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def emit(obj):
    try:
        sys.stderr.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def log(msg):
    emit({"type": "log", "msg": str(msg)})


def progress(pct, stage, msg):
    emit({"type": "progress", "pct": max(0, min(100, int(pct))), "stage": stage, "msg": str(msg)})


# --------------------------------------------------------------------------
# 音频
# --------------------------------------------------------------------------
def read_wav_mono16k(path):
    """读 wav -> (float32 数组[-1,1], 采样率)。优先复用 asr.py 的实现, 保证两个后端
    对同一份音频的读法完全一致(重采样/降混音的行为不出现分叉)。"""
    try:
        from asr import read_wav_mono16k as impl  # asr.py 与本文件同级
        return impl(path)
    except Exception:
        pass

    import numpy as np
    with wave.open(path, "rb") as wf:
        nch, sr, width = wf.getnchannels(), wf.getframerate(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise RuntimeError("只支持 16-bit PCM wav(当前 %d 字节/样本)" % width)
    a = array.array("h")
    a.frombytes(raw)
    data = np.asarray(a, dtype=np.float32) / 32768.0
    if nch > 1:
        usable = (len(data) // nch) * nch
        data = data[:usable].reshape(-1, nch).mean(axis=1)
    if sr != SAMPLE_RATE and len(data):
        new_n = max(1, int(round(len(data) * SAMPLE_RATE / float(sr))))
        data = np.interp(np.linspace(0, len(data) - 1, num=new_n, dtype=np.float32),
                         np.arange(len(data), dtype=np.float32), data).astype(np.float32)
        sr = SAMPLE_RATE
    return data, sr


# --------------------------------------------------------------------------
# log-mel 前端(纯 NumPy, 无第三方音频依赖)
# --------------------------------------------------------------------------
def _hz_to_mel(freq):
    import numpy as np
    freq = np.asarray(freq, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mel = freq / f_sp
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(freq >= min_log_hz,
                    min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) / logstep, mel)


def _mel_to_hz(mel):
    import numpy as np
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3.0
    freqs = f_sp * mel
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(mel >= min_log_mel, min_log_hz * np.exp(logstep * (mel - min_log_mel)), freqs)


class MelFrontend:
    """128 维 Slaney 归一化 mel + log + 逐 bin 归一化, 与 NeMo/parakeet 前端对齐。"""

    def __init__(self, n_mels=MEL_BINS):
        import numpy as np
        self.n_mels = n_mels
        self._window = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float64)

        fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
        pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2.0), n_mels + 2))
        ramps = pts[:, None] - fft_freqs[None, :]
        fb = np.zeros((n_mels, fft_freqs.size), dtype=np.float64)
        for i in range(n_mels):
            lower = -ramps[i] / (pts[i + 1] - pts[i])
            upper = ramps[i + 2] / (pts[i + 2] - pts[i + 1])
            fb[i] = np.maximum(0.0, np.minimum(lower, upper))
        enorm = 2.0 / (pts[2:n_mels + 2] - pts[:n_mels])
        self._fb = fb * enorm[:, None]

    def log_mel(self, samples, normalize=True):
        import numpy as np
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if x.size < WIN_LENGTH:
            x = np.pad(x, (0, WIN_LENGTH - x.size))
        n_frames = 1 + (x.size - WIN_LENGTH) // HOP_LENGTH
        idx = np.arange(WIN_LENGTH)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
        spec = np.fft.rfft(x[idx] * self._window[None, :], n=N_FFT, axis=1)
        power = spec.real ** 2 + spec.imag ** 2
        feat = np.log(power @ self._fb.T + LOG_GUARD).T          # [mel, frames]
        if not normalize:
            return feat.astype(np.float32)
        mean = feat.mean(axis=1, keepdims=True)
        std = feat.std(axis=1, keepdims=True)
        normed = (feat - mean) / (std + NORM_EPS)
        return (np.clip(normed, -NORM_MAX, NORM_MAX)).astype(np.float32)


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
PROVIDER_DEVICES = {
    # 编码器: NPU 最优; 预测/联合网络在 NPU 上编译不过(Level Zero 报错), 退核显
    "npu": ("NPU", "GPU"),
    "gpu": ("GPU", "GPU"),
    "cpu": ("CPU", "CPU"),
}


class NpuRecognizer:
    """OpenVINO 版 parakeet TDT: 编码器 + LSTM 预测 + Joint, TDT 贪心解码。"""

    def __init__(self, model_dir, provider="npu"):
        import numpy as np  # noqa: F401  提前确认 numpy 可导入, 报错更清楚
        import openvino as ov

        self.ov = ov
        self.model_dir = model_dir
        self.mel = MelFrontend()
        want_enc, want_joint = PROVIDER_DEVICES.get(provider, PROVIDER_DEVICES["npu"])

        core = ov.Core()
        # NPU 编译一次要几分钟 —— 打开 blob 缓存, 之后启动只要几秒
        cache_dir = os.path.join(model_dir, ".ov_cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            core.set_property({"CACHE_DIR": os.path.abspath(cache_dir)})
        except Exception as exc:  # noqa: BLE001
            log("编译缓存不可用(忽略): %s" % exc)

        avail = list(core.available_devices)
        log("OpenVINO %s, 可用设备: %s" % (ov.__version__, ", ".join(avail) or "(无)"))

        enc_path = self._pick("encoder-model.onnx", "encoder*.onnx")
        joint_path = self._pick("decoder_joint-model.onnx", "decoder_joint*.onnx")

        t0 = time.time()
        model = core.read_model(enc_path)
        try:
            model.reshape({"audio_signal": [1, MEL_BINS, ENCODER_WINDOW_FRAMES], "length": [1]})
        except Exception as exc:  # noqa: BLE001
            log("编码器静态化跳过: %s" % exc)
        self.encoder, self.enc_device = self._compile(core, model, "编码器", want_enc)

        jm = core.read_model(joint_path)
        try:
            jm.reshape({"encoder_outputs": [1, 1024, 188], "targets": [1, 1], "target_length": [1],
                        "input_states_1": [2, 1, 640], "input_states_2": [2, 1, 640]})
        except Exception as exc:  # noqa: BLE001
            log("预测/联合网络静态化跳过: %s" % exc)
        self.joint, self.joint_device = self._compile(core, jm, "预测/联合网络", want_joint)

        self.enc_req = self.encoder.create_infer_request()
        self.joint_req = self.joint.create_infer_request()

        shape = self.encoder.outputs[0].get_partial_shape()
        dims = [int(shape[i].get_length()) if shape[i].is_static else 0 for i in (1, 2)]
        self.enc_hidden = max(dims) or 1024
        self.dec_hidden = 640
        p = self.joint.input("input_states_1").get_partial_shape()
        if p.rank.is_static and p.rank.get_length() == 3 and p[2].is_static:
            self.dec_hidden = int(p[2].get_length())

        self.vocab = self._load_vocab()
        log("模型加载完成(编码器=%s, 预测/联合=%s), 耗时 %.1fs" % (self.enc_device, self.joint_device, time.time() - t0))

    # ---------------------------------------------------------------- 内部
    def _pick(self, *patterns):
        import glob
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(self.model_dir, pat)))
            hits = [h for h in hits if not h.endswith(".data")]
            if hits:
                return hits[0]
        raise RuntimeError("模型目录缺少 %s: %s" % (patterns[0], self.model_dir))

    def _compile(self, core, model, label, want):
        candidates = [want] + [d for d in ("GPU.0", "GPU", "CPU") if d != want]
        last = None
        for dev in candidates:
            try:
                compiled = core.compile_model(model, dev)
                if dev != want:
                    log("%s 在 %s 上不可用, 已回退到 %s" % (label, want, dev))
                return compiled, dev
            except Exception as exc:  # noqa: BLE001
                last = exc
                log("%s 在 %s 编译失败: %s" % (label, dev, str(exc)[:140]))
        raise RuntimeError("%s 无法编译(尝试过 %s): %s" % (label, candidates, last))

    def _load_vocab(self):
        path = os.path.join(self.model_dir, "vocab.txt")
        if not os.path.exists(path):
            raise RuntimeError("模型目录缺少 vocab.txt: %s" % self.model_dir)
        pieces = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                head, _, tail = line.rpartition(" ")
                pieces.append(head if tail.isdigit() and head else line)
        return pieces

    def _piece(self, tid):
        return self.vocab[tid] if 0 <= tid < len(self.vocab) else ""

    def is_punctuation(self, tid):
        piece = self._piece(tid)
        if piece.startswith(WORD_BOUNDARY):
            piece = piece[len(WORD_BOUNDARY):]
        return piece in (".", "?", "!")

    # ------------------------------------------------------------ 单窗推理
    def _encode(self, window, valid_frames):
        import numpy as np
        signal = np.ascontiguousarray(window.reshape(1, MEL_BINS, -1), dtype=np.float32)
        result = self.enc_req.infer({
            "audio_signal": signal,
            "length": np.array([valid_frames], dtype=np.int64),
        })
        out = np.asarray(result["outputs"], dtype=np.float32)[0].T        # [T, 1024]
        out_len = int(np.asarray(result["encoded_lengths"]).reshape(-1)[0])
        return np.ascontiguousarray(out), max(0, min(out_len, out.shape[0]))

    def _joint_step(self, token, hidden, cell):
        import numpy as np
        req = self.joint_req
        req.set_input_tensor(1, self.ov.Tensor(np.array([[token]], dtype=np.int32)))
        req.set_input_tensor(2, self.ov.Tensor(np.array([1], dtype=np.int32)))
        req.set_input_tensor(3, self.ov.Tensor(np.ascontiguousarray(hidden, dtype=np.float32)))
        req.set_input_tensor(4, self.ov.Tensor(np.ascontiguousarray(cell, dtype=np.float32)))
        req.infer()
        return (req.get_output_tensor(0).data,
                req.get_output_tensor(2).data,
                req.get_output_tensor(3).data)

    def _tdt_loop(self, acts, valid, state, is_last):
        """TDT 贪心解码一个窗口; 就地推进 state(LSTM 状态 + 最后 token)。"""
        import numpy as np
        tokens, timings = [], []
        if valid <= 0:
            return tokens, timings

        enc_out = np.ascontiguousarray(acts.T.reshape(1, self.enc_hidden, -1), dtype=np.float32)
        self.joint_req.set_input_tensor(0, self.ov.Tensor(enc_out))
        vocab = BLANK_TOKEN_ID + 1
        last_token = state["last_token"]
        frame = 0

        def step(token, hidden, cell, at):
            logits_all, h_out, c_out = self._joint_step(token, hidden, cell)
            flat = np.asarray(logits_all, dtype=np.float32).reshape(-1, 1030)
            return flat[at], h_out, c_out

        while frame < valid:
            logits, h_next, c_next = step(last_token, state["hidden"], state["cell"], frame)
            head = logits[:vocab]
            best = int(np.argmax(head))
            dur = DURATION_BINS[int(np.argmax(logits[vocab:vocab + len(DURATION_BINS)]))] or 1
            if best == BLANK_TOKEN_ID:
                frame = min(frame + dur, valid)
                continue
            tokens.append(best)
            timings.append({"id": best, "frame": frame})
            last_token = best
            state["hidden"] = np.ascontiguousarray(h_next, dtype=np.float32)
            state["cell"] = np.ascontiguousarray(c_next, dtype=np.float32)
            state["last_token"] = best
            frame = min(frame + dur, valid)

        if is_last and tokens:
            last_frame = max(0, valid - 1)
            for _ in range(8):
                logits, h_next, c_next = step(last_token, state["hidden"], state["cell"], last_frame)
                head = logits[:vocab]
                best = int(np.argmax(head))
                if best == BLANK_TOKEN_ID:
                    break
                tokens.append(best)
                timings.append({"id": best, "frame": last_frame})
                last_token = best
                state["hidden"] = np.ascontiguousarray(h_next, dtype=np.float32)
                state["cell"] = np.ascontiguousarray(c_next, dtype=np.float32)
                state["last_token"] = best
        return tokens, timings

    # ------------------------------------------------------------ 词级时间轴
    def recognize_words(self, samples, on_progress=None):
        """整段音频 -> [{word, start, anchor}]。

        与 asr.py 的输出现约一致: 标点 token 的时间戳常落在停顿里, 因此额外给出
        ``anchor``(本词最后一个含字母数字 token 的时间), 由调用方据此推算真实结束时间。
        """
        import numpy as np
        mel = self.mel.log_mel(samples, normalize=True)
        frames = mel.shape[1]
        if frames <= 0:
            return []

        stride = ENCODER_WINDOW_FRAMES - OVERLAP_FRAMES
        total_windows = 1 if frames <= ENCODER_WINDOW_FRAMES else \
            1 + (frames - ENCODER_WINDOW_FRAMES + stride - 1) // stride

        state = {
            "hidden": np.zeros((2, 1, self.dec_hidden), dtype=np.float32),
            "cell": np.zeros((2, 1, self.dec_hidden), dtype=np.float32),
            "last_token": BLANK_TOKEN_ID,
        }
        all_tokens, all_timings = [], []
        offset, win_idx = 0, 0
        last_global, have_last = 0, False

        while offset < frames:
            size = min(ENCODER_WINDOW_FRAMES, frames - offset)
            is_last = (offset + size) >= frames
            chunk = mel[:, offset:offset + size]
            window = np.zeros((MEL_BINS, ENCODER_WINDOW_FRAMES), dtype=np.float32)
            window[:, :size] = chunk

            acts, valid = self._encode(window, size)
            tokens, timings = self._tdt_loop(acts, valid, state, is_last)
            for tg in timings:
                tg["frame"] += offset

            if win_idx == 0:
                all_tokens, all_timings = list(tokens), list(timings)
            else:
                # 窗口重叠区会重复解码 —— 用「时间门 + 标点去重 + 最长后缀/前缀重叠」压掉
                skip = 0
                if have_last and timings:
                    while skip < len(timings) and timings[skip]["frame"] <= last_global:
                        skip += 1
                if (skip < len(tokens) and all_tokens
                        and tokens[skip] == all_tokens[-1] and self.is_punctuation(tokens[skip])):
                    skip += 1
                window_n = min(15, len(all_tokens), len(tokens) - skip)
                for length in range(window_n, 1, -1):
                    if all_tokens[-length:] == tokens[skip:skip + length]:
                        skip += length
                        break
                emit_end = len(tokens)
                if not is_last and timings:
                    boundary = size - OVERLAP_FRAMES if size > OVERLAP_FRAMES else 0
                    for idx in range(len(timings)):
                        if timings[idx]["frame"] - offset >= boundary:
                            emit_end = min(emit_end, idx)
                            break
                if skip < emit_end:
                    all_tokens.extend(tokens[skip:emit_end])
                    all_timings.extend(timings[skip:emit_end])
            if all_timings:
                last_global = all_timings[-1]["frame"]
                have_last = True

            win_idx += 1
            if on_progress:
                on_progress(win_idx, total_windows)
            if is_last:
                break
            offset += stride

        return self._tokens_to_words(all_tokens, all_timings)

    def _tokens_to_words(self, tokens, timings):
        """token 流 -> 词列表。词首判定与 asr.py 一致: 带 ▁ 的片段开新词,
        单独的 ▁ token 是显式空格(模型会把 "escape" 切成 ▁many ▁ es ca pe)。"""
        if not tokens or not timings:
            return []
        fps = 1.0 / ENCODER_FRAME_RATE
        words, cur, pending_space = [], None, False

        for tid, tg in zip(tokens, timings):
            piece = self._piece(tid)
            if not piece:
                continue
            if piece == WORD_BOUNDARY:
                pending_space = True
                continue
            leading = piece.startswith(WORD_BOUNDARY)
            clean = piece[len(WORD_BOUNDARY):] if leading else piece
            if not clean:
                continue
            t = tg["frame"] * fps
            if cur is not None and (leading or pending_space):
                words.append(cur)
                cur = None
            pending_space = False
            if cur is None:
                cur = {"word": clean, "start": t, "anchor": t}
            else:
                cur["word"] += clean
            if any(ch.isalnum() for ch in clean):
                cur["anchor"] = t
        if cur:
            words.append(cur)
        return words


# --------------------------------------------------------------------------
# 后处理(与 asr.py 同算法, 保证两个后端的断句/时间行为不分叉)
# --------------------------------------------------------------------------
def refine_word_ends(words, samples, sr):
    """确定词结束时间: 连续语音取下一词起点; 遇停顿用音频能量找语音真正停下的位置。"""
    import numpy as np

    MAX_PAUSE = 0.5
    total_sec = len(samples) / float(sr)
    frame = max(1, int(sr * 0.020))
    hop = max(1, int(sr * 0.010))
    hop_sec = float(hop) / sr
    if len(samples) >= frame:
        n_frames = 1 + (len(samples) - frame) // hop
        idx = np.arange(frame, dtype=np.int64)[None, :] + hop * np.arange(n_frames, dtype=np.int64)[:, None]
        energies = np.sqrt(np.mean(np.square(samples[idx]), axis=1)).astype(np.float32)
    else:
        energies = np.array([float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0], dtype=np.float32)

    thr = max(float(np.percentile(energies, 30)) * 2.0, float(energies.max()) * 0.10, 1e-4) \
        if len(energies) else 1e-4

    def voice_end(t_from, t_to):
        lo = max(0, int(t_from / hop_sec))
        hi = min(len(energies) - 1, int(t_to / hop_sec))
        if hi < lo:
            return t_from
        above = np.nonzero(energies[lo:hi + 1] > thr)[0]
        if len(above) == 0:
            return t_from
        return (lo + int(above[-1]) + 1) * hop_sec

    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else total_sec
        anchor = w.get("anchor", w["start"])
        if nxt - anchor <= MAX_PAUSE:
            end = nxt
        else:
            chars = sum(1 for ch in w["word"] if ch.isalnum())
            est = min(0.6, max(0.15, 0.055 * max(1, chars)))
            end = voice_end(anchor, min(nxt, anchor + est + 0.35))
            if end <= w["start"] + 0.02:
                end = min(nxt, anchor + est)
        w["end"] = max(min(end, total_sec), w["start"] + 0.02)
        w.pop("anchor", None)
    return words


def words_to_segments(words):
    """基础断句: 句末标点 / 长停顿 / 行长兜底(纯本地规则, 不调用 LLM)。"""
    groups, cur = [], []
    for w in words:
        if cur:
            prev = cur[-1]
            too_long = (w["start"] - cur[0]["start"]) > MAX_LINE_SEC or len(cur) >= MAX_LINE_WORDS
            if prev["word"].rstrip().endswith(SENT_END) or (w["start"] - prev["end"]) > PAUSE_SPLIT:
                groups.append(cur)
                cur = []
            elif too_long:
                cut = None
                for j in range(len(cur) - 1, -1, -1):
                    if cur[j]["word"].rstrip().endswith(SOFT_END) and (j + 1) >= 6:
                        cut = j + 1
                        break
                if cut:
                    groups.append(cur[:cut])
                    cur = cur[cut:]
                else:
                    groups.append(cur)
                    cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    out = []
    for i, ws in enumerate(groups):
        out.append({
            "id": i,
            "start": ws[0]["start"],
            "end": ws[-1]["end"],
            "text": " ".join(w["word"] for w in ws).strip(),
            "words": [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                      for w in ws],
        })
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"]:
            out[i]["start"] = out[i - 1]["end"]
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Parakeet TDT 0.6B v2 ASR worker (Intel NPU / OpenVINO)")
    ap.add_argument("--model", required=True, help="模型目录(encoder-model.onnx / decoder_joint-model.onnx / vocab.txt)")
    ap.add_argument("--audio", required=True, help="16kHz 单声道 PCM wav")
    ap.add_argument("--out", required=True, help="结果 JSON 输出路径")
    ap.add_argument("--threads", type=int, default=4, help="保留以兼容统一调用契约(OpenVINO 自管线程)")
    ap.add_argument("--provider", default="npu", choices=["npu", "gpu", "cpu"],
                    help="推理设备: npu(Intel NPU + 核显) / gpu / cpu")
    # 与 asr.py 的契约保持一致: server.js 对两个后端用同一串参数, 这里接受但不启用
    ap.add_argument("--hotwords-file", default="", help="(本后端暂不支持热词, 接受但忽略)")
    ap.add_argument("--hotwords-score", type=float, default=3.0, help="(同上)")
    args = ap.parse_args()

    if args.hotwords_file:
        log("[提示] Intel NPU 后端暂不支持热词偏置, 本次忽略热词文件")
    if args.threads and args.threads != 4:
        log("[提示] --threads 由 OpenVINO 自行管理, 已忽略")

    try:
        log("读取音频 …")
        samples, sr = read_wav_mono16k(args.audio)
        dur = len(samples) / float(sr)
        log("音频 %.1fs @ %dHz" % (dur, sr))
        if dur < 0.2:
            raise RuntimeError("音频过短或无有效采样")

        progress(10, "asr", "准备识别 %d 秒音频" % int(dur))

        def on_progress(done, total):
            progress(25 + int(done / max(1, total) * 60), "asr", "识别中 … 第 %d/%d 块" % (done, total))

        rec = NpuRecognizer(args.model, provider=args.provider)
        t0 = time.time()
        words = rec.recognize_words(samples, on_progress=on_progress)
        if not words:
            raise RuntimeError("未识别到语音内容(模型输出为空)")
        log("识别完成 %d 词, 耗时 %.1fs" % (len(words), time.time() - t0))

        progress(88, "asr", "整理词级时间轴 …")
        words = refine_word_ends(words, samples, sr)
        segments = words_to_segments(words)
        log("断句完成 %d 行" % len(segments))

        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"duration": round(dur, 3), "language": "en", "segments": segments},
                      f, ensure_ascii=False)
        os.replace(tmp, args.out)

        progress(100, "asr", "识别完成: %d 行" % len(segments))
        log("结果已写入 %s" % args.out)
        return 0

    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "msg": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
