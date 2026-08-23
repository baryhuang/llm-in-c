CC ?= cc

CPPFLAGS ?=
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -pthread
OMPFLAGS ?=

BUILD_DIR := build
TARGET_PROBE := $(BUILD_DIR)/target-probe
GEMMA4_LAYER_TEST := $(BUILD_DIR)/gemma4-layer-test
GEMMA4_TASK := $(BUILD_DIR)/gemma4-task
GEMMA4_LAYER_FIXTURE := tests/fixtures/gemma4_layer_v1.bin
QWEN35_GENERIC := models/qwen3.5-0.8b/targets/generic
QWEN35_LAYER_TEST := $(BUILD_DIR)/qwen35-layer-test
QWEN35_LAYER_FIXTURE := tests/fixtures/qwen35_layer_v1.bin
QWEN35_A113X := models/qwen3.5-0.8b/targets/a113x
QWEN35_TASK := $(BUILD_DIR)/qwen35-task
QWEN35_A113X_TASK := $(BUILD_DIR)/qwen35-task-a113x
WHISPER_SMALL := models/whisper-small.en
WHISPER_SMALL_GENERIC := $(WHISPER_SMALL)/targets/generic
WHISPER_SMALL_LOG_MEL_TEST := $(BUILD_DIR)/whisper-small-log-mel-test
WHISPER_SMALL_LOG_MEL_FIXTURE := tests/fixtures/whisper_log_mel_80_v1.bin
WHISPER_ENCODER_STEM_TEST := $(BUILD_DIR)/whisper-encoder-stem-test
WHISPER_ENCODER_STEM_FIXTURE := tests/fixtures/whisper_encoder_stem_v1.bin
WHISPER_ENCODER_BLOCK_TEST := $(BUILD_DIR)/whisper-encoder-block-test
WHISPER_ENCODER_BLOCK_FIXTURE := tests/fixtures/whisper_encoder_block_v1.bin
WHISPER_SMALL_ENCODER_CHECK := $(BUILD_DIR)/whisper-small-encoder-check
WHISPER_SMALL_ENCODER_BENCH := $(BUILD_DIR)/whisper-small-encoder-bench
WHISPER_SMALL_A113X := models/whisper-small.en/targets/a113x
WHISPER_SMALL_A113X_BENCH := $(BUILD_DIR)/whisper-small-encoder-bench-a113x
WHISPER_SMALL_A113X_CHECK := $(BUILD_DIR)/whisper-small-encoder-check-a113x
WHISPER_SMALL_DECODER_CHECK := $(BUILD_DIR)/whisper-small-decoder-check
WHISPER_SMALL_A113X_DECODER_CHECK := $(BUILD_DIR)/whisper-small-decoder-check-a113x
WHISPER_SMALL_TRANSCRIBE := $(BUILD_DIR)/whisper-small-transcribe
WHISPER_SMALL_A113X_TRANSCRIBE := $(BUILD_DIR)/whisper-small-transcribe-a113x
MINIMINDO_GENERIC := models/minimind-o/targets/generic
MINIMINDO_PARALLEL := $(MINIMINDO_GENERIC)/minimindo_parallel.c
MINIMINDO_PARALLEL_TEST := $(BUILD_DIR)/minimindo-parallel-test
MINIMINDO_LAYER_TEST := $(BUILD_DIR)/minimindo-layer-test
MINIMINDO_LAYER_FIXTURE := tests/fixtures/minimindo_layer_v1.bin
MINIMINDO_THINKER := $(BUILD_DIR)/minimindo-thinker
MINIMINDO_CHAT := $(BUILD_DIR)/minimindo-chat
MINIMINDO_OMNI_FORWARD := $(BUILD_DIR)/minimindo-omni-forward
MINIMINDO_MIMI := $(BUILD_DIR)/minimindo-mimi
MINIMINDO_SPEECH := $(BUILD_DIR)/minimindo-speech
MINIMINDO_AUDIO_ENCODER := $(BUILD_DIR)/minimindo-audio-encoder
MINIMINDO_EMBEDDING_COMPARE := $(BUILD_DIR)/minimindo-embedding-compare
WHISPER_TURBO := models/whisper-large-v3-turbo
WHISPER_TURBO_GENERIC := $(WHISPER_TURBO)/targets/generic
WHISPER_TURBO_A113X := $(WHISPER_TURBO)/targets/a113x
WHISPER_TURBO_ENCODER_BENCH := $(BUILD_DIR)/whisper-turbo-encoder-bench
WHISPER_TURBO_A113X_ENCODER_BENCH := $(BUILD_DIR)/whisper-turbo-encoder-bench-a113x
WHISPER_TURBO_TRANSCRIBE := $(BUILD_DIR)/whisper-turbo-transcribe
WHISPER_TURBO_A113X_TRANSCRIBE := $(BUILD_DIR)/whisper-turbo-transcribe-a113x
QWEN38_M3 := models/qwen3.8-27b/targets/apple-m3-pro
QWEN38_COMPILER := compiler/qwen3.8-27b/apple-m3-pro
QWEN38_M3_AIR := $(BUILD_DIR)/qwen38-m3-q4.air
QWEN38_M3_DELTANET_AIR := $(BUILD_DIR)/qwen38-m3-deltanet.air
QWEN38_M3_LAYER_AIR := $(BUILD_DIR)/qwen38-m3-layer.air
QWEN38_M3_ATTENTION_AIR := $(BUILD_DIR)/qwen38-m3-attention.air
QWEN38_M3_GLOBAL_AIR := $(BUILD_DIR)/qwen38-m3-global.air
QWEN38_M3_PREFILL_AIR := $(BUILD_DIR)/qwen38-m3-prefill.air
QWEN38_M3_METALLIB := $(BUILD_DIR)/qwen38-m3-q4.metallib
QWEN38_M3_RUNTIME_OBJECT := $(BUILD_DIR)/qwen38-m3.o
QWEN38_M3_BENCH_OBJECT := $(BUILD_DIR)/qwen38-m3-mlp-bench.o
QWEN38_M3_BENCH := $(BUILD_DIR)/qwen38-m3-mlp-bench
QWEN38_M3_DELTANET_OBJECT := $(BUILD_DIR)/qwen38-m3-deltanet.o
QWEN38_M3_DELTANET_BENCH_OBJECT := $(BUILD_DIR)/qwen38-m3-deltanet-bench.o
QWEN38_M3_DELTANET_BENCH := $(BUILD_DIR)/qwen38-m3-deltanet-bench
QWEN38_M3_LAYER_OBJECT := $(BUILD_DIR)/qwen38-m3-layer.o
QWEN38_M3_LAYER_BENCH_OBJECT := $(BUILD_DIR)/qwen38-m3-layer-bench.o
QWEN38_M3_LAYER_BENCH := $(BUILD_DIR)/qwen38-m3-layer-bench
QWEN38_M3_ATTENTION_OBJECT := $(BUILD_DIR)/qwen38-m3-attention.o
QWEN38_M3_ATTENTION_BENCH_OBJECT := $(BUILD_DIR)/qwen38-m3-attention-bench.o
QWEN38_M3_ATTENTION_BENCH := $(BUILD_DIR)/qwen38-m3-attention-bench
QWEN38_IMPORT := models/qwen3.8-27b/import
QWEN38_SAFETENSORS_INSPECT := $(BUILD_DIR)/qwen38-safetensors-inspect
QWEN38_SAFETENSORS_TEST := $(BUILD_DIR)/qwen38-safetensors-test
QWEN38_SHA256_TEST := $(BUILD_DIR)/qwen38-sha256-test
QWEN38_M3_PACK := $(BUILD_DIR)/qwen38-m3-pack
QWEN38_M3_ATTENTION_PACK := $(BUILD_DIR)/qwen38-m3-attention-pack
QWEN38_M3_GLOBAL_PACK := $(BUILD_DIR)/qwen38-m3-global-pack
QWEN38_M3_OMLX_EXPORT := $(BUILD_DIR)/qwen38-m3-export-omlx
QWEN38_MTP_PACK := $(BUILD_DIR)/qwen38-mtp-pack
QWEN38_M3_DECODE_OBJECT := $(BUILD_DIR)/qwen38-m3-decode.o
QWEN38_M3_DECODE_CLI_OBJECT := $(BUILD_DIR)/qwen38-m3-decode-cli.o
QWEN38_M3_DECODE := $(BUILD_DIR)/qwen38-m3-decode
QWEN38_TOKENIZER_PACK := $(BUILD_DIR)/qwen38-tokenizer-pack
QWEN38_TOKENIZER_OBJECT := $(BUILD_DIR)/qwen38-tokenizer.o
QWEN38_TOKENIZER_CLI_OBJECT := $(BUILD_DIR)/qwen38-tokenizer-cli.o
QWEN38_TOKENIZER_CLI := $(BUILD_DIR)/qwen38-tokenizer
QWEN38_SAMPLER_OBJECT := $(BUILD_DIR)/qwen38-sampler.o
QWEN38_M3_GENERATE_OBJECT := $(BUILD_DIR)/qwen38-m3-generate-cli.o
QWEN38_M3_GENERATE := $(BUILD_DIR)/qwen38-m3-generate
QWEN38_M3_CHAT_OBJECT := $(BUILD_DIR)/qwen38-m3-chat-cli.o
QWEN38_M3_CHAT := $(BUILD_DIR)/qwen38-m3-chat
QWEN38_SAMPLER_TEST := $(BUILD_DIR)/qwen38-sampler-test
QWEN38_M3_API_STATE_TEST := $(BUILD_DIR)/qwen38-m3-api-state-test
QWEN38_M3_PREFILL_PARITY_TEST := $(BUILD_DIR)/qwen38-m3-prefill-parity-test
MINIMAX_H3_GENERIC := models/minimax-h3/targets/generic
MINIMAX_H3_COMPILER := compiler/minimax-h3/apple-m3-pro
MINIMAX_H3_TEST := $(BUILD_DIR)/minimax-h3-test
MINIMAX_H3_REMOTE_INSPECT := $(BUILD_DIR)/minimax-h3-remote-inspect
MINIMAX_H3_Q4_LAYER_INSPECT := $(BUILD_DIR)/minimax-h3-q4-layer-inspect
MINIMAX_H3_TOKENIZER_PACK := $(BUILD_DIR)/minimax-h3-tokenizer-pack
MINIMAX_H3_TOKENIZER_OBJECT := $(BUILD_DIR)/minimax-h3-tokenizer.o
MINIMAX_H3_TOKENIZER_CLI_OBJECT := $(BUILD_DIR)/minimax-h3-tokenizer-cli.o
MINIMAX_H3_TOKENIZER_CLI := $(BUILD_DIR)/minimax-h3-tokenizer
MINIMAX_H3_TOKENIZER_IMAGE := $(BUILD_DIR)/minimax-h3-tokenizer.h3tok
MINIMAX_H3_M3 := models/minimax-h3/targets/apple-m3-pro
MINIMAX_H3_M3_AOT_TEST := $(BUILD_DIR)/minimax-h3-m3-aot-test
MINIMAX_H3_M3_TREE_TEST := $(BUILD_DIR)/minimax-h3-m3-tree-test
MINIMAX_H3_M3_CACHE_TEST := $(BUILD_DIR)/minimax-h3-m3-cache-test
MINIMAX_H3_M3_SPARSE_TEST := $(BUILD_DIR)/minimax-h3-m3-sparse-test
MINIMAX_H3_M3_SELECTOR_TEST := $(BUILD_DIR)/minimax-h3-m3-selector-test
MINIMAX_H3_M3_ATTENTION_AIR := $(BUILD_DIR)/minimax-h3-m3-attention.air
MINIMAX_H3_M3_ATTENTION_METALLIB := $(BUILD_DIR)/minimax-h3-m3-attention.metallib
MINIMAX_H3_M3_PIPELINE_ARCHIVE_TOOL := $(BUILD_DIR)/minimax-h3-m3-pipeline-archive
MINIMAX_H3_M3_PIPELINE_ARCHIVE := $(BUILD_DIR)/minimax-h3-m3-attention.mtlarchive
MINIMAX_H3_M3_ATTENTION_OBJECT := $(BUILD_DIR)/minimax-h3-m3-attention.o
MINIMAX_H3_M3_ATTENTION_BENCH_OBJECT := $(BUILD_DIR)/minimax-h3-m3-attention-bench.o
MINIMAX_H3_M3_ATTENTION_BENCH := $(BUILD_DIR)/minimax-h3-m3-attention-bench
MINIMAX_H3_M3_DENSE_ATTENTION_BENCH := $(BUILD_DIR)/minimax-h3-m3-dense-attention-bench
MINIMAX_H3_M3_ROPE_BENCH := $(BUILD_DIR)/minimax-h3-m3-rope-bench
MINIMAX_H3_M3_VAE_GEMM_BENCH := $(BUILD_DIR)/minimax-h3-m3-vae-gemm-bench
MINIMAX_H3_M3_VAE_ATTENTION_BENCH := $(BUILD_DIR)/minimax-h3-m3-vae-attention-bench
MINIMAX_H3_M3_GEMM_OBJECT := $(BUILD_DIR)/minimax-h3-m3-gemm.o
MINIMAX_H3_M3_GEMM_BENCH_OBJECT := $(BUILD_DIR)/minimax-h3-m3-gemm-bench.o
MINIMAX_H3_M3_GEMM_BENCH := $(BUILD_DIR)/minimax-h3-m3-gemm-bench
MINIMAX_H3_M3_Q8_GEMM_BENCH_OBJECT := $(BUILD_DIR)/minimax-h3-m3-q8-gemm-bench.o
MINIMAX_H3_M3_Q8_GEMM_BENCH := $(BUILD_DIR)/minimax-h3-m3-q8-gemm-bench
MINIMAX_H3_M3_REAL_GEMM_OBJECT := $(BUILD_DIR)/minimax-h3-m3-real-gemm.o
MINIMAX_H3_M3_REAL_GEMM := $(BUILD_DIR)/minimax-h3-m3-real-gemm
MINIMAX_H3_M3_E2E_OBJECT := $(BUILD_DIR)/minimax-h3-m3-e2e.o
MINIMAX_H3_M3_E2E_CLI_OBJECT := $(BUILD_DIR)/minimax-h3-m3-e2e-cli.o
MINIMAX_H3_M3_E2E := $(BUILD_DIR)/minimax-h3-m3-e2e

.PHONY: all a113x clean fixture linux-tools minimindo-tools minimindo-speech minimindo-audio-encoder minimax-h3-m3-attention-bench minimax-h3-m3-dense-attention-bench minimax-h3-m3-rope-bench minimax-h3-m3-vae-gemm-bench minimax-h3-m3-vae-attention-bench minimax-h3-m3-gemm-bench minimax-h3-m3-q8-gemm-bench minimax-h3-m3-pipeline-archive minimax-h3-m3-e2e minimax-h3-tools qwen38-m3-bench qwen38-m3-deltanet-bench qwen38-m3-layer-bench qwen38-m3-attention-bench qwen38-m3-decode qwen38-m3-generate qwen38-m3-chat qwen38-m3-api-state-test qwen38-m3-prefill-parity-test qwen38-tools test whisper-small-tools whisper-turbo-tools

all: $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK)

linux-tools: $(TARGET_PROBE)

a113x: $(QWEN35_A113X_TASK) $(WHISPER_SMALL_A113X_BENCH) \
	$(WHISPER_SMALL_A113X_CHECK) $(WHISPER_SMALL_A113X_TRANSCRIBE) \
	$(WHISPER_SMALL_A113X_DECODER_CHECK) $(WHISPER_TURBO_A113X_ENCODER_BENCH) \
	$(WHISPER_TURBO_A113X_TRANSCRIBE)

whisper-small-tools: $(WHISPER_SMALL_ENCODER_CHECK) $(WHISPER_SMALL_ENCODER_BENCH) \
	$(WHISPER_SMALL_DECODER_CHECK) $(WHISPER_SMALL_TRANSCRIBE)

minimindo-tools: $(MINIMINDO_LAYER_TEST) $(MINIMINDO_THINKER) $(MINIMINDO_CHAT) \
	$(MINIMINDO_OMNI_FORWARD) $(MINIMINDO_MIMI) \
	$(MINIMINDO_EMBEDDING_COMPARE)

minimindo-speech: $(MINIMINDO_SPEECH)

minimindo-audio-encoder: $(MINIMINDO_AUDIO_ENCODER)

whisper-turbo-tools: $(WHISPER_TURBO_ENCODER_BENCH) $(WHISPER_TURBO_TRANSCRIBE)

qwen38-m3-bench: $(QWEN38_M3_BENCH) $(QWEN38_M3_METALLIB)
	$(QWEN38_M3_BENCH) $(QWEN38_M3_METALLIB)

qwen38-m3-deltanet-bench: $(QWEN38_M3_DELTANET_BENCH) $(QWEN38_M3_METALLIB)
	$(QWEN38_M3_DELTANET_BENCH) $(QWEN38_M3_METALLIB)

qwen38-m3-layer-bench: $(QWEN38_M3_LAYER_BENCH) $(QWEN38_M3_METALLIB)

qwen38-m3-attention-bench: $(QWEN38_M3_ATTENTION_BENCH) \
	$(QWEN38_M3_METALLIB)

qwen38-m3-decode: $(QWEN38_M3_DECODE) $(QWEN38_M3_METALLIB)

qwen38-m3-generate: $(QWEN38_M3_GENERATE) $(QWEN38_M3_METALLIB)

qwen38-m3-chat: $(QWEN38_M3_CHAT) $(QWEN38_M3_METALLIB)

minimax-h3-m3-attention-bench: $(MINIMAX_H3_M3_ATTENTION_BENCH) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	$(MINIMAX_H3_M3_ATTENTION_BENCH) $(MINIMAX_H3_M3_ATTENTION_METALLIB)

minimax-h3-m3-dense-attention-bench: $(MINIMAX_H3_M3_DENSE_ATTENTION_BENCH) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	$(MINIMAX_H3_M3_DENSE_ATTENTION_BENCH) \
		$(MINIMAX_H3_M3_ATTENTION_METALLIB)

minimax-h3-m3-rope-bench: $(MINIMAX_H3_M3_ROPE_BENCH) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	$(MINIMAX_H3_M3_ROPE_BENCH) $(MINIMAX_H3_M3_ATTENTION_METALLIB)

minimax-h3-m3-vae-gemm-bench: $(MINIMAX_H3_M3_VAE_GEMM_BENCH) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	$(MINIMAX_H3_M3_VAE_GEMM_BENCH) $(MINIMAX_H3_M3_ATTENTION_METALLIB)

minimax-h3-m3-vae-attention-bench: $(MINIMAX_H3_M3_VAE_ATTENTION_BENCH) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	$(MINIMAX_H3_M3_VAE_ATTENTION_BENCH) \
		$(MINIMAX_H3_M3_ATTENTION_METALLIB)

minimax-h3-m3-gemm-bench: $(MINIMAX_H3_M3_GEMM_BENCH) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	$(MINIMAX_H3_M3_GEMM_BENCH) $(MINIMAX_H3_M3_ATTENTION_METALLIB)

minimax-h3-m3-q8-gemm-bench: $(MINIMAX_H3_M3_Q8_GEMM_BENCH) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)

minimax-h3-m3-pipeline-archive: $(MINIMAX_H3_M3_PIPELINE_ARCHIVE)

minimax-h3-m3-e2e: $(MINIMAX_H3_M3_E2E) \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB) $(MINIMAX_H3_TOKENIZER_PACK)

minimax-h3-tools: $(MINIMAX_H3_REMOTE_INSPECT) \
	$(MINIMAX_H3_Q4_LAYER_INSPECT) $(MINIMAX_H3_TOKENIZER_PACK) \
	$(MINIMAX_H3_TOKENIZER_CLI)

# Needs the packed model directory and metallib, so it is not part of the
# fixture-only `test` target. Run it manually:
#   build/qwen38-m3-api-state-test <model-directory> <metallib>
qwen38-m3-api-state-test: $(QWEN38_M3_API_STATE_TEST) $(QWEN38_M3_METALLIB)

# Live bitwise parity between batched prefill and one-token decode. Run:
#   build/qwen38-m3-prefill-parity-test <model-directory> <metallib>
qwen38-m3-prefill-parity-test: $(QWEN38_M3_PREFILL_PARITY_TEST) \
	$(QWEN38_M3_METALLIB)

qwen38-mtp-pack: $(QWEN38_MTP_PACK)

qwen38-tools: $(QWEN38_SAFETENSORS_INSPECT) $(QWEN38_M3_PACK) \
	$(QWEN38_M3_ATTENTION_PACK) $(QWEN38_M3_GLOBAL_PACK) \
	$(QWEN38_M3_OMLX_EXPORT) $(QWEN38_TOKENIZER_PACK) \
	$(QWEN38_TOKENIZER_CLI)

fixture: $(GEMMA4_LAYER_FIXTURE)

test: $(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE) $(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE) \
	$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_SMALL_LOG_MEL_FIXTURE) \
	$(WHISPER_ENCODER_STEM_TEST) $(WHISPER_ENCODER_STEM_FIXTURE) \
	$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_ENCODER_BLOCK_FIXTURE) \
	$(MINIMINDO_LAYER_TEST) $(MINIMINDO_LAYER_FIXTURE) \
	$(MINIMINDO_PARALLEL_TEST) \
	$(QWEN38_SAFETENSORS_TEST) $(QWEN38_SHA256_TEST) \
	$(QWEN38_SAMPLER_TEST) $(MINIMAX_H3_TEST) $(MINIMAX_H3_M3_AOT_TEST) \
	$(MINIMAX_H3_M3_TREE_TEST) $(MINIMAX_H3_M3_CACHE_TEST) \
	$(MINIMAX_H3_M3_SPARSE_TEST) $(MINIMAX_H3_M3_SELECTOR_TEST)
	python3 -m unittest discover -s tests -p 'test_*.py'
	$(GEMMA4_LAYER_TEST) $(GEMMA4_LAYER_FIXTURE)
	$(QWEN35_LAYER_TEST) $(QWEN35_LAYER_FIXTURE)
	$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_SMALL_LOG_MEL_FIXTURE)
	$(WHISPER_ENCODER_STEM_TEST) $(WHISPER_ENCODER_STEM_FIXTURE)
	$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_ENCODER_BLOCK_FIXTURE)
	$(MINIMINDO_LAYER_TEST) $(MINIMINDO_LAYER_FIXTURE)
	$(MINIMINDO_PARALLEL_TEST)
	$(QWEN38_SAFETENSORS_TEST)
	$(QWEN38_SHA256_TEST)
	$(QWEN38_SAMPLER_TEST)
	$(MINIMAX_H3_TEST)
	$(MINIMAX_H3_M3_AOT_TEST)
	$(MINIMAX_H3_M3_TREE_TEST)
	$(MINIMAX_H3_M3_CACHE_TEST)
	$(MINIMAX_H3_M3_SPARSE_TEST)
	$(MINIMAX_H3_M3_SELECTOR_TEST)

$(MINIMINDO_LAYER_TEST): tests/minimindo_layer_test.c \
	$(MINIMINDO_GENERIC)/minimindo_layer.c \
	$(MINIMINDO_GENERIC)/minimindo_layer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		tests/minimindo_layer_test.c $(MINIMINDO_GENERIC)/minimindo_layer.c \
		-o $@ $(LDFLAGS) $(LDLIBS) -lm

$(MINIMINDO_PARALLEL_TEST): tests/minimindo_parallel_test.c \
	$(MINIMINDO_PARALLEL) $(MINIMINDO_GENERIC)/minimindo_parallel.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		tests/minimindo_parallel_test.c $(MINIMINDO_PARALLEL) \
		-o $@ $(LDFLAGS) $(LDLIBS)

$(MINIMINDO_LAYER_FIXTURE): compiler/generate_minimindo_layer_fixture.py
	mkdir -p $(dir $@)
	python3 $< $@

$(MINIMINDO_THINKER): $(MINIMINDO_GENERIC)/minimindo_thinker_cli.c \
	$(MINIMINDO_GENERIC)/minimindo_thinker.c \
	$(MINIMINDO_GENERIC)/minimindo_thinker.h $(MINIMINDO_PARALLEL)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		$(MINIMINDO_GENERIC)/minimindo_thinker_cli.c \
		$(MINIMINDO_GENERIC)/minimindo_thinker.c \
		$(MINIMINDO_PARALLEL) \
		-o $@ $(LDFLAGS) $(LDLIBS) -lm

$(MINIMINDO_CHAT): $(MINIMINDO_GENERIC)/minimindo_chat.c \
	$(MINIMINDO_GENERIC)/minimindo_thinker.c \
	$(MINIMINDO_GENERIC)/minimindo_thinker.h \
	$(MINIMINDO_GENERIC)/minimindo_tokenizer.c \
	$(MINIMINDO_GENERIC)/minimindo_tokenizer.h $(MINIMINDO_PARALLEL)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		$(MINIMINDO_GENERIC)/minimindo_chat.c \
		$(MINIMINDO_GENERIC)/minimindo_thinker.c \
		$(MINIMINDO_GENERIC)/minimindo_tokenizer.c \
		$(MINIMINDO_PARALLEL) \
		-o $@ $(LDFLAGS) $(LDLIBS) -lm

$(MINIMINDO_OMNI_FORWARD): \
	$(MINIMINDO_GENERIC)/minimindo_omni_forward.c \
	$(MINIMINDO_GENERIC)/minimindo_thinker.c \
	$(MINIMINDO_GENERIC)/minimindo_thinker.h \
	$(MINIMINDO_GENERIC)/minimindo_talker.c \
	$(MINIMINDO_GENERIC)/minimindo_talker.h $(MINIMINDO_PARALLEL)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		$(MINIMINDO_GENERIC)/minimindo_omni_forward.c \
		$(MINIMINDO_GENERIC)/minimindo_thinker.c \
		$(MINIMINDO_GENERIC)/minimindo_talker.c \
		$(MINIMINDO_PARALLEL) \
		-o $@ $(LDFLAGS) $(LDLIBS) -lm

$(MINIMINDO_MIMI): $(MINIMINDO_GENERIC)/minimindo_mimi_cli.c \
	$(MINIMINDO_GENERIC)/minimindo_mimi.c \
	$(MINIMINDO_GENERIC)/minimindo_mimi.h $(MINIMINDO_PARALLEL)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		$(MINIMINDO_GENERIC)/minimindo_mimi_cli.c \
		$(MINIMINDO_GENERIC)/minimindo_mimi.c \
		$(MINIMINDO_PARALLEL) \
		-o $@ $(LDFLAGS) $(LDLIBS) -lm

$(MINIMINDO_SPEECH): $(MINIMINDO_GENERIC)/minimindo_speech.c \
	$(MINIMINDO_GENERIC)/minimindo_thinker.c $(MINIMINDO_GENERIC)/minimindo_thinker.h \
	$(MINIMINDO_GENERIC)/minimindo_talker.c $(MINIMINDO_GENERIC)/minimindo_talker.h \
	$(MINIMINDO_GENERIC)/minimindo_tokenizer.c $(MINIMINDO_GENERIC)/minimindo_tokenizer.h \
	$(MINIMINDO_GENERIC)/minimindo_mimi.c $(MINIMINDO_GENERIC)/minimindo_mimi.h \
	$(MINIMINDO_GENERIC)/minimindo_audio_encoder.c $(MINIMINDO_GENERIC)/minimindo_audio_encoder.h \
	$(MINIMINDO_GENERIC)/minimindo_volume.c $(MINIMINDO_GENERIC)/minimindo_volume.h \
	$(MINIMINDO_PARALLEL)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		$(MINIMINDO_GENERIC)/minimindo_speech.c \
		$(MINIMINDO_GENERIC)/minimindo_thinker.c \
		$(MINIMINDO_GENERIC)/minimindo_talker.c \
		$(MINIMINDO_GENERIC)/minimindo_tokenizer.c \
		$(MINIMINDO_GENERIC)/minimindo_mimi.c \
		$(MINIMINDO_GENERIC)/minimindo_audio_encoder.c \
		$(MINIMINDO_GENERIC)/minimindo_volume.c \
		$(MINIMINDO_PARALLEL) \
		-o $@ $(LDFLAGS) $(LDLIBS) -lm

$(MINIMINDO_AUDIO_ENCODER): $(MINIMINDO_GENERIC)/minimindo_audio_encoder_cli.c \
	$(MINIMINDO_GENERIC)/minimindo_audio_encoder.c \
	$(MINIMINDO_GENERIC)/minimindo_audio_encoder.h $(MINIMINDO_PARALLEL)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMINDO_GENERIC) \
		$(MINIMINDO_GENERIC)/minimindo_audio_encoder_cli.c \
		$(MINIMINDO_GENERIC)/minimindo_audio_encoder.c \
		$(MINIMINDO_PARALLEL) \
		-o $@ $(LDFLAGS) $(LDLIBS) -lm

$(MINIMINDO_EMBEDDING_COMPARE): tests/minimindo_embedding_compare.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) -lm

$(MINIMAX_H3_TEST): tests/minimax_h3_test.c \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_GENERIC) \
		tests/minimax_h3_test.c $(MINIMAX_H3_GENERIC)/minimax_h3.c \
		-o $@ $(LDFLAGS) -lm

$(MINIMAX_H3_REMOTE_INSPECT): \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_inspect.c \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_COMPILER) \
		$(MINIMAX_H3_COMPILER)/minimax_h3_remote_inspect.c \
		$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c \
		-o $@ $(LDFLAGS) -lcurl

$(MINIMAX_H3_Q4_LAYER_INSPECT): \
	$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer_inspect.c \
	$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer.c \
	$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer.h \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_COMPILER) \
		$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer_inspect.c \
		$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer.c \
		$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c \
		-o $@ $(LDFLAGS) -lcurl

$(MINIMAX_H3_TOKENIZER_PACK): $(QWEN38_COMPILER)/qwen38_tokenizer_pack.c \
	$(QWEN38_M3)/qwen38_tokenizer.h \
	$(QWEN38_IMPORT)/qwen38_sha256.c $(QWEN38_IMPORT)/qwen38_sha256.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMINIMAX_H3_TOKENIZER_PROFILE \
		-I$(QWEN38_IMPORT) -I$(QWEN38_M3) \
		$(QWEN38_COMPILER)/qwen38_tokenizer_pack.c \
		$(QWEN38_IMPORT)/qwen38_sha256.c -o $@

$(MINIMAX_H3_TOKENIZER_OBJECT): $(QWEN38_M3)/qwen38_tokenizer.c \
	$(QWEN38_M3)/qwen38_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMINIMAX_H3_TOKENIZER_PROFILE \
		-I$(QWEN38_M3) -c $< -o $@

$(MINIMAX_H3_TOKENIZER_CLI_OBJECT): \
	$(QWEN38_M3)/validation/qwen38_tokenizer.c \
	$(QWEN38_M3)/qwen38_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMINIMAX_H3_TOKENIZER_PROFILE \
		-I$(QWEN38_M3) -c $< -o $@

$(MINIMAX_H3_TOKENIZER_CLI): $(MINIMAX_H3_TOKENIZER_OBJECT) \
	$(MINIMAX_H3_TOKENIZER_CLI_OBJECT)
	$(CC) $^ -o $@ -framework CoreFoundation -licucore

$(MINIMAX_H3_M3_AOT_TEST): tests/minimax_h3_m3_aot_test.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_aot.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_aot.h \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_GENERIC) -I$(MINIMAX_H3_M3) \
		tests/minimax_h3_m3_aot_test.c $(MINIMAX_H3_M3)/minimax_h3_m3_aot.c \
		$(MINIMAX_H3_GENERIC)/minimax_h3.c -o $@ $(LDFLAGS) -lm

$(MINIMAX_H3_M3_TREE_TEST): tests/minimax_h3_m3_tree_test.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.h \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_GENERIC) -I$(MINIMAX_H3_M3) \
		tests/minimax_h3_m3_tree_test.c $(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
		$(MINIMAX_H3_GENERIC)/minimax_h3.c -o $@ $(LDFLAGS) -lm

$(MINIMAX_H3_M3_CACHE_TEST): tests/minimax_h3_m3_cache_test.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_cache.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_cache.h \
	$(MINIMAX_H3_M3)/minimax_h3_m3_aot.h $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_GENERIC) -I$(MINIMAX_H3_M3) \
		tests/minimax_h3_m3_cache_test.c $(MINIMAX_H3_M3)/minimax_h3_m3_cache.c \
		-o $@ $(LDFLAGS)

$(MINIMAX_H3_M3_SPARSE_TEST): tests/minimax_h3_m3_sparse_test.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_sparse.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_sparse.h \
	$(MINIMAX_H3_M3)/minimax_h3_m3_cache.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_cache.h \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_GENERIC) -I$(MINIMAX_H3_M3) \
		tests/minimax_h3_m3_sparse_test.c \
		$(MINIMAX_H3_M3)/minimax_h3_m3_sparse.c \
		$(MINIMAX_H3_M3)/minimax_h3_m3_cache.c \
		$(MINIMAX_H3_GENERIC)/minimax_h3.c -o $@ $(LDFLAGS) -lm

$(MINIMAX_H3_M3_SELECTOR_TEST): tests/minimax_h3_m3_selector_test.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_selector.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_selector.h \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.h \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_GENERIC) -I$(MINIMAX_H3_M3) \
		tests/minimax_h3_m3_selector_test.c \
		$(MINIMAX_H3_M3)/minimax_h3_m3_selector.c \
		$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
		$(MINIMAX_H3_GENERIC)/minimax_h3.c -o $@ $(LDFLAGS) -lm

$(MINIMAX_H3_M3_ATTENTION_AIR): $(MINIMAX_H3_M3)/minimax_h3_attention.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(MINIMAX_H3_M3_ATTENTION_METALLIB): $(MINIMAX_H3_M3_ATTENTION_AIR)
	xcrun -sdk macosx metallib $< -o $@

$(MINIMAX_H3_M3_PIPELINE_ARCHIVE_TOOL): \
	$(MINIMAX_H3_COMPILER)/minimax_h3_m3_pipeline_archive.m
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c17 -Wall -Wextra -Wpedantic -fobjc-arc $< -o $@ \
		-framework Foundation -framework Metal

$(MINIMAX_H3_M3_PIPELINE_ARCHIVE): \
	$(MINIMAX_H3_M3_PIPELINE_ARCHIVE_TOOL) $(MINIMAX_H3_M3_ATTENTION_METALLIB)
	$(MINIMAX_H3_M3_PIPELINE_ARCHIVE_TOOL) \
		$(MINIMAX_H3_M3_ATTENTION_METALLIB) $@

$(MINIMAX_H3_M3_ATTENTION_OBJECT): \
	$(MINIMAX_H3_M3)/minimax_h3_m3_attention.m \
	$(MINIMAX_H3_M3)/minimax_h3_m3_attention.h \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.h $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(MINIMAX_H3_GENERIC) -I$(MINIMAX_H3_M3) -c $< -o $@

$(MINIMAX_H3_M3_ATTENTION_BENCH_OBJECT): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_attention_bench.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_attention.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_M3) -c $< -o $@

$(MINIMAX_H3_M3_ATTENTION_BENCH): $(MINIMAX_H3_M3_ATTENTION_OBJECT) \
	$(MINIMAX_H3_M3_ATTENTION_BENCH_OBJECT) \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c $(MINIMAX_H3_GENERIC)/minimax_h3.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_GENERIC) -I$(MINIMAX_H3_M3) \
		$(MINIMAX_H3_M3_ATTENTION_OBJECT) \
		$(MINIMAX_H3_M3_ATTENTION_BENCH_OBJECT) \
		$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
		$(MINIMAX_H3_GENERIC)/minimax_h3.c -o $@ \
		-framework Foundation -framework Metal -lm

$(MINIMAX_H3_M3_DENSE_ATTENTION_BENCH): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_dense_attention_bench.m \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c17 -Wall -Wextra -Wpedantic -fobjc-arc $< \
		-o $@ -framework Foundation -framework Metal -lm

$(MINIMAX_H3_M3_ROPE_BENCH): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_rope_bench.m \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB) \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c $(MINIMAX_H3_GENERIC)/minimax_h3.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c17 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(MINIMAX_H3_GENERIC) $< $(MINIMAX_H3_GENERIC)/minimax_h3.c \
		-o $@ -framework Foundation -framework Metal -lm

$(MINIMAX_H3_M3_VAE_GEMM_BENCH): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_vae_gemm_bench.m \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c17 -Wall -Wextra -Wpedantic -fobjc-arc $< \
		-o $@ -framework Foundation -framework Metal -lm

$(MINIMAX_H3_M3_VAE_ATTENTION_BENCH): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_vae_attention_bench.m \
	$(MINIMAX_H3_M3_ATTENTION_METALLIB)
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c17 -Wall -Wextra -Wpedantic -fobjc-arc $< \
		-o $@ -framework Foundation -framework Metal -lm

$(MINIMAX_H3_M3_GEMM_OBJECT): \
	$(MINIMAX_H3_M3)/minimax_h3_m3_gemm.m \
	$(MINIMAX_H3_M3)/minimax_h3_m3_gemm.h \
	$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(MINIMAX_H3_M3) -I$(MINIMAX_H3_COMPILER) -c $< -o $@

$(MINIMAX_H3_M3_GEMM_BENCH_OBJECT): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_gemm_bench.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_gemm.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_M3) -c $< -o $@

$(MINIMAX_H3_M3_GEMM_BENCH): $(MINIMAX_H3_M3_GEMM_OBJECT) \
	$(MINIMAX_H3_M3_GEMM_BENCH_OBJECT)
	$(CC) $(MINIMAX_H3_M3_GEMM_OBJECT) \
		$(MINIMAX_H3_M3_GEMM_BENCH_OBJECT) -o $@ \
		-framework Foundation -framework Metal -lm

$(MINIMAX_H3_M3_Q8_GEMM_BENCH_OBJECT): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_q8_gemm_bench.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_gemm.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_M3) -c $< -o $@

$(MINIMAX_H3_M3_Q8_GEMM_BENCH): $(MINIMAX_H3_M3_GEMM_OBJECT) \
	$(MINIMAX_H3_M3_Q8_GEMM_BENCH_OBJECT)
	$(CC) $(MINIMAX_H3_M3_GEMM_OBJECT) \
		$(MINIMAX_H3_M3_Q8_GEMM_BENCH_OBJECT) -o $@ \
		-framework Foundation -framework Metal -lm

$(MINIMAX_H3_M3_REAL_GEMM_OBJECT): \
	$(MINIMAX_H3_M3)/benchmarks/minimax_h3_m3_real_gemm.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_gemm.h \
	$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer.h \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_M3) \
		-I$(MINIMAX_H3_COMPILER) -c $< -o $@

$(MINIMAX_H3_M3_REAL_GEMM): $(MINIMAX_H3_M3_GEMM_OBJECT) \
	$(MINIMAX_H3_M3_REAL_GEMM_OBJECT) \
	$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer.c \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_COMPILER) \
		$(MINIMAX_H3_M3_GEMM_OBJECT) \
		$(MINIMAX_H3_M3_REAL_GEMM_OBJECT) \
		$(MINIMAX_H3_COMPILER)/minimax_h3_q4_layer.c \
		$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c \
		-o $@ -framework Foundation -framework Metal -lcurl -lm

$(MINIMAX_H3_M3_E2E_OBJECT): $(MINIMAX_H3_M3)/minimax_h3_m3_e2e.m \
	$(MINIMAX_H3_M3)/minimax_h3_m3_e2e.h \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.h \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c $(MINIMAX_H3_GENERIC)/minimax_h3.h \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.h \
	$(QWEN38_M3)/qwen38_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c17 -Wall -Wextra -Wpedantic -fobjc-arc \
		-DMINIMAX_H3_TOKENIZER_PROFILE -I$(MINIMAX_H3_COMPILER) \
		-I$(QWEN38_M3) -I$(MINIMAX_H3_M3) -I$(MINIMAX_H3_GENERIC) \
		-c $< -o $@

$(MINIMAX_H3_M3_E2E_CLI_OBJECT): \
	$(MINIMAX_H3_M3)/minimax_h3_m3_e2e_cli.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_e2e.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(MINIMAX_H3_M3) -c $< -o $@

$(MINIMAX_H3_M3_E2E): $(MINIMAX_H3_M3_E2E_OBJECT) \
	$(MINIMAX_H3_M3_E2E_CLI_OBJECT) $(MINIMAX_H3_TOKENIZER_OBJECT) \
	$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c \
	$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
	$(MINIMAX_H3_GENERIC)/minimax_h3.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMINIMAX_H3_TOKENIZER_PROFILE \
		-DMINIMAX_H3_LOCAL_ONLY \
		-I$(MINIMAX_H3_COMPILER) -I$(QWEN38_M3) \
		-I$(MINIMAX_H3_M3) -I$(MINIMAX_H3_GENERIC) \
		$(MINIMAX_H3_M3_E2E_OBJECT) $(MINIMAX_H3_M3_E2E_CLI_OBJECT) \
		$(MINIMAX_H3_TOKENIZER_OBJECT) \
		$(MINIMAX_H3_COMPILER)/minimax_h3_remote_safetensors.c \
		$(MINIMAX_H3_M3)/minimax_h3_m3_tree.c \
		$(MINIMAX_H3_GENERIC)/minimax_h3.c \
		-o $@ -framework Foundation -framework Metal \
		-framework CoreFoundation -licucore -lm

$(TARGET_PROBE): tools/target_probe.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

GEMMA4_GENERIC := models/gemma-4-e2b/targets/generic

$(GEMMA4_LAYER_TEST): tests/gemma4_layer_test.c $(GEMMA4_GENERIC)/gemma4_layer.c $(GEMMA4_GENERIC)/gemma4_layer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(GEMMA4_GENERIC) tests/gemma4_layer_test.c $(GEMMA4_GENERIC)/gemma4_layer.c \
		-o $@ $(LDFLAGS) -lm

$(GEMMA4_TASK): $(GEMMA4_GENERIC)/gemma4_task.c $(GEMMA4_GENERIC)/gemma4_task.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -I$(GEMMA4_GENERIC) $(GEMMA4_GENERIC)/gemma4_task.c \
		-o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(GEMMA4_LAYER_FIXTURE): compiler/generate_gemma4_layer_fixture.py
	python3 $< --output $@

$(QWEN35_LAYER_TEST): tests/qwen35_layer_test.c $(QWEN35_GENERIC)/qwen35_layer.c $(QWEN35_GENERIC)/qwen35_layer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN35_GENERIC) tests/qwen35_layer_test.c $(QWEN35_GENERIC)/qwen35_layer.c \
		-o $@ $(LDFLAGS) -lm

$(QWEN35_LAYER_FIXTURE): compiler/generate_qwen35_layer_fixture.py
	python3 $< --output $@

$(QWEN35_TASK): $(QWEN35_GENERIC)/qwen35_task.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(QWEN35_A113X_TASK): $(QWEN35_A113X)/qwen35_task.c $(QWEN35_A113X)/qwen35_a113x_kernels.h $(QWEN35_GENERIC)/qwen35_task.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 $< \
		-o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_LOG_MEL_TEST): tests/whisper_small_log_mel_test.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tests/whisper_small_log_mel_test.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_LOG_MEL_FIXTURE): compiler/generate_whisper_log_mel_fixture.py
	python3 $< --n-mels 80 --output $@

$(WHISPER_ENCODER_STEM_TEST): tests/whisper_encoder_stem_test.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tests/whisper_encoder_stem_test.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) -lm

$(WHISPER_ENCODER_STEM_FIXTURE): compiler/generate_whisper_encoder_stem_fixture.py
	python3 $< --output $@

$(WHISPER_ENCODER_BLOCK_TEST): tests/whisper_encoder_block_test.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) tests/whisper_encoder_block_test.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_ENCODER_BLOCK_FIXTURE): compiler/generate_whisper_encoder_block_fixture.py
	python3 $< --output $@

$(WHISPER_SMALL_ENCODER_CHECK): $(WHISPER_SMALL)/validation/whisper_small_encoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/validation/whisper_small_encoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_ENCODER_BENCH): $(WHISPER_SMALL)/benchmarks/whisper_small_encoder_bench.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/benchmarks/whisper_small_encoder_bench.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_DECODER_CHECK): $(WHISPER_SMALL)/validation/whisper_small_decoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/validation/whisper_small_decoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_TRANSCRIBE): $(WHISPER_SMALL)/commands/whisper_small_transcribe.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_encoder.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c $(WHISPER_SMALL_GENERIC)/whisper_small_frontend.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c $(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/commands/whisper_small_transcribe.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.c -o $@ $(LDFLAGS) -lm

$(WHISPER_SMALL_A113X_TRANSCRIBE): $(WHISPER_SMALL)/commands/whisper_small_transcribe.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_A113X)/whisper_small_encoder.c $(WHISPER_SMALL_A113X)/whisper_small_a113x_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
	$(WHISPER_SMALL_A113X)/whisper_small_decoder.c \
	$(WHISPER_SMALL_A113X)/whisper_small_a113x_decoder_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/commands/whisper_small_transcribe.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_A113X)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_A113X)/whisper_small_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_TURBO_TRANSCRIBE): $(WHISPER_TURBO)/commands/whisper_turbo_transcribe.c \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_image.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_encoder.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_encoder.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_frontend.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_frontend.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_quant.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -I$(WHISPER_TURBO_GENERIC) \
		$(WHISPER_TURBO)/commands/whisper_turbo_transcribe.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_encoder.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_frontend.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_TURBO_ENCODER_BENCH): $(WHISPER_TURBO)/benchmarks/whisper_turbo_encoder_bench.c \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_image.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_encoder.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_encoder.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_frontend.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_frontend.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_quant.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -I$(WHISPER_TURBO_GENERIC) \
		$(WHISPER_TURBO)/benchmarks/whisper_turbo_encoder_bench.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_encoder.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_frontend.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_TURBO_A113X_TRANSCRIBE): $(WHISPER_TURBO)/commands/whisper_turbo_transcribe.c \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_image.h \
	$(WHISPER_TURBO_A113X)/whisper_turbo_encoder.c $(WHISPER_TURBO_A113X)/whisper_turbo_a113x_kernels.h \
	$(WHISPER_TURBO_A113X)/whisper_turbo_frontend.c \
	$(WHISPER_TURBO_A113X)/whisper_turbo_decoder.c \
	$(WHISPER_TURBO_A113X)/whisper_turbo_a113x_decoder_kernels.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.h $(WHISPER_TURBO_GENERIC)/whisper_turbo_quant.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_TURBO_GENERIC) $(WHISPER_TURBO)/commands/whisper_turbo_transcribe.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c \
		$(WHISPER_TURBO_A113X)/whisper_turbo_encoder.c \
		$(WHISPER_TURBO_A113X)/whisper_turbo_frontend.c \
		$(WHISPER_TURBO_A113X)/whisper_turbo_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_TURBO_A113X_ENCODER_BENCH): \
	$(WHISPER_TURBO)/benchmarks/whisper_turbo_encoder_bench.c \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c $(WHISPER_TURBO_GENERIC)/whisper_turbo_image.h \
	$(WHISPER_TURBO_A113X)/whisper_turbo_encoder.c $(WHISPER_TURBO_A113X)/whisper_turbo_a113x_kernels.h \
	$(WHISPER_TURBO_A113X)/whisper_turbo_frontend.c \
	$(WHISPER_TURBO_A113X)/whisper_turbo_decoder.c \
	$(WHISPER_TURBO_A113X)/whisper_turbo_a113x_decoder_kernels.h \
	$(WHISPER_TURBO_GENERIC)/whisper_turbo_decoder.h $(WHISPER_TURBO_GENERIC)/whisper_turbo_quant.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_TURBO_GENERIC) $(WHISPER_TURBO)/benchmarks/whisper_turbo_encoder_bench.c \
		$(WHISPER_TURBO_GENERIC)/whisper_turbo_image.c \
		$(WHISPER_TURBO_A113X)/whisper_turbo_encoder.c \
		$(WHISPER_TURBO_A113X)/whisper_turbo_frontend.c \
		$(WHISPER_TURBO_A113X)/whisper_turbo_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_A113X_DECODER_CHECK): $(WHISPER_SMALL)/validation/whisper_small_decoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
	$(WHISPER_SMALL_A113X)/whisper_small_decoder.c \
	$(WHISPER_SMALL_A113X)/whisper_small_a113x_decoder_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_decoder.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/validation/whisper_small_decoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c \
		$(WHISPER_SMALL_A113X)/whisper_small_decoder.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_A113X_BENCH): $(WHISPER_SMALL)/benchmarks/whisper_small_encoder_bench.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_A113X)/whisper_small_encoder.c $(WHISPER_SMALL_A113X)/whisper_small_a113x_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/benchmarks/whisper_small_encoder_bench.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_A113X)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(WHISPER_SMALL_A113X_CHECK): $(WHISPER_SMALL)/validation/whisper_small_encoder_check.c \
	$(WHISPER_SMALL_GENERIC)/whisper_small_image.c $(WHISPER_SMALL_GENERIC)/whisper_small_image.h \
	$(WHISPER_SMALL_A113X)/whisper_small_encoder.c $(WHISPER_SMALL_A113X)/whisper_small_a113x_kernels.h \
	$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -mcpu=cortex-a53 -mtune=cortex-a53 \
		-I$(WHISPER_SMALL_GENERIC) $(WHISPER_SMALL)/validation/whisper_small_encoder_check.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_image.c \
		$(WHISPER_SMALL_A113X)/whisper_small_encoder.c \
		$(WHISPER_SMALL_GENERIC)/whisper_small_frontend.c -o $@ $(LDFLAGS) $(OMPFLAGS) -lm

$(QWEN38_M3_AIR): $(QWEN38_M3)/qwen38_q4.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN38_M3_DELTANET_AIR): $(QWEN38_M3)/qwen38_deltanet.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN38_M3_LAYER_AIR): $(QWEN38_M3)/qwen38_layer.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN38_M3_ATTENTION_AIR): $(QWEN38_M3)/qwen38_attention.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN38_M3_GLOBAL_AIR): $(QWEN38_M3)/qwen38_global.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN38_M3_PREFILL_AIR): $(QWEN38_M3)/qwen38_prefill.metal
	mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $@

$(QWEN38_M3_METALLIB): $(QWEN38_M3_AIR) $(QWEN38_M3_DELTANET_AIR) \
	$(QWEN38_M3_LAYER_AIR) $(QWEN38_M3_ATTENTION_AIR) \
	$(QWEN38_M3_GLOBAL_AIR) $(QWEN38_M3_PREFILL_AIR)
	xcrun -sdk macosx metallib $^ -o $@

$(QWEN38_M3_RUNTIME_OBJECT): $(QWEN38_M3)/qwen38_m3.m $(QWEN38_M3)/qwen38_m3.h \
	$(QWEN38_M3)/qwen38_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_BENCH_OBJECT): $(QWEN38_M3)/benchmarks/qwen38_m3_mlp_bench.c $(QWEN38_M3)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_BENCH): $(QWEN38_M3_RUNTIME_OBJECT) $(QWEN38_M3_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_DELTANET_OBJECT): $(QWEN38_M3)/qwen38_m3_deltanet.m \
	$(QWEN38_M3)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_DELTANET_BENCH_OBJECT): $(QWEN38_M3)/benchmarks/qwen38_m3_deltanet_bench.c \
	$(QWEN38_M3)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_DELTANET_BENCH): $(QWEN38_M3_DELTANET_OBJECT) \
	$(QWEN38_M3_DELTANET_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_LAYER_OBJECT): $(QWEN38_M3)/qwen38_m3_layer.m \
	$(QWEN38_M3)/qwen38_m3.h $(QWEN38_M3)/qwen38_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_LAYER_BENCH_OBJECT): $(QWEN38_M3)/benchmarks/qwen38_m3_layer_bench.c \
	$(QWEN38_M3)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_LAYER_BENCH): $(QWEN38_M3_LAYER_OBJECT) \
	$(QWEN38_M3_LAYER_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_ATTENTION_OBJECT): $(QWEN38_M3)/qwen38_m3_attention.m \
	$(QWEN38_M3)/qwen38_m3.h $(QWEN38_M3)/qwen38_m3_image.h \
	$(QWEN38_M3)/qwen38_m3_attention_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_ATTENTION_BENCH_OBJECT): $(QWEN38_M3)/benchmarks/qwen38_m3_attention_bench.c \
	$(QWEN38_M3)/qwen38_m3.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_ATTENTION_BENCH): $(QWEN38_M3_ATTENTION_OBJECT) \
	$(QWEN38_M3_ATTENTION_BENCH_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_DECODE_OBJECT): $(QWEN38_M3)/qwen38_m3_decode.m \
	$(QWEN38_M3)/qwen38_m3_decode.h $(QWEN38_M3)/qwen38_m3.h \
	$(QWEN38_M3)/qwen38_m3_image.h \
	$(QWEN38_M3)/qwen38_m3_attention_image.h \
	$(QWEN38_M3)/qwen38_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -fobjc-arc \
		-I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_DECODE_CLI_OBJECT): $(QWEN38_M3)/validation/qwen38_m3_decode.c \
	$(QWEN38_M3)/qwen38_m3_decode.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_DECODE): $(QWEN38_M3_DECODE_OBJECT) \
	$(QWEN38_M3_DECODE_CLI_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal -lm

$(QWEN38_TOKENIZER_OBJECT): $(QWEN38_M3)/qwen38_tokenizer.c \
	$(QWEN38_M3)/qwen38_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_TOKENIZER_CLI_OBJECT): $(QWEN38_M3)/validation/qwen38_tokenizer.c \
	$(QWEN38_M3)/qwen38_tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_TOKENIZER_CLI): $(QWEN38_TOKENIZER_OBJECT) \
	$(QWEN38_TOKENIZER_CLI_OBJECT)
	$(CC) $^ -o $@ -framework CoreFoundation -licucore

$(QWEN38_SAMPLER_OBJECT): $(QWEN38_M3)/qwen38_sampler.c \
	$(QWEN38_M3)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_SAMPLER_TEST): tests/qwen38_sampler_test.c \
	$(QWEN38_M3)/qwen38_sampler.c $(QWEN38_M3)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) \
		tests/qwen38_sampler_test.c $(QWEN38_M3)/qwen38_sampler.c \
		-o $@ -lm

$(QWEN38_M3_GENERATE_OBJECT): $(QWEN38_M3)/commands/qwen38_m3_generate.c \
	$(QWEN38_M3)/qwen38_m3_decode.h $(QWEN38_M3)/qwen38_tokenizer.h \
	$(QWEN38_M3)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_GENERATE): $(QWEN38_M3_DECODE_OBJECT) \
	$(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
	$(QWEN38_M3_GENERATE_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal \
		-framework CoreFoundation -licucore -lm

$(QWEN38_M3_CHAT_OBJECT): $(QWEN38_M3)/commands/qwen38_m3_chat.c \
	$(QWEN38_M3)/qwen38_m3_decode.h $(QWEN38_M3)/qwen38_tokenizer.h \
	$(QWEN38_M3)/qwen38_sampler.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) -c $< -o $@

$(QWEN38_M3_CHAT): $(QWEN38_M3_DECODE_OBJECT) \
	$(QWEN38_TOKENIZER_OBJECT) $(QWEN38_SAMPLER_OBJECT) \
	$(QWEN38_M3_CHAT_OBJECT)
	$(CC) $^ -o $@ -framework Foundation -framework Metal \
		-framework CoreFoundation -licucore -lm

$(QWEN38_M3_API_STATE_TEST): tests/qwen38_m3_api_state_test.c \
	$(QWEN38_M3_DECODE_OBJECT) $(QWEN38_M3)/qwen38_m3_decode.h \
	$(QWEN38_M3)/qwen38_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) \
		tests/qwen38_m3_api_state_test.c $(QWEN38_M3_DECODE_OBJECT) \
		-o $@ -framework Foundation -framework Metal -lm

$(QWEN38_M3_PREFILL_PARITY_TEST): tests/qwen38_m3_prefill_parity_test.c \
	$(QWEN38_M3_DECODE_OBJECT) $(QWEN38_M3)/qwen38_m3_decode.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) \
		tests/qwen38_m3_prefill_parity_test.c \
		$(QWEN38_M3_DECODE_OBJECT) \
		-o $@ -framework Foundation -framework Metal -lm

$(QWEN38_SAFETENSORS_INSPECT): $(QWEN38_COMPILER)/qwen38_safetensors_inspect.c \
	$(QWEN38_IMPORT)/qwen38_safetensors.c $(QWEN38_IMPORT)/qwen38_safetensors.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) $(QWEN38_COMPILER)/qwen38_safetensors_inspect.c \
		$(QWEN38_IMPORT)/qwen38_safetensors.c -o $@

$(QWEN38_SAFETENSORS_TEST): tests/qwen38_safetensors_test.c \
	$(QWEN38_IMPORT)/qwen38_safetensors.c $(QWEN38_IMPORT)/qwen38_safetensors.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) tests/qwen38_safetensors_test.c \
		$(QWEN38_IMPORT)/qwen38_safetensors.c -o $@

$(QWEN38_SHA256_TEST): tests/qwen38_sha256_test.c \
	$(QWEN38_IMPORT)/qwen38_sha256.c $(QWEN38_IMPORT)/qwen38_sha256.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) tests/qwen38_sha256_test.c \
		$(QWEN38_IMPORT)/qwen38_sha256.c -o $@

$(QWEN38_M3_PACK): $(QWEN38_COMPILER)/qwen38_m3_pack.c \
	$(QWEN38_IMPORT)/qwen38_sha256.c $(QWEN38_IMPORT)/qwen38_sha256.h \
	$(QWEN38_IMPORT)/qwen38_safetensors.c $(QWEN38_IMPORT)/qwen38_safetensors.h \
	$(QWEN38_M3)/qwen38_m3.h $(QWEN38_M3)/qwen38_m3_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) -I$(QWEN38_M3) \
		$(QWEN38_COMPILER)/qwen38_m3_pack.c $(QWEN38_IMPORT)/qwen38_sha256.c \
		$(QWEN38_IMPORT)/qwen38_safetensors.c -o $@ -lm

$(QWEN38_MTP_PACK): $(QWEN38_COMPILER)/qwen38_mtp_pack.c \
	$(QWEN38_IMPORT)/qwen38_sha256.c $(QWEN38_IMPORT)/qwen38_sha256.h \
	$(QWEN38_IMPORT)/qwen38_safetensors.c $(QWEN38_IMPORT)/qwen38_safetensors.h \
	$(QWEN38_M3)/qwen38_m3.h $(QWEN38_M3)/qwen38_m3_image.h \
	$(QWEN38_M3)/qwen38_m3_attention_image.h \
	$(QWEN38_M3)/qwen38_m3_mtp_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) -I$(QWEN38_M3) \
		$(QWEN38_COMPILER)/qwen38_mtp_pack.c \
		$(QWEN38_IMPORT)/qwen38_sha256.c \
		$(QWEN38_IMPORT)/qwen38_safetensors.c -o $@ -lm

$(QWEN38_M3_ATTENTION_PACK): $(QWEN38_COMPILER)/qwen38_m3_attention_pack.c \
	$(QWEN38_IMPORT)/qwen38_sha256.c $(QWEN38_IMPORT)/qwen38_sha256.h \
	$(QWEN38_IMPORT)/qwen38_safetensors.c $(QWEN38_IMPORT)/qwen38_safetensors.h \
	$(QWEN38_M3)/qwen38_m3.h $(QWEN38_M3)/qwen38_m3_image.h \
	$(QWEN38_M3)/qwen38_m3_attention_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) -I$(QWEN38_M3) \
		$(QWEN38_COMPILER)/qwen38_m3_attention_pack.c $(QWEN38_IMPORT)/qwen38_sha256.c \
		$(QWEN38_IMPORT)/qwen38_safetensors.c -o $@

$(QWEN38_M3_GLOBAL_PACK): $(QWEN38_COMPILER)/qwen38_m3_global_pack.c \
	$(QWEN38_IMPORT)/qwen38_sha256.c $(QWEN38_IMPORT)/qwen38_sha256.h \
	$(QWEN38_IMPORT)/qwen38_safetensors.c $(QWEN38_IMPORT)/qwen38_safetensors.h \
	$(QWEN38_M3)/qwen38_m3.h $(QWEN38_M3)/qwen38_m3_image.h \
	$(QWEN38_M3)/qwen38_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) -I$(QWEN38_M3) \
		$(QWEN38_COMPILER)/qwen38_m3_global_pack.c $(QWEN38_IMPORT)/qwen38_sha256.c \
		$(QWEN38_IMPORT)/qwen38_safetensors.c -o $@

$(QWEN38_M3_OMLX_EXPORT): $(QWEN38_M3)/validation/qwen38_m3_export_omlx.c \
	$(QWEN38_M3)/qwen38_m3_image.h \
	$(QWEN38_M3)/qwen38_m3_attention_image.h \
	$(QWEN38_M3)/qwen38_m3_global_image.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_M3) $< -o $@

$(QWEN38_TOKENIZER_PACK): $(QWEN38_COMPILER)/qwen38_tokenizer_pack.c \
	$(QWEN38_M3)/qwen38_tokenizer.h \
	$(QWEN38_IMPORT)/qwen38_sha256.c $(QWEN38_IMPORT)/qwen38_sha256.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(QWEN38_IMPORT) -I$(QWEN38_M3) \
		$(QWEN38_COMPILER)/qwen38_tokenizer_pack.c $(QWEN38_IMPORT)/qwen38_sha256.c \
		-o $@

clean:
	rm -f $(TARGET_PROBE) $(GEMMA4_LAYER_TEST) $(GEMMA4_TASK) $(QWEN35_LAYER_TEST) \
		$(QWEN35_TASK) $(QWEN35_A113X_TASK) \
		$(WHISPER_SMALL_LOG_MEL_TEST) $(WHISPER_ENCODER_STEM_TEST) \
		$(WHISPER_ENCODER_BLOCK_TEST) $(WHISPER_SMALL_ENCODER_CHECK) \
		$(WHISPER_SMALL_ENCODER_BENCH) $(WHISPER_SMALL_A113X_BENCH) \
		$(WHISPER_SMALL_A113X_CHECK) $(WHISPER_SMALL_DECODER_CHECK) \
		$(WHISPER_SMALL_A113X_DECODER_CHECK) $(WHISPER_SMALL_TRANSCRIBE) \
		$(WHISPER_SMALL_A113X_TRANSCRIBE) $(WHISPER_TURBO_TRANSCRIBE) \
		$(WHISPER_TURBO_ENCODER_BENCH) $(WHISPER_TURBO_A113X_ENCODER_BENCH) \
		$(WHISPER_TURBO_A113X_TRANSCRIBE) $(QWEN38_M3_AIR) \
		$(QWEN38_M3_DELTANET_AIR) $(QWEN38_M3_METALLIB) \
		$(QWEN38_M3_LAYER_AIR) \
		$(QWEN38_M3_ATTENTION_AIR) \
		$(QWEN38_M3_GLOBAL_AIR) \
		$(QWEN38_M3_RUNTIME_OBJECT) $(QWEN38_M3_BENCH_OBJECT) $(QWEN38_M3_BENCH) \
		$(QWEN38_M3_DELTANET_OBJECT) $(QWEN38_M3_DELTANET_BENCH_OBJECT) \
		$(QWEN38_M3_DELTANET_BENCH) $(QWEN38_M3_LAYER_OBJECT) \
		$(QWEN38_M3_LAYER_BENCH_OBJECT) $(QWEN38_M3_LAYER_BENCH)
	rm -f $(QWEN38_M3_ATTENTION_OBJECT) $(QWEN38_M3_ATTENTION_BENCH_OBJECT) \
		$(QWEN38_M3_ATTENTION_BENCH)
	rm -f $(QWEN38_M3_DECODE_OBJECT) $(QWEN38_M3_DECODE_CLI_OBJECT) \
		$(QWEN38_M3_DECODE) $(QWEN38_M3_GENERATE_OBJECT) \
		$(QWEN38_M3_GENERATE)
	rm -f $(QWEN38_SAFETENSORS_INSPECT) $(QWEN38_SAFETENSORS_TEST) \
		$(QWEN38_SHA256_TEST) $(QWEN38_M3_PACK) \
		$(QWEN38_M3_ATTENTION_PACK) $(QWEN38_M3_GLOBAL_PACK) \
		$(QWEN38_M3_OMLX_EXPORT) \
		$(QWEN38_TOKENIZER_PACK) $(QWEN38_TOKENIZER_OBJECT) \
		$(QWEN38_TOKENIZER_CLI_OBJECT) $(QWEN38_TOKENIZER_CLI) \
		$(QWEN38_SAMPLER_OBJECT) $(QWEN38_SAMPLER_TEST)
	rm -f $(MINIMAX_H3_TEST) $(MINIMAX_H3_M3_AOT_TEST) \
		$(MINIMAX_H3_M3_TREE_TEST) $(MINIMAX_H3_M3_CACHE_TEST) \
		$(MINIMAX_H3_M3_SPARSE_TEST) $(MINIMAX_H3_M3_SELECTOR_TEST) \
		$(MINIMAX_H3_M3_ATTENTION_AIR) \
		$(MINIMAX_H3_M3_ATTENTION_METALLIB) \
		$(MINIMAX_H3_M3_ATTENTION_OBJECT) \
		$(MINIMAX_H3_M3_ATTENTION_BENCH_OBJECT) \
		$(MINIMAX_H3_M3_ATTENTION_BENCH) \
		$(MINIMAX_H3_M3_GEMM_OBJECT) $(MINIMAX_H3_M3_GEMM_BENCH_OBJECT) \
		$(MINIMAX_H3_M3_GEMM_BENCH) \
		$(MINIMAX_H3_M3_Q8_GEMM_BENCH_OBJECT) \
		$(MINIMAX_H3_M3_Q8_GEMM_BENCH) $(MINIMAX_H3_M3_REAL_GEMM_OBJECT) \
		$(MINIMAX_H3_M3_REAL_GEMM) $(MINIMAX_H3_M3_E2E_OBJECT) \
		$(MINIMAX_H3_M3_E2E_CLI_OBJECT) $(MINIMAX_H3_M3_E2E) \
		$(MINIMAX_H3_REMOTE_INSPECT) $(MINIMAX_H3_Q4_LAYER_INSPECT) \
		$(MINIMAX_H3_TOKENIZER_PACK) $(MINIMAX_H3_TOKENIZER_OBJECT) \
		$(MINIMAX_H3_TOKENIZER_CLI_OBJECT) $(MINIMAX_H3_TOKENIZER_CLI) \
		$(MINIMAX_H3_TOKENIZER_IMAGE)
