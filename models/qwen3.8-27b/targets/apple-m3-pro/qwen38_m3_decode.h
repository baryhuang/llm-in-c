#ifndef QWEN38_M3_DECODE_H
#define QWEN38_M3_DECODE_H

#include <stddef.h>
#include <stdint.h>

typedef struct qwen38_m3_model qwen38_m3_model;

typedef struct {
    uint32_t input_token;
    uint32_t next_token;
    uint32_t position;
    double duration_ms;
    size_t mapped_weight_bytes;
    size_t state_bytes;
    size_t kv_cache_bytes;
    size_t physical_footprint_bytes;
} qwen38_m3_decode_result;

qwen38_m3_model *qwen38_m3_model_open(
    const char *model_directory,
    const char *metallib_path,
    uint32_t context_capacity,
    char *error_message,
    size_t error_message_capacity);

void qwen38_m3_model_reset(qwen38_m3_model *model);

int qwen38_m3_model_decode(
    qwen38_m3_model *model,
    uint32_t token_id,
    uint32_t position,
    qwen38_m3_decode_result *result,
    char *error_message,
    size_t error_message_capacity);

/* Runs one token through the static graph and exposes read-only logits until
 * the next call. This is the C sampling interface; the model owns the data. */
int qwen38_m3_model_forward(
    qwen38_m3_model *model,
    uint32_t token_id,
    uint32_t position,
    qwen38_m3_decode_result *result,
    const float **logits,
    size_t *logit_count,
    char *error_message,
    size_t error_message_capacity);

typedef struct {
    uint32_t token_count;
    uint32_t chunk32_count;
    uint32_t chunk16_count;
    uint32_t single_count;
    double duration_ms;
    double first_chunk_ms;
} qwen38_m3_prefill_result;

/* Process a run of prompt tokens through batched S32/S16 graphs, falling
 * back to one-token forwards for a tail shorter than 16. Layer state after
 * prefill is bitwise-identical to the same tokens pushed one at a time; no
 * logits are produced, so the caller forwards the final prompt token through
 * qwen38_m3_model_forward for sampling. Synchronous; on error the layer
 * state is partially advanced and the caller should reset the model. */
int qwen38_m3_model_prefill(
    qwen38_m3_model *model,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t start_position,
    qwen38_m3_prefill_result *result,
    char *error_message,
    size_t error_message_capacity);

/* Submit one token without waiting for Metal completion. The model owns one
 * workspace and therefore permits exactly one in-flight forward. This split
 * lets the caller overlap CPU detokenization and output with GPU execution. */
int qwen38_m3_model_forward_submit(
    qwen38_m3_model *model,
    uint32_t token_id,
    uint32_t position,
    char *error_message,
    size_t error_message_capacity);

/* Wait for the submitted forward and expose its read-only logits until the
 * next submit. Calling this without an in-flight forward is an error. */
int qwen38_m3_model_forward_wait(
    qwen38_m3_model *model,
    qwen38_m3_decode_result *result,
    const float **logits,
    size_t *logit_count,
    char *error_message,
    size_t error_message_capacity);

/* Multi-token prediction (greedy speculative decoding).
 *
 * qwen38_m3_model_mtp_open loads the MTP draft images; afterwards
 * qwen38_m3_model_prefill also fills the draft layer's cache, and its
 * token_ids argument must carry token_count + 1 entries (the token after
 * the prefilled run). QWEN38_MTP_DEPTH (1..7, read at open) sets the
 * draft depth: each step chains that many draft tokens through the MTP
 * layer, verifies the pending token plus all drafts in one batched
 * forward, and on a partial accept rolls the GDN state back to the last
 * accepted row from the verify's factor checkpoints and hands the
 * corrected token over as the next pending token. A step
 * emits between one and depth + 1 tokens into emitted (up to 8);
 * accepted receives the number of accepted drafts (0..depth).
 * current_token carries the sampled-but-unprocessed token in and the
 * next one out; the caller appends emitted and checks stop tokens. */
int qwen38_m3_model_mtp_open(
    qwen38_m3_model *model,
    const char *layer_image_path,
    const char *extras_image_path,
    char *error_message,
    size_t error_message_capacity);

int qwen38_m3_model_mtp_step(
    qwen38_m3_model *model,
    uint32_t *current_token,
    uint32_t *position,
    uint32_t emitted[8],
    uint32_t *emitted_count,
    int *accepted,
    char *error_message,
    size_t error_message_capacity);

/* Optional context view for lookup drafting: tokens must stay valid and
 * cover the exact sequence the layer states correspond to (prompt plus
 * emitted tokens, excluding the pending token). When the trigram ending
 * at the pending token recurs in this context, the step drafts up to
 * seven follower tokens from the context with no draft-model passes and
 * verifies them in the usual batched forward. Pass NULL/0 to disable. */
void qwen38_m3_model_mtp_context(
    qwen38_m3_model *model,
    const uint32_t *tokens,
    uint32_t count);

/* Conversation checkpoints: save copies the cumulative GDN
 * recurrent/convolution states (the attention KV cache is per-position
 * and needs no copy); restore rewinds to that point so a request that
 * extends the saved prefix can prefill only its new suffix.
 *
 * Two slots, because two different prefixes are worth keeping:
 *
 *   QWEN38_M3_PREFIX_TURN    the last prompt boundary, rewritten every
 *                            request, which serves follow-up turns of
 *                            the conversation in flight.
 *   QWEN38_M3_PREFIX_SYSTEM  a caller-declared prefix that outlives the
 *                            conversation - for an agent front end, the
 *                            system turn holding its instructions and
 *                            tool schemas, which is identical across
 *                            sessions and is otherwise re-prefilled from
 *                            scratch every time one starts.
 *
 * A slot saved with kv_positions == 0 rewinds GDN state only, so the
 * attention KV below the restore position must still hold that prefix's
 * keys and values; that holds within one conversation. A slot saved
 * with kv_positions mirrors the KV span too and therefore survives an
 * unrelated request, including one that reset the caches. */
enum {
    QWEN38_M3_PREFIX_TURN = 0,
    QWEN38_M3_PREFIX_SYSTEM = 1,
    QWEN38_M3_PREFIX_SLOTS = 2
};

/* kv_positions mirrors that many leading positions of the attention KV
 * into the slot as well. Pass 0 for a slot that is only used while the
 * caches are known intact; pass the prefix length for one that must
 * survive an unrelated request prefilling over the low positions. */
int qwen38_m3_model_prefix_save_slot(
    qwen38_m3_model *model,
    uint32_t slot,
    uint32_t kv_positions,
    char *error_message,
    size_t error_message_capacity);

int qwen38_m3_model_prefix_restore_slot(
    qwen38_m3_model *model,
    uint32_t slot,
    char *error_message,
    size_t error_message_capacity);

/* QWEN38_M3_PREFIX_TURN shorthands. */
int qwen38_m3_model_prefix_save(
    qwen38_m3_model *model,
    char *error_message,
    size_t error_message_capacity);

int qwen38_m3_model_prefix_restore(
    qwen38_m3_model *model,
    char *error_message,
    size_t error_message_capacity);

enum {
    QWEN38_M3_STATE_RECURRENT = 0,
    QWEN38_M3_STATE_CONVOLUTION = 1,
    QWEN38_M3_STATE_KEY_CACHE = 2,
    QWEN38_M3_STATE_VALUE_CACHE = 3
};

/* Verification support: copy one layer's persistent state buffer. Returns
 * the state byte count, or 0 if the layer/kind combination does not exist
 * or the destination is too small. Recurrent and convolution state exist on
 * DeltaNet layers; key/value caches exist on attention layers. */
size_t qwen38_m3_model_copy_state(
    qwen38_m3_model *model,
    uint32_t layer_index,
    uint32_t kind,
    void *destination,
    size_t destination_capacity);

void qwen38_m3_model_close(qwen38_m3_model *model);

#endif
