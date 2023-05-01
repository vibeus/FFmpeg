/*
 * RockChip MPP Video Decoder
 * Copyright (c) 2017 Lionel CHAZALLON
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <drm_fourcc.h>
#include <pthread.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/rk_mpi.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <fcntl.h>

#include "avcodec.h"
#include "decode.h"
#include "hwconfig.h"
#include "internal.h"
#include "libavutil/buffer.h"
#include "libavutil/common.h"
#include "libavutil/frame.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_drm.h"
#include "libavutil/imgutils.h"
#include "libavutil/log.h"
#include "libyuv/planar_functions.h"
#include "rga.h"

// HACK: Older BSP kernel use NA12 for NV15.
#ifndef DRM_FORMAT_NV15 // fourcc_code('N', 'V', '1', '5')
#define DRM_FORMAT_NV15 fourcc_code('N', 'A', '1', '2')
#endif

#define FPS_UPDATE_INTERVAL     120

typedef struct {
    MppCtx ctx;
    MppApi *mpi;
    MppBufferGroup frame_group;

    int8_t eos;
    int8_t softconvert;

    AVPacket packet;
    AVBufferRef *frames_ref;
    AVBufferRef *device_ref;

    char print_fps;

    uint64_t last_fps_time;
    uint64_t frames;

    int rga_fd;
} RKMPPDecoder;

typedef struct {
    AVClass *av_class;
    AVBufferRef *decoder_ref;
} RKMPPDecodeContext;

typedef struct {
    MppFrame frame;
    AVBufferRef *decoder_ref;
} RKMPPFrameContext;

static MppCodingType rkmpp_get_codingtype(AVCodecContext *avctx)
{
    switch (avctx->codec_id) {
    case AV_CODEC_ID_H263:          return MPP_VIDEO_CodingH263;
    case AV_CODEC_ID_H264:          return MPP_VIDEO_CodingAVC;
    case AV_CODEC_ID_HEVC:          return MPP_VIDEO_CodingHEVC;
    case AV_CODEC_ID_AV1:           return MPP_VIDEO_CodingAV1;
    case AV_CODEC_ID_VP8:           return MPP_VIDEO_CodingVP8;
    case AV_CODEC_ID_VP9:           return MPP_VIDEO_CodingVP9;
    case AV_CODEC_ID_MPEG1VIDEO:    /* fallthrough */
    case AV_CODEC_ID_MPEG2VIDEO:    return MPP_VIDEO_CodingMPEG2;
    case AV_CODEC_ID_MPEG4:         return MPP_VIDEO_CodingMPEG4;
    default:                        return MPP_VIDEO_CodingUnused;
    }
}

static uint32_t rkmpp_get_frameformat(MppFrameFormat mppformat)
{
    switch (mppformat & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:          return DRM_FORMAT_NV12;
    case MPP_FMT_YUV420SP_10BIT:    return DRM_FORMAT_NV15;
    case MPP_FMT_YUV422SP:          return DRM_FORMAT_NV16;
    default:                        return 0;
    }
}

static uint32_t rkmpp_get_avformat(MppFrameFormat mppformat)
{
    switch (mppformat & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:          return AV_PIX_FMT_NV12;
    case MPP_FMT_YUV420SP_10BIT:    return AV_PIX_FMT_NONE;
    case MPP_FMT_YUV422SP:          return AV_PIX_FMT_NV16;
    default:                        return 0;
    }
}

static uint32_t rkmpp_get_rgaformat(MppFrameFormat mppformat)
{
    switch (mppformat & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:          return RGA_FORMAT_YCbCr_420_SP;
    case MPP_FMT_YUV420SP_10BIT:    return RGA_FORMAT_YCbCr_420_SP_10B;
    case MPP_FMT_YUV422SP:          return RGA_FORMAT_YCbCr_422_SP;
    default:                        return RGA_FORMAT_UNKNOWN;
    }
}

static int rkmpp_close_decoder(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;

    av_packet_unref(&decoder->packet);

    av_buffer_unref(&rk_context->decoder_ref);
    return 0;
}

static void rkmpp_release_decoder(void *opaque, uint8_t *data)
{
    RKMPPDecoder *decoder = (RKMPPDecoder *)data;

    if (decoder->mpi) {
        decoder->mpi->reset(decoder->ctx);
        mpp_destroy(decoder->ctx);
        decoder->ctx = NULL;
    }

    if (decoder->frame_group) {
        mpp_buffer_group_put(decoder->frame_group);
        decoder->frame_group = NULL;
    }

    if (decoder->rga_fd) {
        close(decoder->rga_fd);
        decoder->rga_fd = 0;
    }

    av_buffer_unref(&decoder->frames_ref);
    av_buffer_unref(&decoder->device_ref);

    av_free(decoder);
}

static int rkmpp_init_decoder(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = NULL;
    MppCodingType codectype = MPP_VIDEO_CodingUnused;
    char *env;
    int ret;

    avctx->pix_fmt = ff_get_format(avctx, avctx->codec->pix_fmts);

    // create a decoder and a ref to it
    decoder = av_mallocz(sizeof(RKMPPDecoder));
    if (!decoder) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    env = getenv("FFMPEG_RKMPP_LOG_FPS");
    if (env != NULL)
        decoder->print_fps = !!atoi(env);

    rk_context->decoder_ref = av_buffer_create((uint8_t *)decoder, sizeof(*decoder), rkmpp_release_decoder,
                                               NULL, AV_BUFFER_FLAG_READONLY);
    if (!rk_context->decoder_ref) {
        av_free(decoder);
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    av_log(avctx, AV_LOG_DEBUG, "Initializing RKMPP decoder.\n");

    codectype = rkmpp_get_codingtype(avctx);
    if (codectype == MPP_VIDEO_CodingUnused) {
        av_log(avctx, AV_LOG_ERROR, "Unknown codec type (%d).\n", avctx->codec_id);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    ret = mpp_check_support_format(MPP_CTX_DEC, codectype);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Codec type (%d) unsupported by MPP\n", avctx->codec_id);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    // Create the MPP context
    ret = mpp_create(&decoder->ctx, &decoder->mpi);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create MPP context (code = %d).\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    ret = 1;
    decoder->mpi->control(decoder->ctx, MPP_DEC_SET_PARSER_FAST_MODE, &ret);

    // initialize mpp
    ret = mpp_init(decoder->ctx, MPP_CTX_DEC, codectype);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to initialize MPP context (code = %d).\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    ret = mpp_buffer_group_get_internal(&decoder->frame_group, MPP_BUFFER_TYPE_ION);
    if (ret) {
       av_log(avctx, AV_LOG_ERROR, "Failed to get buffer group (code = %d)\n", ret);
       ret = AVERROR_UNKNOWN;
       goto fail;
    }

    decoder->rga_fd = open("/dev/rga", O_RDWR);
    if (decoder->rga_fd < 0) {
       av_log(avctx, AV_LOG_WARNING, "Failed to open RGA, Falling back to libyuv\n");
    }

    ret = decoder->mpi->control(decoder->ctx, MPP_DEC_SET_EXT_BUF_GROUP, decoder->frame_group);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to assign buffer group (code = %d)\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    decoder->mpi->control(decoder->ctx, MPP_DEC_SET_DISABLE_ERROR, NULL);

    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to prepare decoder (code = %d)\n", ret);
        goto fail;
    }

    av_log(avctx, AV_LOG_DEBUG, "RKMPP decoder initialized successfully.\n");

    decoder->device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_DRM);
    if (!decoder->device_ref) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ret = av_hwdevice_ctx_init(decoder->device_ref);
    if (ret < 0)
        goto fail;

    return 0;

fail:
    av_log(avctx, AV_LOG_ERROR, "Failed to initialize RKMPP decoder.\n");
    rkmpp_close_decoder(avctx);
    return ret;
}

static void rkmpp_release_frame(void *opaque, uint8_t *data)
{
    AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor *)data;
    AVBufferRef *framecontextref = (AVBufferRef *)opaque;
    RKMPPFrameContext *framecontext = (RKMPPFrameContext *)framecontextref->data;

    mpp_frame_deinit(&framecontext->frame);
    av_buffer_unref(&framecontext->decoder_ref);
    av_buffer_unref(&framecontextref);

    av_free(desc);
}

static void rkmpp_release_buffer(void *opaque, uint8_t *data)
{
    MppFrame mppframe = opaque;
    mpp_frame_deinit(&mppframe);
}

static int rkmpp_convert_frame(AVCodecContext *avctx, AVFrame *frame,
                               MppFrame mppframe, MppBuffer buffer)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;

    char *src = mpp_buffer_get_ptr(buffer);
    char *dst_y = frame->data[0];
    char *dst_u = frame->data[1];
    char *dst_v = frame->data[2];
    enum rga_surf_format rgaformat = rkmpp_get_rgaformat(mpp_frame_get_fmt(mppframe));
    int width = mpp_frame_get_width(mppframe);
    int height = mpp_frame_get_height(mppframe);
    int hstride = mpp_frame_get_hor_stride(mppframe);
    int vstride = mpp_frame_get_ver_stride(mppframe);
    int y_pitch = frame->linesize[0];
    int u_pitch = frame->linesize[1];
    int v_pitch = frame->linesize[2];

    int ret;

    int dst_height = (dst_u - dst_y) / y_pitch;

    if (decoder->softconvert || decoder->rga_fd < 0)
        goto softconvert;

    struct rga_req req = {
        .src = {
            .yrgb_addr = mpp_buffer_get_fd(buffer),
            .v_addr = hstride * vstride,
            .format = rgaformat,
            .act_w = width,
            .act_h = height,
            .vir_w = hstride,
            .vir_h = vstride,
            .rd_mode = RGA_RASTER_MODE,
        },
        .dst = {
            .uv_addr = (uintptr_t) dst_y,
            .v_addr = (uintptr_t) dst_u,
            .format = RGA_FORMAT_YCbCr_420_P,
            .act_w = width,
            .act_h = height,
            .vir_w = y_pitch,
            .vir_h = dst_height,
            .rd_mode = RGA_RASTER_MODE,
        },
        .mmu_info = {
            .mmu_en = 1,
            .mmu_flag = 0x80000521,
        },
    };

    ret = ioctl(decoder->rga_fd, RGA_BLIT_SYNC, &req);
    if (ret < 0){
        decoder->softconvert = 1;
        av_log(avctx, AV_LOG_WARNING, "RGA failed with code %d, falling back to soft conversion of uv planes\n", ret);
        goto softconvert;
    }

    rkmpp_release_buffer(mppframe, NULL);
    return 0;

softconvert:
    //data[0] points to buf[1] where the mppbuffer is referenced for y plane
    //so that we can still use y plane without extra copies
    //data[1,2] points to allready allocated AVBuffer Pool (buf[0]), we will convert to
    //that buffer only u+v planes, which is half the size operation
    frame->data[0] = mpp_buffer_get_ptr(buffer);
    frame->buf[1] = av_buffer_create(frame->data[0], mpp_buffer_get_size(buffer),
            rkmpp_release_buffer, mppframe,
            AV_BUFFER_FLAG_READONLY);
    if (!frame->buf[1]) {
        return AVERROR(ENOMEM);
    }
    frame->linesize[0] = hstride;

    src += hstride * vstride;

    SplitUVPlane(src, hstride, dst_u, u_pitch, dst_v, v_pitch,
            (frame->width + 1) >> 1, (frame->height + 1) >> 1);
    return 0;
}

static void rkmpp_update_fps(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    struct timeval tv;
    uint64_t curr_time;
    float fps;

    if (!decoder->print_fps)
        return;

    if (!decoder->last_fps_time) {
        gettimeofday(&tv, NULL);
        decoder->last_fps_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    }

    if (++decoder->frames % FPS_UPDATE_INTERVAL)
        return;

    gettimeofday(&tv, NULL);
    curr_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;

    fps = 1000.0f * FPS_UPDATE_INTERVAL / (curr_time - decoder->last_fps_time);
    decoder->last_fps_time = curr_time;

    av_log(avctx, AV_LOG_INFO,
           "[FFMPEG RKMPP] FPS: %6.1f || Frames: %" PRIu64 "\n",
           fps, decoder->frames);
}

static int rkmpp_get_frame(AVCodecContext *avctx, AVFrame *frame, int timeout)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    RKMPPFrameContext *framecontext = NULL;
    AVBufferRef *framecontextref = NULL;
    int ret;
    MppFrame mppframe = NULL;
    MppBuffer buffer = NULL;
    AVDRMFrameDescriptor *desc = NULL;
    AVDRMLayerDescriptor *layer = NULL;
    int mode;
    MppFrameFormat mppformat;
    uint32_t drmformat;

    // should not provide any frame after EOS
    if (decoder->eos)
        return AVERROR_EOF;

    decoder->mpi->control(decoder->ctx, MPP_SET_OUTPUT_TIMEOUT, (MppParam)&timeout);

    ret = decoder->mpi->decode_get_frame(decoder->ctx, &mppframe);
    if (ret != MPP_OK && ret != MPP_ERR_TIMEOUT) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get frame (code = %d)\n", ret);
        return AVERROR_UNKNOWN;
    }

    if (!mppframe) {
        av_log(avctx, AV_LOG_DEBUG, "Timeout getting decoded frame.\n");
        return AVERROR(EAGAIN);
    }

    if (mpp_frame_get_eos(mppframe)) {
        av_log(avctx, AV_LOG_DEBUG, "Received a EOS frame.\n");
        decoder->eos = 1;
        ret = AVERROR_EOF;
        goto fail;
    }

    if (mpp_frame_get_discard(mppframe)) {
        av_log(avctx, AV_LOG_DEBUG, "Received a discard frame.\n");
        ret = AVERROR(EAGAIN);
        goto fail;
    }

    if (mpp_frame_get_errinfo(mppframe)) {
        av_log(avctx, AV_LOG_ERROR, "Received a errinfo frame.\n");
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    if (mpp_frame_get_info_change(mppframe)) {
        AVHWFramesContext *hwframes;

        av_log(avctx, AV_LOG_INFO, "Decoder noticed an info change (%dx%d), format=%d\n",
               (int)mpp_frame_get_width(mppframe), (int)mpp_frame_get_height(mppframe),
               (int)mpp_frame_get_fmt(mppframe));

        avctx->width = mpp_frame_get_width(mppframe);
        avctx->height = mpp_frame_get_height(mppframe);

        // chromium would align planes' width and height to 32, adding this
        // hack to avoid breaking the plane buffers' contiguous.
        avctx->coded_width = FFALIGN(avctx->width, 64);
        avctx->coded_height = FFALIGN(avctx->height, 64);

        decoder->mpi->control(decoder->ctx, MPP_DEC_SET_FRAME_INFO, (MppParam) mppframe);
        decoder->mpi->control(decoder->ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);

        av_buffer_unref(&decoder->frames_ref);

        decoder->frames_ref = av_hwframe_ctx_alloc(decoder->device_ref);
        if (!decoder->frames_ref) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }

        mppformat = mpp_frame_get_fmt(mppframe);

        hwframes = (AVHWFramesContext*)decoder->frames_ref->data;
        hwframes->format    = AV_PIX_FMT_DRM_PRIME;
        hwframes->sw_format = rkmpp_get_avformat(mppformat);
        hwframes->width     = avctx->width;
        hwframes->height    = avctx->height;
        ret = av_hwframe_ctx_init(decoder->frames_ref);
        if (!ret)
            ret = AVERROR(EAGAIN);

        goto fail;
    }

    // here we should have a valid frame
    av_log(avctx, AV_LOG_DEBUG, "Received a frame.\n");

    // now setup the frame buffer info
    buffer = mpp_frame_get_buffer(mppframe);
    if (!buffer) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get the frame buffer, frame is dropped (code = %d)\n", ret);
        ret = AVERROR(EAGAIN);
        goto fail;
    }

    rkmpp_update_fps(avctx);

    if (avctx->pix_fmt != AV_PIX_FMT_DRM_PRIME) {
        ret = ff_get_buffer(avctx, frame, 0);
        if (ret < 0)
            goto fail;
    }

    // setup general frame fields
    frame->format           = avctx->pix_fmt;
    frame->width            = mpp_frame_get_width(mppframe);
    frame->height           = mpp_frame_get_height(mppframe);
    frame->pts              = mpp_frame_get_pts(mppframe);
    frame->reordered_opaque = frame->pts;
    frame->color_range      = mpp_frame_get_color_range(mppframe);
    frame->color_primaries  = mpp_frame_get_color_primaries(mppframe);
    frame->color_trc        = mpp_frame_get_color_trc(mppframe);
    frame->colorspace       = mpp_frame_get_colorspace(mppframe);

    mode = mpp_frame_get_mode(mppframe);
    frame->interlaced_frame = ((mode & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_DEINTERLACED);
    frame->top_field_first  = ((mode & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_TOP_FIRST);

    if (avctx->pix_fmt != AV_PIX_FMT_DRM_PRIME) {
        return rkmpp_convert_frame(avctx, frame, mppframe, buffer);
    }

    mppformat = mpp_frame_get_fmt(mppframe);
    drmformat = rkmpp_get_frameformat(mppformat);

    desc = av_mallocz(sizeof(AVDRMFrameDescriptor));
    if (!desc) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    desc->nb_objects = 1;
    desc->objects[0].fd = mpp_buffer_get_fd(buffer);
    desc->objects[0].size = mpp_buffer_get_size(buffer);

    desc->nb_layers = 1;
    layer = &desc->layers[0];
    layer->format = drmformat;
    layer->nb_planes = 2;

    layer->planes[0].object_index = 0;
    layer->planes[0].offset = 0;
    layer->planes[0].pitch = mpp_frame_get_hor_stride(mppframe);

    layer->planes[1].object_index = 0;
    layer->planes[1].offset = layer->planes[0].pitch * mpp_frame_get_ver_stride(mppframe);
    layer->planes[1].pitch = layer->planes[0].pitch;

    // we also allocate a struct in buf[0] that will allow to hold additionnal information
    // for releasing properly MPP frames and decoder
    framecontextref = av_buffer_allocz(sizeof(*framecontext));
    if (!framecontextref) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    // MPP decoder needs to be closed only when all frames have been released.
    framecontext = (RKMPPFrameContext *)framecontextref->data;
    framecontext->decoder_ref = av_buffer_ref(rk_context->decoder_ref);
    framecontext->frame = mppframe;

    frame->data[0]  = (uint8_t *)desc;
    frame->buf[0]   = av_buffer_create((uint8_t *)desc, sizeof(*desc), rkmpp_release_frame,
                                       framecontextref, AV_BUFFER_FLAG_READONLY);

    if (!frame->buf[0]) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    frame->hw_frames_ctx = av_buffer_ref(decoder->frames_ref);
    if (!frame->hw_frames_ctx) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    return 0;

fail:
    if (mppframe)
        mpp_frame_deinit(&mppframe);

    if (framecontext)
        av_buffer_unref(&framecontext->decoder_ref);

    if (framecontextref)
        av_buffer_unref(&framecontextref);

    if (desc)
        av_free(desc);

    return ret;
}

static int rkmpp_send_packet(AVCodecContext *avctx, AVPacket *packet)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    MppPacket mpkt;
    int64_t pts = packet->pts;
    int ret;

    if (!pts || pts == AV_NOPTS_VALUE)
        pts = avctx->reordered_opaque;

    ret = mpp_packet_init(&mpkt, packet->data, packet->size);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init MPP packet (code = %d)\n", ret);
        return AVERROR_UNKNOWN;
    }

    mpp_packet_set_pts(mpkt, pts);

    ret = decoder->mpi->decode_put_packet(decoder->ctx, mpkt);
    mpp_packet_deinit(&mpkt);

    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_DEBUG, "Buffer full\n");
        return AVERROR(EAGAIN);
    }

    av_log(avctx, AV_LOG_DEBUG, "Wrote %d bytes to decoder\n", packet->size);
    return 0;
}

static int rkmpp_send_eos(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    MppPacket mpkt;
    int ret;

    ret = mpp_packet_init(&mpkt, NULL, 0);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init EOS packet (code = %d)\n", ret);
        return AVERROR_UNKNOWN;
    }

    mpp_packet_set_eos(mpkt);

    do {
        ret = decoder->mpi->decode_put_packet(decoder->ctx, mpkt);
    } while (ret != MPP_OK);
    mpp_packet_deinit(&mpkt);

    return 0;
}

static int rkmpp_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    AVPacket *packet = &decoder->packet;
    int ret;
    int gettimeout = MPP_TIMEOUT_NON_BLOCK;;

    if (decoder->eos)
        return AVERROR_EOF;

    // get packet if not already available from previous iteration
    if (!packet->size){
        ret = ff_decode_get_packet(avctx, packet);
        if (ret == AVERROR_EOF) {
            av_log(avctx, AV_LOG_DEBUG, "Draining.\n");
            // send EOS and start draining
            rkmpp_send_eos(avctx);
            // we can get all the decoder backlog blocking here
            gettimeout = MPP_TIMEOUT_BLOCK;
        }
    }

    // when there are packets available to push to decoder
    if (packet->size) {
        ret = rkmpp_send_packet(avctx, packet);
        if (ret == AVERROR(EAGAIN)) {
            // decoder input buffer is full, no need to poll packets unless we receive a frame
            gettimeout = MPP_TIMEOUT_BLOCK;
        } else if (ret < 0) {
            // error handling
            av_log(avctx, AV_LOG_ERROR, "Failed to send data (code = %d)\n", ret);
            return ret;
        } else {
            // successful decoder write
            av_packet_unref(packet);
        }
    }

    // always try to consume decoder because it is more likely to be full rather than the packet inputs
    // decoder is the bottleneck here
     ret = rkmpp_get_frame(avctx, frame, gettimeout);
    if (ret == AVERROR_EOF) {
        av_log(avctx, AV_LOG_DEBUG, "End of Stream.\n");
        return rkmpp_get_frame(avctx, frame, MPP_TIMEOUT_BLOCK);
    }

    return ret;
}

static void rkmpp_flush(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;

    av_log(avctx, AV_LOG_DEBUG, "Flush.\n");

    decoder->mpi->reset(decoder->ctx);

    decoder->eos = 0;
    decoder->softconvert = 0;

    decoder->last_fps_time = decoder->frames = 0;

    av_packet_unref(&decoder->packet);
}

static const AVCodecHWConfigInternal *const rkmpp_hw_configs[] = {
    HW_CONFIG_INTERNAL(DRM_PRIME),
    HW_CONFIG_INTERNAL(YUV420P),
    NULL
};

#define RKMPP_DEC_CLASS(NAME) \
    static const AVClass rkmpp_##NAME##_dec_class = { \
        .class_name = "rkmpp_" #NAME "_dec", \
        .version    = LIBAVUTIL_VERSION_INT, \
    };

#define RKMPP_DEC(NAME, ID, BSFS) \
    RKMPP_DEC_CLASS(NAME) \
    AVCodec ff_##NAME##_rkmpp_decoder = { \
        .name           = #NAME "_rkmpp", \
        .long_name      = NULL_IF_CONFIG_SMALL(#NAME " (rkmpp)"), \
        .type           = AVMEDIA_TYPE_VIDEO, \
        .id             = ID, \
        .priv_data_size = sizeof(RKMPPDecodeContext), \
        .init           = rkmpp_init_decoder, \
        .close          = rkmpp_close_decoder, \
        .receive_frame  = rkmpp_receive_frame, \
        .flush          = rkmpp_flush, \
        .priv_class     = &rkmpp_##NAME##_dec_class, \
        .capabilities   = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_AVOID_PROBING | AV_CODEC_CAP_HARDWARE, \
        .caps_internal  = FF_CODEC_CAP_CONTIGUOUS_BUFFERS, \
        .pix_fmts       = (const enum AVPixelFormat[]) { AV_PIX_FMT_DRM_PRIME, \
                                                         AV_PIX_FMT_YUV420P, \
                                                         AV_PIX_FMT_NONE}, \
        .hw_configs     = rkmpp_hw_configs, \
        .bsfs           = BSFS, \
        .wrapper_name   = "rkmpp", \
    };

RKMPP_DEC(h263,  AV_CODEC_ID_H263,          NULL)
RKMPP_DEC(h264,  AV_CODEC_ID_H264,          "h264_mp4toannexb")
RKMPP_DEC(hevc,  AV_CODEC_ID_HEVC,          "hevc_mp4toannexb")
RKMPP_DEC(av1,   AV_CODEC_ID_AV1,           NULL)
RKMPP_DEC(vp8,   AV_CODEC_ID_VP8,           NULL)
RKMPP_DEC(vp9,   AV_CODEC_ID_VP9,           NULL)
RKMPP_DEC(mpeg1, AV_CODEC_ID_MPEG1VIDEO,    NULL)
RKMPP_DEC(mpeg2, AV_CODEC_ID_MPEG2VIDEO,    NULL)
RKMPP_DEC(mpeg4, AV_CODEC_ID_MPEG4,         "mpeg4_unpack_bframes")
