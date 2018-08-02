/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "mkldnn_types.h"
#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_u8s8s32x_convolution.hpp"
#include <iostream>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_call_s *);

#define wht_blk_off(d, g, ...) \
        (conf_.with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

template <bool with_relu, data_type_t dst_type>
void _jit_avx512_core_u8s8s32x_convolution_fwd_t<with_relu, dst_type>::
execute_forward()
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());


    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const size_t bia_dt_size = conf_.with_bias()
        ? types::data_type_size(conf_.cdesc()->bias_desc.data_type) : 0;
    const size_t padding_dt_size = conf_.with_value_padding()
        ? types::data_type_size(conf_.cdesc()->padding_desc.data_type) : 0;

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    auto padding = src;
    if (jcp.with_value_padding) {
        padding = reinterpret_cast<const src_data_t *>(this->input_memory(3));
    }

    const auto &oscales = conf_.attr()->output_scales_;
    
    if (jcp.with_value_padding) {
        auto src_tmp = reinterpret_cast<const wei_data_t *>(this->input_memory(0));
       /*
       for (int mb = 0; mb < jcp_padding.mb; ++mb) {
            for (int ih = 0; ih < jcp_padding.ih; ++ih) {
                 for (int iw = 0; iw < jcp_padding.iw; ++iw) {
                      for (int ic = 0; ic < jcp_padding.ic; ++ic) {
                           int index = mb * jcp_padding.ih * jcp_padding.iw * jcp_padding.ic
                                + ih * jcp_padding.iw * jcp_padding.ic
                                + iw * jcp_padding.ic
                                + ic;
                            std::cout << "src_origin = " << int(*(src_tmp + index)) << std::endl;             
                       }
                  }
             }
       }
    */
        
        auto jcp_padding = jcp_origin;
        int r_pad = nstl::max(0, (jcp_padding.ow - 1) * jcp_padding.stride_w
                + (jcp_padding.kw - 1) * (jcp_padding.dilate_w + 1)
                - (jcp_padding.iw + jcp_padding.l_pad - 1));
        int b_pad = nstl::max(0, (jcp_padding.oh - 1) * jcp_padding.stride_h
                + (jcp_padding.kh - 1) * (jcp_padding.dilate_h + 1)
                - (jcp_padding.ih + jcp_padding.t_pad - 1));
        int iw_with_pad = jcp_padding.iw + jcp_padding.l_pad + r_pad;
        int ih_with_pad = jcp_padding.ih + jcp_padding.t_pad + b_pad;
 
       for (int mb = 0; mb < jcp_padding.mb; ++mb) {
           for (int ih = 0; ih < ih_with_pad; ++ih) {
               for (int iw = 0; iw < iw_with_pad; ++iw) {
                   for (int ic = 0; ic < jcp_padding.ic; ++ic) {
                       int index = mb * ih_with_pad * iw_with_pad * jcp_padding.ic
                           + ih * iw_with_pad * jcp_padding.ic
                           + iw * jcp_padding.ic
                           + ic;
                       if (iw < jcp_padding.l_pad || iw >= jcp_padding.l_pad + jcp_padding.iw ||
                           ih < jcp_padding.t_pad || ih >= jcp_padding.t_pad + jcp_padding.ih) {
                           *(src_with_pad + index) = *(padding + ic);
                       } else {
                           int index_src = mb * jcp_padding.ih * jcp_padding.iw * jcp_padding.ic 
                              + (ih - jcp_padding.t_pad) * jcp_padding.iw * jcp_padding.ic 
                              + (iw - jcp_padding.l_pad) * jcp_padding.ic
                              + ic;
                           *(src_with_pad + index) = src_data_t(*(src_tmp + index_src) + abs(*(padding + ic)));
                       } 
                       //std::cout << "src_with_pad = " << *(src_with_pad + index) << std::endl;                     
                   }
               }
           }
       }
       src = reinterpret_cast<src_data_t *>(src_with_pad);
    }

#   pragma omp parallel
    {
        int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();

        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int nb_groups = jcp.nb_ch;
        int group_block = jcp.ch_block;

        int start{0}, end{0};
        int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();

        auto ws_l = ws_ + ithr * ws_per_thread_;

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);
        if (jcp.with_value_padding) {
           src_h_stride = jcp.iw * jcp.ic;
        }

        int n{0}, gb{0}, occ{0}, oh_s{0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, occ, oc_chunks, gb, nb_groups, n, jcp.mb,
                    oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start, gb, nb_groups, n, jcp.mb, occ, oc_chunks,
                    oh_s, jcp.oh);
        else if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, n, jcp.mb, gb, nb_groups, occ, oc_chunks,
                    oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");
        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g = gb * group_block;
            int g_oc = (g * jcp.nb_oc + ocb) * jcp.oc_block;

            int g_ic = g * jcp.nb_ic * jcp.oc_block;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            auto bias_w = bias ? bias + (bias_d.blk_off(g_oc) * bia_dt_size) : 0;
            auto padding_w = padding; 
            if (jcp.with_value_padding) {
               padding_w = padding + (bias_d.blk_off(g_oc) * padding_dt_size);
             
            }
            auto dst_w = dst + dst_d.blk_off(n, g_oc, oh_s);
            auto src_w = src + src_d.blk_off(n, g_ic, ih_s);
            auto wht_w = weights + wht_blk_off(weights_d, gb, ocb, 0);
            size_t src_blk_off = src_d.blk_off(n, g_ic, ih_s);
            if (jcp.with_value_padding) {
               src_blk_off = n * jcp.ic * jcp.ih * jcp.iw
                    + ih_s * jcp.iw * jcp.ic + g_ic;
               src_w = src + src_blk_off; 
            }

            auto scales = &oscales.scales_[jcp.is_oc_scale * g_oc];

            for (int icc = 0; icc < ic_chunks; ++icc) {
                auto src_c = src_w;
                auto dst_c = dst_w;
                auto ws_c = ws_l;

                int icb = icc * jcp.nb_ic_blocking;

                for (int oj = oh_s, ij = ih_s;
                        oj < oh_e; ++oj, ij += jcp.stride_h)
                {
                    int dilate_h = jcp.dilate_h + 1;
                    int i_t_overflow = div_up(max(0, -ij), dilate_h);
                    int i_b_overflow = div_up(
                            max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                            dilate_h);
                    int kh_padding = nstl::max(0,
                        jcp.kh - i_t_overflow - i_b_overflow);

                    p.src = src_c + i_t_overflow * dilate_h * src_h_stride;
                    p.dst = dst_c;
                    p.filt = wht_w + i_t_overflow * wht_h_stride;
                    p.bias = bias_w;
                    p.padding = padding_w;
                    p.acc_s32 = ws_c;
                    p.channel = icb;
                    p.kh_padding = kh_padding;
                    p.scales = scales;

                    kernel_->jit_ker(&p);

                    src_c += src_h_stride * jcp.stride_h;
                    dst_c += dst_h_stride;
                    ws_c += jcp.ow * jcp.oc_block * jcp.nb_oc_blocking;
                }
                src_w += jcp.ic_block * jcp.nb_ic_blocking;
                wht_w += wht_ic_stride * jcp.nb_ic_blocking;
            }
            if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end, occ, oc_chunks, gb, nb_groups, n,
                        jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_jump(start, end, gb, nb_groups, n, jcp.mb, occ,
                        oc_chunks, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_ngc)
                nd_iterator_jump(start, end, n, jcp.mb, gb, nb_groups, occ,
                        oc_chunks, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    }
}

template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<false, data_type::u8>;
template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<true, data_type::u8>;

template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<false, data_type::s8>;
template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<true, data_type::s8>;

template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<false, data_type::s32>;
template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<true, data_type::s32>;

template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<false, data_type::f32>;
template struct _jit_avx512_core_u8s8s32x_convolution_fwd_t<true, data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
