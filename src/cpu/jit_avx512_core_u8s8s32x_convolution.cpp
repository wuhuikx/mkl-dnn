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

template <typename data_t>
void print_func(const data_t *data_ptr, std::vector<int> c, std::string str){
     for (int mb = 0; mb < c[0]; ++mb) {
         for (int oh = 0; oh < c[1]; ++oh) {
             for (int ow = 0; ow < c[2]; ++ow) {
                 std::cout << "--------------ow = " << ow << std::endl;
                 for (int oc = 0; oc < c[3]; ++oc) {
                     int index = mb * c[1] * c[2] * c[3] +
                                 oh * c[2] * c[3] +
                                 ow * c[3] +
                                 oc;
                     std::cout<< str << " = " << int(*(data_ptr + index)) << std::endl;
                     if ((oc + 1)%16 == 0) {
                         std::cout<< " " << std::endl;
                     }
                 }
             }
         }
     }
}



template <bool with_relu, data_type_t dst_type>
void _jit_avx512_core_u8s8s32x_convolution_fwd_t<with_relu, dst_type>::
execute_forward()
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory(0));

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));
    
    const size_t bia_dt_size = conf_.with_bias()
        ? types::data_type_size(conf_.cdesc()->bias_desc.data_type) : 0;

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    const auto &oscales = conf_.attr()->output_scales_;

    const dst_data_t *src_concat = dst;
    dst_data_t *dst_concat = dst;
    dst_data_t *dst_concat_tmp = dst;
    if (jcp.with_concat) {
       src_concat = reinterpret_cast<const dst_data_t *>(this->input_memory(3));
       dst_concat = reinterpret_cast<dst_data_t *>(this->memory(1));
       dst_concat_tmp = reinterpret_cast<dst_data_t *>(this->memory(1));
       //dst_concat_tmp = dst_concat;

       const memory_desc_wrapper dst_concat_d(conf_.cdesc()->dst_concat_desc);
       format_perm(dst_concat_d.ndims(), dst_concat_d.blocking_desc().strides[0], perm_,  iperm_);
       int *perm = perm_, *iperm = iperm_;
            
       const memory_desc_wrapper src_concat_d(conf_.cdesc()->src_concat_desc); 
       int concat_dim = jcp.concat_dim;
       int i = std::max(0, perm[concat_dim]-1);
       int o_d_blk_off = size_t(src_concat_d.blocking_desc().strides[0][iperm[i]]); 
       dst_concat = dst_concat + o_d_blk_off;

       //std::vector<int> src_concat_size = {src_concat_d.dims()[0], src_concat_d.dims()[2], src_concat_d.dims()[1], src_concat_d.dims()[3]};
       //print_func<dst_data_t>(src_concat, src_concat_size, "src_concat");
      
       //std::cout << "------ printf" << std::endl;
       //int num = 0;
       //for(int i=0; i<128; i++){
       //   if(num == 16){
       //      printf("\n");
       //      num = 0;
       //   }
       //   printf("%d,", int(src_concat[i]));
       //   num ++;
                                    }
       //   printf("\n"); 
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
 
        size_t dst_concat_h_stride = 0;
        if (jcp.with_concat) {
            const memory_desc_wrapper dst_concat_d(conf_.cdesc()->dst_concat_desc);
            dst_concat_h_stride = dst_concat_d.blk_off(0, 0, 1);
            //size_t dst_concat_h_stride = jcp.ow_concat * jcp.oc_concat;
       
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

            auto dst_w = dst + dst_d.blk_off(n, g_oc, oh_s);
            auto src_w = src + src_d.blk_off(n, g_ic, ih_s);
            auto wht_w = weights + wht_blk_off(weights_d, gb, ocb, 0);
          
            auto dst_w_concat = dst_w;
            if (jcp.with_concat) {
               size_t dst_concat_blk_off = n * jcp.oh_concat * jcp.ow_concat * jcp.oc_concat
                             + oh_s * jcp.ow_concat * jcp.oc_concat
                             + g_oc;
               dst_w_concat = dst_concat + dst_concat_blk_off;
            }

            auto scales = &oscales.scales_[jcp.is_oc_scale * g_oc];

            for (int icc = 0; icc < ic_chunks; ++icc) {
                auto src_c = src_w;
                auto dst_c = dst_w;
                auto ws_c = ws_l;
                auto dst_c_concat = dst_w_concat;  

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
                    p.acc_s32 = ws_c;
                    p.channel = icb;
                    p.kh_padding = kh_padding;
                    p.scales = scales;
                    
                    if (jcp.with_concat) {
                       p.dst_concat = dst_c_concat;
                    }

                    kernel_->jit_ker(&p);

                    src_c += src_h_stride * jcp.stride_h;
                    dst_c += dst_h_stride;
                    ws_c += jcp.ow * jcp.oc_block * jcp.nb_oc_blocking;
                    if (jcp.with_concat) {
                       dst_c_concat += dst_concat_h_stride;
                    }
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
    
    //std::vector<int> dst_size = {jcp.mb, jcp.oh, jcp.ow, jcp.oc};
    //print_func<dst_data_t>(dst, dst_size, "dst");
    
   
    if (jcp.with_concat) {
       int num_arrs = 2;
       int max_num_arrs = 12;
       const dst_data_t *input_ptrs[max_num_arrs];
       dst_data_t *output_ptrs[max_num_arrs];
       size_t nelems_to_copy[max_num_arrs];
       strides_t is[max_num_arrs];
       int concat_dim = jcp.concat_dim;
       
       const memory_desc_wrapper dst_concat_d(conf_.cdesc()->dst_concat_desc);
       format_perm(dst_concat_d.ndims(), dst_concat_d.blocking_desc().strides[0], perm_,  iperm_);
       int *perm = perm_, *iperm = iperm_;
      
       //int current_concat_dim_offset = 0;
       for (int a = 1; a < num_arrs; ++a) {
            const memory_desc_wrapper i_d(conf_.cdesc()->src_concat_desc); 
            //const memory_desc_wrapper o_d(conf_.cdesc()->dst_concat_desc);
           
            //int i = std::max(0, perm[concat_dim]-1);
            //int o_d_blk_off = size_t(dst_d.blocking_desc().strides[0][iperm[i]]);
            input_ptrs[a] =  src_concat + i_d.blk_off(0);
            //output_ptrs[a] = dst_concat + o_d_blk_off;
            output_ptrs[a] = dst_concat_tmp;
            nelems_to_copy[a] = nelems_to_concat(concat_dim, perm, iperm, i_d);
            for (int i = 0; i < perm[concat_dim]; i++) {
                 is[a][i] = size_t(i_d.blocking_desc().strides[0][iperm[i]]);
            }
       }
        
       const memory_desc_wrapper o_d(conf_.cdesc()->dst_concat_desc);
       auto &blk = o_d.blocking_desc();
       strides_t os = { 0 };
       for (int i = 0; i < perm[concat_dim]; i++)
           os[i] = o_d.blocking_desc().strides[0][iperm[i]];
       dims_t phys_dims;
       for (size_t i = 0; i < sizeof(phys_dims)/sizeof(phys_dims[0]); i++)
            phys_dims[i] = (i < (size_t)perm[concat_dim]) ?
                  o_d.dims()[iperm[i]] / blk.block_dims[iperm[i]] : 1;

       switch (perm[concat_dim]) {
       case (0): {
            for (int a = 1; a < num_arrs; ++a) {
                const dst_data_t *i = &input_ptrs[a][0];
                dst_data_t *o = &output_ptrs[a][0];
#               pragma omp parallel for
                for (ptrdiff_t e = 0; e < (ptrdiff_t)nelems_to_copy[a]; ++e)
                    o[e] = i[e];
            }
            break;
       }
       default: {
#           pragma omp parallel for collapse(6) schedule(static)
            for (int n0 = 0; n0 < phys_dims[0]; ++n0)
                for (int n1 = 0; n1 < phys_dims[1]; ++n1)
                    for (int n2 = 0; n2 < phys_dims[2]; ++n2)
                        for (int n3 = 0; n3 < phys_dims[3]; ++n3)
                            for (int n4 = 0; n4 < phys_dims[4]; ++n4)
                                for (int a = 1; a < num_arrs; ++a) {
                                    size_t in_off = is[a][0] * n0 + is[a][1] * n1+ is[a][2] * n2 + is[a][3] * n3 + is[a][4] * n4;
                                    size_t out_off = os[0] * n0 + os[1] * n1 + os[2] * n2 + os[3] * n3 + os[4] * n4;
                                    const dst_data_t *i = &input_ptrs[a][in_off];
                                    dst_data_t *o = &output_ptrs[a][out_off];
                                    PRAGMA_OMP_SIMD()
                                    for (size_t e = 0; e < nelems_to_copy[a]; ++e)
                                        o[e] = i[e];
                                }
       }
       }
       //std::vector<int> dst_concat_size = {jcp.mb, jcp.oh_concat, jcp.ow_concat, jcp.oc_concat};
       //print_func<dst_data_t>(dst_concat, dst_concat_size, "dst_concat");
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
