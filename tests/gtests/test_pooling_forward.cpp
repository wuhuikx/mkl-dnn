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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"
#include <omp.h>
#include <numeric>
#include "mkldnn.hpp"
#include <iostream>
#include <sys/time.h>

void clear_cache(char* p, size_t n) {
#pragma omp parallel for 
     for (size_t i = 0; i < n; ++i) {
         char write = char(3*i), read = char(4*i);  // give a little cal
         *(p+i) = read;
         read = p[i];
         *(p+i) = write;
     }
}


//namespace mkldnn {
using namespace mkldnn;

struct test_pool_desc_t {
    int mb, c;
    int ih, iw;
    int oh, ow;
    int kh, kw;
    int padt, padl;
    int strh, strw;
};

struct pool_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    algorithm aalgorithm;
    memory::format src_format;
    memory::format dst_format;
    test_pool_desc_t test_pd;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};


template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_relu_fwd(const test_convolution_sizes_t &c,
      const memory &src, const memory &weights, const memory &bias,
      const memory &dst, bool w_bias, float negative_slope)
{
     data_t_src *src_data = (data_t_src *)src.get_data_handle();
     data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();
     data_t_dst *bias_data
             = (data_t_dst *)(w_bias ? bias.get_data_handle() : nullptr);
     data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

     const memory::desc src_d = src.get_primitive_desc().desc();
     const memory::desc weights_d = weights.get_primitive_desc().desc();
     const memory::desc dst_d = dst.get_primitive_desc().desc();

#pragma omp parallel for collapse(5) schedule(static)
     for (int n = 0; n < c.mb; n++) {
         for (int g = 0; g < c.ng; g++) {
             for (int oc = 0; oc < c.oc / c.ng; oc++) {
                 for (int oh = 0; oh < c.oh; oh++) {
                     for (int ow = 0; ow < c.ow; ow++) {
                         int oidx = n * c.oc * c.oh * c.ow
                                 + g * c.oc / c.ng * c.oh * c.ow
                                 + oc * c.oh * c.ow + oh * c.ow + ow;
                         dst_data[map_index(dst_d, oidx)] = bias_data ?
                                 bias_data[map_index(
                                         bias.get_primitive_desc().desc(),
                                         g * c.oc / c.ng + oc)] :
                                 data_t_dst{0};

                         for (int ic = 0; ic < c.ic / c.ng; ic++) {
                             for (int kh = 0; kh < c.kh; kh++) {
                                 for (int kw = 0; kw < c.kw; kw++) {
                                     int iw = ow * c.strw
                                           - c.padw + kw * (1 + c.dilw);
                                     int ih = oh * c.strh
                                         - c.padh + kh * (1 + c.dilh);
                                     if (iw < 0 || iw >= c.iw) continue;
                                     if (ih < 0 || ih >= c.ih) continue;
                                     int iidx = n * c.ic * c.ih * c.iw
                                         + g * c.ic / c.ng * c.ih * c.iw
                                         + ic * c.ih * c.iw + ih * c.iw + iw;
                                     int widx = g * c.oc / c.ng * c.ic
                                         / c.ng * c.kh * c.kw
                                         + oc * c.ic / c.ng * c.kh * c.kw
                                         + ic * c.kh * c.kw + kh * c.kw + kw;

                                     dst_data[map_index(dst_d, oidx)]
                                         += src_data[map_index(src_d, iidx)]
                                         * weights_data[map_index(
                                                 weights_d, widx)];
                                }
                             }
                         }

                         if (dst_data[map_index(dst_d, oidx)] < 0) {
                             dst_data[map_index(dst_d, oidx)] =
                                 static_cast<data_t_dst>( negative_slope
                                         * dst_data[map_index(dst_d, oidx)] );
                         }
                     }
                 }
             }
         }
     }
}







template <typename data_t>
void check_pool_fwd(const pool_test_params &p, const memory &src,
        const memory &dst, const memory &ws)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    auto ws_data = [=](size_t idx) -> int {
        auto w = (unsigned char *)ws.get_data_handle();
        if (w == nullptr) return -1;
        if (ws.get_primitive_desc().desc().data.data_type == mkldnn_u8)
            return (int)w[idx];
        else
            return ((int *)w)[idx];
    };

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();
    const memory::desc ws_d  = ws.get_primitive_desc().desc();

    auto pd = p.test_pd;

#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < pd.mb; n++) {
        for (int c = 0; c < pd.c; c++) {
            for (int oh = 0; oh < pd.oh; oh++) {
                for (int ow = 0; ow < pd.ow; ow++) {
                    int oidx = n * pd.c * pd.oh * pd.ow + c * pd.oh * pd.ow
                            + oh * pd.ow + ow;
                    data_t out = dst_data[map_index(dst_d, oidx)];
                    int out_index = -1;
                    if(p.aalgorithm == pooling_max
                        && p.aprop_kind == prop_kind::forward_training) {
                        out_index = ws_data(map_index(ws_d, oidx));
                    }
                    data_t out_ref = data_t(0);
                    int out_ref_index = 0;
                    bool is_initialized = false;
                    int num_summands = 0;

                    for (int kh = 0; kh < pd.kh; ++kh) {
                        for (int kw = 0; kw < pd.kw; ++kw) {
                            const int ih = oh * pd.strh - pd.padt + kh;
                            const int iw = ow * pd.strw - pd.padl + kw;

                            if (ih < 0 || ih >= pd.ih) continue;
                            if (iw < 0 || iw >= pd.iw) continue;

                            int iidx = n * pd.c * pd.ih * pd.iw
                                    + c * pd.ih * pd.iw + ih * pd.iw + iw;

                            data_t d = src_data[map_index(src_d, iidx)];
                            if (p.aalgorithm == pooling_max) {
                                if (!is_initialized) {
                                    out_ref = d;
                                    out_ref_index = kh* pd.kh + kw;
                                    is_initialized = true;
                                } else {
                                    if (out_ref < d) {
                                        out_ref = d;
                                        out_ref_index = kh* pd.kh + kw;
                                    }
                                }
                            } else if (p.aalgorithm == pooling_avg_include_padding ||
                                       p.aalgorithm == pooling_avg_exclude_padding) {
                                out_ref += d;
                                num_summands++;
                            }
                        }
                    }

                    if (p.aalgorithm == pooling_avg_include_padding) {
                        num_summands = pd.kw * pd.kh;
                    }

                    if (p.aalgorithm == pooling_avg_include_padding ||
                        p.aalgorithm == pooling_avg_exclude_padding) {
                        out_ref = out_round<data_t>(
                                (float)out_ref / num_summands);
                    }
                    EXPECT_NEAR(out, out_ref, 1e-6);
                    if(p.aalgorithm == pooling_max
                        && p.aprop_kind == forward_training) {
                        EXPECT_EQ(out_index, out_ref_index) << " n = " << n
                             << " c = " << c << " oh = " << oh << " ow = " << ow;
                    }
                }
            }
        }
    }
}


template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class conv_relu_pooling_test : public ::testing::TestWithParam<pool_test_params> {
protected:
    virtual void SetUp()
    {
       
/************************ create conv primitive ************************/       
       
        int batch_size = 2;
        int group_num = 1;
        int conv_ic = 16, conv_oc = 16;
        int conv_ih = 4, conv_iw = 4;
        int conv_oh = 2, conv_ow = 2;
        int kh = 3, kw = 3;
        int padh = 0, padw = 0;
        int strh = 1, strw = 1;
        float negative_slope = 0.0;
        auto eng = engine(engine::cpu, 0);

        test_convolution_sizes_t cd(batch_size, 
                                    group_num,
                                    conv_ic, conv_ih, conv_iw,
                                    conv_oc, conv_oh, conv_ow,
                                    kh, kw,
                                    padh, padw,
                                    strh, strw);

       /*********** create conv memory *************/
        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;

        auto c_src_desc = create_md({cd.mb, cd.ic, cd.ih, cd.iw},
                data_type_src, memory::format::nhwc);
        auto c_weights_desc = create_md({cd.oc, cd.ic, cd.kh, cd.kw},
                data_type_wei, memory::format::OIhw4i16o4i);
        auto c_dst_desc = create_md({cd.mb, cd.oc, cd.oh, cd.ow},
                data_type_dst, memory::format::nhwc);
        auto c_bias_desc = create_md({cd.oc}, data_type_dst, memory::format::x);

        auto c_src = memory({c_src_desc, eng});
        auto c_weights = memory({c_weights_desc, eng});
        auto c_dst = memory({c_dst_desc, eng});
        auto c_bias = memory({c_bias_desc, eng});

        auto dst_ref = memory({c_dst_desc, eng});

        fill_data<data_t_src>(c_src.get_primitive_desc().get_size() / sizeof(data_t_src),
                (data_t_src *)c_src.get_data_handle());
        data_t_src *src_data = (data_t_src *)c_src.get_data_handle();
        const int mb_chunk = static_cast<int>(
            (c_src.get_primitive_desc().get_size() / sizeof(data_t_src))
            / cd.mb );
        for (int i = 0; i < cd.mb * mb_chunk; ++i) {
            if ((i / mb_chunk) % 2) src_data[i] *= (data_t_src)-1.;
        }

        fill_data<data_t_wei>(c_weights.get_primitive_desc().get_size() / sizeof(data_t_wei),
                (data_t_wei *)c_weights.get_data_handle());
        fill_data<data_t_dst>(c_bias.get_primitive_desc().get_size() / sizeof(data_t_dst),
                (data_t_dst *)c_bias.get_data_handle(), 1., true);

       // data_t_src *src_ptr = (data_t_src *)c_src.get_data_handle();
       // for (int i = 0; i < cd.ih * cd.iw * cd.ic; ++i)
       //     std::cout << "conv_src = " << int(*(src_ptr + i)) << std::endl;
        
       // data_t_wei *wei_ptr = (data_t_wei *)c_weights.get_data_handle();
       // for (int i = 0; i < cd.oc * cd.kh * cd.kw * cd.ic; ++i)
       //     std::cout << "conv_wei = " << int(*(wei_ptr + i)) << std::endl;
        
       
        std::vector<int> padR_conv = {cd.padh, cd.padw};
        for (int i = 0; i < 2; ++i) {
              if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR_conv[0])
                  / cd.strh + 1 != cd.oh)
                  ++padR_conv[0];
              if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR_conv[1])
                  / cd.strw + 1 != cd.ow)
                  ++padR_conv[1];
        }

      /********** create conv primitive ***********/
       auto conv_desc = 
           convolution_forward::desc(prop_kind::forward_scoring, convolution_direct,
               c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
               {cd.strh, cd.strw}, {cd.padh, cd.padw}, padR_conv, padding_kind::zero);
       auto conv_relu_desc = 
           convolution_relu_forward::desc(conv_desc, negative_slope);

       auto conv_primitive_desc = 
           convolution_relu_forward::primitive_desc( conv_relu_desc, eng);
       auto conv = convolution_relu_forward(conv_primitive_desc,
               c_src, c_weights, c_bias, c_dst);


       std::vector<primitive> pipeline;
       pipeline.push_back(conv);
       //stream(stream::kind::lazy).submit(pipeline).wait();

       //compute_ref_conv_relu_fwd<data_t_src, data_t_wei, data_t_wei, data_t_dst>(
       //        cd, c_src, c_weights, c_bias, dst_ref, true, negative_slope);
       //compare_data<data_t_dst>(c_dst, dst_ref);

       //data_t_dst *dst_ptr = (data_t_dst *)c_dst.get_data_handle();
       //for (int i = 0; i < cd.oc * cd.oh * cd.ow; ++i)
       //     std::cout << "conv_dst = " << int(*(dst_ptr + i)) << std::endl;


//#ifdef pooling
/************************ create pooling primitive ************************/       
        pool_test_params p
                = ::testing::TestWithParam<pool_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        //auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t_dst>::data_type;
/*
struct test_pool_desc_t {
    int mb, c;
    int ih, iw;
    int oh, ow;
    int kh, kw;
    int padt, padl;
    int strh, strw;
};
*/
   
       test_pool_desc_t pd = p.test_pd;
       /* test_pool_desc_t pd{
            batch_size, conv_oc,
            conv_oh, conv_ow,
            conv_oh / 2, conv_ow / 2,
            2, 2,
            0, 0,
            2, 2     
        };
*/
        auto p_src_desc
                = create_md({ pd.mb, pd.c, pd.ih, pd.iw }, data_type, p.src_format);
        auto p_dst_desc
                = create_md({ pd.mb, pd.c, pd.oh, pd.ow }, data_type, p.dst_format);

        //auto p_src = memory({p_src_desc, eng});
        auto p_src = c_dst;
        auto p_dst = memory({p_dst_desc, eng});

       // fill_data<data_t_dst>(p_src.get_primitive_desc().get_size()/ sizeof(data_t_dst),
       //         (data_t_dst *)p_src.get_data_handle());

       // data_t_dst *data_ptr = (data_t_dst *)p_src.get_data_handle();
       // for (int i = 0; i < cd.oh * cd.ow * cd.oc; ++i)
       //     std::cout << "conv_result = " << *(data_ptr + i) << std::endl;
        /*
        data_t_dst *data_ptr = (data_t_dst *)p_src.get_data_handle();
        std::cout<< "pd.ih = " << pd.ih << std::endl;
        std::cout<< "pd.iw = " << pd.iw << std::endl;
        for (int i = 0; i < pd.ih * pd.iw * pd.c; ++i)
            *(data_ptr + i) = data_t_dst(0); */

       /* for (int i = 0; i < pd.ih; i=i+2){
            //for (int j = 0; j < pd.iw; ++j){
                *(data_ptr + i * pd.iw + 0) = data_t_dst(1);
                *(data_ptr + i * pd.iw + 1) = data_t_dst(0);
                *(data_ptr + i * pd.iw + 2) = data_t_dst(1);
                *(data_ptr + i * pd.iw + 3) = data_t_dst(0);
            //}
                *(data_ptr + (i+1) * pd.iw + 0) = data_t_dst(0);
                *(data_ptr + (i+1) * pd.iw + 1) = data_t_dst(0);
                *(data_ptr + (i+1) * pd.iw + 2) = data_t_dst(0);
                *(data_ptr + (i+1) * pd.iw + 3) = data_t_dst(0);
        }*/
       /* for (int i = 0; i < pd.ih * pd.iw; ++i)
             std::cout<< "data = " << *(data_ptr + i) << std::endl;
         */   

        std::vector<int> padR = { pd.padt, pd.padl };
        for (int i = 0; i < 2; ++i) {
        if ((pd.ih + pd.padt + padR[0] - pd.kh)/pd.strh + 1 < pd.oh) ++padR[0];
        if ((pd.iw + pd.padl + padR[1] - pd.kw)/pd.strw + 1 < pd.ow) ++padR[1];
        }

        std::shared_ptr<memory> p_workspace;

        auto test = [&]() {
            auto pool_desc = pooling_forward::desc(p.aprop_kind, p.aalgorithm,
                    p_src_desc, p_dst_desc,
                    {pd.strh, pd.strw}, {pd.kh, pd.kw}, {pd.padt, pd.padl},
                    padR, padding_kind::zero);

            auto pool_prim_desc
                = pooling_forward::primitive_desc(pool_desc, eng);

            bool with_workspace = true
                && p.aprop_kind == prop_kind::forward_training
                && p.aalgorithm == pooling_max;
            auto p_workspace_desc = with_workspace
                ? pool_prim_desc.workspace_primitive_desc()
                : memory::primitive_desc( {{}, data_type, p.dst_format}, eng);
            p_workspace.reset(new memory(p_workspace_desc));

            auto pool = with_workspace
                ? pooling_forward(pool_prim_desc, p_src, p_dst, *p_workspace)
                : pooling_forward(pool_prim_desc, p_src, p_dst);

            pipeline.push_back(pool);

           // stream(stream::kind::lazy).submit(pipeline).wait();
          auto get_current_ms = []() -> double {
                struct timeval time;
                gettimeofday(&time, NULL);
                return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
           };
         
            const size_t PAGE_4M = 4 * 1024 *1024;   // skx, L3: 1.375M*n
            int max_nthr = omp_get_max_threads();
            std::cout <<"max threads: " << max_nthr << std::endl;
            const size_t total_size = PAGE_4M * max_nthr;
            char* dummy_data = (char*) malloc(total_size);
 
            int burning_iter = 1;
            double sum_time = 0;
            int count_time = 0;
            for (auto i = 0; i < burning_iter; ++i) {
                clear_cache(dummy_data, total_size);
                
                auto s1 = get_current_ms();
                stream(stream::kind::lazy).submit(pipeline).wait();
                auto s2 = get_current_ms();
                sum_time += (s2 - s1);

                clear_cache(dummy_data, total_size);
            }
            std::cout << "avg time: " << sum_time / (double) burning_iter << " ms" << std::endl;

        };


       if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
            return;
//#endif
       check_pool_fwd<data_t_dst>(p, p_src, p_dst, *p_workspace);
        //data_t *dst_ptr = (data_t *)p_dst.get_data_handle();
        //std::cout<<*dst_ptr<<std::endl;
    }
};
//using pooling_test_s8 = pooling_test<int8_t>;
using pooling_test_s32 = conv_relu_pooling_test<uint8_t, int8_t, int32_t, int32_t>;
TEST_P(pooling_test_s32, TestsPooling)
{
}
INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardS8, pooling_test_s32, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 16, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2 } } //,
           // pool_test_params{ prop_kind::forward_inference,
           // engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
           // memory::format::nhwc, {1, 256, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));
/*
        TestPoolingForwardAvgS8, pooling_test_s8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
            ));
*/
/*
using pooling_test_float = pooling_test<float>;
using pooling_test_s8 = pooling_test<int8_t>;
using pooling_test_u8 = pooling_test<uint8_t>;
using pooling_test_s32 = pooling_test<int32_t>;
using pool_test_params_float = pool_test_params;
TEST_P(pooling_test_s8, TestsPooling)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardS8, pooling_test_s8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 96, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 256, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 256, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxS8, pooling_test_s8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgS8, pooling_test_s8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

TEST_P(pooling_test_u8, TestsPooling)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxU8, pooling_test_u8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgU8, pooling_test_u8, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

TEST_P(pooling_test_s32, TestsPooling)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardS32, pooling_test_s32, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 96, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 256, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {1, 256, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxS32, pooling_test_s32, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgS32, pooling_test_s32, ::testing::Values(
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_include_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding,
            memory::format::nhwc, memory::format::nhwc,
            {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

TEST_P(pooling_test_float, TestsPooling)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardEF, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 0, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
            true, mkldnn_invalid_arguments},
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 0, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
            true, mkldnn_invalid_arguments},
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 0, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
            true, mkldnn_invalid_arguments},
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 7, 7, 3, 3, 1, 1, 1, 1 },
            true, mkldnn_invalid_arguments},
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 4, 2, 3, 3, 1, 1, 1, 1 },
            true, mkldnn_invalid_arguments}
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMax, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
            ));


INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxNHWC, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
            memory::format::nhwc, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }

            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlockedPerf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgBlockedPerf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 1, 8, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }

            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardMaxBlocked16Perf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardAvgBlocked16Perf, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
            algorithm::pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));


INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardMaxNCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardMaxBlocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardMaxBlocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxBlockedStride1, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxCIFAR10NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgCIFAR10NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxCIFAR10Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgCIFAR10Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxCIFAR10Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgCIFAR10Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxGoogleNetV1NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxGoogleNetV1Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxGoogleNetV1Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxResnet50NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
            memory::format::nchw, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxResnet50Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingMaxResnet50Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgGoogleNetV1NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgGoogleNetV1Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgGoogleNetV1Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgResnet50NCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nchw, memory::format::nchw,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgResnet50Blocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAvgResnet50Blocked16, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw16c, memory::format::nChw16c,
            { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAsymmPadding, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}

            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}

            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_include_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
            ,pool_test_params_float{ prop_kind::forward_inference,
            engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
            memory::format::nChw8c, memory::format::nChw8c,
            {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }

            ));
*/
//}
