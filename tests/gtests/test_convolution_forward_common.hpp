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

#ifndef TEST_CONVOLUTION_FORWARD_COMMON_H
#define TEST_CONVOLUTION_FORWARD_COMMON_H

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
#include <stdint.h>

#include <math.h>

namespace mkldnn {

template <typename data_t>
void print_func(data_t *data_ptr, std::vector<int> c, std::string str){
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

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_fwd(const test_convolution_sizes_t &c,
        const test_convolution_attr_t &attr,
        const memory::desc &src_d,
        const memory::desc &weights_d,
        const memory::desc &bias_d,
        const memory::desc &dst_d,
        const memory &src,
        const memory &weights,
        const memory &bias,
        const memory &dst)
{
    const bool w_bias = bias_d.data.format != memory::format::format_undef;
    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();

    data_t_dst *bias_data = w_bias ? (data_t_dst *)bias.get_data_handle() : nullptr;
    data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

#pragma omp parallel for collapse(5) schedule(static)
    for (int n = 0; n < c.mb; n++) {
        for (int g = 0; g < c.ng; g++) {
            for (int oc = 0; oc < c.oc / c.ng; oc++) {
                for (int oh = 0; oh < c.oh; oh++) {
                    for (int ow = 0; ow < c.ow; ow++) {
                        data_t_acc a = 0;
                        for (int ic = 0; ic < c.ic / c.ng; ic++) {
                            for (int kh = 0; kh < c.kh; kh++) {
                                for (int kw = 0; kw < c.kw; kw++) {
                                    int iw = ow * c.strw
                                          - c.padw + kw * (1 + c.dilw);
                                    int ih = oh * c.strh
                                          - c.padh + kh * (1 + c.dilh);
                                    if (iw < 0 || iw >= c.iw) continue;
                                    if (ih < 0 || ih >= c.ih) continue;
                                    size_t iidx = n * padded_ic * c.ih * c.iw
                                        + g * padded_ic / c.ng * c.ih * c.iw
                                        + ic * c.ih * c.iw + ih * c.iw + iw;
                                    size_t widx = g * padded_oc / c.ng * padded_ic
                                        / c.ng * c.kh * c.kw
                                        + oc * padded_ic / c.ng * c.kh * c.kw
                                        + ic * c.kh * c.kw + kh * c.kw + kw;
                                    a += ((data_t_acc)
                                        src_data[map_index(src_d, iidx)])
                                        *  weights_data[map_index(
                                        weights_d, widx)];
                                }
                            }
                        }

                        float a_fp = (float)a;

                        a_fp += (float)(bias_data ?
                            bias_data[map_index(bias_d,
                                        g * c.oc / c.ng + oc)] :
                            0);


                        if (attr.oscale.is_def()) {
                            const auto &s = attr.oscale;
                            using P = test_convolution_attr_t::scale_t;
                            if (s.policy == P::policy_t::COMMON) {
                                a_fp *= s.scale;
                            }
                        }

                        using D = memory::data_type;
                        if (data_traits<data_t_dst>::data_type != D::f32){
                            using R = mkldnn::round_mode;
                            switch (attr.rmode) {
                                case R::round_down: a_fp = floorf(a_fp); break;
                                case R::round_nearest: a_fp = nearbyintf(a_fp); break;
                            }
                        }

                        size_t oidx = n * padded_oc * c.oh * c.ow
                                 + g * padded_oc / c.ng * c.oh * c.ow
                                 + oc * c.oh * c.ow + oh * c.ow + ow;
                        dst_data[map_index(dst_d, oidx)] = (data_t_dst)a_fp;
                    }
                }
            }
        }
    }
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class convolution_forward_test
        : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<test_convolution_params_t>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        auto p = ::testing::TestWithParam<test_convolution_params_t>::GetParam();
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = engine(p.engine_kind, 0);

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        test_convolution_attr_t attr = p.attr;
        attr.mkldnn_attr_recreate();

        auto aprop_kind = prop_kind::forward;
        bool with_bias = p.formats.bias_format != memory::format::format_undef;

        // convolution memory desc
        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw },
            data_type_src, p.formats.src_format);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        data_type_wei, p.formats.weights_format) :
                create_md({ cd.oc, cd.ic, cd.kh, cd.kw },
                        data_type_wei,p.formats.weights_format);
        auto c_dst_desc = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
                data_type_dst, p.formats.dst_format);
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, data_type_dst, p.formats.bias_format) :
                create_md({}, data_type_dst, p.formats.bias_format);
       
        /* This code use to concat array1 and array2 */
        /* Array1 is the convolution result (array1 denoted as c_dst_desc, c_dst)*/ 
        /* Array2 will write after array1, (array2 denoted as c_src_concat_desc, c_src_concat)*/
        /* The concat result is denoted as (c_dst_concat_desc, c_dst_concat)*/
        /* c_dst_concat = c_dst + c_src_concat*/

        bool with_concat = true;
        int concat_dim = 1; // concat_dim [0, 1, 2, 3] for [n, c, h, w]
        int mb_src = cd.mb, oc_src = cd.oc, oh_src = cd.oh, ow_src = cd.ow; // for array1
        int mb_concat = cd.mb, oc_concat = cd.oc, oh_concat = cd.oh, ow_concat = cd.ow; // for concat result
       
        srand((unsigned)time(NULL));
        int rand_num = rand() % 100;  
        rand_num = 16;
        switch(concat_dim){
            case 0: mb_src = rand_num; mb_concat += mb_src; break;
            case 1: oc_src = rand_num; oc_concat += oc_src; break;
            case 2: oh_src = rand_num; oh_concat += oh_src; break;
            case 3: ow_src = rand_num; ow_concat += ow_src; break;
        }
        
        // concat memory desc
        auto c_src_concat_desc = create_md({ mb_src, oc_src, oh_src, ow_src },
                data_type_dst, p.formats.dst_format);
        auto c_dst_concat_desc = create_md({ mb_concat, oc_concat, oh_concat, ow_concat },
                data_type_dst, p.formats.dst_format);

        // convolution memory
        auto c_src = test_memory(c_src_desc, eng);
        auto c_weights = test_memory(c_weights_desc, eng);
        auto c_bias = test_memory(c_bias_desc, eng);
        auto c_dst = memory(memory::primitive_desc(c_dst_desc, eng));

        // concat memory
        auto c_src_concat = memory(memory::primitive_desc(c_src_concat_desc, eng));
        auto c_dst_concat = memory(memory::primitive_desc(c_dst_concat_desc, eng));
        auto c_dst_concat_fuse = memory(memory::primitive_desc(c_dst_concat_desc, eng));

        // fill data for convolution and concat array1
        // Only true for dense format
        fill_data<data_t_dst>(c_dst.get_primitive_desc().get_size() / sizeof(data_t_dst),
                (data_t_dst *)c_dst.get_data_handle());
        fill_data<data_t_src>(c_src.get_size() / sizeof(data_t_src),
                (data_t_src *)c_src.get().get_data_handle());
        fill_data<data_t_wei>(c_weights.get_size() / sizeof(data_t_wei),
                (data_t_wei *)c_weights.get().get_data_handle());
        if (with_bias) {
            fill_data<data_t_dst>(c_bias.get_size() / sizeof(data_t_dst),
                    (data_t_dst *)c_bias.get().get_data_handle());
        }
        fill_data<data_t_dst>(c_src_concat.get_primitive_desc().get_size()/ sizeof(data_t_dst),
                (data_t_dst *)c_src_concat.get_data_handle());
        fill_data<data_t_dst>(c_dst_concat.get_primitive_desc().get_size()/ sizeof(data_t_dst),
                (data_t_dst *)c_dst_concat.get_data_handle());
        fill_data<data_t_dst>(c_dst_concat_fuse.get_primitive_desc().get_size()/ sizeof(data_t_dst),
                (data_t_dst *)c_dst_concat_fuse.get_data_handle());
        
        check_zero_tail<data_t_src>(1, c_src.get());
        check_zero_tail<data_t_wei>(1, c_weights.get());
        check_zero_tail<data_t_dst>(1, c_dst);

        std::vector<int> padR = {
            right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
            right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)
        };

        auto conv_desc = with_bias
            ? convolution_forward::desc(aprop_kind, p.aalgorithm,
                    c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                    { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                    { cd.padh, cd.padw }, padR, padding_kind::zero)
            : convolution_forward::desc(aprop_kind, p.aalgorithm,
                    c_src_desc, c_weights_desc, c_dst_desc,
                    { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                    { cd.padh, cd.padw }, padR, padding_kind::zero);

        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, attr.mkl_attr, eng);

        auto conv = with_bias ?
            convolution_forward(conv_primitive_desc, c_src.get(),
                    c_weights.get(), c_bias.get(), c_dst) :
            convolution_forward(conv_primitive_desc, c_src.get(),
                    c_weights.get(), c_dst);

        auto mpd1 = c_dst.get_primitive_desc(); 
        auto mpd2 = c_src_concat.get_primitive_desc();
        std::vector<memory::primitive_desc> srcs_pd{mpd2, mpd1};
        std::vector<memory> srcs{c_src_concat, c_dst};
        //std::vector<memory::primitive_desc> srcs_pd{mpd1, mpd2};
        //std::vector<memory> srcs{c_dst, c_src_concat};
        
        auto concat_pd = concat::primitive_desc(c_dst_concat_desc, concat_dim, srcs_pd);
        std::vector<primitive::at> inputs{srcs[0], srcs[1]};
        auto con = concat(concat_pd, inputs, c_dst_concat);

        // compute conv+concat reference by mkl-dnn
        std::vector<primitive> pipeline;
        pipeline.push_back(conv);
        pipeline.push_back(con);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();
       
        // create conv_concat fuse primitive and submit
        if (with_concat) {
            auto  conv_concat_desc = 
                    convolution_forward::desc(aprop_kind, p.aalgorithm,
                    c_src_desc, c_weights_desc, c_bias_desc,
                    c_src_concat_desc,
                    c_dst_desc,
                    c_dst_concat_desc,
                    { concat_dim },
                    { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                    { cd.padh, cd.padw }, padR, padding_kind::zero);
           
           auto conv_concat_primitive_desc = convolution_forward::primitive_desc(
                conv_concat_desc, attr.mkl_attr, eng);

           auto conv_concat = 
                convolution_forward(conv_concat_primitive_desc, c_src.get(),
                    c_weights.get(), c_bias.get(), c_src_concat, c_dst, c_dst_concat_fuse);
       
           std::vector<primitive> pipeline_concat;
           pipeline_concat.push_back(conv_concat);
           auto s = stream(stream::kind::lazy);
           s.submit(pipeline_concat).wait();
        } 
        
        // compare reference non-fuse result with conv_concat fuse result 
        compare_data<data_t_dst>(c_dst_concat, c_dst_concat_fuse);
 
        // print and checkout result 
        /*
        std::vector<int> dst_size{cd.mb, cd.oh, cd.ow, cd.oc};
        data_t_dst * dst_ptr = (data_t_dst *)c_dst.get_data_handle();
        print_func<data_t_dst>(dst_ptr, dst_size, "array1");
        */
        std::vector<int> src_concat_size{mb_src, oh_src, ow_src, oc_src};
        data_t_dst * src_concat_ptr = (data_t_dst *)c_src_concat.get_data_handle();
        print_func<data_t_dst>(src_concat_ptr, src_concat_size, "array2");
        

        std::vector<int> dst_concat_size{mb_concat, oh_concat, ow_concat, oc_concat};
        data_t_dst * dst_concat_ptr = (data_t_dst *)c_dst_concat.get_data_handle();
        print_func<data_t_dst>(dst_concat_ptr, dst_concat_size, "non-fuse result");
        

        data_t_dst * dst_concat_fuse_ptr = (data_t_dst *)c_dst_concat_fuse.get_data_handle();
        print_func<data_t_dst>(dst_concat_fuse_ptr, dst_concat_size, "fuse result");
    }
};

}
#endif
