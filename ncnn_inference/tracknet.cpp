// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/dnn.hpp>

#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>

cv::Mat hwc2chw(const cv::Mat& src_mat9) 
{
    std::vector<cv::Mat> bgr_channels(9);
    cv::split(src_mat9, bgr_channels);
    for (size_t i = 0; i < bgr_channels.size(); i++)
    {
        bgr_channels[i] = bgr_channels[i].reshape(1, 1); // reshape为1通道，1行，n列
    }
    cv::Mat dst_mat;
    cv::hconcat(bgr_channels, dst_mat);
    return dst_mat;
}


void print_shape(const ncnn::Mat &in)
{
    std::cout << "d: " << in.d << " c: " << in.c << " w: " << in.w << " h: " << in.h << " cstep: " << in.cstep << std::endl;
}

std::tuple<int, int, int> get_shuttle_position(const cv::Mat binary_pred)
{
    if (cv::countNonZero(binary_pred) <= 0)
     {
        // (visible, cx, cy)
        return std::make_tuple(0, 0, 0);
    } 
    else
    {
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(binary_pred, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        assert(cnts.size()!= 0);

        std::vector<cv::Rect> rects;
        for (const auto& ctr : cnts) {
            rects.push_back(cv::boundingRect(ctr));
        }

        int max_area_idx = 0;
        int max_area = rects[max_area_idx].width * rects[max_area_idx].height;

        for (size_t ii = 0; ii < rects.size(); ++ii) {
            int area = rects[ii].width * rects[ii].height;
            if (area > max_area) {
                max_area_idx = ii;
                max_area = area;
            }
        }

        cv::Rect target = rects[max_area_idx];
        int cx = target.x + target.width / 2;
        int cy = target.y + target.height / 2;

        // (visible, cx, cy)
        return std::make_tuple(1, cx, cy);
    }
}


static int detect_tracknet(const char* video_path)
{
    ncnn::Net tracknet;

    // GPU
    tracknet.opt.use_vulkan_compute = false;

    if(tracknet.load_param("./last_opt.ncnn.param"))
        exit(-1);
    if(tracknet.load_model("./last_opt.ncnn.bin"))
        exit(-1);


    cv::VideoCapture vid_cap(video_path);
    bool video_end = false;

    int video_len = vid_cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = vid_cap.get(cv::CAP_PROP_FPS);
    int w = vid_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = vid_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    int iw = 512;
    int ih = 288;
    int ic = 9;

    int count = 0;
    while (vid_cap.isOpened()) 
    {
        std::vector<cv::Mat> imgs;
        for (int i = 0; i < 3; ++i) 
        {
            cv::Mat img;
            bool ret = vid_cap.read(img);
            if (!ret) 
            {
                video_end = true;
                break;
            }

            imgs.push_back(img);
        }

        if (video_end) 
            break;

        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<cv::Mat> imgs_hwc;
        for (int i=0; i<3; i++)
        {
            cv::Mat img;
            imgs[i].convertTo(img, CV_32F, 1.0 / 255.0);

            cv::resize(img, img, cv::Size(iw, ih), 0, 0, cv::INTER_LINEAR);

            std::vector<cv::Mat> bgr_channels(3);
            cv::split(img, bgr_channels);

            // 210: bgr - > rgb
            imgs_hwc.push_back(bgr_channels[2].reshape(1, 1));
            imgs_hwc.push_back(bgr_channels[1].reshape(1, 1));
            imgs_hwc.push_back(bgr_channels[0].reshape(1, 1));
        }

        // inference need chw !
        cv::Mat imgs_chw;
        cv::hconcat(imgs_hwc, imgs_chw);
        
        // d = 1,c = 9, w = 512, h = 288, cstep = 147456
        ncnn::Mat in(iw, ih, ic, (void*)imgs_chw.data);
        // print_shape(in);
        std::chrono::duration<double, std::milli> elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << "Preprocess time taken: " << elapsed.count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        ncnn::Extractor ex = tracknet.create_extractor();
        ex.input("in0", in);

        ncnn::Mat out;
        ex.extract("out0", out);
        elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << "Inference time taken: " << elapsed.count() << " ms" << std::endl;

        // post process
        // c = 3
        start = std::chrono::high_resolution_clock::now();
        for (int i=0; i<out.c; i++)
        {
            cv::Mat pred(out.h, out.w, CV_32FC1, (void*)(const float*)out.channel(i));
            pred.convertTo(pred, CV_8U, 255.0);

            cv::Mat binary_pred;
            cv::threshold(pred, binary_pred, 127, 255, cv::THRESH_BINARY);

            auto [visible, cx_pred, cy_pred] = get_shuttle_position(binary_pred);
            int cx = int(cx_pred*w/iw);
            int cy = int(cy_pred*h/ih);

            std::cout << visible << " " << cx << " " << cy << std::endl;

            // cv::circle(imgs[i], cv::Point(cx, cy), 8, cv::Scalar(0, 0, 255), -1);
            // cv::imshow("predict", imgs[i]);
            // cv::imwrite("./predict/" + std::to_string(count) + ".png", imgs[i]);
            count++;
        }
        elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << "Postprocess time taken: " << elapsed.count() << " ms" << std::endl << std::endl;

    }

    return 0;
}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s video_path\n", argv[0]);
        return -1;
    }

    const char* video_path = argv[1];

    detect_tracknet(video_path);

    return 0;
}
