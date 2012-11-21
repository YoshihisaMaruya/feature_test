//
//  myMatcherTest.h
//  feature_test
//
//  Created by 佳久 丸谷 on 12/11/18.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//  投票機の性能評価を行う

#ifndef feature_test_myMatcherTest_h
#define feature_test_myMatcherTest_h
#include<string>
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
//for OpenCV 2.4.x (2.3.x系の場合はいらない)
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp>
using namespace std;
using namespace cv;
class myMatcherTest{
private:
    vector<Mat> descriptors;
    string IntToString(const int);
    int exe_vote(const vector<DMatch>);
    Mat mst_data;
    void exe_brute(const string);
    void exe_flan(const string);
    void exe_lsh(const string);
    int mst_feature_image_count;
public:
    myMatcherTest();
    void read_from_mst_training_images(const std::string, const int);
    void execute();
};


#endif
