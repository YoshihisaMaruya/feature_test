//
//  myMatcherTest.cpp
//  feature_test
//
//  Created by 佳久 丸谷 on 12/11/18.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//  特徴量検出器の時間測定をするクラス

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
//for OpenCV 2.4.x (2.3.x系の場合はいらない)
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "myMatcherTest.h"
#include "myTimer.h"

using namespace std;
using namespace cv;

string myMatcherTest::IntToString(const int number)
{
    stringstream ss;
    ss << number;
    return ss.str();
}

myMatcherTest::myMatcherTest(){
    
    
}

//mst_dataから読み込む
void myMatcherTest::read_from_mst_training_images(const string dir_path, const int count)
{
    mst_feature_image_count = count;
    for(int i = 0; i < count; i++){
        vector<Mat> trainDescriptorses;
        Mat training_image_mat = imread(dir_path + "/" + IntToString(i) + ".jpg");
        //SURF検出器に基づく特徴点検出
        SurfFeatureDetector surf_detector; //SURF特徴点検出器 TODO: 引数について: http://opencv.jp/opencv-2.2/c/features2d_feature_detection_and_description.html
        vector<KeyPoint> trainKeypoints;
        vector<KeyPoint>::iterator itk;
        Scalar color(100,255,50);
        Mat gray(training_image_mat.rows,training_image_mat.cols,CV_8UC1); //グレーイメジに変換
        cvtColor(training_image_mat,gray,CV_RGBA2GRAY,0);
        normalize(gray, gray, 0, 255, NORM_MINMAX);
        surf_detector.detect(gray,trainKeypoints);
        // SURFに基づくディスクリプタ抽出器
        SurfDescriptorExtractor surf_extractor; //SURF特徴量抽出機
        Mat trainDescriptors;
        surf_extractor.compute(gray,trainKeypoints,trainDescriptors);
        descriptors.push_back(trainDescriptors);
        if(i == 1) mst_data = trainDescriptors;
        
    }
}

vector<Mat> getDesVectors(std::vector<Mat>::iterator begin,std::vector<Mat>::iterator end){
    std::vector<Mat>::iterator mti;
    vector<Mat> result;
    for(mti = begin; mti != end; ++mti){
        result.push_back(*mti);
    }
    return result;
}

//投票実行
int myMatcherTest::exe_vote(const vector<DMatch> matches){
    int votes[mst_feature_image_count];
    for(int i = 0; i < mst_feature_image_count; i++) votes[i] = 0;
    for(int i = 0; i < matches.size(); i++){
        votes[matches[i].imgIdx]++;
    }
    
    // 投票数の多い画像のIDを調査
    int maxImageId = -1;
    int maxVotes = 0;
    for(int i = 0; i < mst_feature_image_count; i++){
        if(votes[i] > maxVotes){
            maxImageId = i;  //マッチした特徴点を一番多く持つ学習画像のID
            maxVotes = votes[i]; //マッチした特徴点の数
        }
    }
    return maxImageId;
}

const int exc_num = 5;
//brute(線形)
void myMatcherTest::exe_brute(const string result_file_path){
    std::vector<Mat>::iterator mti;
    //
    ofstream ofs(result_file_path.c_str(),ios::trunc);
    int i = 1;
    ofs << "count,time,id" << endl;
    for(mti = descriptors.begin(); mti!=descriptors.end(); ++mti) {
        BruteForceMatcher< L2<float> > matcher; //特徴量照合器
        matcher.add(getDesVectors(descriptors.begin(), mti));
        double sum_result_time = 0;
        int result = -1;
        for(int j = 0; j < exc_num; j++){
            vector<DMatch> matches;
            myTimer::getInstance().start();
            matcher.match(mst_data,  matches);
            sum_result_time += myTimer::getInstance().stop();
            result = exe_vote(matches);
        }
        ofs << i << "," << sum_result_time / exc_num << "," << result << endl;
        i++;
    }
    ofs.close();
}

//flann(kd-tree)
void myMatcherTest::exe_flan(const string result_file_path){
    std::vector<Mat>::iterator mti;
    //
    ofstream ofs(result_file_path.c_str(),ios::trunc);
    int i = 1;
    ofs << "count,time,id" << endl;
    for(mti = descriptors.begin(); mti!=descriptors.end(); ++mti) {
        FlannBasedMatcher matcher; //特徴量照合器

        matcher.add(getDesVectors(descriptors.begin(), mti));
        double sum_result_time = 0;
        int result = -1;
        for(int j = 0; j < exc_num; j++){
            vector<DMatch> matches;
            myTimer::getInstance().start();
            matcher.match(mst_data,  matches);
            sum_result_time += myTimer::getInstance().stop();       
            result = exe_vote(matches);
        }
        ofs << i << "," << sum_result_time / exc_num << "," << result << endl;
        i++;
    }
    ofs.close();
}

//lsh
void myMatcherTest::exe_lsh(const string rfp){
    
}

//matcernに１つずつ特徴量を追加していく
void myMatcherTest::execute(){
    exe_brute("/Applications/workspace/OpenCV/native/feature_test/brute_result.csv");
    exe_flan("/Applications/workspace/OpenCV/native/feature_test/flan_result.csv");
}