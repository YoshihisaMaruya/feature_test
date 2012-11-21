//
//  myCreateFeature.cpp
//  feature_test
//
//  Created by 佳久 丸谷 on 12/11/17.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//

#include <iostream>
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
#include "myCreateFeature.h"

using namespace cv;
using namespace std;

string myCreateFeature::IntToString(const int number)
{
    stringstream ss;
    ss << number;
    return ss.str();
}



//SURFの特徴点、特徴量書き出し
void myCreateFeature::writeSurfFeature(Mat training_image_mat,const string detector_file_name,const string extactor_file_name){
    
    //SURF検出器に基づく特徴点検出
    SurfFeatureDetector surf_detector; //SURF特徴点検出器 TODO: 引数について: http://opencv.jp/opencv-2.2/c/features2d_feature_detection_and_description.html
    vector<KeyPoint> trainKeypoints;
    vector<KeyPoint>::iterator itk;
    Scalar color(100,255,50);
    Mat gray(training_image_mat.rows,training_image_mat.cols,CV_8UC1); //グレーイメジに変換
    cvtColor(training_image_mat,gray,CV_RGBA2GRAY,0);
    normalize(gray, gray, 0, 255, NORM_MINMAX);
    surf_detector.detect(gray,trainKeypoints);
    
    //特徴点を表示した画像を、指定されたファイルに書き出し
    for(itk = trainKeypoints.begin(); itk != trainKeypoints.end(); ++ itk){
        circle(training_image_mat,itk->pt,1,color,-1);
        //circle(training_image_mat, itk->pt, itk->size, color, 1, CV_AA);
        /*if(itk->angle >= 0){
            Point pt2(itk->pt.x + cos(itk->angle) * itk->size, itk->pt.y + sin(itk->angle) * itk->size);
            line(training_image_mat,itk->pt,pt2,color,1,CV_AA);
        }*/
        imwrite(detector_file_name, training_image_mat);
    }
    
    
    
    // SURFに基づくディスクリプタ抽出器
    SurfDescriptorExtractor surf_extractor; //SURF特徴量抽出機
    Mat trainDescriptors;
    surf_extractor.compute(gray,trainKeypoints,trainDescriptors);
    ofstream ofs(extactor_file_name.c_str() ,ios::trunc);
    ofs << trainKeypoints.size() << " " << trainDescriptors.cols <<endl;
    for(int i = 0; i < trainDescriptors.rows; ++i){
        Mat d(trainDescriptors,Rect(0,i,trainDescriptors.cols,1));
        cout<< d.cols << endl;
        ofs << i << ": " << d << endl;
    }
    ofs.close();
}

//Orbの特徴点、特徴量書き出し
void myCreateFeature::writeOrbFeature(Mat training_image_mat,const string detector_file_name,const string extactor_file_name){

    cv::Mat gray_img;
    cv::cvtColor(training_image_mat, gray_img, CV_BGR2GRAY);
    cv::normalize(gray_img, gray_img, 0, 255, cv::NORM_MINMAX);
    
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint>::iterator itk;
    
    // ORB 検出器に基づく特徴点検出
    // n_features=300, params=default
    cv::OrbFeatureDetector detector;
    cv::Scalar color(200,250,255);
    detector.detect(gray_img, keypoints);
    for(itk = keypoints.begin(); itk!=keypoints.end(); ++itk) {
        cv::circle(training_image_mat, itk->pt, 1, color, -1);
       // cv::circle(training_image_mat, itk->pt, itk->size, color, 1, CV_AA);
    }
    imwrite(detector_file_name, training_image_mat);
    
    // ORB に基づくディスクリプタ抽出器
    Mat descriptors;
    cv::OrbDescriptorExtractor extractor;
    extractor.compute(gray_img, keypoints, descriptors);
    
    ofstream ofs(extactor_file_name.c_str() ,ios::trunc);
    ofs << keypoints.size() << " " << descriptors.cols <<endl;
    // 32次元の特徴量 x keypoint数
    for(int i=0; i<descriptors.rows; ++i) {
        Mat d(descriptors, Rect(0,i,descriptors.cols,1));
        ofs << i << ": " << d << endl;
    }
}

//
void myCreateFeature::execute(const std::string training_iamge_dir_path,const std::string detector_path,const std::string extozoractor_path,const int image_count){
    for(int i = 0; i < image_count; i++){
        string fp = training_iamge_dir_path + "/" + IntToString(i) + ".jpg";
        Mat training_image_mat = imread(training_iamge_dir_path + "/" + IntToString(i) + ".jpg");
        string detector_file_name = detector_path + "/" + IntToString(i) + ".jpg";
        string extactor_file_name = extozoractor_path + "/" + IntToString(i) + ".se";
        writeSurfFeature(training_image_mat, detector_file_name,extactor_file_name);
    }
    while(1);
    
}