//
//  myCreateFeature.h
//  feature_test
//
//  Created by 佳久 丸谷 on 12/11/17.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//  カラーイメージから特徴量を作成するクラス

#ifndef feature_test_myCreateFeature_h
#define feature_test_myCreateFeature_h

#include <string>
#include <iostream>

class myCreateFeature{
private:
    std::string IntToString(int);
    void writeSurfFeature(cv::Mat,const std::string,const std::string);
    void writeOrbFeature(cv::Mat,const std::string,const std::string);
public:
    void execute(const std::string,const std::string,const std::string,const int);
    static myCreateFeature&  getInstance(){
        static myCreateFeature singleton;
        return singleton;
    }
};

#endif
