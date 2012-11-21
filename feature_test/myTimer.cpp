//
//  myTimer.cpp
//  feature_test
//
//  Created by 佳久 丸谷 on 12/11/16.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "myTimer.h"

const double myTimer::f = 1000.0/cv::getTickFrequency();

void myTimer::start(){
    myTimer::start_time = cv::getTickCount();
}

double myTimer::stop(){
    return (cv::getTickCount()-myTimer::start_time)*myTimer::f; 
}