//
//  myTimer.h
//  feature_test
//
//  Created by 佳久 丸谷 on 12/11/16.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//  OpenCV時間測定用クラス

#ifndef feature_test_myTimer_h
#define feature_test_myTimer_h
#include <opencv2/core/core.hpp>

class myTimer{
private:
    myTimer(void){}
    int64 start_time;
    static const double f;
public:
    void start(void);
    double stop(void);
    static myTimer&  getInstance(){
        static myTimer singleton;
        return singleton;
    }
};
#endif
