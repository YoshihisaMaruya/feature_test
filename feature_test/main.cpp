/***************************
 detect_image_sampleのデバッグ用ネイティブ
 互換性があるよう、主要な部分は同じ形式(c)で書く
 ***************************/

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

//for OpenCV 2.4.x (2.3.x系の場合はいらない)
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp>

//my define
#include "myTimer.h"
#include "myCreateFeature.h"
#include "myMatcherTest.h"


using namespace cv;
using namespace std;

int face[] = {cv::FONT_HERSHEY_SIMPLEX, cv::FONT_HERSHEY_PLAIN, cv::FONT_HERSHEY_DUPLEX, cv::FONT_HERSHEY_COMPLEX, 
    cv::FONT_HERSHEY_TRIPLEX, cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 
    cv::FONT_HERSHEY_SCRIPT_COMPLEX, cv::FONT_ITALIC};


//macherの結果
typedef struct MATCHED_INFO{
    int id;
    int similarity;
}matched_info;


//NDKでも動くようにcで書く
extern "C"{
    // グローバル変数
    float THRESHOLD = 45; //閾値
    const int IMAGE_NUM = 10;   //学習画像の枚数
    
    OrbFeatureDetector orb_detector(300); //ORB特徴点検出器
    OrbDescriptorExtractor orb_extractor; //ORB特徴量抽出機
    
    SurfFeatureDetector surf_detector(1000); //SURF特徴点検出器
    SurfDescriptorExtractor surf_extractor; //SURF特徴量抽出機
    
    FeatureDetector* detector =  &surf_detector; 
    DescriptorExtractor* extractor = &surf_extractor;
    
   // BruteForceMatcher< Hamming > haming_matcher; //特徴量照合器(hamming)
    BruteForceMatcher< L2<float> > norm_matcher; //特徴量照合機(norm)
   // FlannBasedMatcher flan_matcher;
   // gpu::BruteForceMatcher_GPU< L2<float> > norm_matcher_gpu;
   // BruteForceMatcher_GPU_base< Hamming> haming_matcher_gpu;
    
    
    
    /*
     refer : Java_com_example_detectimage_DetectImageActivity_setTrainingImages
     全部まねると面倒なので、ファイル読み込み部分は妥協
     */
    void setTrainingImages(vector<Mat> training_images){
        vector<Mat> trainDescriptorses;
        vector<KeyPoint>trainKeypoints;
        Mat trainDescriptors;
        
        ////各画像に対し、特徴量を抽出し特徴量照合器(matcher)へ登録
        for(int i = 0; i < IMAGE_NUM; i++){
            Mat ti = training_images[i];
            //matをグレースケールに変換
            Mat gray(ti.rows,ti.cols,CV_8UC1);
            cvtColor(ti,gray,CV_RGBA2GRAY,0);
            
            //TODO: surfの場合normalizeが必要か
            normalize(gray, gray, 0, 255, NORM_MINMAX);
            //END
            detector->detect(gray, trainKeypoints); // 特徴点をtrainKeypointsへ格納
            extractor->compute(gray, trainKeypoints, trainDescriptors); //各特徴点の特徴ベクトルをtrainDescriptorsへ格納
            trainDescriptorses.push_back(trainDescriptors);
        }
        norm_matcher.add(trainDescriptorses);//照合器へ全ての学習画像の特徴ベクトルを登録
    }
    
    /*
        取得した画像をフレームごとに解析
     */
    matched_info detectImage(Mat capture_image){
        vector<KeyPoint> queryKeypoints;
        Mat queryDescriptors;
        
        //matをグレースケールに変換
        Mat gray(capture_image.rows,capture_image.cols,CV_8UC1);
        cvtColor(capture_image,gray,CV_RGBA2GRAY,0);
        

        detector->detect(gray, queryKeypoints); //特徴点を検出
        extractor->compute(gray, queryKeypoints, queryDescriptors); //特徴量を検出
        
        
        // BrustForceMatcher による画像マッチング
       /* vector<DMatch> matches;
        norm_matcher.match(queryDescriptors, matches);
        
        int votes[IMAGE_NUM]; // 学習画像の投票箱
        for(int i = 0; i < IMAGE_NUM; i++) votes[i] = 0;
        
        // キャプチャ画像の各特徴点に対して、ハミング距離が閾値より小さい特徴点を持つ学習画像へ投票
        for(int i = 0; i < matches.size(); i++){
            if(matches[i].distance < THRESHOLD){
                votes[matches[i].imgIdx]++;
            }
        }
        
        // 投票数の多い画像のIDを調査
        int maxImageId = -1;
        int maxVotes = 0;
        for(int i = 0; i < IMAGE_NUM; i++){
            if(votes[i] > maxVotes){
                maxImageId = i;  //マッチした特徴点を一番多く持つ学習画像のID
                maxVotes = votes[i]; //マッチした特徴点の数
            }
        }
        
        vector<Mat> trainDescs = norm_matcher.getTrainDescriptors();
        
        float similarity = (float)maxVotes/trainDescs[maxImageId].rows*100;
        if(similarity < 5){
            maxImageId = -1; // マッチした特徴点の数が全体の5%より少なければ、未検出とする
        }*/
               
        int maxImageId = 0;
        int similarity = 0;
        matched_info mi;
        mi.id = maxImageId;
        mi.similarity = similarity;
            
        return mi;
    }
    
}



int main () {
    
   /* VideoCapture cap(0);
    Mat frm;
    vector<Mat> training_images;
    
    for(int i = 1; i <= IMAGE_NUM; i++){
        char str_path[100];
        sprintf(str_path,"/Applications/workspace/OpenCV/native/feature_test/feature_test/training_images/%d.jpg",i);
        string sp = string(str_path);
        cout << sp << endl;
        training_images.push_back( imread(sp) );
    }
    
    setTrainingImages(training_images);
    
    while (waitKey(1) != 32) {
        myTimer::getInstance().start();
        
        char str[100];
        cap >> frm;
        matched_info mi = detectImage(frm);
  
        double total_time = myTimer::getInstance().stop();

        sprintf(str,"id : %d, similarity : %d, fps : %lf",mi.id, mi.similarity,1 / (total_time * 0.001));
        putText(frm, str, cv::Point(50,50), face[0], 1.2, cv::Scalar(0,0,200), 2, CV_AA);
        
        imshow("sample", frm);
    }*/
    
    /*myCreateFeature::getInstance().execute("/Applications/workspace/OpenCV/native/feature_test/mst_data/training_images", "/Applications/workspace/OpenCV/native/feature_test/mst_data/detector", 
        "/Applications/workspace/OpenCV/native/feature_test/mst_data/extractor",10);*/
       myMatcherTest* mmt = new myMatcherTest();
       mmt->read_from_mst_training_images("/Applications/workspace/OpenCV/native/feature_test/mst_data/training_images", 20);
      mmt->execute();
     free(mmt);
       
}



