#include <ros/ros.h>
#include "HackathonPipeline.h"
#include "HackathonPipeline.cpp"
#include <std::vector>
#include <opencv2>
#include <igvc_msgs/lineArray>
using namespace grip;
using namespace cv;
ros::Publisher out_pub;



int main(int argc, char** argv){
    ros::init(argc, argv, "openCV");
    ros::NodeHandle nh;
    Mat frame;
    VideoCapture cap;
    cap.open(0 + cv::CAP_ANY);
    if(!cap.isOpened()){
        cerr<< "cant open camera :(" << endl;
    }
    cap.read(frame);

    HackathonPipeline::Process(frame);
    vector<Line> lines = HackathonPipeline::GetFilterLinesOutput();
    out_pub = nh.advertise("/openCV/lines", 1);
    ros::spin()


}