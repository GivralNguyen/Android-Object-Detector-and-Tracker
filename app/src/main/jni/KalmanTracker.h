//
// Created by cuongvt on 25/09/2020.
//

#ifndef RTSP_ZMQ_SAST_EXAMPLE_KALMANTRACKER_H
#define RTSP_ZMQ_SAST_EXAMPLE_KALMANTRACKER_H

#include "opencv2/video/tracking.hpp"
//#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

namespace KalmanTrackerNS {

#define StateType Rect_<float>

// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker {
private:
    void init_kf(StateType stateMat)
    {
        int stateNum = 7;
        int measureNum = 4;
        kf = KalmanFilter(stateNum, measureNum, 0);

        measurement = Mat::zeros(measureNum, 1, CV_32F);

        kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
                1, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 1);

        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
        setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(kf.errorCovPost, Scalar::all(1));

        // initialize state vector with bounding box in [cx,cy,s,r] style
        kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
        kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
        kf.statePost.at<float>(2, 0) = stateMat.area();
        kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
    }
public:
    KalmanTracker()
    {
        init_kf(StateType());
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        //kf_count++;
        m_boxId = -1;
    }

    KalmanTracker(StateType initRect)
    {
        init_kf(initRect);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;
        m_boxId = -1;
    }

    ~KalmanTracker()
    {
        m_history.clear();
    }

    StateType predict()
    {
        // predict
        Mat p = kf.predict();
        m_age += 1;

        if (m_time_since_update > 0)
            m_hit_streak = 0;
        m_time_since_update += 1;

        StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

        m_history.push_back(predictBox);
        return m_history.back();
    }

    //void update(StateType stateMat)
    void update(StateType stateMat, int boxId)
    {
        m_time_since_update = 0;
        m_history.clear();
        m_hits += 1;
        m_hit_streak += 1;

        // measurement
        measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
        measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
        measurement.at<float>(2, 0) = stateMat.area();
        measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

        // update
        kf.correct(measurement);
        //box id
        m_boxId = boxId;
    }

    StateType get_state()
    {
        Mat s = kf.statePost;
        return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
    }

    Mat get_full_state(){
        Mat s = kf.statePost;
        return s;
    }


    StateType get_rect_xysr(float cx, float cy, float s, float r)
    {
        float w = sqrt(s * r);
        float h = s / w;
        float x = (cx - w / 2);
        float y = (cy - h / 2);

        if (x < 0 && cx > 0)
            x = 0;
        if (y < 0 && cy > 0)
            y = 0;

        return StateType(x, y, w, h);
    }

public:
    static int kf_count;

    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;
    int m_boxId;

private:
    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<StateType> m_history;
};//end class KalmanTracker

int KalmanTracker::kf_count = 0;

}//end namespace KalmanTracker

#endif //RTSP_ZMQ_SAST_EXAMPLE_KALMANTRACKER_H
