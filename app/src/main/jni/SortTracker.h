//
// Created by cuongvt on 25/09/2020.
//

#ifndef RTSP_ZMQ_SAST_EXAMPLE_SORTTRACKER_H
#define RTSP_ZMQ_SAST_EXAMPLE_SORTTRACKER_H

#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>
#include <vector>

//#include "opencv2/video/tracking.hpp"
using namespace std;
//using namespace cv;

#include "Hungarian.h"
#include "KalmanTracker.h"
using namespace KalmanTrackerNS;

namespace SortTrackerNS {

typedef struct TrackingBox
{
    int frame;
    int id;
    int boxId;
    Rect_<float> box;
    //other
    int update_hits;
    int predict_hits;

    float vx;
    float vy;
}TrackingBox;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

class SortTracker {
public:
    SortTracker()
    {
        KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
    }
    ~SortTracker(){}

    /*
    static SortTracker& getInstance(){
        static SortTracker theInstance;
        return theInstance;
    }
    */

    /*
    void doPredict(){
        frame_count++;
        if (trackers.size() == 0) return;// the first frame met
        for (auto it = trackers.begin(); it != trackers.end();) {
            Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0){}
            else {
                it = trackers.erase(it);
            };
        }
    }
     */

    //void doSortTrack(vector<TrackingBox> detFrameData)
    //vector<TrackingBox> doSortTrack(vector<TrackingBox> detFrameData)
    vector<TrackingBox> doSortTrack(vector<Rect_<float>> detRectData)
    {
        frame_count++;
        //cout << frame_count << endl;

        // I used to count running time using clock(), but found it seems to conflict with cv::cvWaitkey(),
        // when they both exists, clock() can not get right result. Now I use cv::getTickCount() instead.
        start_time = getTickCount();

        if (trackers.size() == 0) // the first frame met
        {
            // initialize kalman trackers using first detections.
            for (unsigned int i = 0; i < detRectData.size(); i++)//detFrameData[fi]
            {
                KalmanTracker trk = KalmanTracker(detRectData[i]);//detFrameData[i].box
                trackers.push_back(trk);
            }
            // output the first frame detections
            /*
            for (unsigned int id = 0; id < detFrameData.size(); id++)
            {
                TrackingBox tb = detFrameData[id];
                //resultsFile << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
            }
            */
            frameTrackingResult.clear();
            return frameTrackingResult;//null?
        }

        ///////////////////////////////////////
        // 3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();)
        {
            Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                it++;
            }
            else
            {
                it = trackers.erase(it);
                //cerr << "Box invalid at frame: " << frame_count << endl;
            }
        }

        ///////////////////////////////////////
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
        // dets : detFrameData[fi]
        trkNum = predictedBoxes.size();
        detNum = detRectData.size();//detFrameData

        iouMatrix.clear();
        iouMatrix.resize(trkNum, vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++)
            {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detRectData[j]);//detFrameData[j].box
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        assignment.clear();
        HungarianNS::Solve(iouMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }
        else
            ;

        // filter out matched with low IOU
        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        ///////////////////////////////////////
        // 3.3. updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker
        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detRectData[detIdx], detIdx);//detFrameData[detIdx].box
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detRectData[umd]);//detFrameData[umd].box
            trackers.push_back(tracker);
        }

        // get trackers' output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
        {
            TrackingBox res;
            if (((*it).m_time_since_update < 1) &&
                ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
            {
                /*
                TrackingBox res;
                res.box = (*it).get_state();
                res.id = (*it).m_id + 1;
                res.frame = frame_count;
                res.boxId = (*it).m_boxId;//boxId
                frameTrackingResult.push_back(res);
                it++;
                */
                res.boxId = (*it).m_boxId;//boxId
            } else {
                res.boxId = -1;//boxId
                //it++;
            }
            //TODO - cuongvt2 - update include predict
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = frame_count;
            res.update_hits = (*it).m_hit_streak;//kalman.update
            res.predict_hits = (*it).m_time_since_update;//kalman.predict

            res.vx = (*it).get_full_state().at<float>(4,0);
            res.vy = (*it).get_full_state().at<float>(5,0);

            frameTrackingResult.push_back(res);
            it++;

            // remove dead tracklet
            if (it != trackers.end() && (*it).m_time_since_update >= max_age)//>
                it = trackers.erase(it);
        }

        cycle_time = (double)(getTickCount() - start_time);
        //total_time += cycle_time / getTickFrequency();

        return frameTrackingResult;
    }

    //vector<TrackingBox> getTrackResults() { return  frameTrackingResult; }
public:
    int frame_count = 0;
    int max_age = 10;//50;//1s~10frame;
    int min_hits = 2;//3;
    double iouThreshold = 0.01;//0.05;//0.3;
    vector<KalmanTracker> trackers;
private:
    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycle_time = 0.0;
    int64 start_time = 0;
};
}

#endif //RTSP_ZMQ_SAST_EXAMPLE_SORTTRACKER_H
