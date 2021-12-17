package com.qualcomm.qti.snpe.imageclassifiers.sortJni;

import android.graphics.RectF;
import android.util.Log;


import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;
import com.qualcomm.qti.snpe.imageclassifiers.detector.Recognition;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

//import com.securityandsafetythings.examples.tflitedetector.detector.Bbox;

public class nativeObjectTracker {
    static final String LOGTAG = nativeObjectTracker.class.getSimpleName();
    public static final float SORT_PADDING_SCALE = 0.3f;//0.2f;//0.15f;
    public static final int SORTTRACK_TIMEOUT = 20;//1s~10 frame

    private long            mNativePointer;
    private AtomicBoolean   mIsInited;

    public nativeObjectTracker() {
        mNativePointer = initNative();
        mIsInited = new AtomicBoolean(false);
    }
    //init
    public native long initNative();
    //do tracker
    public native TrackUtils.TrackResult[] nativeTrackSort(float[][] bboxesObjArr, boolean isInited, boolean isPredict, long cppSortTrackerPtr);

    public synchronized TrackUtils.TrackResult[] prepareTrackSortFace(List<Bbox> bboxes){
        if(bboxes.size() > 0) {
            mIsInited.set(true);

            float[][] bboxesObjArr = new float[bboxes.size()][4];

            for (int i = 0; i < bboxes.size(); i++) {
                Bbox box = bboxes.get(i);
                //addPaddingScale(box);//for better tracking
                //add padding for better tracking
                float width = box.x2 - box.x1;
                float height = box.y2 - box.y1;
                //
                int paddingWidth = (int)(width * SORT_PADDING_SCALE);
                int paddingHeight = (int)(height * SORT_PADDING_SCALE);
                //
                bboxesObjArr[i] = new float[]{box.x1 - paddingWidth, box.y1 - paddingHeight, box.x2 + paddingWidth, box.y2 + paddingHeight};
            }

            TrackUtils.TrackResult[] trResults = nativeTrackSort(bboxesObjArr, mIsInited.get(), false, mNativePointer);

//            for (TrackUtils.TrackResult tmp : trResults){
//                Log.d(LOGTAG + "_checkResultTrackNative", "native track info: frame=" + tmp.frame +
//                        " id=" + tmp.id +
//                        " boxId=" + tmp.boxId +
//                        " uphit=" + tmp.update_hits +
//                        " prehit="+ tmp.predict_hits +
//                        " box={" + tmp.x1 + "," + tmp.y1 + "," + tmp.width + "," + tmp.height + "}"
//                );
//            }
            Log.d(LOGTAG, "prepareTrackSort " + trResults.length + " boxes " + bboxes.size());
            return trResults;
        } else {
            if(mIsInited.get()) nativeTrackSort(null, mIsInited.get(), true, mNativePointer);
            return null;
        }
    }

    public synchronized void addPaddingScale(Bbox box){
        //add padding for better tracking
        float width = box.x2 - box.x1;
        float height = box.y2 - box.y1;
        //
        int paddingWidth = (int)(width * SORT_PADDING_SCALE);
        int paddingHeight = (int)(height * SORT_PADDING_SCALE);
        //
        box.x1 -= paddingWidth;
        box.x2 += paddingWidth;
        box.y1 -= paddingHeight;
        box.y2 += paddingHeight;
    }
    public synchronized TrackUtils.TrackResult[] prepareTrackSortHuman(List<Recognition> bboxes){
        if(bboxes.size() > 0) {
            mIsInited.set(true);

            float[][] bboxesObjArr = new float[bboxes.size()][4];

            for (int i = 0; i < bboxes.size(); i++) {
                //Recognition box = bboxes.get(i);
                RectF loc = bboxes.get(i).getLocation();
                bboxesObjArr[i] = new float[]{loc.left, loc.top, loc.right, loc.bottom};
            }

            TrackUtils.TrackResult[] trResults = nativeTrackSort(bboxesObjArr, mIsInited.get(), false, mNativePointer);

            Log.d(LOGTAG, "prepareTrackSortHuman " + trResults.length + " mNativePointer " + mNativePointer);
            for (TrackUtils.TrackResult trRes : trResults) {
                double v = Math.sqrt(trRes.vx*trRes.vx + trRes.vy*trRes.vy);
            }

            return trResults;
        }else {
            if(mIsInited.get()) nativeTrackSort(null, mIsInited.get(), true, mNativePointer);
            return null;
        }
    }
    //do predict
    //public native void predictNative(long cppSortTrackerPtr);
    //public synchronized void predict(){predictNative(mNativePointer);}
    //relese
    public native void releaseNative(long cppSortTrackerPtr);
    public void release(){releaseNative(mNativePointer);}
}