package com.qualcomm.qti.snpe.imageclassifiers.sortJni;

import android.graphics.Bitmap;


import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;

import java.util.HashMap;

public class TrackLabel {
    public enum StatusPosition {
        isLeft,
        isRight,
        NA
    }
    public enum StatusGoing {
        comeIn,
        goOut,
        NA
    }

    public TrackUtils.TrackResult trackResult;

    public Bbox lastBbox;
    //public Recognition lastRecognition;
    public String label = "";
    public long lastTimeUpdated = System.currentTimeMillis();
    public Bitmap eventFace;
    //public Bitmap alignSave;
    //public Mat alignMat;
    public int mergeOldId = -1;
    public int frameCount;
    public HashMap<String, Float> scoreMap = new HashMap<>();

    public TrackLabel(TrackUtils.TrackResult trackResult){
        this.trackResult = trackResult;
        this.frameCount = 0;
    }
}