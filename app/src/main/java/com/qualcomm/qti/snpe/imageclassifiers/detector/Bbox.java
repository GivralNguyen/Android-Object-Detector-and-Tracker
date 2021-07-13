package com.qualcomm.qti.snpe.imageclassifiers.detector;

import android.graphics.Bitmap;
import android.graphics.PointF;
import android.graphics.RectF;

import org.opencv.core.Mat;

import java.time.LocalDateTime;
import java.util.ArrayList;

public class Bbox implements Comparable<Bbox> {
    public float x1;
    public float y1;
    public float x2;
    public float y2;
    public Float conf;
    public PointF landmarks[];// = new PointF[5];
    //for tracking
    public boolean isTracked = false;

    //align face
    public Mat alignInput = null;
    public Bitmap alignBitmap = null;

    //Fr
    public float min_group_distance = 0.0f;
    public float min_distance = 0.0f;
    public int index = -1;
    public int personId = 0;
    public int groupId = 0;
    public String label = "Unknown";
    public String groupLabel = "Unknown";

    //Align Ratio
    public float alignRatio = 0.0f;
    public float iou = 0.0f;
    public float distanceEyesRatio = 0.0f;
    public float noseRatio = 0.0f;

    //time
    public Long lastUpdated = System.currentTimeMillis();
    public Long lastFrExecuted = System.currentTimeMillis();
    public LocalDateTime recognizedDate = LocalDateTime.now();

    //feature
    public float[] feature;

    //trueface
    public float faceProb = 0.0f;
    public float finalProb = 0.0f;

    //postIQA
    public float eyesRatio = 0.0f;
    public float eyesDistance = 0.0f;
    public float sharpness = 0.0f;

    //RPY
    public double yawDeg = 0.0f;
    public double rollDeg = 0.0f;
    public double pitchDeg = 0.0f;
    /*
    public Bbox(float x1, float y1, float x2, float y2, float conf, PointF[] landmarks) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.conf = conf;
        this.landmarks  = landmarks;
    }
     */

    public Bbox() {
        landmarks  = new PointF[5];
    }

    public Bbox(float x1, float y1, float x2, float y2, Float conf, String label) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.conf = conf;
        this.label = label;
        landmarks  = new PointF[5];

    }

    // duplicates into another Box
    public void copyTo(Bbox b){
        b.x1 = this.x1;
        b.y1 = this.y1;
        b.x2 = this.x2;
        b.y2 = this.y2;
        //b.conf = this.conf;
        b.landmarks = this.landmarks;
        //b.label = this.label;
        //b.min_distance = this.min_distance;
        //b.index = this.index;
        //b.isTracked = this.isTracked;
    }

    //public void calcAlignRatio()
    public void calcRatios()
    {
        //bbox width
        float width = this.x2 - this.x1;
        //alignRatio
        double dist1 = Math.sqrt(Math.pow(landmarks[0].x-landmarks[2].x,2) + Math.pow(landmarks[0].y-landmarks[2].y,2));
        float dx1 = Math.abs(x2 - landmarks[2].x);
        double dist2 = Math.sqrt(Math.pow(landmarks[1].x-landmarks[2].x,2) + Math.pow(landmarks[1].y-landmarks[2].y,2));
        float dx2 = Math.abs(x1 - landmarks[2].x);
        //alignRatio = (float)(Math.min(dist1, dist2) / Math.max(dist1, dist2));
        alignRatio = (float)(Math.min(dx1, dx2) / Math.max(dx1, dx2));
        //eyes
        double eyeDist = Math.sqrt(Math.pow(landmarks[0].x-landmarks[1].x,2) + Math.pow(landmarks[0].y-landmarks[1].y,2));
        distanceEyesRatio =  (float)(eyeDist/width);
        //nose
        float noseWidth = landmarks[2].x - this.x1;
        noseRatio = (float)(noseWidth/width);
    }

    // convenience function
    public static ArrayList<Bbox> createBoxes(int count) {
        final ArrayList<Bbox> boxes = new ArrayList<>();
        for (int i = 0; i < count; ++i)
            boxes.add(new Bbox());
        return boxes;
    }

    @Override
    public int compareTo(Bbox o) {
        return o.conf.compareTo(this.conf);
    }

    public String getLabel() {
        return label;
    }

    public Float getConfidence() {
        return conf;
    }

    public RectF getLocation() {
        return new RectF(x1, y1, x2, y2);
    }

    public float getFaceProb() {
        return faceProb;
    }

    public void setFaceProb(float faceProb) {
        this.faceProb = faceProb;
    }

    public float getFinalProb() {
        return finalProb;
    }

    public void setFinalProb(float finalProb) {
        this.finalProb = finalProb;
    }
}