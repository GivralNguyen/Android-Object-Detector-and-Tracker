package com.qualcomm.qti.snpe.imageclassifiers.sortJni;

import android.graphics.Bitmap;


import com.qualcomm.qti.snpe.imageclassifiers.detector.Recognition;

import java.util.ArrayList;
import java.util.List;

public class TrackHuman {
    private int MAX_TRACK_BOXES = 10;

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

    public List<Recognition> listBbox = new ArrayList<>();
    public long lastTimeUpdated = System.currentTimeMillis();
    public int frameCount;

    public TrackHuman(TrackUtils.TrackResult trackResult){
        this.trackResult = trackResult;
        this.frameCount = 0;
    }

    public void addBbox(Recognition bbox) {
        if (listBbox.size() >= MAX_TRACK_BOXES) {
            listBbox.remove(0);
        }
        listBbox.add(bbox);
    }

    public float checkBoxIntersection(Bitmap bitmap, Recognition currBbox) {
        // Determine the (x, y) - coordinates of the intersected rectangle
        Recognition lastBbox = listBbox.get(0);
        if (lastBbox != null) {
//            Renderer.bboxRenderLastBox(new Canvas(bitmap), lastBbox);
            ;
            float xA = Math.max(lastBbox.mLocation.left, currBbox.mLocation.left);
            float yA = Math.max(lastBbox.mLocation.top, currBbox.mLocation.top);
            float xB = Math.min(lastBbox.mLocation.right, currBbox.mLocation.right);
            float yB = Math.min(lastBbox.mLocation.bottom, currBbox.mLocation.bottom);

            if (xB < xA || yB < yA) {
                return (float) 0.0;
            }
            // Compute the area of intersected rectangle
            float interArea = (xB - xA) * (yB - yA);

            // Compute the area of both the predicted and ground-truth rectangles
            float predictedArea = (currBbox.mLocation.right - currBbox.mLocation.left) *
                    (currBbox.mLocation.bottom - currBbox.mLocation.top);
            float groundTruthArea = (lastBbox.mLocation.right - lastBbox.mLocation.left) *
                    (lastBbox.mLocation.bottom - lastBbox.mLocation.top);

            // Compute the intersection over union by taking the intersection
            // area and dividing it by the sum of prediction + ground-truth
            // areas - the intersection area
            float iou = interArea / (predictedArea + groundTruthArea - interArea);
//            Log.d("LastBbox", "index: " + currBbox.trackId + " iou: " + iou);

            return iou;
        } else {
            return 1f;
        }
    }
}