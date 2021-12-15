package com.qualcomm.qti.snpe.imageclassifiers.thread;

import android.graphics.Bitmap;

import com.qualcomm.qti.snpe.FloatTensor;

import java.util.List;
import java.util.Map;

public class DetectorResult {
    private Bitmap frame;
    private List<float[]> outputs;
    private int frame_id_detector;
    public DetectorResult(Bitmap frame, List<float[]> outputs,int frame_id_detector){
        this.frame = frame;
        this.outputs = outputs;
        this.frame_id_detector = frame_id_detector;
    }

    public int getFrame_id_detector() {
        return frame_id_detector;
    }

    public void setFrame_id_detector(int frame_id_detector) {
        this.frame_id_detector = frame_id_detector;
    }

    public Bitmap getFrame() {
        return frame;
    }

    public void setFrame(Bitmap frame) {
        this.frame = frame;
    }

    public List<float[]> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<float[]> outputs) {
        this.outputs = outputs;
    }
}
