package com.qualcomm.qti.snpe.imageclassifiers.thread;

import android.graphics.Bitmap;

import com.qualcomm.qti.snpe.FloatTensor;

import java.util.List;
import java.util.Map;

public class DetectorResult {
    private Bitmap frame;
    private List<float[]> outputs;
    public DetectorResult(Bitmap frame, List<float[]> outputs){
        this.frame = frame;
        this.outputs = outputs;
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
