package com.qualcomm.qti.snpe.imageclassifiers.thread;

import android.graphics.Bitmap;

import com.qualcomm.qti.snpe.FloatTensor;

import java.util.HashMap;
import java.util.Map;

public class PreprocessResult {
    private Bitmap frame;
    private float[] inputs ;

    public PreprocessResult(Bitmap frame, float[] inputs){
        this.frame = frame;
        this.inputs = inputs;
    }

    public Bitmap getFrame() {
        return frame;
    }

    public void setFrame(Bitmap frame) {
        this.frame = frame;
    }

    public float[] getInputs() {
        return inputs;
    }

    public void setInputs(float[] inputs) {
        this.inputs = inputs;
    }
}
