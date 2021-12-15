package com.qualcomm.qti.snpe.imageclassifiers.thread;

import android.graphics.Bitmap;

import com.qualcomm.qti.snpe.FloatTensor;

import java.util.HashMap;
import java.util.Map;

public class PreprocessResult {
    private Bitmap frame;
    private float[] inputs ;
    private int frame_id_preprocess;
    public PreprocessResult(Bitmap frame, float[] inputs,int frame_id_preprocess){
        this.frame = frame;
        this.inputs = inputs;
        this.frame_id_preprocess = frame_id_preprocess;
    }

    public int getFrame_id_preprocess() {
        return frame_id_preprocess;
    }

    public void setFrame_id_preprocess(int frame_id_preprocess) {
        this.frame_id_preprocess = frame_id_preprocess;
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
