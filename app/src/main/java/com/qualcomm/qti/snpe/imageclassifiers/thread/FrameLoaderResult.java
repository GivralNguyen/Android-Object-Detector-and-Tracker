package com.qualcomm.qti.snpe.imageclassifiers.thread;

import android.graphics.Bitmap;

public class FrameLoaderResult {
    private Bitmap frame;
    private int frame_id_loader;
    public FrameLoaderResult(Bitmap frame, int frame_id_loader){
        this.frame = frame;
        this.frame_id_loader = frame_id_loader;
    }

    public Bitmap getFrame() {
        return frame;
    }

    public void setFrame(Bitmap frame) {
        this.frame = frame;
    }

    public int getFrame_id_loader() {
        return frame_id_loader;
    }

    public void setFrame_id_loader(int frame_id_loader) {
        this.frame_id_loader = frame_id_loader;
    }
}