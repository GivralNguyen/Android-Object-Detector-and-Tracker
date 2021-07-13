package com.qualcomm.qti.snpe.imageclassifiers.detector;

public class Anchor {
    public float cx;
    public float cy;
    public float sx;
    public float sy;

    public Anchor(float cx, float cy, float sx, float sy) {
        this.cx = cx;
        this.cy = cy;
        this.sx = sx;
        this.sy = sy;
    }

    public Anchor() { }
}
