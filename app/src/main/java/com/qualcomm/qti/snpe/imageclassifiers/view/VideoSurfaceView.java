package com.qualcomm.qti.snpe.imageclassifiers.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceView;

public class VideoSurfaceView extends SurfaceView {
    public VideoSurfaceView(Context context) {
        super(context);
    }

    public VideoSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public VideoSurfaceView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public VideoSurfaceView(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public void drawBitmap(Bitmap bmp)
    {
        Canvas canvas = getHolder().lockCanvas();
        int left = 0;
        int top = 0;
        int width = getWidth();
        int height = getHeight();
        if(bmp.getWidth() > width || bmp.getHeight() > height) {
            float scale = Math.min((float)width/bmp.getWidth(), (float)height/bmp.getHeight());
            bmp = Bitmap.createScaledBitmap(bmp, (int)(scale*bmp.getWidth()), (int)(scale*bmp.getHeight()), true);
        } else {
            left = (width - bmp.getWidth())/2;
            top = (height - bmp.getHeight())/2;
        }
        canvas.drawBitmap(bmp, left, top, null);
        getHolder().unlockCanvasAndPost(canvas);
    }
}
