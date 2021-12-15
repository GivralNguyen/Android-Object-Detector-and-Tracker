package com.qualcomm.qti.snpe.imageclassifiers.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.util.AttributeSet;
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
        canvas.drawBitmap(bmp, 0, 0, null);
        getHolder().unlockCanvasAndPost(canvas);
    }
}
