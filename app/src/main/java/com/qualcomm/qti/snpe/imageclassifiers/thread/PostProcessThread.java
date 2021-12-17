package com.qualcomm.qti.snpe.imageclassifiers.thread;

import static org.bytedeco.javacpp.Loader.getCacheDir;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.ImageView;

import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;
import com.qualcomm.qti.snpe.imageclassifiers.view.VideoSurfaceView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import com.qualcomm.qti.snpe.imageclassifiers.sortJni.nativeObjectTracker;

public class PostProcessThread extends Thread{
    private static final String LOGTAG = PostProcessThread.class.getSimpleName();
    private static final int MAX_QUEUE_SIZE = 20;
    private static final String AI_POST_PROCESS_THREAD = "AI post process thread";

    private Context mContext;
    private VideoSurfaceView mImageView;
    private LinkedBlockingDeque<DetectorResult> PostProcessQueue;
    nativeObjectTracker tracker;


    public PostProcessThread(Context context) {
        this.mContext = context;
        this.PostProcessQueue = new LinkedBlockingDeque<>();
        this.setName(AI_POST_PROCESS_THREAD);
        tracker = new nativeObjectTracker();
    }
    @Override
    public void run() {
        while(true){
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (PostProcessQueue.size() > 0){
                try{
                    long postProcessStart = System.currentTimeMillis();
                    DetectorResult detectorResult = PostProcessQueue.takeFirst();
                    List<float[]> outputs = detectorResult.getOutputs();
                    Bitmap frame_pp = detectorResult.getFrame();
                    int postProcess_frame_id = detectorResult.getFrame_id_detector();
                    float[] outputConf = outputs.get(0);
                    float[] outputClass = outputs.get(1);
                    float[] outputBbox = outputs.get(2);
                    List<Bbox> listBox = new ArrayList<>();

                    for(int i = 0; i< outputConf.length;i++){
                        Bbox r = new Bbox( outputBbox[i*4+1], outputBbox[i*4],  outputBbox[i*4+3], outputBbox[i*4+2],outputConf[i], (int) outputClass[i]);
                        listBox.add(r);

                    }
                    tracker.prepareTrackSortFace(listBox);


//                    final Bitmap bmpcopy = frame_pp.copy(Bitmap.Config.ARGB_8888, true);
//                    Canvas canvasMerge = new Canvas(bmpcopy);
//
//                    Paint paintMerge = new Paint();
//                    //paint.setAlpha(0xA0); // the transparency
//                    paintMerge.setColor(Color.RED); // color is red
//                    paintMerge.setStyle(Paint.Style.STROKE); // stroke or fill or ...
//                    paintMerge.setStrokeWidth(1); // the stroke width
//                    for(int i = 0; i< outputConf.length;i++){
//                        Rect r = new Rect((int) outputBbox[i*4+1], (int) outputBbox[i*4], (int) outputBbox[i*4+3], (int) outputBbox[i*4+2]);
//                        canvasMerge.drawRect(r, paintMerge);
//                        canvasMerge.drawText(Float.toString(outputClass[i]),(int) outputBbox[i*4+1], (int) outputBbox[i*4+0],paintMerge );
//                    }
//                    String formatted = String.format("%06d", postProcess_frame_id);
//                    String filenameMerge = "detectresult"+ formatted;
//                    savebitmap(bmpcopy, filenameMerge);
//                    displayVideo(bmpcopy);
                    long postProcessTime = System.currentTimeMillis()- postProcessStart;
                    Log.d(LOGTAG, "postprocess: "+ postProcessTime);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    public boolean addItem(DetectorResult detectorResult) {
        if (PostProcessQueue.size() > MAX_QUEUE_SIZE){
            try {
                PostProcessQueue.takeFirst();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        return PostProcessQueue.offerLast(detectorResult);
    }

    public File savebitmap(Bitmap bmp, String filename) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 60, bytes);
        File f = new File(this.mContext.getCacheDir()
                + File.separator + filename +".jpg");
        Log.d(LOGTAG + "fpath", "file-path= " + (getCacheDir()
                + File.separator + filename +".jpg"));
        f.createNewFile();
        FileOutputStream fo = new FileOutputStream(f);
        fo.write(bytes.toByteArray());
        fo.close();
        return f;
    }

    public void displayVideo(Bitmap bmp) {
        if(mImageView != null) {
            mImageView.drawBitmap(bmp);
        }
    }

    public void setImageView(VideoSurfaceView mImageView) {
        this.mImageView = mImageView;
    }
}

