/*
 * Copyright (c) 2016 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;
import static org.bytedeco.javacpp.Loader.getCacheDir;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;

import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;
import com.qualcomm.qti.snpe.imageclassifiers.detector.MobilenetDetector;
import com.qualcomm.qti.snpe.imageclassifiers.detector.Recognition;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {
    /** Declare variables **/
    private static final String LOGTAG = MainActivity.class.getSimpleName();
    private MobilenetDetector mDetector1;
    private Bitmap testImageBitmap;

    /** Declare variables **/

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Loader.load(opencv_java.class);
        long loadBmpStart = System.currentTimeMillis();
        testImageBitmap = loadBmpImage(R.raw.car300);/**Load bitmap image**/
        long loadBmpTime = System.currentTimeMillis()- loadBmpStart;
        Log.d(LOGTAG,"loadBmpTime_time: "+ loadBmpTime);
        mDetector1 = new MobilenetDetector(this, this.getApplication(), R.raw.mb1_ssd_sim_int8); /**load mobilenet model**/

        Thread t1 = new Thread(new Runnable() {
            @Override
            public void run() {
                /** Loop running detection **/
                while(true) {

                    try {
                        startAIFlowDetect(mDetector1, testImageBitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
                /** Loop running detection **/
            }
        });
        t1.setName("Ai Thread 1");
        t1.start();

    }

    private void startAIFlowDetect(MobilenetDetector mDetector, Bitmap bmp) throws IOException {
        List<Bbox> outputs = new ArrayList<>();
        long detectFrameStart = System.currentTimeMillis();
        outputs = mDetector.detectFrame(bmp);
        long detectFrameTime = System.currentTimeMillis()- detectFrameStart;
        Log.d(LOGTAG, "detectframe: "+ detectFrameTime);

        final Bitmap bmpcopy = bmp.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvasMerge = new Canvas(bmpcopy);

        Paint paintMerge = new Paint();
        //paint.setAlpha(0xA0); // the transparency
        paintMerge.setColor(Color.RED); // color is red
        paintMerge.setStyle(Paint.Style.STROKE); // stroke or fill or ...
        paintMerge.setStrokeWidth(1); // the stroke width

        for (Bbox mBox : outputs) {
            Rect r = new Rect((int) mBox.x1, (int) mBox.y1, (int) mBox.x2, (int) mBox.y2);
            canvasMerge.drawRect(r, paintMerge);
            canvasMerge.drawText(Integer.toString(mBox.label), mBox.x1, mBox.y1,paintMerge );
        }
        String filenameMerge = "detectresult";
        savebitmap(bmpcopy, filenameMerge);


    }

    private Bitmap loadBmpImage(int Input){
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inScaled = false;
        Bitmap testBmp = BitmapFactory.decodeResource(getResources(),Input,o);

        return testBmp;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mDetector1 != null){
            mDetector1.close();
        }

    }

    public File savebitmap(Bitmap bmp, String filename) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 60, bytes);
        File f = new File(this.getCacheDir()
                + File.separator + filename +".jpg");
        Log.d(LOGTAG + "fpath", "file-path= " + (getCacheDir()
                + File.separator + filename +".jpg"));
        f.createNewFile();
        FileOutputStream fo = new FileOutputStream(f);
        fo.write(bytes.toByteArray());
        fo.close();
        return f;
    }




}
