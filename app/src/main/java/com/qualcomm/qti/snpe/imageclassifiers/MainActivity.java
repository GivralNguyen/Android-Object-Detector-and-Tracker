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

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;
import com.qualcomm.qti.snpe.imageclassifiers.detector.TFMobilenetQuantizeDetector;
import com.qualcomm.qti.snpe.imageclassifiers.detector.Recognition;
import com.qualcomm.qti.snpe.imageclassifiers.thread.DetectorThread;
import com.qualcomm.qti.snpe.imageclassifiers.thread.PostProcessThread;
import com.qualcomm.qti.snpe.imageclassifiers.thread.PreprocessThread;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MainActivity extends Activity {
    /** Declare variables **/
    private static final String LOGTAG = MainActivity.class.getSimpleName();
    private Bitmap testImageBitmap;

    /** Declare variables **/

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Loader.load(opencv_java.class);
        long loadBmpStart = System.currentTimeMillis();
        testImageBitmap = loadBmpImage(R.raw.image2);/**Load bitmap image**/
        long loadBmpTime = System.currentTimeMillis()- loadBmpStart;
        Log.d(LOGTAG,"loadBmpTime_time: "+ loadBmpTime);
        final PostProcessThread postProcessThread = new PostProcessThread(this);
        final DetectorThread detectorThread = new DetectorThread(this.getApplication(),this,postProcessThread);
        final PreprocessThread preprocessThread = new PreprocessThread(detectorThread);

        Thread t1 = new Thread(new Runnable() {
            @Override
            public void run() {
                /** Loop running detection **/
                while(true) {
                    preprocessThread.addItem(testImageBitmap);

                }
                /** Loop running detection **/
            }
        });
        t1.setName("Ai Thread load img");
        t1.start();

        Thread t2 = new Thread(new Runnable() {
            @Override
            public void run() {
                /** Loop running detection **/
                while(true) {
                    preprocessThread.run();
                }
                /** Loop running detection **/
            }
        });
        t2.setName("Ai Thread preprocess");
        t2.start();
        Thread t3 = new Thread(new Runnable() {
            @Override
            public void run() {
                /** Loop running detection **/
                while(true) {
                    detectorThread.run();
                }
                /** Loop running detection **/
            }
        });
        t3.setName("Ai Thread preprocess");
        t3.start();
        Thread t4 = new Thread(new Runnable() {
            @Override
            public void run() {
                /** Loop running detection **/
                while(true) {
                    postProcessThread.run();
                }
                /** Loop running detection **/
            }
        });
        t4.setName("Ai Thread preprocess");
        t4.start();
    }

    private Bitmap loadBmpImage(int Input){
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inScaled = false;
        Bitmap testBmp = BitmapFactory.decodeResource(getResources(),Input,o);

        return testBmp;
    }



}
