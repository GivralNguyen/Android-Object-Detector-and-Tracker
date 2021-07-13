/*
 * Copyright (c) 2016 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.Activity;
import android.app.FragmentTransaction;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;

import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;
import com.qualcomm.qti.snpe.imageclassifiers.detector.RetinaDetector;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.osgi.OpenCVNativeLoader;

import java.util.ArrayList;
import java.util.List;

import static com.qualcomm.qti.snpe.imageclassifiers.detector.RetinaDetector.IMG_HEIGHT;
import static com.qualcomm.qti.snpe.imageclassifiers.detector.RetinaDetector.IMG_WIDTH;

public class MainActivity extends Activity {
    private static final String LOGTAG = MainActivity.class.getSimpleName();
    private RetinaDetector mDetector1;
    private RetinaDetector mDetector2;
    private RetinaDetector mDetector3;
    private RetinaDetector mDetector4;

    private Bitmap testImageBitmap;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Loader.load(opencv_java.class);

        testImageBitmap = BitmapFactory.decodeResource(this.getResources(), R.raw.test_image);

        /** Load model & create create instance */
        mDetector1 = new RetinaDetector(this, this.getApplication(), R.raw.retina_mb_nosm_h288_w512_quantized);
        mDetector2 = new RetinaDetector(this, this.getApplication(), R.raw.retina_mb_nosm_h288_w512_quantized);
        mDetector3 = new RetinaDetector(this, this.getApplication(), R.raw.retina_mb_nosm_h288_w512_quantized);
        mDetector4 = new RetinaDetector(this, this.getApplication(), R.raw.retina_mb_nosm_h288_w512_quantized);


        new Thread(new Runnable() {
            @Override
            public void run() {
                while(true) {
                    startAIFlowDetect(mDetector1, testImageBitmap);
                    Log.d(LOGTAG +"_thread1", "thread_1 live");
                }
            }
        }).start();

        new Thread(new Runnable() {
            @Override
            public void run() {
                while(true) {
                    startAIFlowDetect(mDetector2, testImageBitmap);
                    Log.d(LOGTAG +"_thread1", "thread_2 live");
                }
            }
        }).start();

        new Thread(new Runnable() {
            @Override
            public void run() {
                while(true) {
                    startAIFlowDetect(mDetector3, testImageBitmap);
                    Log.d(LOGTAG +"_thread1", "thread_3 live");
                }
            }
        }).start();

        new Thread(new Runnable() {
            @Override
            public void run() {
                while(true) {
                    startAIFlowDetect(mDetector4, testImageBitmap);
                    Log.d(LOGTAG +"_thread1", "thread_4 live");
                }
            }
        }).start();

    }

    private void startAIFlowDetect(RetinaDetector mDetector, Bitmap bmp){
        List<Bbox> detectedBoxes = new ArrayList<>();
        /** Preprocess input image */
        final float scaleX = IMG_WIDTH / (float) bmp.getWidth();
        final float scaleY = IMG_HEIGHT / (float) bmp.getHeight();

        final Matrix scalingMatrix = new Matrix();
        scalingMatrix.postScale(scaleX, scaleY);

        final Bitmap resizedBmp = Bitmap.createBitmap(bmp,
                0, 0,
                bmp.getWidth(), bmp.getHeight(),
                scalingMatrix, false);

        detectedBoxes = mDetector.detectFrame(resizedBmp);
        Log.d(LOGTAG +"_boxResultInfo", "box count= " + detectedBoxes.size());
        for (Bbox box0 : detectedBoxes){
            Log.d(LOGTAG + "singleBoxInfo", "co-ord= " + box0.x1 + " " + box0.x2 +" " + box0.y1 + " " + box0.y2 + " | conf = " + box0.conf);
        }
    }



    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mDetector1 != null){
            mDetector1.close();
        }
        if (mDetector2 != null){
            mDetector2.close();
        }
        if (mDetector3 != null){
            mDetector3.close();
        }
        if (mDetector4 != null){
            mDetector4.close();
        }

    }
}
