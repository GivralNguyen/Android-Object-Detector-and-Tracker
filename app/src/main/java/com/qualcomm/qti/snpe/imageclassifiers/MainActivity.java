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

import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooKeeper;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.osgi.OpenCVNativeLoader;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

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

        /** Preprocess input image */
        final float scaleX = IMG_WIDTH / (float) testImageBitmap.getWidth();
        final float scaleY = IMG_HEIGHT / (float) testImageBitmap.getHeight();

        final Matrix scalingMatrix = new Matrix();
        scalingMatrix.postScale(scaleX, scaleY);

        final Bitmap resizedBmp1 = Bitmap.createBitmap(testImageBitmap,
                0, 0,
                testImageBitmap.getWidth(), testImageBitmap.getHeight(),
                scalingMatrix, false);




        /** Load model & create create instance */
        mDetector1 = new RetinaDetector(this, this.getApplication(), R.raw.retina_480x850_quantize_v11);

        Thread t1 = new Thread(new Runnable() {
            @Override
            public void run() {
                //createZooApache();

                while(true) {
                    startAIFlowDetect(mDetector1, resizedBmp1);
                    Log.d(LOGTAG +"_thread1", "thread_1 live");
                }
            }
        });



        t1.setName("Ai Thread 1");
        t1.start();

    }

    private void startAIFlowDetect(RetinaDetector mDetector, Bitmap bmp){
        List<Bbox> detectedBoxes = new ArrayList<>();


        detectedBoxes = mDetector.detectFrame(bmp);
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

    public void createZooApache() {
        System.out.println("Connecting to Zookeeper" + Thread.currentThread());
        final CountDownLatch connectedSignal = new CountDownLatch(1);
        // init zookeeper
        try {
            ZooKeeper zooKeeper = new ZooKeeper("192.168.0.221:2181", 5000, new Watcher() {
                @Override
                public void process(WatchedEvent event) {
                    System.out.println("WatchedEvent");
                    if (event.getState() == Watcher.Event.KeeperState.SyncConnected) {
                        connectedSignal.countDown();
                        System.out.println("Connected to zookeeper");
                    }
                }
            });
            connectedSignal.await();
            System.out.println("Connected!");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error on connect to zoo " + e.getMessage());
        }
    }


}
